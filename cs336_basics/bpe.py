from collections import Counter, defaultdict
from multiprocessing import Manager, Process, Queue
from queue import Empty
import os
from typing import BinaryIO
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = min(4, os.cpu_count() or 1)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab : dict[int, bytes] = {x : bytes([x]) for x in range(256)}
    current_index = 256
    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            vocab[current_index] = token_bytes
            current_index += 1
    return vocab

def pair_counts(word_counter: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pairs_freqs: dict[tuple[int, int], int] = {}
    for word, count in word_counter.items():
        for a, b in zip(word, word[1:]):
            pairs_freqs[(a, b)] = pairs_freqs.get((a, b), 0) + count
    return pairs_freqs

def get_most_frequent_pair(
    pair_counter: dict[tuple[int, int], int], vocab: dict[int, bytes]
) -> tuple[int, int]:
    max_freq = max(pair_counter.values())

    candidates = [
        (pair, (vocab[pair[0]], vocab[pair[1]])) for pair, freq in pair_counter.items() if freq == max_freq
    ]
    candidates.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)

    return candidates[0][0]

def add_pair_to_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    index1, index2 = pair
    vocab[len(vocab)] = vocab[index1] + vocab[index2]
    return len(vocab) - 1

def merge(indices, most_frequent_pair, new_index) -> list:
    new_indices = []
    i = 0

    while i < len(indices):
        if (
            i + 1 < len(indices)
            and indices[i] == most_frequent_pair[0]
            and indices[i + 1] == most_frequent_pair[1]
        ):
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1

    return new_indices

def split_by_special_tokens(text: str, special_tokens: list[str], include_special: bool = False) -> list[str]:
    if not special_tokens:
        return [text]

    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

    if include_special:
        special_chunks = re.split(f"({pattern})", text)
    else:
        # Split without capturing the special tokens
        special_chunks = re.split(pattern, text)

    return special_chunks


def string_to_bytes(s: str, return_int: bool = False) -> list[int] | list[bytes]:
    byte_array = s.encode("utf-8")
    return list(map(int, byte_array)) if return_int else [bytes([b]) for b in byte_array]


def pre_tokenize(string: str, special_tokens: list[str], including_special: bool = False) -> Counter:
    word_counter = Counter()

    chunks = split_by_special_tokens(string, special_tokens, include_special=including_special)

    for chunk in chunks:
        if including_special and chunk in special_tokens:
            word_counter[tuple(string_to_bytes(chunk))] += 1
        else:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                word_encoded = tuple(string_to_bytes(word, return_int=True))
                word_counter[word_encoded] += 1

    return word_counter

def pre_tokenize_string_worker(*args):
    input_path, special_tokens, queue, start, end, include_special = args

    # Read the chunk from the file
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    word_counter = pre_tokenize(chunk, special_tokens, include_special)

    # Put the result in the queue
    queue.put(word_counter)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)

    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, desired_num_chunks=kwargs.get("desired_num_chunks", NUM_PROCESSES), split_special_token=b"<|endoftext|>"
        )
    
    manager = Manager()
    queue = manager.Queue()
    processes: list[Process] = []
    
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize_string_worker,
            args=(input_path, special_tokens, queue, start, end, False),
        )
        processes.append(p)
        p.start()

    word_counter = Counter()

    for _ in range(len(processes)):
        try:
            partial_counter = queue.get(timeout=10)
            word_counter.update(partial_counter)
        except Empty:
            continue
    for p in processes:
        p.join()
    
    pairs_counter: Counter = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for word, count in word_counter.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs_counter[pair] += count
            pair_to_words[pair].add(word)

    for _ in range(num_merges):
        if not pairs_counter:
            break

        best_pair = get_most_frequent_pair(pairs_counter, vocab)

        # Record merge before modifying vocab
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        new_index = add_pair_to_vocab(vocab, best_pair)

        # Incremental update: only process words that contain the merged pair
        affected_words = list(pair_to_words.get(best_pair, set()))

        for word in affected_words:
            count = word_counter.get(word, 0)
            if count == 0:
                continue

            # Build new word by replacing best_pair with new_index
            new_word_list = []
            i = 0
            L = len(word)
            while i < L:
                if i + 1 < L and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word_list.append(new_index)
                    i += 2
                else:
                    new_word_list.append(word[i])
                    i += 1
            new_word = tuple(new_word_list)

            # Aggregate old pair counts (handle duplicate pairs in same word)
            old_delta: Counter = Counter()
            for a, b in zip(word, word[1:]):
                old_delta[(a, b)] += count

            new_delta: Counter = Counter()
            for a, b in zip(new_word, new_word[1:]):
                new_delta[(a, b)] += count

            # Subtract old pair contributions
            for pair_key, delta in old_delta.items():
                current = pairs_counter.get(pair_key, 0) - delta
                if current <= 0:
                    pairs_counter.pop(pair_key, None)
                else:
                    pairs_counter[pair_key] = current
                pair_to_words[pair_key].discard(word)
                if not pair_to_words[pair_key]:
                    pair_to_words.pop(pair_key, None)

            # Add new pair contributions
            for pair_key, delta in new_delta.items():
                pairs_counter[pair_key] = pairs_counter.get(pair_key, 0) + delta
                pair_to_words.setdefault(pair_key, set()).add(new_word)

            # Update word counter
            del word_counter[word]
            word_counter[new_word] = word_counter.get(new_word, 0) + count

        # Ensure merged pair is fully removed
        pairs_counter.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)

    return vocab, merges
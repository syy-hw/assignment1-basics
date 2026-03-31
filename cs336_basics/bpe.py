from collections import Counter
import os
import regex as re


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

def get_most_frequent_pair(pairs_freqs: dict[tuple[int, int], int]) -> tuple[int, int]:
    max_freq = max(pairs_freqs.values())
    candidates = [pair for pair, freq in pairs_freqs.items() if freq == max_freq]
    res = max(candidates)
    return res

def add_pair_to_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    index1, index2 = pair
    vocab[len(vocab)] = vocab[index1] + vocab[index2]
    return len(vocab) - 1

def merge_pair_ids(
    word_counter: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_index: int
)-> tuple[dict[tuple[int, ...], int], dict[tuple[int, int], int]]:
    new_word_counter: dict[tuple[int, ...], int] = {}
    new_pairs_freqs: dict[tuple[int, int], int] = {}
    for word, count in word_counter.items():
        new_word = []
        i = 0
        L = len(word)
        while i < L:
            if i + 1 < L and word[i:i+2] == pair:
                new_word.append(new_index)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_counter[tuple(new_word)] = count
        for a, b in zip(new_word, new_word[1:]):
            new_pairs_freqs[(a, b)] = new_pairs_freqs.get((a, b), 0) + count
    return new_word_counter, new_pairs_freqs

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


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)

    merges: dict[tuple[int, int], int] = {}

    word_counter = pre_tokenize(string, special_tokens, including_special=False)

    pairs_freqs = pair_counts(word_counter)

    for _ in range(num_merges):
        most_common_pair = get_most_frequent_pair(pairs_freqs)
        new_index = add_pair_to_vocab(vocab, most_common_pair)
        merges[most_common_pair] = new_index
        word_counter, pairs_freqs = merge_pair_ids(word_counter, most_common_pair, new_index)
    
    return vocab, merges
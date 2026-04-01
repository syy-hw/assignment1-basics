from bpe import train_bpe

if __name__ == '__main__':
    vocab, merges = train_bpe(
        input_path='data/TinyStoriesV2-GPT4-valid.txt',
        vocab_size=500,
        special_tokens=['<|endoftext|>'],
    )
    print(vocab)
    print(merges)
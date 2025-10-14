from datasets import load_from_disk
from transformers import AutoTokenizer


def load_tokenized_dataset(tokenizer_path, data_path, max_length=1024):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def tokenize(example):
        return tokenizer(
            example["input"] + example["output"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )

    dataset = load_from_disk(data_path)
    return dataset.map(tokenize)

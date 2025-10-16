from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH, SEED
import random

def load_and_prepare(tokenizer_name=MODEL_NAME):
    """
    Returns:
      - train_all (shuffled)
      - test dataset
      - tokenizer
      - tokenize_fn (callable for datasets.map)
    """
    raw = load_dataset("stanfordnlp/imdb")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # deterministic shuffle for reproducible subset selection
    train_all = raw["train"].shuffle(seed=SEED)
    test = raw["test"]

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    return train_all, test, tokenizer, tokenize_fn

def get_train_subset(train_all, size):
    # select first `size` examples deterministically after shuffle
    if size >= len(train_all):
        return train_all
    return train_all.select(range(size))

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

def load_data(batch_size=16):
    dataset = load_dataset("imdb")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size)

    return train_loader, test_loader, tokenizer
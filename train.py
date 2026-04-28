import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from data_loader import load_data
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    train_loader, test_loader, tokenizer = load_data()
    model = get_model().to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(2):
        model.train()
        total_loss = 0
        preds, labels = [], []

        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label
            )

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy())

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch} | Loss: {total_loss:.3f} | Accuracy: {acc:.3f}")

    model.save_pretrained("../models/bert_model")
    tokenizer.save_pretrained("../models/bert_model")

if __name__ == "__main__":
    train()
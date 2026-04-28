import torch
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_data
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    _, test_loader, _ = load_data()
    model = get_model()
    model.load_state_dict(torch.load("../models/bert_model/pytorch_model.bin"))
    model.to(device)
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy())

    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds))

if __name__ == "__main__":
    evaluate()
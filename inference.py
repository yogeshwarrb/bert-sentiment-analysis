import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained("../models/bert_model").to(device)
tokenizer = BertTokenizer.from_pretrained("../models/bert_model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

    return "Positive" if pred == 1 else "Negative"

if __name__ == "__main__":
    text = input("Enter text: ")
    print(predict(text))
# BERT Fine-Tuning for Sentiment Analysis

## 📌 Overview
This project fine-tunes a pretrained BERT model (bert-base-uncased) for binary sentiment classification using the IMDb dataset.

## 📊 Dataset
- Source: Hugging Face IMDb dataset
- 50,000 movie reviews labeled as positive or negative

## 🧠 Model
- Pretrained BERT (bert-base-uncased)
- Fine-tuned for classification

## ⚙️ Training
- Optimizer: AdamW
- Learning Rate: 2e-5
- Epochs: 2

## 📈 Results
- Accuracy: ~88-92% (varies)
- Evaluation Metrics:
  - Precision
  - Recall
  - F1-score

## 📊 Evaluation
- Confusion Matrix
- Classification Report

### Install dependencies
```bash
pip install -r requirements.txt

python src/train.py

python src/evaluate.py

Input:
This movie was amazing!

Output:
Positive

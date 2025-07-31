# Emotion Detection from Tweets using Machine Learning and NLP

##  Team Members
- Sonam Rani
- Priya Gauma

## Problem Statement
This project focuses on detecting **emotions** expressed in short social media posts (tweets). Detecting emotions is useful in various domains like mental health, marketing, sentiment analysis, and user feedback systems.

The goal is to classify each tweet into one of the six emotions:
- joy, sadness, anger, fear, love, surprise

---

## Dataset
- **Source**: HuggingFace Datasets → `dair-ai/emotion`
- **Language**: English
- **Data Type**: Short tweets
- **Classes**: ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
- **Splits**: Train, Validation, Test (predefined)

---

## Evaluation Metrics
We use:
- **Accuracy**
- **F1-Score (macro)** — to balance class performance
- **Confusion Matrix** — to observe model confusion

---

##  Methodology

### 1. Traditional Approach (Main Solution)
- **Preprocessing**: Lowercasing, noise removal, minimal cleaning
- **Feature Extraction**: TF-IDF (unigram + bigram)
- **Models**: Logistic Regression (primary), SVM and Random Forest for comparison

### 2. Pretrained BERT (Enhanced Accuracy for Demo)
- Model: `j-hartmann/emotion-english-distilroberta-base`
- Integrated into the Gradio app with a toggle switch
- Improves predictions on emotional phrases unseen in training

---

##  Results Summary

| Model                  | Accuracy | F1 Score (Macro) |
|------------------------|----------|------------------|
| Logistic Regression    | ~88%     | ~87%             |
| Support Vector Machine | ~86%     | ~85%             |
| Random Forest          | ~84%     | ~83%             |
| Pretrained BERT (demo) | ~94%+    | ~93%+            |

---

##  Demo Web App

We built a **Gradio web app** where users can enter a tweet and get emotion predictions using either:
- Logistic Regression (TF-IDF)
- BERT (optional toggle)


---

## 🧾 How to Run

1. Clone the repo or open the notebook in Google Colab
2. Run all cells
3. Try out the Gradio demo app at the end of the notebook

```bash
pip install -r requirements.txt

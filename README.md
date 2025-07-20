

---

# 📊 Sentiment Analysis using BERT (Transformers)

This project demonstrates **Sentiment Analysis** on app review text data using **BERT (Bidirectional Encoder Representations from Transformers)**. We fine-tune the pre-trained BERT model from Hugging Face to classify user reviews into **Positive**, **Neutral**, or **Negative** sentiments.

---

## 🔍 Problem Statement

Given a set of app reviews and their associated rating scores, the goal is to:

* Preprocess and map numeric ratings to sentiment classes.
* Tokenize the review texts using BERT Tokenizer.
* Fine-tune a BERT-based model for multi-class sentiment classification.
* Evaluate model performance and make predictions on new text inputs.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Transformers (Hugging Face)
* Scikit-learn
* Pandas, NumPy, Seaborn, Matplotlib
* Google Colab (runtime environment)

---

## 📁 Dataset

We use a `reviews.csv` file with the following columns:

* `content`: The review text
* `score`: The star rating (1 to 5)

Other metadata columns are also available but are not directly used in training.

---

## 📘 Project Structure

```bash
.
├── reviews.csv                 # Input dataset (example format shown in notebook)
├── best_model.bin             # Saved best model (after training)
├── SentimentAnalysisBERT.ipynb # Jupyter notebook/Colab script
├── README.md                  # Project documentation
```

---

---

## 🚀 Features

* Text preprocessing using `BERT tokenizer`
* Sentiment label creation (0 = Negative, 1 = Neutral, 2 = Positive)
* PyTorch `Dataset` and `DataLoader`
* BERT-based sentiment classifier with dropout regularization
* Training with `AdamW`, learning rate scheduler, gradient clipping
* Evaluation on validation/test set
* Inference on new user text input with confidence scores

---
## 🧠 Technologies Used

* Python 3.x
* PyTorch
* Hugging Face Transformers
* scikit-learn
* pandas, numpy, seaborn, matplotlib
* Google Colab for training

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/your-username/sentiment-analysis-bert.git
cd sentiment-analysis-bert
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install transformers torch scikit-learn pandas matplotlib seaborn
```

---
## 🧾 Step-by-Step Guide

### ✅ Step 1: Install and Import Required Libraries

Install packages:

```bash
!pip install transformers torch seaborn scikit-learn --quiet
```

Import essential libraries for modeling, evaluation, and visualization.

---

### ✅ Step 2: Load Dataset

```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('reviews.csv')
```

---

### ✅ Step 3: Preprocessing

* Remove missing values.
* Map ratings to sentiment classes:

  * 1–2 → Negative (0)
  * 3   → Neutral (1)
  * 4–5 → Positive (2)
* Plot sentiment class distribution.

---

### ✅ Step 4: Tokenization using BERT

* Load `bert-base-cased` tokenizer.
* Visualize token length distribution.
* Set `MAX_LEN = 160` based on token distribution.

---

### ✅ Step 5: Create Dataset and DataLoader

Create a custom `Dataset` class and data loaders:

```python
ReviewDataset(Dataset)
```

* Split data into train, validation, and test sets.
* Batch size: 16

---

### ✅ Step 6: Build BERT-based Sentiment Classifier

Define a model using `BertModel` with:

* Dropout
* Fully connected output layer
* Number of classes: 3

---

### ✅ Step 7: Define Optimizer, Scheduler, and Loss Function

* Optimizer: `AdamW`
* Learning rate: `2e-5`
* Scheduler: `get_linear_schedule_with_warmup`
* Loss function: `CrossEntropyLoss`

---

### ✅ Step 8: Training and Evaluation Functions

Two core functions:

* `train_epoch`: Forward, backward pass, and optimization.
* `eval_model`: Evaluation on validation or test sets.

Use gradient clipping (`max_norm=1.0`) to avoid exploding gradients.

---

### ✅ Step 9: Train the Model

Train for 4 epochs and track accuracy/loss:

```python
torch.save(model.state_dict(), 'best_model.bin')
```

Store the best-performing model based on validation accuracy.
---

## 📈 Training Performance

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Val Loss |
| ----- | -------------- | ------------------- | ---------- | -------- |
| 1     | 73.10%         | 75.90%              | 0.68       | 0.59     |
| 2     | 80.24%         | 76.54%              | 0.50       | 0.61     |
| 3     | 86.28%         | 75.66%              | 0.37       | 0.69     |
| 4     | 90.12%         | 75.42%              | 0.27       | 0.82     |

---

### ✅ Step 10: Evaluate the Model on Test Data

Load the saved model:

```python
model.load_state_dict(torch.load('best_model.bin'))
```

Evaluate using:

* Accuracy
* CrossEntropy loss
* Confusion matrix (optional)
* Classification report (optional)

---

### ✅ Step 11: Inference on New Text

Use `predict_sentiment()` to predict sentiment for any new text. Outputs:

* Sentiment class
* Confidence score
* Probability for each class

---

## 💻 Sample Output

```plaintext
Text: I love this product! It's amazing and works perfectly!
Predicted: Positive (Confidence: 0.99)
Probabilities: {'negative': 0.0048, 'neutral': 0.0071, 'positive': 0.9880}
```

---

## 📈 Performance Summary

| Metric         | Value   |
| -------------- | ------- |
| Train Accuracy | \~90.1% |
| Val Accuracy   | \~76.5% |
| Test Accuracy  | \~74.7% |

---

## 📦 Requirements

* Python >= 3.6
* PyTorch
* Hugging Face Transformers
* scikit-learn
* seaborn
* pandas, numpy
* matplotlib

---

## 📚 References

* [BERT Paper (Devlin et al.)](https://arxiv.org/abs/1810.04805)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch Documentation](https://pytorch.org/)

---

## 🧠 Future Improvements

* Implement multi-lingual BERT (mBERT) for multi-language reviews.
* Hyperparameter tuning (batch size, learning rate).
* Add text cleaning (stop words, emojis, etc.)
* Use model explainability tools (e.g., SHAP, LIME)

---

## 🤝 Contributing

If you’d like to contribute, feel free to fork the repo and submit a pull request. Suggestions, issues, and improvements are welcome!

---
## 📬 Contact

**Author:** Jitendra Kumar Gupta
**Email:** [jitendraguptaaur@gmail.com](mailto:jitendraguptaaur@gmail.com)
**LinkedIn:** [jitendra-gupta-iitk](https://www.linkedin.com/in/jitendra-kumar-30a78216a/)

---



---

# 📊 Sentiment Analysis Using BERT (Bidirectional Encoder Representations from Transformers)

This project demonstrates how to build a **Sentiment Analysis** model using **BERT** from Hugging Face Transformers. The model is trained on **app reviews** data to classify sentiments as **Negative**, **Neutral**, or **Positive**.

---

## 📁 Project Structure

```
.
├── reviews.csv                # Raw dataset
├── sentiment_analysis_bert.ipynb  # Jupyter Notebook implementation
├── best_model.bin            # Best performing model weights
├── README.md                 # Project documentation
```

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

## 📘 Step-by-Step Guide

### ✅ Step 1: Import Libraries

Import required libraries and set the random seed for reproducibility. Use GPU if available.

---

### 📂 Step 2: Load Dataset

Upload or load `reviews.csv` dataset which contains user reviews from an app store.

Essential columns:

* `content`: Review text
* `score`: Rating (1 to 5)

---

### 🧹 Step 3: Preprocess Data

* Drop missing values
* Map ratings to sentiment labels:

  * 1–2 → Negative (0)
  * 3 → Neutral (1)
  * 4–5 → Positive (2)
* Visualize class distribution using seaborn

---

### 🧾 Step 4: Tokenization

Use Hugging Face `bert-base-cased` tokenizer:

* Encode each review
* Visualize token length distribution
* Set `MAX_LEN = 160` for consistency

---

### 📦 Step 5: Dataset & DataLoader

* Define a custom `ReviewDataset` class
* Create DataLoaders for train, validation, and test using `torch.utils.data.DataLoader`

---

### 🏗️ Step 6: Build the BERT Classifier

Define a `SentimentClassifier` model:

* BERT base encoder
* Dropout
* Fully connected layer for classification

---

### ⚙️ Step 7: Optimizer, Scheduler, and Loss

* Optimizer: `AdamW`
* Scheduler: `get_linear_schedule_with_warmup`
* Loss: `CrossEntropyLoss`

---

### 🔁 Step 8: Training & Evaluation Functions

Define `train_epoch()` and `eval_model()` functions:

* Handles forward pass, backward pass, optimizer step
* Uses gradient clipping and learning rate scheduling

---

### 📊 Step 9: Train the Model

Train the model for 4 epochs:

* Save the best model (with highest validation accuracy) as `best_model.bin`

---

### 🧪 Step 10: Evaluate the Model

Evaluate model performance on the **test set** using:

* Accuracy
* Loss
* Confusion matrix / classification report (can be added optionally)

---

### 🔮 Step 11: Predict Sentiment on New Text

Predict sentiment on unseen user input using:

```python
predict_sentiment("Your input text", model, tokenizer)
```

Returns:

* Sentiment label
* Confidence score
* Class-wise probabilities

---

## 📈 Sample Output

```text
Text: I love this product! It's amazing and works perfectly!
Predicted: Positive (Confidence: 0.99)
Probabilities: {'Negative': 0.0048, 'Neutral': 0.0071, 'Positive': 0.9880}
```

---

## 📌 Notes

* You can increase the number of epochs or experiment with other BERT variants like `bert-base-uncased`, `distilbert-base-uncased` for speed.
* Fine-tuning on more balanced and larger datasets improves model performance.

---

## 📚 References

* [BERT Paper (Devlin et al.)](https://arxiv.org/abs/1810.04805)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## 🙌 Acknowledgments

Thanks to the open-source community for enabling powerful NLP tools. Built with ❤️ using Hugging Face and PyTorch.

---

## 📬 Contact

**Author:** Jitendra Kumar Gupta
**Email:** [jitendraguptaaur@gmail.com](mailto:jitendraguptaaur@gmail.com)
**LinkedIn:** [jitendra-gupta-iitk](https://www.linkedin.com/in/jitendra-kumar-30a78216a/)

---




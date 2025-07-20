

---

```markdown
# 🧠 Sentiment Analysis using BERT (Bidirectional Encoder Representations from Transformers)

This project demonstrates how to perform **Sentiment Analysis** on Google Play Store reviews using the **BERT transformer model** (from Hugging Face). The model classifies reviews into **Positive**, **Neutral**, or **Negative** sentiments based on the review content.

---

## 📌 Project Highlights

- Preprocessing real-world app review data
- Tokenization using BERT tokenizer
- Fine-tuning `bert-base-cased` for multi-class sentiment classification
- PyTorch-based model training and evaluation
- Predicting sentiment on new, unseen text

---

## 📁 Dataset

The dataset used is a CSV file (`reviews.csv`) containing app reviews from the Google Play Store, with the following relevant columns:

- `content`: The review text
- `score`: Review score (1 to 5)

Sentiment labels are created as:
- **1-2** → Negative (Label 0)
- **3**   → Neutral (Label 1)
- **4-5** → Positive (Label 2)

---

## 🧩 Project Structure

```

📦 Sentiment-BERT
├── reviews.csv              # Input dataset
├── sentiment\_analysis.ipynb # Main training and prediction notebook
├── best\_model.bin           # Saved best model weights
├── README.md                # Project documentation

````

---

## ⚙️ Installation

Make sure you have Python 3.7+ installed. Use the following commands to install the required packages.

```bash
pip install torch transformers scikit-learn seaborn matplotlib pandas
````

---

## 🧠 Model Architecture

* **Base Model**: `bert-base-cased`
* **Final Layers**: Dropout (0.3) → Linear Layer (3 output classes)
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: AdamW with learning rate 2e-5
* **Learning Scheduler**: Linear warmup

---

## 🚀 Step-by-Step Workflow

### ✅ Step 1: Import Libraries & Set Configs

Set up necessary imports, configurations, and randomness seeds for reproducibility.

### 📥 Step 2: Load Dataset

Upload and load the `reviews.csv` file, then explore the data.

### 🧹 Step 3: Preprocess Data

Clean and label the data based on the score column to form sentiment categories.

### 🔡 Step 4: Tokenization

Use `BertTokenizer` to tokenize text. Token length distribution is visualized to decide `MAX_LEN`.

### 📦 Step 5: Create Dataset & DataLoaders

Define a custom PyTorch `Dataset` and create train, validation, and test `DataLoaders`.

### 🧱 Step 6: Build Model

Define a custom `SentimentClassifier` using pretrained `BertModel`.

### 🔧 Step 7: Optimizer, Scheduler, and Loss

Set up the optimizer (`AdamW`), learning rate scheduler, and cross-entropy loss.

### 🔁 Step 8: Training and Evaluation

Implement functions for training and evaluating the model, with model saving on best validation accuracy.

### 🎯 Step 9: Train the Model

Train for 4 epochs and log accuracy and loss on training and validation sets.

### 📊 Step 10: Evaluate on Test Set

Load the best model and evaluate its performance on the test data.

### 💬 Step 11: Predict on New Text

Define a function to predict sentiment from any new sentence input.

---

## 📈 Training Performance

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Val Loss |
| ----- | -------------- | ------------------- | ---------- | -------- |
| 1     | 73.10%         | 75.90%              | 0.68       | 0.59     |
| 2     | 80.24%         | 76.54%              | 0.50       | 0.61     |
| 3     | 86.28%         | 75.66%              | 0.37       | 0.69     |
| 4     | 90.12%         | 75.42%              | 0.27       | 0.82     |

---

## 📌 Sample Predictions

```python
"This product is absolutely terrible! I hate it!"
→ Predicted: Negative (Confidence: 0.99)

"The item was okay, nothing special."
→ Predicted: Negative (Confidence: 0.93)

"I love this product! It's amazing and works perfectly!"
→ Predicted: Positive (Confidence: 0.99)
```

---

## 🧪 Model Evaluation

```python
Test Accuracy: 74.72%
```

---

## 🛠️ Tools & Libraries

* [Transformers (Hugging Face)](https://huggingface.co/transformers/)
* [PyTorch](https://pytorch.org/)
* [scikit-learn](https://scikit-learn.org/)
* [seaborn](https://seaborn.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)

---

## 📎 Future Improvements

* Handle class imbalance via class weights or oversampling
* Use more advanced models like `RoBERTa`, `DistilBERT`
* Add hyperparameter tuning
* Integrate Flask for a web-based prediction interface

---

## 🤝 Contribution

Feel free to fork the repo, raise issues or submit pull requests. Contributions are welcome!

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Jitendra Kumar Gupta**
🔗 [LinkedIn](https://www.linkedin.com/in/jitendraguptaiitk/) | 
✉️ [jitendraguptaaur@gmail.com](mailto:jitendraguptaaur@gmail.com)

---

```

```

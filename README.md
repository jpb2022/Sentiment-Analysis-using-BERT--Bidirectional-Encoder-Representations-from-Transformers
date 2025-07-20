# 🚀 BERT-Powered Sentiment Analysis for App Reviews

## 🔍 Overview
This project implements a state-of-the-art sentiment analysis system using BERT (Bidirectional Encoder Representations from Transformers) to classify app reviews into three sentiment categories: Negative, Neutral, and Positive. The model achieves **74.72% accuracy** on real-world app review data.

## 🌟 Key Features
- **BERT-based architecture** leveraging transformer models
- **Custom sentiment mapping** (1-2⭐ → Negative, 3⭐ → Neutral, 4-5⭐ → Positive)
- **Dynamic text processing** with intelligent tokenization
- **Comprehensive training pipeline** with learning rate scheduling
- **Production-ready inference** with confidence scoring

## 🛠️ Technical Stack
| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| NLP Model | BERT-base-cased |
| Tokenizer | HuggingFace Transformers |
| Utilities | pandas, numpy, scikit-learn |
| Visualization | matplotlib, seaborn |

## 📂 Repository Structure
```
.
├── data/
│   └── reviews.csv                # Raw review dataset
├── models/
│   └── best_model.bin             # Trained model weights
├── notebooks/
│   └── sentiment_analysis.ipynb   # Complete implementation
├── requirements.txt               # Dependency list
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- GPU recommended for training

### Installation
```bash
git clone https://github.com/your-username/sentiment-analysis-bert.git
cd sentiment-analysis-bert
pip install -r requirements.txt
```

## 🧠 Model Architecture
```python
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.3)  # Regularization
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
```

## ⚙️ Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 4 |
| Max Sequence Length | 160 tokens |
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss |

## 📊 Performance Metrics
| Epoch | Train Accuracy | Validation Accuracy |
|-------|----------------|---------------------|
| 1 | 73.10% | 75.90% |
| 2 | 80.24% | 76.54% |
| 3 | 86.28% | 75.66% |
| 4 | 90.12% | 75.42% |

**Final Test Accuracy: 74.72%**

## 💡 Usage Examples

### Making Predictions
```python
sample_text = "This app has completely transformed my productivity!"
result = predict_sentiment(sample_text, model, tokenizer)

print(f"Text: {result['text']}")
print(f"Predicted: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
print("Probabilities:", result['probabilities'])
```

### Expected Output
```
Text: This app has completely transformed my productivity!
Predicted: Positive (Confidence: 0.97)
Probabilities: {'Negative': 0.012, 'Neutral': 0.018, 'Positive': 0.970}
```

## 🛠️ Customization Options
1. **Model Variants**: Try `bert-base-uncased` or `distilbert` for faster inference
2. **Hyperparameters**: Adjust batch size, learning rate, or dropout probability
3. **Data Augmentation**: Incorporate back-translation for small datasets

## 📚 Learn More
- [Original BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers Documentation](https://huggingface.co/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## 🤝 Contributing
We welcome contributions! Please open an issue or submit a pull request for:
- Bug fixes
- Performance improvements
- Additional features

## 📬 Contact
**Project Maintainer**: Jitendra Kumar Gupta  
**Email**: [jitendraguptaaur@gmail.com](mailto:jitendraguptaaur@gmail.com)  
**LinkedIn**: [Connect with me](https://www.linkedin.com/in/jitendra-kumar-30a78216a/)

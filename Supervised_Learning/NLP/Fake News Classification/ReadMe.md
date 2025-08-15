# 📰 Fake News Classification using Natural Language Processing (NLP)

## Overview

This project involves building a machine learning model to detect fake news articles using Natural Language Processing (NLP) techniques. It utilizes a labeled dataset of real and fake news articles to train a classification model that can effectively distinguish between truthful and false news content.

---

## Dataset Description

- **Dataset Source**: [Kaggle - Fake News Classification](https://www.kaggle.com/datasets/aadyasingh55/fake-news-classification)
- **Total Records**: 45,000+
- **Languages**: English
- **Target Variable**:
  - `label = 1`: Real News
  - `label = 0`: Fake News

### Features
| Column | Description                      |
|--------|----------------------------------|
| `id`   | Unique identifier for the article |
| `title` | Title of the article             |
| `text` | Content of the news article       |
| `label` | 1 = True, 0 = Fake               |

### Data Splits
| Split       | Records |
|-------------|---------|
| Training    | 24,353  |
| Validation  | 8,117   |
| Test        | 8,117   |

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Label distribution visualization
- Article length statistics
- Sample article inspection

### 2. Data Preprocessing
- Duplicate and null value removal
- Text normalization: lowercase conversion, punctuation removal
- Tokenization
- Stopwords removal
- Stemming using `PorterStemmer`

### 3. Feature Extraction
- TF-IDF Vectorization using `TfidfVectorizer`

### 4. Model Building
- Logistic Regression Model
- Performance evaluated on both **validation** and **test** datasets

---

## Classification Reports

### Test Data
| Metric      | Class 0 (Fake) | Class 1 (Real) |
|-------------|----------------|----------------|
| Precision   | 0.97           | 0.97           |
| Recall      | 0.97           | 0.97           |
| F1-Score    | 0.97           | 0.97           |
| Accuracy    | **0.97**       |                |

### Validation Data
| Metric      | Class 0 (Fake) | Class 1 (Real) |
|-------------|----------------|----------------|
| Precision   | 0.96           | 0.97           |
| Recall      | 0.96           | 0.97           |
| F1-Score    | 0.96           | 0.97           |
| Accuracy    | **0.96**       |                |

---

## Performance Visualization

The model's performance metrics were visualized using bar plots for precision, recall, and F1-score for each class, providing intuitive insights into the classifier's behavior.

---

## Tools and Technologies

- **Languages**: Python
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `nltk` (for NLP preprocessing)
  - `scikit-learn` (for model building and evaluation)


---

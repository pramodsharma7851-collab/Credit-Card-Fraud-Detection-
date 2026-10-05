# Credit Card Fraud Detection using Anomaly Detection

## 📌 Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent only a very small proportion of all transactions. This project focuses on identifying fraudulent credit card transactions using **unsupervised anomaly detection techniques**.

The project evaluates three anomaly detection algorithms:

- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **One-Class Support Vector Machine (One-Class SVM)**

The models are compared using accuracy, precision, recall, F1-score, and the number of classification errors, with particular attention given to the performance on the minority **fraud class**.

---

## 📊 Dataset

The dataset contains anonymized credit card transactions made by European cardholders during September 2013.

### Dataset Statistics

| Description | Value |
|---|---:|
| Total transactions | **116,705** |
| Fraudulent transactions | **244** |
| Legitimate transactions | **116,461** |
| Fraud percentage | **~0.21%** |

The dataset is therefore **highly imbalanced**, meaning that legitimate transactions greatly outnumber fraudulent transactions.

### Features

The dataset contains:

- `V1` to `V28` — anonymized numerical features obtained through **Principal Component Analysis (PCA)**
- `Time` — seconds elapsed between the transaction and the first transaction in the dataset
- `Amount` — transaction amount
- `Class` — target variable:
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction

Due to confidentiality concerns, the original feature descriptions are not available.

---

## 🎯 Problem Statement

The objective of this project is to identify potentially fraudulent transactions by treating fraudulent transactions as **anomalies or outliers**.

Because the fraud class represents only around 0.21% of the complete dataset, conventional accuracy alone is not sufficient for evaluating a fraud detection system. Therefore, fraud-class **precision, recall, and F1-score** are also considered.

---

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand the structure and distribution of the transaction data.

The analysis includes:

- Checking the dataset structure and data types
- Checking for missing values
- Examining the distribution of legitimate and fraudulent transactions
- Comparing transaction amounts between fraud and legitimate transactions
- Analysing transaction time and amount
- Examining correlations between features using a correlation heatmap

The analysis highlights the severe class imbalance present in the dataset.

---

## ⚙️ Data Preparation

For computational efficiency, the notebook works with a **10% random sample** of the dataset.

For the new dataset:

- Full dataset: **116,705 transactions**
- Evaluation sample: **11,670 transactions**
- Legitimate transactions in evaluation sample: **11,653**
- Fraudulent transactions in evaluation sample: **17**

The `Class` column is separated as the target variable, while the remaining columns are used as input features.

---

# 🤖 Models Used

## 1. Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm based on the idea that anomalies are:

- Few in number
- Different from normal observations
- Easier to isolate from the rest of the data

The algorithm creates isolation trees by randomly selecting features and split values. Anomalous observations generally require fewer splits to be isolated.

### Configuration

```python
IsolationForest(
    n_estimators=100,
    max_samples=len(X),
    contamination=outlier_fraction,
    random_state=state
)
```

---

## 2. Local Outlier Factor (LOF)

Local Outlier Factor is an unsupervised outlier detection technique that compares the **local density** of a data point with the density of its neighbouring observations.

An observation is considered an outlier when its local density is substantially lower than that of its neighbours.

### Configuration

```python
LocalOutlierFactor(
    n_neighbors=20,
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    p=2,
    contamination=outlier_fraction
)
```

---

## 3. One-Class SVM

One-Class Support Vector Machine is an unsupervised anomaly detection method that learns a boundary around the normal observations and identifies observations outside that boundary as anomalies.

### Configuration

```python
OneClassSVM(
    kernel='rbf',
    degree=3,
    gamma=0.1,
    nu=0.05,
    max_iter=-1
)
```

---

# 📈 Model Evaluation

The models were evaluated on the sampled dataset containing **11,670 transactions**, including **17 fraudulent transactions**.

### Overall Results

| Model | Accuracy | Errors |
|---|---:|---:|
| **Isolation Forest** | **99.82%** | **21** |
| **Local Outlier Factor** | **99.70%** | **35** |
| **One-Class SVM** | **63.92%** | **4,211** |

### Fraud-Class Performance

| Model | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| **Isolation Forest** | **39%** | **41%** | **40%** |
| **Local Outlier Factor** | **0%** | **0%** | **0%** |
| **One-Class SVM** | **0%** | **29%** | **0%** |

---

## 🏆 Key Findings

### Isolation Forest

Isolation Forest produced the best overall results among the three evaluated models.

It achieved:

- **99.82% overall accuracy**
- **21 classification errors**
- **39% fraud precision**
- **41% fraud recall**
- **40% fraud F1-score**

The **41% recall for the fraud class** means that the model successfully identified approximately 41% of the actual fraudulent transactions in the evaluation sample.

### Local Outlier Factor

LOF achieved a high overall accuracy of **99.70%**, but this result is misleading when considered alone because of the severe class imbalance.

Its fraud-class performance was:

- Precision: **0%**
- Recall: **0%**
- F1-score: **0%**

This indicates that LOF was unable to effectively identify the fraudulent transactions in this evaluation.

### One-Class SVM

One-Class SVM achieved:

- **63.92% overall accuracy**
- **4,211 errors**
- **29% fraud recall**

Although its fraud recall was higher than LOF's, its overall performance and fraud precision were poor.

---

## ⚠️ Why Accuracy Alone Is Not Enough

This dataset is highly imbalanced. Only approximately **0.21% of the complete dataset consists of fraudulent transactions**.

Therefore, a model could achieve very high accuracy simply by predicting most transactions as legitimate.

For fraud detection, metrics such as:

- **Precision**
- **Recall**
- **F1-score**
- **Precision-Recall AUC (AUPRC)**

are more informative than accuracy alone.

In this project, particular attention is given to **fraud recall**, because failing to identify an actual fraudulent transaction can be more important than correctly classifying an already legitimate transaction.

---

## 💡 Future Improvements

The current results can potentially be improved through:

- Hyperparameter tuning
- Increasing the amount of data used for training/evaluation
- Experimenting with additional anomaly detection algorithms
- Using advanced machine learning techniques
- Exploring deep-learning-based anomaly detection
- Optimizing the decision threshold
- Using precision-recall curves and **AUPRC** for model selection
- Applying appropriate techniques for highly imbalanced datasets

More complex models may provide better fraud detection performance, but they can also require greater computational resources.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — machine learning and evaluation
- **Jupyter Notebook** — development environment

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Anamoly Detection (1).ipynb
├── creditcard.csv
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Credit-Card-Fraud-Detection
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

### 4. Open the notebook

```bash
jupyter notebook
```

Open:

```text
Anamoly Detection (1).ipynb
```

Make sure `creditcard.csv` is placed in the same directory as the notebook.

---

## 📌 Conclusion

This project demonstrates the application of **unsupervised anomaly detection** to credit card fraud detection. Among the three evaluated approaches, **Isolation Forest provided the strongest overall performance**, achieving **99.82% accuracy and 41% fraud recall** on the evaluation sample.

The results also demonstrate an important challenge in fraud detection: **a very high overall accuracy does not necessarily mean that a model is effective at detecting fraud**. Evaluating the minority class using recall, precision, and F1-score is essential when working with highly imbalanced datasets.

---

## 📚 Dataset Reference

The dataset is based on the widely used **Credit Card Fraud Detection** dataset containing anonymized transactions from European cardholders. The original features are confidential and have been transformed using PCA.


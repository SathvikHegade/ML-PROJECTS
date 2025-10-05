# Machine Learning Essentials

## 📊 Overview

This repository provides a concise guide to key **Machine Learning (ML)** concepts, covering **Data Wrangling**, **Supervised Learning**, and **Unsupervised Learning**. It is designed for students, developers, and anyone looking to understand ML workflows and algorithms quickly.

---

## 1. 🧹 Data Wrangling

**What it is:**
Data Wrangling is the process of **preparing raw data** for analysis or modeling. Clean and structured data is crucial for accurate ML results.

**Steps to follow:**

* **Import Data:** Load datasets from CSV, Excel, or databases.
* **Handle Missing Values:** Fill, drop, or impute gaps in data.
* **Encode Categories:** Convert text labels to numbers (One-Hot or Label Encoding).
* **Scale Features:** Normalize or standardize numerical values.
* **Remove Outliers:** Detect anomalies and handle them.
* **Feature Engineering:** Create new features to improve predictions.

**Tools:** `pandas`, `numpy`, `scikit-learn`

> **💡If you want detailed explanations and step-by-step code, open the README inside the folder above.**

---

## 2. 🏷️ Supervised Learning

**Definition:**
Supervised learning trains models on **labeled data** to predict outcomes.

**Popular Algorithms:**

### a) Logistic Regression

* For **binary classification** problems.
* Uses **sigmoid function** to predict probabilities.
* Evaluate with accuracy, precision, recall, ROC-AUC.

### b) Decision Tree

* Splits data into branches based on feature values.
* Easy to interpret but can overfit.
* Splitting criteria: **Gini Index** or **Entropy**.

### c) Random Forest

* Ensemble of multiple decision trees.
* Reduces overfitting and improves accuracy.
* Key parameters: `n_estimators`, `max_depth`.

### d) AdaBoost

* Combines weak classifiers to make a strong one.
* Focuses on misclassified data in each iteration.
* Sensitive to noisy data.

**Libraries:** `sklearn.linear_model`, `sklearn.tree`, `sklearn.ensemble`

> **💡 If you want detailed explanations and step-by-step code, open the README inside the folder above.**

---

## 3. 🔍 Unsupervised Learning

**Definition:**
Unsupervised learning finds patterns in **unlabeled data**.

**Popular Methods:**

### a) K-Means Clustering

* Groups data into **K clusters** based on similarity.

### b) Hierarchical Clustering

* Builds a **tree of clusters** (dendrogram).

### c) DBSCAN

* Density-based clustering for arbitrary-shaped clusters.

**Applications:** Customer segmentation, anomaly detection, pattern discovery.

**Libraries:** `sklearn.cluster`

> **💡 If you want detailed explanations and step-by-step code, open the README inside the folder above.**

---

## 4. 🚀 ML Workflow

```text
Data Collection → Data Wrangling → Feature Engineering → Model Selection
     → Supervised / Unsupervised Learning → Evaluation → Deployment
```

**Tip:** Start with clean data, choose the right model, evaluate carefully, and iterate.

---

## 📚 References

* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*

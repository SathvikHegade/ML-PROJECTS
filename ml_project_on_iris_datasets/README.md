# 📊 Machine Learning Experiments Repository

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Build](https://img.shields.io/badge/build-passing-brightgreen)



## 📚 Table of Contents
1. [Repository Overview](#repository-overview)  
2. [Projects Included](#projects-included)  
   - [1. Accuracy Comparison: Logistic Regression vs Random Forest](#1-accuracy-comparison-logistic-regression-vs-random-forest)  
   - [2. Clustering: K-Means & Hierarchical](#2-clustering-k-means--hierarchical)  
   - [3. Decision Tree Pruning](#3-decision-tree-pruning)  
3. [Installation & Setup](#installation--setup)  
4. [Author](#author)

---

## Repository Overview
This repository consolidates **three major machine learning experiments**, each in its own folder with **code + detailed README**:

1. Accuracy comparison between Logistic Regression and Random Forest  
2. Clustering analysis using K-Means and Hierarchical algorithms  
3. Decision Tree Pruning (Pre-Pruning and Post-Pruning)  

Each sub-folder is **self-contained** with a Jupyter Notebook implementing the code and a README explaining methodology, results, and insights.

---

## Projects Included

### 1. Accuracy Comparison: Logistic Regression vs Random Forest ⚡
- **Purpose:** Compare predictive performance of Logistic Regression and Random Forest classifiers.  
- **Key Focus:** Accuracy, precision, recall, F1-score  
- **Folder:** `ACCURACY_COMPARISION/`  

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

> **💡 HIGHLIGHT:** If you want more detailed explanations and step-by-step code, open the README inside this folder.

---

### 2. Clustering: K-Means & Hierarchical 🧩
- **Purpose:** Explore unsupervised learning techniques to discover hidden patterns.  
- **Algorithms:** K-Means, Hierarchical Clustering  
- **Folder:** `CLUSTERING_METHODS/`  

```python
from sklearn.cluster import KMeans, AgglomerativeClustering

# K-Means Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierarchical Clustering
hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_labels = hier.fit_predict(X_scaled)
```

> **💡 HIGHLIGHT:** For detailed explanations, metrics, and visualizations, open the README inside this folder.

---

### 3. Decision Tree Pruning 🌳
- **Purpose:** Optimize tree-based models using Pre-Pruning and Post-Pruning techniques.  
- **Techniques:** max_depth, min_samples_leaf, cost complexity pruning (ccp_alpha)  
- **Folder:** `PRUNING_TREE/`  

```python
from sklearn.tree import DecisionTreeClassifier

# Pre-Pruning Example
pre_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
pre_tree.fit(X_train, y_train)

# Post-Pruning Example
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
```

> **💡 HIGHLIGHT:** For comprehensive analysis and code examples, open the README inside this folder.

---

## Evaluation Metrics 📏
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **Silhouette Score / WCSS** for clustering

---

## Visualizations 📊
- Cluster plots for K-Means and Hierarchical clustering  
- Decision Tree diagrams (pre- and post-pruning)  
- Feature importance plots  
- Accuracy comparison bar charts

---

## 🛠️ Installation & Setup
### Prerequisites
- Python 3.8+  
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`  

### Installation
```bash
git clone https://github.com/SathvikHegade11/Machine_Learning_Experiments.git
cd Machine_Learning_Experiments
pip install -r requirements.txt
```

### Usage
- Navigate to any project folder  
- Open the Jupyter Notebook and follow the steps  
- **Each folder contains its own README for detailed instructions and explanations.**

---

## 👨‍💻 Author
**Sathvik Hegade**  
Machine Learning Engineer | Data Science Enthusiast | Python Developer  

✨ Dive into clustering patterns 🧩, tree pruning effects 🌳, and model accuracy insights ⚡—all in one repository!  
**💡 HIGHLIGHT:** For more detailed explanations, open the README in each project folder.


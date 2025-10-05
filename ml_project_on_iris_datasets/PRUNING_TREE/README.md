# 🌳 Decision Tree Pruning: Enhancing Model Performance with Pre-Pruning & Post-Pruning

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Build](https://img.shields.io/badge/build-passing-brightgreen)

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Understanding Decision Tree Pruning](#understanding-decision-tree-pruning)
   - [Pre-Pruning](#pre-pruning)
   - [Post-Pruning](#post-pruning)
3. [Implementation Details](#implementation-details)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Visualizations](#visualizations)
6. [Installation & Setup](#installation-setup)
7. [Author](#author)

---

## 📌 Project Overview

This project delves into the intricacies of **Decision Tree Pruning** using the **Iris dataset**. It explores two pivotal techniques:

- **Pre-Pruning (Early Stopping)**: Halting the growth of the tree during its construction to prevent overfitting.
- **Post-Pruning (Cost Complexity Pruning)**: Allowing the tree to grow fully and then pruning it to remove nodes that provide little predictive power.

The goal is to enhance the model's generalization capabilities, ensuring it performs well on unseen data.

---

## 🔍 Understanding Decision Tree Pruning

### Pre-Pruning

Pre-Pruning involves setting constraints during the tree construction process to limit its growth. Common parameters include:

- `max_depth`: Maximum depth of the tree.
- `min_samples_split`: Minimum number of samples required to split an internal node.
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node.

These parameters help in creating a simpler model that is less likely to overfit.

### Post-Pruning

Post-Pruning, also known as Cost Complexity Pruning, involves growing the tree fully and then removing nodes that have little importance. This is achieved by:

- Calculating the cost-complexity path using `cost_complexity_pruning_path`.
- Fitting the tree with different values of `ccp_alpha` (the complexity parameter).
- Selecting the optimal `ccp_alpha` based on cross-validation performance.

This technique helps in reducing the tree's complexity without sacrificing accuracy.

---

## ⚙️ Implementation Details

### Pre-Pruning Example

```python
from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Classifier with pre-pruning parameters
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)

# Fit the model
clf.fit(X_train, y_train)
```

### Post-Pruning Example

```python
from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Classifier without pruning
clf = DecisionTreeClassifier(random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Get the effective alpha values and the corresponding total leaf impurities
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Fit the model with each alpha value
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
```

---

## 📊 Evaluation Metrics

The models were evaluated using the following metrics:

- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were identified correctly.
- **F1-Score**: Harmonic mean of precision and recall.

These metrics provide a comprehensive understanding of the model's performance.

---

## 📈 Visualizations

- **Decision Tree Visualization**: Illustrates the structure of the decision tree before and after pruning.
- **Feature Importance Plot**: Shows the importance of each feature in the decision-making process.
- **Accuracy Comparison Graph**: Compares the accuracy of pre-pruned and post-pruned models.

These visualizations aid in interpreting the model's behavior and the impact of pruning techniques.

---

## 🛠️ Installation & Setup

### Prerequisites

Ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/SathvikHegade11/ml_project_on_iris_datasets.git
cd ml_project_on_iris_datasets
pip install -r requirements.txt
```

### Usage

Run the Jupyter Notebook to explore the implementation and results:

```bash
jupyter notebook Decision_Tree_Pruning.ipynb
```

---

## 👨‍💻 Author

**Sathvik Hegade**  
Machine Learning Engineer | Data Science Enthusiast | Python Developer

✨ Explore the effects of pruning and see **how tree complexity impacts predictive performance**!
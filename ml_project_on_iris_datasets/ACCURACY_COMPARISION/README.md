
# 🌸 Iris Flower Classification with Missing Value Handling  

## 📌 Overview  
This project applies **machine learning classification** to the Iris dataset, focusing on **missing value handling** and **model performance comparison**. It demonstrates the importance of preprocessing before training ML models.  

## ❓ Problem Statement  
The Iris dataset is widely used for classification. However, real-world datasets often contain missing values. In this project, artificial missing values were introduced and imputed using **statistical techniques** to make the model more robust.  

## 📂 Dataset  
- **Source:** Scikit-learn `load_iris()` dataset  
- **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width  
- **Target:** Species (Setosa, Versicolor, Virginica)  

## 🛠 Steps Involved  
1. **Data Preprocessing**  
   - Introduced missing values (~5% of data)  
   - Imputed numeric features using **mean**  
   - Imputed categorical values using **mode**  

2. **Feature Scaling**  
   - Standardized features with `StandardScaler`  

3. **Model Building**  
   - Logistic Regression  
   - Random Forest Classifier  
   - Hyperparameter tuning with **GridSearchCV**  

4. **Evaluation Metrics**  
   - Accuracy Score  
   - Confusion Matrix  
   - Classification Report  

## 📊 Results  
- **Logistic Regression Accuracy:** ~96%  
- **Random Forest Accuracy:** ~100%  
- Random Forest outperformed Logistic Regression with tuned hyperparameters  

## ✅ Conclusion  
- Handling missing values improves data quality and model reliability  
- Ensemble methods like **Random Forest** can achieve superior accuracy  
- This project shows the full ML workflow: preprocessing → model training → evaluation  

## 🧰 Tools & Libraries  
- Python  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib, Seaborn  

---

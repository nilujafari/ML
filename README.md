# 🎗️ Breast Cancer Classification using Machine Learning

## 📌 About the Project
This project compares multiple machine learning models to classify breast tumors as benign or malignant.
The goal is to evaluate model performance and find the most reliable approach, with a focus on recall due to its importance in medical diagnosis.
The Dataset detailed information is available on : 
- **Source:** [Breast Cancer Dataset – Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)


## 📊 Dataset 
 - Samples: 569
 - Features: 30 numerical features
 - Target:
    - 0 → Benign
    - 1 → Malignant

## 📦 Requirements
Python 3.x
 - pandas
 - numpy
 - matplotlib
 - seaborn
 - scikit-learn

## ⚙️ Models Used 
- ### The following models were trained and evaluated:
 - K-Nearest Neighbors (KNN)
 - Decision Tree
 - Random Forest
 - Support Vector Machine (SVM)
 - Logistic Regression
 - Artificial Neural Network (ANN)

## 📈 Evaluation Metrics
 - ### Models were evaluated using:
 - Accuracy
 - Precision
 - Recall (Critical im medical diagnosis)
 - F1-score
 - Confusion Matrix

## 📁 Project Structure
### breast_cancer_model.ipynb → main notebook


## ▶️ How to Run the Project

1. Clone the repository:
```bash
git clone https:/
```

2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the jupyter notebook:
```bash
jupyter notebook
```
## 🧪 Results
KNN and Decision Tree showed acceptable performance but had lower recall, missing more cancer cases.<br>
Random Forest and SVM improved overall performance with better accuracy and balance.<br>
Logistic Regression demonstrated strong and stable results.<br>
ANN achieved the best strongest overall performance.<br>

## 🔍 Conclusion
ANN and Logistic Regression detected the same number of cancer cases. However, ANN achieved slightly higher accuracy and F1-score.<br>

In medical diagnosis, recall is more important than accuracy, since missing a cancer case (false negative) can be critical.<br>

Overall, ANN is selected as the best model due to its strong balance between accuracy and recall.


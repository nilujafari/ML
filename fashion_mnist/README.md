# Fashion-MNIST Classification using Machine Learning

## Overview

This project implements and compares multiple Machine Learning algorithms for image classification using the Fashion-MNIST dataset.

The objective is to classify grayscale clothing images into one of ten categories and evaluate the performance of different machine learning models.

---

## Dataset

**Fashion-MNIST**

- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10

### Classes

| Label | Class |
|-------|----------------|
| 0 | T-shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## Project Workflow

### 1. Data Loading

- Load Fashion-MNIST training dataset
- Load Fashion-MNIST testing dataset
- Explore dataset dimensions

---

### 2. Exploratory Data Analysis (EDA)

- Display sample images
- Visualize class distribution
- Verify dataset balance
- Inspect dataset structure

---

### 3. Data Preprocessing

- Split features and labels
- Normalize pixel values using **MinMaxScaler**
- Prepare training and testing sets

---

## Machine Learning Models

The following models were trained and evaluated:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- Artificial Neural Network (MLPClassifier)

---

## Evaluation Metrics

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## Model Comparison

| Model | Accuracy |
|-----------------------|----------|
| Random Forest | **88.46%** |
| Artificial Neural Network | **86.24%** |
| K-Nearest Neighbors | **85.91%** |
| Support Vector Machine | **85.72%** |
| Logistic Regression | **85.23%** |
| Decision Tree | **80.82%** |

---

## Results

Random Forest achieved the highest classification accuracy among all evaluated models.

Performance ranking:

1. Random Forest
2. ANN (MLPClassifier)
3. KNN
4. SVM
5. Logistic Regression
6. Decision Tree

---

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Visualizations

The project includes:

- Sample image visualization
- Class distribution chart
- Confusion Matrix
- Classification Reports
- Model Accuracy Comparison Chart

---

## Project Structure

```
Fashion-MNIST/
│
├── dataset/
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
│
├── main.ipynb
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/Fashion-MNIST-Classification.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook

```bash
jupyter notebook
```

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
jupyter
```

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross Validation
- Deep Learning with TensorFlow/Keras
- CNN implementation
- Model deployment using Flask or FastAPI

---

## Conclusion

Six machine learning algorithms were trained and evaluated on the Fashion-MNIST dataset.

Among all models, **Random Forest** achieved the highest accuracy (88.46%), demonstrating superior performance for this classification task.

Artificial Neural Network (ANN) and KNN also produced competitive results, while Decision Tree showed the lowest accuracy.

This project demonstrates a complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, evaluation, visualization, and performance comparison.

---

## Author

Developed as a Machine Learning course project.

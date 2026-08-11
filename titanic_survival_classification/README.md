Titanic Survival Prediction

Overview

This project predicts passenger survival in the Titanic dataset usingmachine learning classification models and compares three featurerepresentations: Original, PCA (Principal Component Analysis), and SVD(Singular Value Decomposition).

The goal is to investigate whether dimensionality reduction can reducethe feature space while preserving or improving classificationperformance.

Dataset

Target: Survived

0 = Did not survive

1 = Survived

Training samples: 712

Test/validation samples: 179

Original features: 8

Machine Learning Models

KNN --- K-Nearest Neighbors

GaussianNB --- Gaussian Naive Bayes

Decision Tree

Random Forest

Logistic Regression

SVM --- Support Vector Machine

ANN --- Artificial Neural Network

Dimensionality Reduction

PCA

PCA transforms the original feature space into a lower-dimensionalrepresentation while retaining important information.

SVD

The project uses:

TruncatedSVD(n_components=5)

SVD therefore reduces the representation from 8 features to 5components.

Evaluation Metrics

The models are evaluated using:

Accuracy

Precision

Recall

F1 Score

Results

Accuracy

Model             Train  Train PCA  Train SVD   Validation   Validation   ValidationOriginal                           Original          PCA          SVD

KNN            0.825843   0.808989   0.808989     0.798883     0.798883     0.798883

GaussianNB     0.787921   0.783708   0.783708     0.837989     0.843575     0.843575

Decision       0.835674   0.856742   0.856742     0.821229     0.776536     0.776536Tree

Random         0.855337   0.869382   0.866573     0.837989     0.821229     0.826816Forest

Logistic       0.799157   0.780899   0.780899     0.798883     0.793296     0.793296Regression

SVM            0.834270   0.807584   0.807584     0.832402     0.832402     0.832402

ANN            0.849719   0.828652   0.825843     0.826816     0.826816     0.815642

Best validation accuracy: GaussianNB with PCA/SVD --- 0.843575.

SVD Precision

Model                   SVD Precision

KNN                          0.775862GaussianNB                   0.857143Decision Tree                0.759259Random Forest                0.836364Logistic Regression          0.754098SVM                          0.865385ANN                          0.872340

Best SVD precision: ANN --- 0.872340.

Recall

The recall comparison shows that PCA/SVD improve or preserve recall forseveral models. GaussianNB reaches approximately 0.70, while RandomForest improves with SVD compared with its original representation.Logistic Regression remains relatively stable, whereas ANN loses recallwith SVD.

SVD F1 Score

Model                   SVD F1 Score

KNN                         0.714286GaussianNB                  0.774194Decision Tree               0.672131Random Forest               0.747967Logistic Regression         0.713178SVM                         0.750000ANN                         0.713043

Best SVD F1 score: GaussianNB --- 0.774194.

PCA vs SVD

PCA and SVD produce very similar results for several models. Both reducethe feature space and preserve competitive performance, especially forGaussianNB.

However, dimensionality reduction does not automatically improve everyclassifier. Its effect depends on both the model and the evaluationmetric.

Best Results

Metric                Best Model/Configuration        Score

Validation Accuracy   GaussianNB + PCA/SVD         0.843575SVD Precision         ANN                          0.872340SVD Recall            GaussianNB                    ~0.706SVD F1 Score          GaussianNB                   0.774194

Conclusion

This project demonstrates a machine learning workflow for Titanicsurvival prediction including preprocessing, classification,dimensionality reduction, evaluation, and visualization.

The experiments show that:

PCA and SVD can reduce the feature space while maintainingcompetitive performance.

GaussianNB benefits most consistently from dimensionality reduction.

ANN and SVM achieve strong precision values.

Random Forest remains a strong classifier.

PCA/SVD do not improve every model, so model selection shouldconsider multiple metrics rather than accuracy alone.

Technologies

Python

Jupyter Notebook

NumPy

Pandas

Matplotlib

Scikit-learn

Project Structure

Titanic-Survival/
├── main.ipynb
├── README.md
└── dataset/
    └── titanic.csv

Author

Titanic Survival Prediction --- Machine Learning Classification andDimensionality Reduction Project.
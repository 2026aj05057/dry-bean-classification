# Dry Bean Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to identify different varieties of dry beans based on their physical characteristics.

The project uses numerical features such as Area, Perimeter, Major Axis Length, Minor Axis Length, Eccentricity, Compactness, and other shape-related measurements to classify dry beans into seven different categories.

Five machine learning classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The performance of these models is evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The objective is to identify the best-performing model for the given dataset.

## b. Dataset Description

The Dry Bean Dataset is used for this classification task.

- **Total instances:** 13,611
- **Total input features:** 16
- **Target variable:** Class
- **Number of classes:** 7
- **Missing values:** No missing values in the dataset.

The seven dry bean varieties are:

1. SEKER
2. BARBUNYA
3. BOMBAY
4. CALI
5. HOROZ
6. SIRA
7. DERMASON

The dataset contains numerical measurements describing the size, shape, and geometric properties of dry beans.

Examples of input features include Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, and Shape Factors.

The dataset was divided into training and testing sets using an 80:20 split:

- **Training samples:** 10,888
- **Testing samples:** 2,723

Feature scaling was performed using StandardScaler for Logistic Regression, KNN, Gaussian Naive Bayes, and SVM. The Decision Tree model was trained using the original unscaled features.

## c. GitHub Repository Link

**GitHub Repository:**  
https://github.com/2026aj05057/dry-bean-classification

The repository contains the trained machine learning models, preprocessing artifact, Streamlit application, test dataset, requirements file, README, and source notebook.

## d. Models Used and Evaluation

Five machine learning classification algorithms were implemented and evaluated using the same training and testing dataset.

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8920 | 0.9334 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 92.14% accuracy with an AUC score of 0.9934. It demonstrated strong classification performance with balanced precision and recall. It performed consistently well across the evaluation metrics. |
| Decision Tree | Achieved 89.20% accuracy and an AUC score of 0.9334. It provided reasonable classification performance but recorded the lowest AUC, accuracy, and MCC among the five models. |
| KNN | Achieved 91.66% accuracy and an AUC score of 0.9812. It performed well in identifying bean varieties, with good precision, recall, and F1 score. Its performance was slightly lower than Logistic Regression and SVM. |
| Naive Bayes | Achieved 89.79% accuracy with an AUC score of 0.9902. Although it demonstrated strong class discrimination based on AUC, its accuracy and other classification metrics were comparatively lower. |
| Support Vector Machine | Achieved the highest accuracy of 92.21% and the highest AUC score of 0.9937. It also recorded the highest precision, recall, F1 score, and MCC, demonstrating the strongest overall performance among the evaluated models. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** is the overall winner for the Dry Bean Classification dataset.

It achieved the highest accuracy of **92.21%**, AUC of **0.9937**, precision of **0.9224**, recall of **0.9221**, F1 score of **0.9222**, and MCC of **0.9059**.

These results indicate that SVM provides the best overall classification performance among the five evaluated machine learning algorithms.

Therefore, SVM is selected as the best-performing model for this classification task.

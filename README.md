# Dry Bean Classification Using Machine Learning

## Project Overview
This project predicts Dry Bean varieties using five machine learning classification algorithms.

## Dataset
- Dataset: UCI Dry Bean Dataset
- Total Instances: 13,611
- Input Features: 16
- Target Variable: Class
- Number of Classes: 7

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Evaluation Metrics
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Data Preprocessing
- Handled the dataset using Pandas and NumPy.
- Split the data into 80% training and 20% testing.
- Applied StandardScaler to the training and testing data where required.
- Used stratified splitting to preserve class distribution.

## Model Performance
| Model | Accuracy | AUC |
|---|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 |
| Decision Tree | 0.8920 | 0.9334 |
| KNN | 0.9166 | 0.9812 |
| Gaussian Naive Bayes | 0.8979 | 0.9902 |
| SVM | 0.9221 | 0.9937 |

## Project Structure
- app.py: Streamlit application
- requirements.txt: Required Python libraries
- test_data.csv: Test dataset
- model/: Saved five classifiers and scaler
- source_code/: Original Jupyter notebook

## Installation
Install dependencies using:

pip install -r requirements.txt

## Run Application
Start the application using:

streamlit run app.py

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn and Joblib.

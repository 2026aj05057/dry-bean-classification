
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification")
st.write("Predict Dry Bean varieties using five Machine Learning classifiers.")

# Model directory
MODEL_DIR = "model"

# Available models
models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# Expected input features
features = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

# Models that require feature scaling
scaled_models = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Support Vector Machine"
]

# Calculate multiclass AUC
def calculate_auc(model, X_input, actual):

    classes = model.classes_

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_input)

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_input)

    else:
        return np.nan

    binary_actual = label_binarize(
        actual,
        classes=classes
    )

    auc_values = []

    for i in range(len(classes)):

        if len(np.unique(binary_actual[:, i])) < 2:
            continue

        auc = roc_auc_score(
            binary_actual[:, i],
            scores[:, i]
        )

        auc_values.append(auc)

    return np.mean(auc_values) if auc_values else np.nan


# Select model for displaying individual predictions
selected_model = st.selectbox(
    "Choose a classification model for prediction results",
    list(models.keys())
)

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload test CSV containing all 16 features and the Class column",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    required_columns = features + ["Class"]

    # Validate uploaded CSV
    if not all(column in data.columns for column in required_columns):

        st.error(
            "Uploaded CSV must contain all 16 required features "
            "and the target column 'Class'."
        )

    elif data.empty:

        st.error("Uploaded CSV contains no records.")

    elif data[required_columns].isnull().any().any():

        st.error(
            "Uploaded CSV contains missing values. "
            "Please upload a cleaned test dataset."
        )

    else:

        X = data[features]
        actual = data["Class"]

        # Load saved scaler
        scaler = joblib.load(
            os.path.join(MODEL_DIR, "scaler.joblib")
        )

        # Store predictions and evaluation metrics
        all_predictions = {}
        evaluation_results = []

        st.info(
            "Evaluating all five previously trained models. "
            "No model retraining is performed."
        )

        # Evaluate all five saved models
        for model_name, model_file in models.items():

            # Load saved model
            model = joblib.load(
                os.path.join(MODEL_DIR, model_file)
            )

            # Apply scaling where required
            if model_name in scaled_models:
                X_input = scaler.transform(X)
            else:
                X_input = X

            # Generate predictions
            predictions = model.predict(X_input)

            all_predictions[model_name] = predictions

            # Calculate evaluation metrics
            accuracy = accuracy_score(actual, predictions)

            auc = calculate_auc(
                model,
                X_input,
                actual
            )

            precision = precision_score(
                actual,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                actual,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                actual,
                predictions,
                average="weighted",
                zero_division=0
            )

            mcc = matthews_corrcoef(
                actual,
                predictions
            )

            evaluation_results.append({
                "ML Model Name": model_name,
                "Accuracy": accuracy,
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "MCC": mcc
            })

        # Display evaluation comparison
        st.success("All five models evaluated successfully!")

        st.subheader("📊 Model Performance Comparison")

        metrics_df = pd.DataFrame(evaluation_results)

        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Display selected model prediction results
        st.subheader("Prediction Results")

        selected_predictions = all_predictions[selected_model]

        results = data.copy()

        results["Predicted_Class"] = selected_predictions

        results["Prediction_Status"] = [
            "Correct" if a == p else "Incorrect"
            for a, p in zip(actual, selected_predictions)
        ]

        correct = (actual == selected_predictions).sum()
        incorrect = len(selected_predictions) - correct

        accuracy = accuracy_score(
            actual,
            selected_predictions
        ) * 100

        st.write(f"Selected Model: **{selected_model}**")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Predictions", len(selected_predictions))
        col2.metric("Accuracy", f"{accuracy:.2f}%")
        col3.metric("Correct Predictions", correct)
        col4.metric("Incorrect Predictions", incorrect)

        st.dataframe(
            results,
            use_container_width=True
        )

        # Download predictions
        st.download_button(
            label="Download Predictions",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="dry_bean_predictions.csv",
            mime="text/csv"
        )

        # Download model comparison
        st.download_button(
            label="Download Model Comparison",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="model_performance_comparison.csv",
            mime="text/csv"
        )

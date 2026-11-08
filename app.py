
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import accuracy_score

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

# Select model
selected_model = st.selectbox(
    "Choose a classification model",
    list(models.keys())
)

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload a CSV file containing the 16 Dry Bean features",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if not all(feature in data.columns for feature in features):
        st.error("Uploaded CSV must contain all 16 required features.")

    else:
        X = data[features]

        # Load selected model
        model = joblib.load(
            os.path.join(MODEL_DIR, models[selected_model])
        )

        # Apply scaling where required
        if selected_model in [
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
            "Support Vector Machine"
        ]:
            scaler = joblib.load(
                os.path.join(MODEL_DIR, "scaler.joblib")
            )
            X_input = scaler.transform(X)
        else:
            X_input = X

        # Generate predictions
        predictions = model.predict(X_input)

        results = data.copy()
        results["Predicted_Class"] = predictions

        st.success("Predictions generated successfully!")

        # Evaluation Summary (only when actual labels are available)
        if "Class" in data.columns:

            actual = data["Class"]

            correct = (actual == predictions).sum()
            incorrect = len(predictions) - correct

            accuracy = accuracy_score(actual, predictions) * 100

            st.subheader("📊 Evaluation Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Predictions", len(predictions))
            col2.metric("Accuracy", f"{accuracy:.2f}%")
            col3.metric("Correct Predictions", correct)
            col4.metric("Incorrect Predictions", incorrect)

            results["Prediction_Status"] = [
                "Correct" if a == p else "Incorrect"
                for a, p in zip(actual, predictions)
            ]

        else:
            st.info(
                "Actual 'Class' column not found. "
                "Accuracy and correct/incorrect counts cannot be calculated."
            )

        # Prediction Results
        st.subheader("Prediction Results")
        st.dataframe(results)

        # Download predictions
        st.download_button(
            label="Download Predictions",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="dry_bean_predictions.csv",
            mime="text/csv"
        )

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)

# =====================================
# ⚙️ Streamlit Page Configuration
# =====================================
st.set_page_config(page_title="Customer Conversion Analysis", layout="wide")
st.title("🛒 Customer Conversion Analysis App")
st.write("Analyze customer behavior, predict conversions, estimate revenue, and segment users interactively.")

# =====================================
# 🧠 Load Models from artifacts folder
# =====================================
@st.cache_resource
def load_models():
    try:
        base_path = "artifacts"
        classification_model = joblib.load(os.path.join(base_path, "classifier.joblib"))
        regression_model = joblib.load(os.path.join(base_path, "regressor.joblib"))
        clustering_model = joblib.load(os.path.join(base_path, "kmeans.joblib"))
        cluster_scaler = joblib.load(os.path.join(base_path, "cluster_scaler.joblib"))
        st.sidebar.success("✅ Models loaded successfully!")
        return classification_model, regression_model, clustering_model, cluster_scaler
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")
        return None, None, None, None

classification_model, regression_model, clustering_model, cluster_scaler = load_models()

# =====================================
# ⚙️ Helper Functions
# =====================================
def detect_model_type(file_name: str):
    name = file_name.lower()
    if "class" in name:
        return "Classification"
    elif "reg" in name:
        return "Regression"
    elif "cluster" in name:
        return "Clustering"
    else:
        return None

def align_features(df, model):
    """Align dataset columns with model's training columns"""
    if hasattr(model, "feature_names_in_"):
        model_cols = model.feature_names_in_
        missing_cols = set(model_cols) - set(df.columns)
        extra_cols = set(df.columns) - set(model_cols)

        if missing_cols:
            st.warning(f"⚠️ Missing columns filled with 0: {missing_cols}")
        if extra_cols:
            st.info(f"ℹ️ Extra columns ignored: {extra_cols}")

        df_aligned = df.reindex(columns=model_cols, fill_value=0)
        return df_aligned
    else:
        return df

# =====================================
# 🎛 Sidebar Navigation
# =====================================
option = st.sidebar.selectbox(
    "Select Mode",
    ["Upload Dataset", "Manual Prediction"]
)

# =====================================
# 📂 Upload Dataset Mode
# =====================================
if option == "Upload Dataset":
    st.subheader("📂 Upload Your CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        model_type = detect_model_type(uploaded_file.name)

        if not model_type:
            st.error("⚠️ Could not determine dataset type. Please rename your file with 'class', 'reg', or 'cluster'.")
        else:
            st.success(f"✅ Detected as {model_type} dataset")

            # ===== Classification =====
            if model_type == "Classification" and classification_model:
                df_aligned = align_features(df, classification_model)
                preds = classification_model.predict(df_aligned)
                df["Predicted_Conversion"] = preds
                st.success("✅ Conversion Prediction Completed!")
                st.dataframe(df.head())

                if "target" in df.columns:
                    y_true = df["target"]
                    y_pred = preds
                    st.subheader("📊 Classification Metrics")
                    st.write(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
                    st.write(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.3f}")
                    st.write(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.3f}")
                    st.write(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.3f}")
                    st.write(f"ROC-AUC: {roc_auc_score(y_true, y_pred):.3f}")

            # ===== Regression =====
            elif model_type == "Regression" and regression_model:
                df_aligned = align_features(df, regression_model)
                preds = regression_model.predict(df_aligned)
                df["Predicted_Revenue"] = preds
                st.success("✅ Revenue Estimation Completed!")
                st.dataframe(df.head())

                if "target" in df.columns:
                    y_true = df["target"]
                    y_pred = preds
                    st.subheader("📈 Regression Metrics")
                    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
                    st.write(f"MAE: {mean_absolute_error(y_true, y_pred):.3f}")
                    st.write(f"R²: {r2_score(y_true, y_pred):.3f}")

            # ===== Clustering =====
            elif model_type == "Clustering" and clustering_model:
                df_scaled = cluster_scaler.transform(df.select_dtypes(include=["number"]))
                clusters = clustering_model.predict(df_scaled)
                df["Cluster"] = clusters
                st.success("✅ Clustering Completed!")
                st.dataframe(df.head())

                st.subheader("📊 Clustering Metrics")
                st.write(f"Silhouette Score: {silhouette_score(df_scaled, clusters):.3f}")
                st.write(f"Davies-Bouldin Index: {davies_bouldin_score(df_scaled, clusters):.3f}")

                # Visualization
                st.subheader("📈 Cluster Visualization")
                if df_scaled.shape[1] >= 2:
                    plt.figure(figsize=(7, 5))
                    plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=clusters, cmap="viridis")
                    plt.title("Customer Segments Visualization")
                    st.pyplot(plt)

            # ===== EDA Visualizations =====
            st.subheader("📊 EDA Visualizations")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column for visualization", numeric_cols)
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                sns.histplot(df[selected_col], kde=True, ax=axes[0])
                axes[0].set_title("Histogram")

                df[selected_col].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[1])
                axes[1].set_title("Pie Chart")

                sns.barplot(x=df[selected_col].value_counts().index, y=df[selected_col].value_counts().values, ax=axes[2])
                axes[2].set_title("Bar Chart")
                st.pyplot(fig)

# =====================================
# ✋ Manual Prediction Mode
# =====================================
elif option == "Manual Prediction":
    st.subheader("🔮 Manual Customer Prediction")
    model_choice = st.selectbox("Select Prediction Type", ["Classification", "Regression"])

    if model_choice == "Classification" and classification_model:
        st.write("Enter customer details for conversion prediction:")
        country = st.selectbox("Country", ["India", "USA", "UK", "Germany", "France"])
        page_main = st.selectbox("Page 1 (Main Category)", [1, 2, 3, 4])
        color = st.selectbox("Colour", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        price = st.number_input("Price (USD)", min_value=0.0, max_value=1000.0, step=10.0)
        price2 = st.selectbox("Price higher than average?", [1, 2])
        page = st.slider("Page number within e-store", 1, 5, 1)

        if st.button("Predict Conversion"):
            sample = pd.DataFrame({
                'country': [country],
                'page1_main_category': [page_main],
                'colour': [color],
                'price': [price],
                'price2': [price2],
                'page': [page]
            })
            sample_aligned = align_features(sample, classification_model)
            prediction = classification_model.predict(sample_aligned)
            st.success(f"Predicted Conversion: {'✅ Yes' if prediction[0] == 1 else '❌ No'}")

    elif model_choice == "Regression" and regression_model:
        st.write("Enter customer details for revenue estimation:")
        price = st.number_input("Price (USD)", min_value=0.0, max_value=1000.0, step=10.0)
        page = st.slider("Page number", 1, 5, 1)
        clicks = st.number_input("Number of clicks", min_value=1, max_value=100)
        session_length = st.number_input("Session length (seconds)", min_value=10, max_value=600)

        if st.button("Estimate Revenue"):
            sample = pd.DataFrame({
                'price': [price],
                'page': [page],
                'clicks': [clicks],
                'session_length': [session_length]
            })
            sample_aligned = align_features(sample, regression_model)
            revenue_pred = regression_model.predict(sample_aligned)
            st.success(f"Estimated Revenue: ${revenue_pred[0]:.2f}")

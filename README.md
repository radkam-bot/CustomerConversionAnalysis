# 🛒 Customer Conversion Analysis App

This project predicts customer conversion likelihood, estimates revenue, and segments customers using machine learning models.  
It provides an interactive **Streamlit web app** for easy visualization and real-time predictions.

---

## 🚀 Features

- **Classification** – Predict whether a customer will complete a purchase.  
- **Regression** – Estimate potential revenue based on browsing behavior.  
- **Clustering** – Segment customers into meaningful groups.  
- **EDA Visualizations** – Interactive charts including bar, pie, and histogram views.  
- **Manual Prediction Mode** – Input new customer data directly for prediction.

---

## 🧠 Models Used

All trained models are stored in the `artifacts/` folder:

- `classifier.joblib` – For conversion classification  
- `regressor.joblib` – For revenue estimation  
- `kmeans.joblib` – For customer segmentation  
- `cluster_scaler.joblib` – For clustering feature scaling

---

## 🧰 Tech Stack

- **Python**  
- **Streamlit** – for the web interface  
- **Scikit-learn** – for ML models  
- **Pandas, NumPy** – for data processing  
- **Matplotlib, Seaborn** – for visualization  

---

## ⚙️ How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/radkam-bot/CustomerConversionAnalysis.git
   cd CustomerConversionAnalysis

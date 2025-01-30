# 🚗 **Fault Prediction for New Energy Vehicles (NEV) using Machine Learning & Deep Learning**
**An end-to-end ML pipeline for predicting faults in New Energy Vehicles using Random Forest and Deep Learning models, integrated with MLflow for model tracking and FastAPI for deployment.**

---

## 📌 **Project Overview**
This project builds a **fault prediction system** for **New Energy Vehicles (NEV)** using **Random Forest** and **Deep Learning (Neural Network)** models. The pipeline includes **data preprocessing, model training, performance tracking with MLflow, model deployment using FastAPI, and visualization of feature importance**.  

✅ **Key Features:**
- **ETL Pipeline**: Data preprocessing, feature engineering, and dataset splitting.
- **Machine Learning & Deep Learning Models**: Random Forest vs. Neural Network (Keras).
- **Automated Model Tracking**: MLflow for logging experiments and version control.
- **Best Model Selection**: Automatically selects and saves the best-performing model.
- **Model Deployment**: Deploys the best model via FastAPI for real-time predictions.
- **Feature Importance Analysis**: Visualizes key factors influencing predictions.

---

## 📂 **Project Structure**
```
fault_prediction_project/
│── data/                   # Raw and processed data
│── models/                 # Saved best models (.pkl for RF, .keras for NN)
│── reports/                # Model performance reports and visualizations
│── src/                    # Source code for model training and deployment
│   ├── data_processor.py   # Handles data loading and preprocessing
│   ├── model_trainer.py    # Trains RF & NN models, logs experiments in MLflow
│   ├── model_deployer.py   # Saves, logs, and serves the best model
│   ├── visualization.py    # Feature importance plots
│── app/                    # FastAPI application for model serving
│   ├── fastapi_app.py      # REST API for real-time predictions
│── notebooks/              # Jupyter notebooks for exploratory analysis
│── mlruns/                 # MLflow experiment tracking data
│── README.md               # Project documentation (this file)
│── requirements.txt        # Required Python packages
│── config.yaml             # Configuration settings (optional)
```

---

## 📊 **Dataset**
- **Source**: [Kaggle]
- **Data Description**:
  - **Features**: Vehicle sensor data, driving patterns, road conditions, environmental factors, maintenance logs.
  - **Target Variable**: `fault_type` (categorical, indicating vehicle faults).

---

## 🚀 **Installation & Setup**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/niloysamadder/fault_prediction_project.git
cd fault_prediction_project
```

### **2️⃣ Create a Virtual Environment (Recommended)**
```sh
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

---

## 🏗️ **How to Run the Project**
### **1️⃣ Train the Models**
Run the training script to preprocess the data and train both Random Forest and Deep Learning models.
```sh
python src/model_trainer.py
```
> The best model will be saved in the `models/` directory.

### **2️⃣ Deploy the Best Model using FastAPI**
Run the FastAPI server to serve real-time predictions.
```sh
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```
> The API will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### **3️⃣ Run MLflow UI for Experiment Tracking**
```sh
mlflow ui --backend-store-uri "file:mlruns"
```
> Access MLflow UI at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🔬 **Results & Model Comparison**
| Model           | Accuracy | F1 Score |
|----------------|---------|----------|
| Random Forest  | 0.66    | 0.62     |
| Neural Network | 0.64    | 0.60     |

- **Feature Importance (Random Forest)**: Visualizes top factors influencing fault prediction.
- **Neural Network Feature Analysis**: Compares the effect of input features on fault classification.

---

## 📊 **Feature Importance Visualization**
```sh
python src/visualization.py
```
> Generates plots comparing feature contributions in **Random Forest** and **Deep Learning** models.

---

## 🚀 **Endpoints & API Usage**
| Method | Endpoint      | Description |
|--------|--------------|-------------|
| `POST` | `/predict`   | Get predictions from the best model |

### **Example Request**
```json
{
  "data": [[0.5, 1.2, 3.4, 2.1, 4.5, 1.1, 0.8, 3.2, 2.5, 1.0]]
}
```

### **Example Response**
```json
{
  "predictions": ["engine_overheating"]
}
```

---

## 🛠️ **Technologies Used**
- **Programming**: Python (Pandas, NumPy, Scikit-learn, TensorFlow, FastAPI)
- **Machine Learning**: Random Forest, Neural Networks
- **Model Tracking**: MLflow
- **Deployment**: FastAPI, Uvicorn
- **Visualization**: Matplotlib, Seaborn

---

## 📌 **Future Improvements**
- Adding **hyperparameter tuning** for optimal model performance.
- Improving **feature engineering** for better predictions.
- Deploying the API using **Docker & cloud services (AWS/Azure/GCP)**.

---

## 📝 **Acknowledgments**
- **Data Source**: Kaggle
- **Libraries**: Scikit-learn, TensorFlow, FastAPI, MLflow

---

## 💡 **Contributions & Issues**
- Feel free to **fork the repo**, open issues, or submit pull requests.
- If you find a bug or have suggestions, please create an **issue**.

📌 **Author**: [Niloy Samadder](https://github.com/niloysamadder)  
📌 **License**: MIT License  


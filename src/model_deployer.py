import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable TensorFlow oneDNN optimizations

import mlflow
import mlflow.keras
import mlflow.sklearn
import joblib
import numpy as np
import tensorflow as tf

class ModelDeployer:
    @staticmethod
    def save_model(model, path, model_type="sklearn"):
        """Save a trained model to the given path."""
        if model_type == "sklearn":
            joblib.dump(model, path)
        else:
            model.save(path)
        print(f"✅ Model saved at {path}")

    @staticmethod
    def serve_model(model_name, input_data):
        """Load the model from MLflow and make predictions."""
        model_uri = f"models:/{model_name}/latest"
        print(f"🔄 Loading model from MLflow: {model_uri}")
        
        # Load the appropriate model type
        if model_name == "best_rf_model":
            model = mlflow.sklearn.load_model(model_uri)
        else:
            model = mlflow.keras.load_model(model_uri)

        # ✅ Ensure input is a NumPy array
        input_data = np.array(input_data, dtype=np.float32)

        # ✅ Ensure correct shape (batch_size, features)
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        predictions = model.predict(input_data)
        return predictions


# 🔥 **Execute Deployment**
if __name__ == "__main__":
    # **Load the best model**
    print("🔄 Loading trained model for deployment...")

    # Model paths
    rf_model_path = "F:/Portfolio Projects/fault_prediction_project/models/best_rf_model.pkl"
    nn_model_path = "F:/Portfolio Projects/fault_prediction_project/models/best_nn_model.keras"

    best_model = None
    best_model_name = None
    best_model_type = None

    if os.path.exists(rf_model_path):  # Load Random Forest if exists
        print("🌲 Best model: Random Forest")
        best_model = joblib.load(rf_model_path)
        best_model_name = "best_rf_model"
        best_model_type = "sklearn"

    elif os.path.exists(nn_model_path):  # Load Neural Network if exists
        print("🤖 Best model: Neural Network")
        best_model = tf.keras.models.load_model(nn_model_path)
        best_model_name = "best_nn_model"
        best_model_type = "keras"

    if best_model is None:
        raise FileNotFoundError("❌ No best model found! Train and save models first.")

    # **Save the best model for deployment**
    deployed_model_path = f"F:/Portfolio Projects/fault_prediction_project/models/deployed_model.{best_model_type}"
    print("💾 Saving best model for deployment...")
    ModelDeployer.save_model(best_model, deployed_model_path, best_model_type)

    # **Log the model to MLflow**
    print("📢 Logging model to MLflow...")
    mlflow.set_tracking_uri("file:F:/Portfolio Projects/fault_prediction_project/mlruns")  # Set MLflow local tracking

    if best_model_type == "sklearn":
        model_info = mlflow.sklearn.log_model(best_model, best_model_name)
    else:
        model_info = mlflow.keras.log_model(best_model, best_model_name)

    # **Register the best model in MLflow**
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(best_model_name)
        print(f"✅ Model registered in MLflow as '{best_model_name}'")
    except mlflow.exceptions.MlflowException:
        print(f"⚠️ Model '{best_model_name}' already exists in MLflow, skipping registration.")

    # **Create a new model version**
    model_uri = model_info.model_uri
    try:
        client.create_model_version(name=best_model_name, source=model_uri, run_id=mlflow.active_run().info.run_id)
        print(f"✅ Model version created for '{best_model_name}'")
    except mlflow.exceptions.MlflowException:
        print(f"⚠️ Model version for '{best_model_name}' already exists.")

    # ✅ Register the model manually
client = mlflow.tracking.MlflowClient()

try:
    client.create_registered_model("best_rf_model")
    print("✅ Model registered as 'best_rf_model'")
except mlflow.exceptions.MlflowException:
    print("⚠️ Model 'best_rf_model' already exists.")

# ✅ Create a new version of the model
model_uri = "F:/Portfolio Projects/fault_prediction_project/models/deployed_model.sklearn"

try:
    client.create_model_version(name="best_rf_model", source=model_uri, run_id=mlflow.active_run().info.run_id)
    print("✅ Model version created successfully!")
except mlflow.exceptions.MlflowException:
    print("⚠️ Model version already exists.")

    # **Test serving (optional)**
    print("🚀 Loading and serving the model via MLflow...")
    test_input = [[0.5] * 17]  # Example input (adjust dimensions based on your model)
    predictions = ModelDeployer.serve_model(best_model_name, test_input)
    print(f"📌 Model Predictions: {predictions}")
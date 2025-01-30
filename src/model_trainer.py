import sys
import os

# Set the working directory to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
print(f"📌 Project root added to Python path: {project_root}")

# Disable TensorFlow oneDNN optimizations to avoid floating-point rounding issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import mlflow
import mlflow.sklearn
import mlflow.keras
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, experiment_name="NEV Fault Prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def train_rf(self, X_train, y_train, X_test, y_test):
        """Train a Random Forest model and log it to MLflow."""
        print("🔄 Initializing Random Forest training...")
        with mlflow.start_run(run_name="Random Forest Classifier"):
            try:
                # Train the model
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                print("✅ Random Forest training complete.")

                # Predictions
                y_pred = rf_model.predict(X_test)

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                print(f"🎯 Random Forest - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

                # Log model to MLFlow
                mlflow.log_param("Model", "Random Forest")
                mlflow.log_param("n_estimators", 100)
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("F1 Score", f1)
                mlflow.sklearn.log_model(rf_model, "random_forest_model")
                print("✅ Random Forest model logged to MLflow.")

                return rf_model, accuracy  # ✅ Return model and accuracy

            except Exception as e:
                print(f"❌ Error during Random Forest training: {e}")

    def train_nn(self, X_train, y_train, X_test, y_test, input_dim):
        """Train a Neural Network model and log it to MLflow."""
        print("🔄 Initializing Neural Network training...")
        print(f"📏 Input Shape: {X_train.shape}, Target Shape: {y_train.shape}")

        # ✅ Encode categorical target labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        print(f"✅ Encoded Target Labels: {list(label_encoder.classes_)}")

        # ✅ Convert labels to one-hot encoding
        y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=len(label_encoder.classes_))
        y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=len(label_encoder.classes_))

        with mlflow.start_run(run_name="Deep Learning Model"):
            try:
                # Define Neural Network
                model = Sequential([
                    Dense(128, activation='relu', input_dim=input_dim),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(len(label_encoder.classes_), activation='softmax')
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), 
                              loss='categorical_crossentropy', 
                              metrics=['accuracy'])

                # Train the model
                print("🚀 Training the Neural Network...")
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train_onehot, 
                          validation_data=(X_test, y_test_onehot), 
                          epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)
                print("✅ Neural Network training complete.")

                # Save the model locally
                model_dir = "F:/Portfolio Projects/fault_prediction_project/models"
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "deep_learning_model.keras")
                model.save(model_path)
                print(f"✅ Neural Network model saved at {model_path}")

                # Log model to MLFlow
                loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
                y_pred = model.predict(X_test)
                y_pred_classes = tf.argmax(y_pred, axis=1)
                f1 = f1_score(y_test_encoded, y_pred_classes, average="weighted")

                mlflow.log_param("Model", "Deep Learning") 
                mlflow.log_param("Learning Rate", 0.001)
                mlflow.log_param("Batch Size", 32)
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("F1 Score", f1)
                mlflow.keras.log_model(model, "deep_learning_model")

                print(f"🎯 Neural Network - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

                return model, accuracy  # ✅ Return model and accuracy

            except Exception as e:
                print(f"❌ Error during Neural Network training or saving: {e}")


# ✅ Execute the script
if __name__ == "__main__":
    from src.data_processor import DataProcessor

    # File path
    data_file_path = "F:/Portfolio Projects/fault_prediction_project/data/Fault_nev_dataset.csv"

    # Initialize DataProcessor
    processor = DataProcessor(data_file_path)

    # Load and preprocess the data
    print("\n📂 Loading Dataset...")
    data = processor.load_data()
    print(f"✅ Dataset Loaded: {data.shape}")

    print("\n🛠️ Preprocessing Dataset...")
    X, y = processor.preprocess(target_column="fault_type", categorical_columns=["road_condition"])
    print(f"✅ Features Shape: {X.shape}, Target Shape: {y.shape}")

    print("\n📊 Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Training Set: {X_train.shape}, {y_train.shape}")
    print(f"✅ Testing Set: {X_test.shape}, {y_test.shape}")

    # Initialize ModelTrainer
    trainer = ModelTrainer()

    # 🔥 Train and evaluate both models
    print("\n🌲 Training Random Forest...")
    rf_model, rf_accuracy = trainer.train_rf(X_train, y_train, X_test, y_test)
    
    print("\n🤖 Training Neural Network...")
    nn_model, nn_accuracy = trainer.train_nn(X_train, y_train, X_test, y_test, input_dim=X_train.shape[1])

    # ✅ Save the best model
    best_model = rf_model if rf_accuracy >= nn_accuracy else nn_model
    best_model_name = "best_rf_model.pkl" if rf_accuracy >= nn_accuracy else "best_nn_model.keras"
    best_model_path = f"F:/Portfolio Projects/fault_prediction_project/models/{best_model_name}"

    if isinstance(best_model, RandomForestClassifier):
        joblib.dump(best_model, best_model_path)
    else:
        best_model.save(best_model_path)

    print(f"🏆 ✅ Best model saved as {best_model_name} with accuracy: {max(rf_accuracy, nn_accuracy):.2f}")
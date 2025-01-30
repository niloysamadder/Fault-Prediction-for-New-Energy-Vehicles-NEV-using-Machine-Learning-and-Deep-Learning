import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import mlflow

client = mlflow.tracking.MlflowClient()

# 🔍 Check all registered models
print("\n📌 Registered Models in MLflow:")
for rm in client.search_registered_models():
    print(f"🔹 {rm.name}")

# 🔍 Check versions of best_rf_model
try:
    latest_version = client.get_latest_versions("best_rf_model")[0].version
    print("\n🔎 Latest Model Version:", latest_version)
except Exception as e:
    print("\n❌ Model 'best_rf_model' is not registered:", e)


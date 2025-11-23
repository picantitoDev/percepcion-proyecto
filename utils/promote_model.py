import mlflow
from mlflow import MlflowClient
import dagshub

# Configurar conexión
dagshub.init(
    repo_owner='picantitoDev',
    repo_name='percepcion-proyecto',
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/picantitoDev/percepcion-proyecto.mlflow")

client = MlflowClient()

# Obtener la última versión del modelo
model_name = "ResNet18"

try:
    # Obtener todas las versiones
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if versions:
        # Obtener la versión más reciente
        latest_version = versions[0]
        version_number = latest_version.version
        
        print(f"Found model version: {version_number}")
        print(f"Current stage: {latest_version.current_stage}")
        
        # Promover a Production
        client.transition_model_version_stage(
            name=model_name,
            version=version_number,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"✅ Model version {version_number} promoted to Production!")
    else:
        print(f"❌ No versions found for model '{model_name}'")
        
except Exception as e:
    print(f"❌ Error: {e}")

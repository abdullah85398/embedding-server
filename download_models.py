import os
import yaml
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """
    Downloads models specified in models.yaml to the local cache.
    This is intended to be run during the Docker build process.
    """
    config_path = os.getenv("MODEL_CONFIG_PATH", "models.yaml")
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Skipping model download.")
        return

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to read {config_path}: {e}")
        return

    models = config.get("models", {})
    
    for alias, model_info in models.items():
        # Only download if preload is true
        if not model_info.get("preload", False):
            logger.info(f"Skipping model {alias} (preload=false)")
            continue

        model_name = model_info.get("name")
        if model_name:
            logger.info(f"Downloading model: {alias} ({model_name})...")
            try:
                # This triggers the download and caching
                SentenceTransformer(model_name)
                logger.info(f"Successfully downloaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                # We might want to fail the build if a model fails to download
                raise e

if __name__ == "__main__":
    download_models()

import logging
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the lifecycle of embedding models (loading, unloading, caching).
    """
    def __init__(self):
        self.models: Dict[str, SentenceTransformer] = {}
        # Determine default device
        self.default_device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.default_device = "mps"
        
        logger.info(f"ModelManager initialized. Default device: {self.default_device}")
        
        # We need to load config differently or pass it in. 
        # For now, let's lazy load config inside methods or constructor
        from app.config.settings import model_config
        self.config = model_config
        self._load_preloaded_models()

    def _load_preloaded_models(self):
        """Loads models marked as 'preload: true' in the configuration."""
        for alias, conf in self.config.items():
            if conf.get("preload", True):
                device = conf.get("device", self.default_device)
                logger.info(f"Preloading model: {alias} ({conf['name']}) on {device}")
                self.load_model(alias, device=device)

    def load_model(self, alias: str, model_name: Optional[str] = None, device: Optional[str] = None) -> Optional[SentenceTransformer]:
        """
        Loads a model into memory.

        Args:
            alias (str): The alias/ID of the model.
            model_name (str, optional): The HuggingFace model name. Defaults to config if not provided.
            device (str, optional): The device to load the model on (cpu, cuda, mps).

        Returns:
            SentenceTransformer: The loaded model instance.

        Raises:
            ValueError: If the model alias is not found in config and no model_name is provided.
            Exception: If model loading fails.
        """
        if alias in self.models:
            return self.models[alias]
        
        target_name = model_name
        target_device = device or self.default_device

        if not target_name:
            if alias not in self.config:
                raise ValueError(f"Model alias '{alias}' not found in configuration.")
            conf = self.config[alias]
            target_name = conf["name"]
            target_device = conf.get("device", target_device)
        
        try:
            logger.info(f"Loading model: {target_name} on {target_device}")
            model = SentenceTransformer(target_name, device=target_device)
            self.models[alias] = model
            
            if alias not in self.config:
                self.config[alias] = {"name": target_name, "preload": False, "device": target_device}
                
            return model
        except Exception as e:
            logger.error(f"Failed to load model {target_name}: {e}")
            raise e

    def unload_model(self, alias: str):
        """
        Unloads a model from memory to free up resources.

        Args:
            alias (str): The alias of the model to unload.
        """
        if alias in self.models:
            logger.info(f"Unloading model: {alias}")
            del self.models[alias]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.warning(f"Model {alias} not loaded, cannot unload.")

    def get_model(self, alias: str) -> SentenceTransformer:
        """
        Retrieves a loaded model, or loads it if not present.

        Args:
            alias (str): The model alias.

        Returns:
            SentenceTransformer: The model instance.
        """
        if alias not in self.models:
            return self.load_model(alias)
        return self.models[alias]

model_manager = ModelManager()

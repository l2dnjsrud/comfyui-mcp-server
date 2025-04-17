import requests
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger("ModelManager")

class ModelManager:
    """
    Manages model discovery, validation and usage for ComfyUI.
    Handles different model types including checkpoints, LoRAs, ControlNets, etc.
    """
    
    def __init__(self, comfyui_base_url: str = "http://localhost:8188"):
        self.comfyui_base_url = comfyui_base_url
        # Cache for model information
        self.model_cache = {}
        # Cache for node information
        self.node_info_cache = {}
    
    def get_node_info(self, node_class: str) -> Dict[str, Any]:
        """Get information about a specific node class"""
        if node_class in self.node_info_cache:
            return self.node_info_cache[node_class]
            
        try:
            response = requests.get(f"{self.comfyui_base_url}/object_info/{node_class}")
            if response.status_code != 200:
                logger.warning(f"Failed to fetch info for node {node_class}")
                return {}
                
            data = response.json()
            self.node_info_cache[node_class] = data.get(node_class, {})
            return self.node_info_cache[node_class]
        except Exception as e:
            logger.error(f"Error fetching node info for {node_class}: {e}")
            return {}
    
    def get_available_models(self, model_type: str = "checkpoint") -> List[str]:
        """
        Get available models of a specific type
        
        Args:
            model_type: Type of model to fetch (e.g., "checkpoint", "lora", "controlnet")
            
        Returns:
            List of available model names
        """
        cache_key = f"models_{model_type}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
            
        models = []
        
        try:
            # Determine the appropriate node class based on model type
            node_class = self._get_node_class_for_model_type(model_type)
            if not node_class:
                logger.warning(f"Unknown model type: {model_type}")
                return []
                
            # Get node info
            node_info = self.get_node_info(node_class)
            if not node_info:
                return []
                
            # Extract model list from the appropriate input field
            input_field = self._get_input_field_for_model_type(model_type)
            if not input_field or "input" not in node_info:
                return []
                
            required_inputs = node_info.get("input", {}).get("required", {})
            if input_field in required_inputs and len(required_inputs[input_field]) > 0:
                models = required_inputs[input_field][0]
                
            # Cache the result
            self.model_cache[cache_key] = models
            return models
            
        except Exception as e:
            logger.error(f"Error fetching {model_type} models: {e}")
            return []
    
    def get_available_checkpoints(self) -> List[str]:
        """Get available checkpoint models"""
        return self.get_available_models("checkpoint")
    
    def get_available_loras(self) -> List[str]:
        """Get available LoRA models"""
        return self.get_available_models("lora")
    
    def get_available_controlnets(self) -> List[str]:
        """Get available ControlNet models"""
        return self.get_available_models("controlnet")
    
    def get_available_upscalers(self) -> List[str]:
        """Get available upscaler models"""
        return self.get_available_models("upscaler")
    
    def get_available_embeddings(self) -> List[str]:
        """Get available textual inversion embeddings"""
        return self.get_available_models("embedding")
    
    def get_model_metadata(self, model_name: str, model_type: str = "checkpoint") -> Dict[str, Any]:
        """
        Get metadata for a specific model if available
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            
        Returns:
            Dictionary with model metadata
        """
        cache_key = f"metadata_{model_type}_{model_name}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
            
        # Default metadata
        metadata = {
            "name": model_name,
            "type": model_type,
            "size": None,
            "hash": None,
            "path": None
        }
        
        try:
            # For now, attempt to get file stats from the expected location
            model_path = self._get_model_path(model_name, model_type)
            if model_path and os.path.exists(model_path):
                metadata["size"] = os.path.getsize(model_path)
                metadata["path"] = model_path
                
            # Cache the result
            self.model_cache[cache_key] = metadata
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching metadata for {model_type} model {model_name}: {e}")
            return metadata
    
    def get_model_folders(self) -> Dict[str, str]:
        """
        Get the folder paths for different model types
        
        Returns:
            Dictionary mapping model types to folder paths
        """
        try:
            response = requests.get(f"{self.comfyui_base_url}/system_stats")
            if response.status_code != 200:
                logger.warning("Failed to fetch system stats")
                return {}
                
            data = response.json()
            return data.get("model_folders", {})
            
        except Exception as e:
            logger.error(f"Error fetching model folders: {e}")
            return {}
    
    def get_available_samplers(self) -> List[str]:
        """
        Get available sampling methods
        
        Returns:
            List of available sampler names
        """
        if "samplers" in self.model_cache:
            return self.model_cache["samplers"]
            
        try:
            node_info = self.get_node_info("KSampler")
            if not node_info or "input" not in node_info:
                return []
                
            required_inputs = node_info.get("input", {}).get("required", {})
            samplers = []
            
            if "sampler_name" in required_inputs and len(required_inputs["sampler_name"]) > 0:
                samplers = required_inputs["sampler_name"][0]
                
            self.model_cache["samplers"] = samplers
            return samplers
            
        except Exception as e:
            logger.error(f"Error fetching samplers: {e}")
            return []
    
    def get_available_schedulers(self) -> List[str]:
        """
        Get available scheduler methods
        
        Returns:
            List of available scheduler names
        """
        if "schedulers" in self.model_cache:
            return self.model_cache["schedulers"]
            
        try:
            node_info = self.get_node_info("KSampler")
            if not node_info or "input" not in node_info:
                return []
                
            required_inputs = node_info.get("input", {}).get("required", {})
            schedulers = []
            
            if "scheduler" in required_inputs and len(required_inputs["scheduler"]) > 0:
                schedulers = required_inputs["scheduler"][0]
                
            self.model_cache["schedulers"] = schedulers
            return schedulers
            
        except Exception as e:
            logger.error(f"Error fetching schedulers: {e}")
            return []
    
    def get_controlnet_preprocessors(self) -> List[str]:
        """
        Get available ControlNet preprocessors
        
        Returns:
            List of available preprocessor types
        """
        if "preprocessors" in self.model_cache:
            return self.model_cache["preprocessors"]
            
        try:
            # Find all ControlNet preprocessor nodes
            response = requests.get(f"{self.comfyui_base_url}/object_info")
            if response.status_code != 200:
                logger.warning("Failed to fetch object info")
                return []
                
            data = response.json()
            preprocessors = []
            
            for node_name, node_info in data.items():
                if "ControlNetPreprocessor" in node_name:
                    preprocessors.append(node_name)
                    
            self.model_cache["preprocessors"] = preprocessors
            return preprocessors
            
        except Exception as e:
            logger.error(f"Error fetching ControlNet preprocessors: {e}")
            return []
    
    def clear_cache(self):
        """Clear the model cache to force re-fetching data"""
        self.model_cache.clear()
        self.node_info_cache.clear()
        logger.info("Model cache cleared")
    
    def _get_node_class_for_model_type(self, model_type: str) -> str:
        """Get the appropriate node class for a model type"""
        type_mapping = {
            "checkpoint": "CheckpointLoaderSimple",
            "lora": "LoraLoader",
            "controlnet": "ControlNetLoader",
            "upscaler": "UpscalerLoader",
            "embedding": "CLIPTextEncode",  # Not exactly right, but close
            "vae": "VAELoader"
        }
        return type_mapping.get(model_type, "")
    
    def _get_input_field_for_model_type(self, model_type: str) -> str:
        """Get the appropriate input field name for a model type"""
        field_mapping = {
            "checkpoint": "ckpt_name",
            "lora": "lora_name",
            "controlnet": "control_net_name",
            "upscaler": "upscaler_name",
            "embedding": "embedding_name",  # Approximate
            "vae": "vae_name"
        }
        return field_mapping.get(model_type, "")
    
    def _get_model_path(self, model_name: str, model_type: str) -> Optional[str]:
        """
        Attempt to construct the path to a model file
        
        Note: This is a best-effort function and may not work for all setups
        """
        # Get folder paths from ComfyUI if possible
        folders = self.get_model_folders()
        
        # Mapping of model types to their typical folder names
        folder_mapping = {
            "checkpoint": "checkpoints",
            "lora": "loras",
            "controlnet": "controlnet",
            "upscaler": "upscalers",
            "embedding": "embeddings",
            "vae": "vae"
        }
        
        # Try to get the folder path
        folder = folders.get(folder_mapping.get(model_type, ""), "")
        
        # If we can't get the folder from ComfyUI, use a generic guess
        if not folder:
            comfyui_dir = os.path.dirname(os.path.dirname(self.comfyui_base_url.replace("http://", "").replace("https://", "").split(":")[0]))
            folder = os.path.join(comfyui_dir, "models", folder_mapping.get(model_type, ""))
        
        # Check if model_name already includes the folder path
        if os.path.exists(model_name):
            return model_name
            
        # Check if the model exists in the expected folder
        if folder and os.path.exists(folder):
            model_path = os.path.join(folder, model_name)
            if os.path.exists(model_path):
                return model_path
                
        return None
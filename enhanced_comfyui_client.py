import requests
import json
import time
import logging
import os
import base64
from typing import Dict, List, Optional, Any, Tuple, Union
from workflow_analyzer import WorkflowAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUIClient")

class ComfyUIClient:
    """
    Enhanced ComfyUI client with support for multiple workflow types
    and advanced parameter handling.
    """
    
    def __init__(self, base_url: str, workflows_dir: str = "workflows"):
        self.base_url = base_url
        self.workflows_dir = workflows_dir
        self.available_models = self._get_available_models()
        self.workflow_analyzer = WorkflowAnalyzer()
        self.samplers = self._get_available_samplers()
        
        # Create workflows directory if it doesn't exist
        os.makedirs(workflows_dir, exist_ok=True)
        
        # Preload and analyze available workflows
        self.available_workflows = self._analyze_available_workflows()
        
    def _analyze_available_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all workflows in the workflows directory"""
        workflows = {}
        
        try:
            for filename in os.listdir(self.workflows_dir):
                if filename.endswith(".json"):
                    workflow_id = os.path.splitext(filename)[0]
                    workflow_path = os.path.join(self.workflows_dir, filename)
                    
                    # Get available parameters for this workflow
                    available_params = self.workflow_analyzer.get_available_params(workflow_path)
                    
                    workflows[workflow_id] = {
                        "path": workflow_path,
                        "parameters": available_params
                    }
            
            logger.info(f"Analyzed {len(workflows)} workflows")
            return workflows
            
        except Exception as e:
            logger.error(f"Error analyzing workflows: {e}")
            return {}

    def _get_available_models(self) -> List[str]:
        """Fetch list of available checkpoint models from ComfyUI"""
        try:
            response = requests.get(f"{self.base_url}/object_info/CheckpointLoaderSimple")
            if response.status_code != 200:
                logger.warning("Failed to fetch model list; using default handling")
                return []
            data = response.json()
            models = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            logger.info(f"Available models: {models}")
            return models
        except Exception as e:
            logger.warning(f"Error fetching models: {e}")
            return []
            
    def _get_available_samplers(self) -> List[str]:
        """Fetch list of available samplers from ComfyUI"""
        try:
            response = requests.get(f"{self.base_url}/object_info/KSampler")
            if response.status_code != 200:
                logger.warning("Failed to fetch sampler list; using default options")
                return ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", 
                        "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde"]
                
            data = response.json()
            samplers = data["KSampler"]["input"]["required"]["sampler_name"][0]
            logger.info(f"Available samplers: {samplers}")
            return samplers
        except Exception as e:
            logger.warning(f"Error fetching samplers: {e}")
            return ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", 
                    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde"]

    def list_available_workflows(self) -> Dict[str, List[str]]:
        """Return a list of available workflows and their parameters"""
        result = {}
        
        for workflow_id, workflow_info in self.available_workflows.items():
            result[workflow_id] = workflow_info["parameters"]
            
        return result
        
    def list_available_models(self) -> List[str]:
        """Return a list of available checkpoint models"""
        return self.available_models
        
    def list_available_samplers(self) -> List[str]:
        """Return a list of available samplers"""
        return self.samplers

    def generate_image(self, prompt: str, width: int = 512, height: int = 512, 
                      workflow_id: str = "basic_api_test", model: Optional[str] = None, 
                      **kwargs) -> str:
        """
        Generate an image using a text prompt and optional parameters.
        
        Args:
            prompt: The text prompt for image generation
            width: Image width in pixels
            height: Image height in pixels
            workflow_id: ID of the workflow to use
            model: Model checkpoint to use
            **kwargs: Additional parameters to pass to the workflow
            
        Returns:
            URL of the generated image
        """
        return self._run_workflow(
            workflow_id=workflow_id,
            params={
                "prompt": prompt,
                "width": width,
                "height": height,
                "model": model,
                **kwargs
            }
        )
        
    def img2img(self, prompt: str, image_data: Union[str, bytes], 
               strength: float = 0.75, width: int = 512, height: int = 512,
               workflow_id: str = "img2img", model: Optional[str] = None, 
               **kwargs) -> str:
        """
        Generate an image based on another image and a text prompt.
        
        Args:
            prompt: The text prompt for image generation
            image_data: Base64 encoded image data or raw image bytes
            strength: How much to transform the input image (0-1)
            width: Output image width in pixels
            height: Output image height in pixels
            workflow_id: ID of the workflow to use
            model: Model checkpoint to use
            **kwargs: Additional parameters to pass to the workflow
            
        Returns:
            URL of the generated image
        """
        # Convert raw image bytes to base64 if needed
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode('utf-8')
            
        # Remove base64 prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',', 1)[1]
            
        return self._run_workflow(
            workflow_id=workflow_id,
            params={
                "prompt": prompt,
                "image": image_data,
                "strength": strength,
                "width": width,
                "height": height,
                "model": model,
                **kwargs
            }
        )
        
    def inpaint(self, prompt: str, image_data: Union[str, bytes], 
               mask_data: Union[str, bytes], width: int = 512, height: int = 512,
               workflow_id: str = "inpaint", model: Optional[str] = None, 
               **kwargs) -> str:
        """
        Inpaint parts of an image based on a mask.
        
        Args:
            prompt: The text prompt for inpainting
            image_data: Base64 encoded image data or raw image bytes
            mask_data: Base64 encoded mask data or raw mask bytes
            width: Output image width in pixels
            height: Output image height in pixels
            workflow_id: ID of the workflow to use
            model: Model checkpoint to use
            **kwargs: Additional parameters to pass to the workflow
            
        Returns:
            URL of the generated image
        """
        # Convert raw image/mask bytes to base64 if needed
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode('utf-8')
        if isinstance(mask_data, bytes):
            mask_data = base64.b64encode(mask_data).decode('utf-8')
            
        # Remove base64 prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',', 1)[1]
        if mask_data.startswith('data:image'):
            mask_data = mask_data.split(',', 1)[1]
            
        return self._run_workflow(
            workflow_id=workflow_id,
            params={
                "prompt": prompt,
                "image": image_data,
                "mask": mask_data,
                "width": width,
                "height": height,
                "model": model,
                **kwargs
            }
        )

    def _run_workflow(self, workflow_id: str, params: Dict[str, Any]) -> str:
        """
        Run a ComfyUI workflow with the given parameters.
        
        Args:
            workflow_id: ID of the workflow to use
            params: Parameters to set in the workflow
            
        Returns:
            URL of the generated image or other output
        """
        try:
            # Validate workflow exists
            if workflow_id not in self.available_workflows:
                raise ValueError(f"Workflow '{workflow_id}' not found. Available workflows: {list(self.available_workflows.keys())}")
                
            workflow_path = self.available_workflows[workflow_id]["path"]
            
            # Load workflow
            with open(workflow_path, "r") as f:
                workflow = json.load(f)
                
            # Validate model if specified
            if "model" in params and params["model"]:
                model = params["model"]
                # Strip accidental quotes
                if model.endswith("'") or model.endswith('"'):
                    model = model.rstrip("'\"")
                    params["model"] = model
                    logger.info(f"Corrected model name: {model}")
                # Validate model is available
                if self.available_models and model not in self.available_models:
                    raise ValueError(f"Model '{model}' not in available models: {self.available_models}")
            
            # Use the workflow analyzer to apply parameters
            modified_workflow = self.workflow_analyzer.apply_parameters(workflow, params)
            
            # Submit workflow to ComfyUI
            logger.info(f"Submitting workflow {workflow_id} to ComfyUI...")
            response = requests.post(f"{self.base_url}/prompt", json={"prompt": modified_workflow})
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to queue workflow: {response.status_code} - {response.text}")

            prompt_id = response.json()["prompt_id"]
            logger.info(f"Queued workflow with prompt_id: {prompt_id}")

            # Wait for completion and get results
            max_attempts = 30
            for _ in range(max_attempts):
                history = requests.get(f"{self.base_url}/history/{prompt_id}").json()
                if history.get(prompt_id):
                    outputs = history[prompt_id]["outputs"]
                    logger.info("Workflow outputs: %s", json.dumps(outputs, indent=2))
                    
                    # Find image nodes in output
                    image_nodes = []
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            image_nodes.append((node_id, node_output["images"]))
                    
                    if not image_nodes:
                        raise RuntimeError(f"No output node with images found: {outputs}")
                        
                    # Use the last image node's output (usually the final result)
                    last_node_id, images = image_nodes[-1]
                    image_filename = images[0]["filename"]
                    image_url = f"{self.base_url}/view?filename={image_filename}&subfolder=&type=output"
                    
                    logger.info(f"Generated image URL: {image_url}")
                    return image_url
                    
                time.sleep(1)
                
            raise RuntimeError(f"Workflow {prompt_id} didn't complete within {max_attempts} seconds")

        except FileNotFoundError:
            raise FileNotFoundError(f"Workflow file for '{workflow_id}' not found")
        except KeyError as e:
            raise KeyError(f"Workflow error - invalid node or input: {e}")
        except requests.RequestException as e:
            raise RuntimeError(f"ComfyUI API error: {e}")
        
    def upload_workflow(self, workflow_id: str, workflow_data: Dict[str, Any]) -> bool:
        """
        Upload a new workflow to the workflows directory.
        
        Args:
            workflow_id: ID for the new workflow
            workflow_data: Workflow data in ComfyUI API format
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Sanitize workflow ID to be a valid filename
            safe_id = "".join(c for c in workflow_id if c.isalnum() or c in "-_")
            
            workflow_path = os.path.join(self.workflows_dir, f"{safe_id}.json")
            
            with open(workflow_path, "w") as f:
                json.dump(workflow_data, f, indent=2)
                
            # Analyze the new workflow
            available_params = self.workflow_analyzer.get_available_params(workflow_path)
            
            # Add to available workflows
            self.available_workflows[safe_id] = {
                "path": workflow_path,
                "parameters": available_params
            }
            
            logger.info(f"Uploaded workflow '{safe_id}' with parameters: {available_params}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading workflow: {e}")
            return False
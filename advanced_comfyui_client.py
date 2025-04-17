import requests
import json
import time
import logging
import os
import base64
from typing import Dict, List, Optional, Any, Tuple, Union
from workflow_analyzer import WorkflowAnalyzer
from model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedComfyUIClient")

class AdvancedComfyUIClient:
    """
    Advanced ComfyUI client with support for ControlNet, LoRA, and other 
    advanced features for image generation.
    """
    
    def __init__(self, base_url: str, workflows_dir: str = "workflows"):
        self.base_url = base_url
        self.workflows_dir = workflows_dir
        self.workflow_analyzer = WorkflowAnalyzer()
        self.model_manager = ModelManager(base_url)
        
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

    def list_available_workflows(self) -> Dict[str, List[str]]:
        """Return a list of available workflows and their parameters"""
        result = {}
        
        for workflow_id, workflow_info in self.available_workflows.items():
            result[workflow_id] = workflow_info.get("parameters", [])
            
        return result
        
    def list_available_models(self) -> List[str]:
        """Return a list of available checkpoint models"""
        return self.model_manager.get_available_checkpoints()
    
    def list_available_loras(self) -> List[str]:
        """Return a list of available LoRA models"""
        return self.model_manager.get_available_loras()
    
    def list_available_controlnets(self) -> List[str]:
        """Return a list of available ControlNet models"""
        return self.model_manager.get_available_controlnets()
    
    def list_available_samplers(self) -> List[str]:
        """Return a list of available samplers"""
        return self.model_manager.get_available_samplers()

    def generate_image(self, prompt: str, width: int = 512, height: int = 512, 
                      workflow_id: str = "basic_api_test", model: Optional[str] = None, 
                      negative_prompt: str = "", seed: int = -1, steps: int = 20,
                      cfg_scale: float = 7.0, scheduler: str = "normal",
                      sampler: Optional[str] = None, **kwargs) -> str:
        """
        Generate an image using a text prompt and optional parameters.
        
        Args:
            prompt: The text prompt for image generation
            width: Image width in pixels
            height: Image height in pixels
            workflow_id: ID of the workflow to use
            model: Model checkpoint to use
            negative_prompt: Negative prompt for generation
            seed: Seed for generation (-1 for random)
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            scheduler: Sampler scheduler
            sampler: Sampler algorithm
            **kwargs: Additional parameters to pass to the workflow
            
        Returns:
            URL of the generated image
        """
        return self._run_workflow(
            workflow_id=workflow_id,
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "scheduler": scheduler,
                "sampler": sampler,
                **kwargs
            }
        )
        
    def img2img(self, prompt: str, image_data: Union[str, bytes], 
               strength: float = 0.75, width: int = 512, height: int = 512,
               workflow_id: str = "img2img", model: Optional[str] = None, 
               negative_prompt: str = "", seed: int = -1, steps: int = 20,
               cfg_scale: float = 7.0, scheduler: str = "normal",
               sampler: Optional[str] = None, **kwargs) -> str:
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
            negative_prompt: Negative prompt for generation
            seed: Seed for generation (-1 for random)
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            scheduler: Sampler scheduler
            sampler: Sampler algorithm
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
                "negative_prompt": negative_prompt,
                "image": image_data,
                "strength": strength,
                "width": width,
                "height": height,
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "scheduler": scheduler,
                "sampler": sampler,
                **kwargs
            }
        )
        
    def inpaint(self, prompt: str, image_data: Union[str, bytes], 
               mask_data: Union[str, bytes], width: int = 512, height: int = 512,
               workflow_id: str = "inpaint", model: Optional[str] = None, 
               negative_prompt: str = "", seed: int = -1, steps: int = 20,
               cfg_scale: float = 7.0, scheduler: str = "normal",
               sampler: Optional[str] = None, **kwargs) -> str:
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
            negative_prompt: Negative prompt for generation
            seed: Seed for generation (-1 for random)
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            scheduler: Sampler scheduler
            sampler: Sampler algorithm
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
                "negative_prompt": negative_prompt,
                "image": image_data,
                "mask": mask_data,
                "width": width,
                "height": height,
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "scheduler": scheduler,
                "sampler": sampler,
                **kwargs
            }
        )
    
    def controlnet(self, prompt: str, control_image: Union[str, bytes], 
                  control_type: str = "canny", control_strength: float = 1.0,
                  width: int = 512, height: int = 512, workflow_id: str = "controlnet",
                  model: Optional[str] = None, controlnet_model: Optional[str] = None,
                  negative_prompt: str = "", seed: int = -1, steps: int = 20,
                  cfg_scale: float = 7.0, scheduler: str = "normal",
                  sampler: Optional[str] = None, **kwargs) -> str:
        """
        Generate an image using ControlNet guidance.
        
        Args:
            prompt: The text prompt for image generation
            control_image: Base64 encoded control image or raw image bytes
            control_type: Type of control ("canny", "depth", "pose", etc.)
            control_strength: Strength of control (0-1)
            width: Output image width in pixels
            height: Output image height in pixels
            workflow_id: ID of the workflow to use
            model: Main model checkpoint to use
            controlnet_model: Specific ControlNet model to use
            negative_prompt: Negative prompt for generation
            seed: Seed for generation (-1 for random)
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            scheduler: Sampler scheduler
            sampler: Sampler algorithm
            **kwargs: Additional parameters to pass to the workflow
            
        Returns:
            URL of the generated image
        """
        # Convert raw image bytes to base64 if needed
        if isinstance(control_image, bytes):
            control_image = base64.b64encode(control_image).decode('utf-8')
            
        # Remove base64 prefix if present
        if control_image.startswith('data:image'):
            control_image = control_image.split(',', 1)[1]
            
        return self._run_workflow(
            workflow_id=workflow_id,
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "control_image": control_image,
                "control_type": control_type,
                "control_strength": control_strength,
                "width": width,
                "height": height,
                "model": model,
                "controlnet_model": controlnet_model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "scheduler": scheduler,
                "sampler": sampler,
                **kwargs
            }
        )
    
    def lora_generation(self, prompt: str, lora_name: str, lora_strength: float = 0.8,
                      width: int = 512, height: int = 512, workflow_id: str = "lora",
                      model: Optional[str] = None, negative_prompt: str = "", 
                      seed: int = -1, steps: int = 20, cfg_scale: float = 7.0, 
                      scheduler: str = "normal", sampler: Optional[str] = None, **kwargs) -> str:
        """
        Generate an image using a text prompt with LoRA model.
        
        Args:
            prompt: The text prompt for image generation
            lora_name: Name of the LoRA model to use
            lora_strength: Strength of LoRA effect (0-1)
            width: Image width in pixels
            height: Image height in pixels
            workflow_id: ID of the workflow to use
            model: Base model checkpoint to use
            negative_prompt: Negative prompt for generation
            seed: Seed for generation (-1 for random)
            steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            scheduler: Sampler scheduler
            sampler: Sampler algorithm
            **kwargs: Additional parameters to pass to the workflow
            
        Returns:
            URL of the generated image
        """
        return self._run_workflow(
            workflow_id=workflow_id,
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "lora_name": lora_name,
                "lora_strength": lora_strength,
                "width": width,
                "height": height,
                "model": model,
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "scheduler": scheduler,
                "sampler": sampler,
                **kwargs
            }
        )
    
    def upscale(self, image_data: Union[str, bytes], scale_factor: float = 2.0,
               upscaler: str = "ESRGAN_4x", workflow_id: str = "upscale", **kwargs) -> str:
        """
        Upscale an image using an upscaler model.
        
        Args:
            image_data: Base64 encoded image data or raw image bytes
            scale_factor: Factor to scale the image by
            upscaler: Name of the upscaler model to use
            workflow_id: ID of the workflow to use
            **kwargs: Additional parameters to pass to the workflow
            
        Returns:
            URL of the upscaled image
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
                "image": image_data,
                "scale_factor": scale_factor,
                "upscaler": upscaler,
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
            with open(workflow_path, "r", encoding="utf-8") as f:
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
                available_models = self.model_manager.get_available_checkpoints()
                if available_models and model not in available_models:
                    raise ValueError(f"Model '{model}' not in available models: {available_models}")
            
            # Validate LoRA if specified
            if "lora_name" in params and params["lora_name"]:
                lora = params["lora_name"]
                # Validate LoRA is available
                available_loras = self.model_manager.get_available_loras()
                if available_loras and lora not in available_loras:
                    raise ValueError(f"LoRA '{lora}' not in available models: {available_loras}")
            
            # Validate ControlNet if specified
            if "controlnet_model" in params and params["controlnet_model"]:
                controlnet = params["controlnet_model"]
                # Validate ControlNet is available
                available_controlnets = self.model_manager.get_available_controlnets()
                if available_controlnets and controlnet not in available_controlnets:
                    raise ValueError(f"ControlNet '{controlnet}' not in available models: {available_controlnets}")
            
            # Validate upscaler if specified
            if "upscaler" in params and params["upscaler"]:
                upscaler = params["upscaler"]
                # Validate upscaler is available
                available_upscalers = self.model_manager.get_available_upscalers()
                if available_upscalers and upscaler not in available_upscalers:
                    raise ValueError(f"Upscaler '{upscaler}' not in available models: {available_upscalers}")
            
            # Validate sampler if specified
            if "sampler" in params and params["sampler"]:
                sampler = params["sampler"]
                # Validate sampler is available
                available_samplers = self.model_manager.get_available_samplers()
                if available_samplers and sampler not in available_samplers:
                    raise ValueError(f"Sampler '{sampler}' not in available samplers: {available_samplers}")
            
            # Handle seed value for KSampler node
            if "seed" in params and params["seed"] < 0:
                # ComfyUI requires non-negative seeds, so we'll use a positive value
                params["seed"] = int(time.time()) % 2**32  # Use current time as seed
                logger.info(f"Using auto-generated seed: {params['seed']}")
                
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
            # 타임아웃 시간을 30초에서 180초로 늘림
            max_attempts = 180  # 3분으로 증가
            for attempt_num in range(max_attempts):
                try:
                    history_response = requests.get(f"{self.base_url}/history/{prompt_id}")
                    if history_response.status_code != 200:
                        logger.warning(f"History API returned {history_response.status_code} on attempt {attempt_num+1}")
                        time.sleep(1)
                        continue
                        
                    history = history_response.json()
                    
                    if not history:
                        logger.warning(f"Empty history response on attempt {attempt_num+1}")
                        time.sleep(1)
                        continue
                    
                    if prompt_id in history:
                        outputs = history[prompt_id]["outputs"]
                        logger.info("Workflow outputs: %s", json.dumps(outputs, indent=2))
                        
                        # Find image nodes in output
                        image_nodes = []
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                image_nodes.append((node_id, node_output["images"]))
                        
                        if not image_nodes:
                            logger.warning(f"No output node with images found: {outputs}")
                            # 이미지 노드가 없지만 워크플로우가 완료되었다면 계속 기다리지 않음
                            if all(node.get("status", {}).get("completed", False) for node in outputs.values()):
                                raise RuntimeError(f"Workflow completed but no images were produced")
                        else:
                            # Use the last image node's output (usually the final result)
                            last_node_id, images = image_nodes[-1]
                            image_filename = images[0]["filename"]
                            image_url = f"{self.base_url}/view?filename={image_filename}&subfolder=&type=output"
                            
                            logger.info(f"Generated image URL: {image_url}")
                            return image_url
                    else:
                        logger.info(f"Prompt {prompt_id} not found in history yet, waiting... (attempt {attempt_num+1}/{max_attempts})")
                        
                except Exception as e:
                    logger.error(f"Error checking workflow status on attempt {attempt_num+1}: {e}")
                
                # 첫 10초는 빠르게 폴링, 그 이후로는 조금 더 천천히
                if attempt_num < 10:
                    time.sleep(1)
                else:
                    time.sleep(2)
                
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
            
            with open(workflow_path, "w", encoding="utf-8") as f:
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
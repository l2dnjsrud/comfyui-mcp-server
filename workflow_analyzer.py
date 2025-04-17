import json
import logging
from typing import Dict, Tuple, List, Optional, Any

logger = logging.getLogger("WorkflowAnalyzer")

class WorkflowAnalyzer:
    """
    Analyzes ComfyUI workflows to dynamically map parameters to the correct nodes.
    This allows for more flexible parameter handling without hardcoded node mappings.
    """
    
    # Common node types and their typical parameter names
    NODE_TYPE_MAPPING = {
        "CLIPTextEncode": {"prompt": "text"},
        "EmptyLatentImage": {"width": "width", "height": "height"},
        "KSampler": {
            "seed": "seed", 
            "steps": "steps", 
            "cfg": "cfg", 
            "sampler": "sampler_name", 
            "scheduler": "scheduler",
            "denoise": "denoise"
        },
        "CheckpointLoaderSimple": {"model": "ckpt_name"}
    }
    
    # Node types that typically produce image outputs
    IMAGE_OUTPUT_NODES = ["SaveImage", "PreviewImage", "VaeDecodeForInpaint"]
    
    def __init__(self):
        self.cache = {}  # Cache analyzed workflows
    
    def analyze_workflow(self, workflow_path: str) -> Dict[str, Dict[str, Tuple[str, str]]]:
        """
        Analyzes a workflow file and returns parameter mappings.
        Returns a dictionary mapping parameter names to (node_id, input_name) tuples.
        """
        # Check cache first
        if workflow_path in self.cache:
            return self.cache[workflow_path]
            
        try:
            with open(workflow_path, "r") as f:
                workflow = json.load(f)
                
            # Result mapping will contain:
            # {
            #   "inputs": {"param_name": ("node_id", "node_input_name")},
            #   "outputs": {"image": "node_id"}
            # }
            result = {
                "inputs": {},
                "outputs": {}
            }
            
            # Find nodes by their class types
            for node_id, node_data in workflow.items():
                if not isinstance(node_data, dict) or "class_type" not in node_data:
                    continue
                    
                class_type = node_data["class_type"]
                
                # Map input parameters
                if class_type in self.NODE_TYPE_MAPPING:
                    for param_name, node_input_name in self.NODE_TYPE_MAPPING[class_type].items():
                        result["inputs"][param_name] = (node_id, node_input_name)
                
                # Identify output nodes
                if class_type in self.IMAGE_OUTPUT_NODES:
                    result["outputs"]["image"] = node_id
            
            # Cache the result
            self.cache[workflow_path] = result
            logger.info(f"Analyzed workflow {workflow_path}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing workflow {workflow_path}: {e}")
            return {"inputs": {}, "outputs": {}}
    
    def apply_parameters(self, workflow: Dict[str, Any], params: Dict[str, Any], 
                         mapping: Optional[Dict[str, Tuple[str, str]]] = None) -> Dict[str, Any]:
        """
        Applies parameters to a workflow based on dynamic or provided mapping.
        Returns the modified workflow.
        """
        # Create a deep copy to avoid modifying the original
        workflow_copy = json.loads(json.dumps(workflow))
        
        # Use provided mapping or default behavior
        param_mapping = mapping or {}
        
        for param_name, value in params.items():
            # Skip if value is None
            if value is None:
                continue
                
            # If we have an explicit mapping for this parameter
            if param_name in param_mapping:
                node_id, input_name = param_mapping[param_name]
                if node_id in workflow_copy:
                    workflow_copy[node_id]["inputs"][input_name] = value
                    logger.info(f"Set parameter {param_name}={value} to node {node_id}.{input_name}")
            else:
                # Try to find a matching node based on class types
                for node_id, node_data in workflow_copy.items():
                    if not isinstance(node_data, dict) or "class_type" not in node_data:
                        continue
                        
                    class_type = node_data["class_type"]
                    if class_type in self.NODE_TYPE_MAPPING and param_name in self.NODE_TYPE_MAPPING[class_type]:
                        input_name = self.NODE_TYPE_MAPPING[class_type][param_name]
                        workflow_copy[node_id]["inputs"][input_name] = value
                        logger.info(f"Auto-mapped parameter {param_name}={value} to {node_id}.{input_name}")
        
        return workflow_copy
                
    def get_available_params(self, workflow_path: str) -> List[str]:
        """
        Returns a list of parameter names that can be set for a given workflow.
        """
        mapping = self.analyze_workflow(workflow_path)
        return list(mapping["inputs"].keys())

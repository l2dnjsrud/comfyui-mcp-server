import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from datetime import datetime
from contextlib import asynccontextmanager

import websockets
from mcp.server.fastmcp import FastMCP

# Import our enhanced components
from advanced_comfyui_client import AdvancedComfyUIClient
from progress_tracker import ProgressTracker
from auth import AuthManager
from workflow_analyzer import WorkflowAnalyzer
from model_manager import ModelManager
from job_queue import JobQueue, Job
from db_manager import SQLiteDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/mcp_server.log')
    ]
)
logger = logging.getLogger("MCP_Server")

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Define application context for dependency injection
class AppContext:
    """Application context with all service dependencies"""
    
    def __init__(self, 
                 comfyui_client: AdvancedComfyUIClient,
                 progress_tracker: ProgressTracker,
                 auth_manager: AuthManager,
                 workflow_analyzer: WorkflowAnalyzer,
                 model_manager: ModelManager,
                 job_queue: JobQueue,
                 db_manager: SQLiteDatabaseManager):
        self.comfyui_client = comfyui_client
        self.progress_tracker = progress_tracker
        self.auth_manager = auth_manager
        self.workflow_analyzer = workflow_analyzer
        self.model_manager = model_manager
        self.job_queue = job_queue
        self.db_manager = db_manager
        
        # WebSocket clients for broadcasting events
        self.ws_clients = set()

# Load configuration from environment or config file
def load_config() -> Dict[str, Any]:
    """Load server configuration from environment or file"""
    config = {
        "comfyui_url": os.environ.get("COMFYUI_URL", "http://localhost:8188"),
        "comfyui_ws_url": os.environ.get("COMFYUI_WS_URL", "ws://localhost:8188/ws"),
        "mcp_server_host": os.environ.get("MCP_SERVER_HOST", "localhost"),
        "mcp_server_port": int(os.environ.get("MCP_SERVER_PORT", "9000")),
        "dashboard_host": os.environ.get("DASHBOARD_HOST", "localhost"),
        "dashboard_port": int(os.environ.get("DASHBOARD_PORT", "8080")),
        "enable_auth": os.environ.get("ENABLE_AUTH", "false").lower() == "true",
        "auth_config_path": os.environ.get("AUTH_CONFIG_PATH", "auth_config.json"),
        "workflows_dir": os.environ.get("WORKFLOWS_DIR", "workflows"),
        "db_path": os.environ.get("DB_PATH", "data/mcp_server.db"),
        "queue_save_path": os.environ.get("QUEUE_SAVE_PATH", "data/queue_state.json"),
        "max_concurrent_jobs": int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
    }
    
    # Try to load from config file if specified
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    return config

# Initialize configuration
config = load_config()

# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(config["db_path"]), exist_ok=True)

# Lifespan management for application startup/shutdown
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AppContext:
    """Manage application lifecycle with proper initialization and cleanup"""
    logger.info("Starting MCP server lifecycle...")
    
    # Initialize components
    comfyui_client = AdvancedComfyUIClient(
        base_url=config["comfyui_url"],
        workflows_dir=config["workflows_dir"]
    )
    
    workflow_analyzer = WorkflowAnalyzer()
    
    model_manager = ModelManager(config["comfyui_url"])
    
    auth_manager = AuthManager(enable_auth=config["enable_auth"])
    if config["enable_auth"]:
        auth_manager.load_api_keys(config["auth_config_path"])
    
    progress_tracker = ProgressTracker(comfyui_ws_url=config["comfyui_ws_url"])
    await progress_tracker.start()
    
    # Initialize database
    db_manager = SQLiteDatabaseManager(config["db_path"])
    await db_manager.initialize()
    
    # Initialize job queue
    job_queue = JobQueue(
        max_concurrent_jobs=config["max_concurrent_jobs"],
        save_path=config["queue_save_path"]
    )
    
    # Set up job executor function
    async def job_executor(job: Job) -> Dict[str, Any]:
        """Execute a generation job"""
        job_type = job.job_type
        params = job.params
        
        try:
            # Check which job type to execute
            if job_type == "txt2img":
                image_url = comfyui_client.generate_image(**params)
            elif job_type == "img2img":
                image_url = comfyui_client.img2img(**params)
            elif job_type == "inpaint":
                image_url = comfyui_client.inpaint(**params)
            elif job_type == "controlnet":
                image_url = comfyui_client.controlnet(**params)
            elif job_type == "lora":
                image_url = comfyui_client.lora_generation(**params)
            elif job_type == "upscale":
                image_url = comfyui_client.upscale(**params)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            
            # Save job to database
            await db_manager.save_job({
                "job_id": job.job_id,
                "mode": job_type,
                "prompt": params.get("prompt", ""),
                "width": params.get("width", 512),
                "height": params.get("height", 512),
                "workflow_id": params.get("workflow_id", "basic_api_test"),
                "model": params.get("model"),
                "image_url": image_url,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "parameters": {k: v for k, v in params.items() if k not in ["prompt", "width", "height", "workflow_id", "model"]}
            })
            
            return {"image_url": image_url}
        except Exception as e:
            logger.error(f"Error executing job {job.job_id}: {e}")
            return {"error": str(e)}
    
    # Set job executor
    job_queue.set_job_executor(job_executor)
    
    # Set up progress updater function
    async def progress_updater(job: Job, update: Dict[str, Any]):
        """Handle job progress updates"""
        # Forward progress updates to all client callbacks
        pass
    
    # Set progress updater
    job_queue.set_progress_updater(progress_updater)
    
    # Start job queue
    await job_queue.start()
    
    try:
        # Create and yield the application context
        context = AppContext(
            comfyui_client=comfyui_client,
            progress_tracker=progress_tracker,
            auth_manager=auth_manager,
            workflow_analyzer=workflow_analyzer,
            model_manager=model_manager,
            job_queue=job_queue,
            db_manager=db_manager
        )
        
        logger.info("MCP server components initialized")
        yield context
        
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down MCP server components...")
        await job_queue.stop()
        await progress_tracker.stop()
        await db_manager.close()
        logger.info("MCP server shutdown complete")

# Initialize FastMCP with lifespan
mcp = FastMCP("ComfyUI_MCP_Server", lifespan=app_lifespan)

# Define MCP tools
@mcp.tool()
async def generate_image(params: str) -> Dict[str, Any]:
    """
    Generate an image using ComfyUI.
    
    Args:
        params: JSON string with parameters:
            - prompt: Text prompt for image generation
            - width: Image width in pixels (default: 512)
            - height: Image height in pixels (default: 512)
            - workflow_id: Workflow to use (default: "basic_api_test")
            - model: Model checkpoint to use (optional)
            - Additional parameters depending on workflow
            
    Returns:
        Dictionary with image URL or error message
    """
    context = await mcp.context_getter()
    logger.info(f"Received generate_image request")
    try:
        param_dict = json.loads(params)
        
        # Add job to queue
        job_id = await context.job_queue.add_job(
            job_type="txt2img",
            params=param_dict
        )
        
        # Return job ID for tracking
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Image generation job queued"
        }
    except Exception as e:
        logger.error(f"Error in generate_image: {e}")
        return {"error": str(e)}

@mcp.tool()
async def img2img(params: str) -> Dict[str, Any]:
    """
    Generate an image based on another image and a text prompt.
    
    Args:
        params: JSON string with parameters:
            - prompt: Text prompt for image generation
            - image: Base64 encoded image data
            - strength: How much to transform the input image (0-1)
            - width: Output image width in pixels (default: 512)
            - height: Output image height in pixels (default: 512)
            - workflow_id: Workflow to use (default: "img2img")
            - model: Model checkpoint to use (optional)
            - Additional parameters depending on workflow
            
    Returns:
        Dictionary with image URL or error message
    """
    context = await mcp.context_getter()
    logger.info("Received img2img request")
    try:
        param_dict = json.loads(params)
        
        # Validate image data exists
        if "image" not in param_dict or not param_dict["image"]:
            raise ValueError("Image data is required for img2img")
        
        # Add job to queue
        job_id = await context.job_queue.add_job(
            job_type="img2img",
            params=param_dict
        )
        
        # Return job ID for tracking
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Image-to-image job queued"
        }
    except Exception as e:
        logger.error(f"Error in img2img: {e}")
        return {"error": str(e)}

@mcp.tool()
async def inpaint(params: str) -> Dict[str, Any]:
    """
    Inpaint parts of an image based on a mask.
    
    Args:
        params: JSON string with parameters:
            - prompt: Text prompt for inpainting
            - image: Base64 encoded image data
            - mask: Base64 encoded mask data
            - width: Output image width in pixels (default: 512)
            - height: Output image height in pixels (default: 512)
            - workflow_id: Workflow to use (default: "inpaint")
            - model: Model checkpoint to use (optional)
            - Additional parameters depending on workflow
            
    Returns:
        Dictionary with image URL or error message
    """
    context = await mcp.context_getter()
    logger.info("Received inpaint request")
    try:
        param_dict = json.loads(params)
        
        # Validate image and mask data exist
        if "image" not in param_dict or not param_dict["image"]:
            raise ValueError("Image data is required for inpainting")
        if "mask" not in param_dict or not param_dict["mask"]:
            raise ValueError("Mask data is required for inpainting")
        
        # Add job to queue
        job_id = await context.job_queue.add_job(
            job_type="inpaint",
            params=param_dict
        )
        
        # Return job ID for tracking
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Inpainting job queued"
        }
    except Exception as e:
        logger.error(f"Error in inpaint: {e}")
        return {"error": str(e)}

@mcp.tool()
async def controlnet(params: str) -> Dict[str, Any]:
    """
    Generate an image using ControlNet guidance.
    
    Args:
        params: JSON string with parameters:
            - prompt: Text prompt for image generation
            - control_image: Base64 encoded control image
            - control_type: Type of control ("canny", "depth", "pose", etc.)
            - control_strength: Strength of control (0-1)
            - width: Output image width in pixels (default: 512)
            - height: Output image height in pixels (default: 512)
            - workflow_id: Workflow to use (default: "controlnet")
            - model: Model checkpoint to use (optional)
            - controlnet_model: ControlNet model to use (optional)
            - Additional parameters depending on workflow
            
    Returns:
        Dictionary with image URL or error message
    """
    context = await mcp.context_getter()
    logger.info("Received controlnet request")
    try:
        param_dict = json.loads(params)
        
        # Validate control image exists
        if "control_image" not in param_dict or not param_dict["control_image"]:
            raise ValueError("Control image is required for ControlNet")
        
        # Add job to queue
        job_id = await context.job_queue.add_job(
            job_type="controlnet",
            params=param_dict
        )
        
        # Return job ID for tracking
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "ControlNet job queued"
        }
    except Exception as e:
        logger.error(f"Error in controlnet: {e}")
        return {"error": str(e)}

@mcp.tool()
async def lora_generation(params: str) -> Dict[str, Any]:
    """
    Generate an image using LoRA model.
    
    Args:
        params: JSON string with parameters:
            - prompt: Text prompt for image generation
            - lora_name: Name of LoRA model to use
            - lora_strength: Strength of LoRA effect (0-1)
            - width: Output image width in pixels (default: 512)
            - height: Output image height in pixels (default: 512)
            - workflow_id: Workflow to use (default: "lora")
            - model: Base model checkpoint to use (optional)
            - Additional parameters depending on workflow
            
    Returns:
        Dictionary with image URL or error message
    """
    context = await mcp.context_getter()
    logger.info("Received lora_generation request")
    try:
        param_dict = json.loads(params)
        
        # Validate LoRA name exists
        if "lora_name" not in param_dict or not param_dict["lora_name"]:
            raise ValueError("LoRA name is required for LoRA generation")
        
        # Add job to queue
        job_id = await context.job_queue.add_job(
            job_type="lora",
            params=param_dict
        )
        
        # Return job ID for tracking
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "LoRA generation job queued"
        }
    except Exception as e:
        logger.error(f"Error in lora_generation: {e}")
        return {"error": str(e)}

@mcp.tool()
async def upscale(params: str) -> Dict[str, Any]:
    """
    Upscale an image using an upscaler model.
    
    Args:
        params: JSON string with parameters:
            - image: Base64 encoded image data
            - scale_factor: Factor to scale the image by (default: 2.0)
            - upscaler: Name of upscaler model to use (default: "ESRGAN_4x")
            - workflow_id: Workflow to use (default: "upscale")
            
    Returns:
        Dictionary with image URL or error message
    """
    context = await mcp.context_getter()
    logger.info("Received upscale request")
    try:
        param_dict = json.loads(params)
        
        # Validate image data exists
        if "image" not in param_dict or not param_dict["image"]:
            raise ValueError("Image data is required for upscaling")
        
        # Add job to queue
        job_id = await context.job_queue.add_job(
            job_type="upscale",
            params=param_dict
        )
        
        # Return job ID for tracking
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Upscale job queued"
        }
    except Exception as e:
        logger.error(f"Error in upscale: {e}")
        return {"error": str(e)}

@mcp.tool()
async def get_job_status(params: str) -> Dict[str, Any]:
    """
    Get status of a job.
    
    Args:
        params: JSON string with parameters:
            - job_id: ID of the job to check
            
    Returns:
        Dictionary with job status
    """
    context = await mcp.context_getter()
    try:
        param_dict = json.loads(params)
        job_id = param_dict.get("job_id")
        
        if not job_id:
            raise ValueError("job_id is required")
        
        # Get job status from queue
        job_status = await context.job_queue.get_job_status(job_id)
        
        if job_status:
            return job_status
        
        # If not in queue, check database
        db_job = await context.db_manager.get_job(job_id)
        if db_job:
            return db_job
            
        return {"error": f"Job {job_id} not found"}
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return {"error": str(e)}

@mcp.tool()
async def list_workflows() -> Dict[str, Any]:
    """
    List available workflows and their parameters.
    
    Returns:
        Dictionary with workflow information
    """
    context = await mcp.context_getter()
    try:
        workflows = context.comfyui_client.list_available_workflows()
        return {"workflows": workflows}
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        return {"error": str(e)}

@mcp.tool()
async def list_models() -> Dict[str, Any]:
    """
    List available model checkpoints.
    
    Returns:
        Dictionary with model information
    """
    context = await mcp.context_getter()
    try:
        models = context.model_manager.get_available_checkpoints()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"error": str(e)}

@mcp.tool()
async def list_loras() -> Dict[str, Any]:
    """
    List available LoRA models.
    
    Returns:
        Dictionary with LoRA information
    """
    context = await mcp.context_getter()
    try:
        loras = context.model_manager.get_available_loras()
        return {"loras": loras}
    except Exception as e:
        logger.error(f"Error listing LoRAs: {e}")
        return {"error": str(e)}

@mcp.tool()
async def list_controlnets() -> Dict[str, Any]:
    """
    List available ControlNet models.
    
    Returns:
        Dictionary with ControlNet information
    """
    context = await mcp.context_getter()
    try:
        controlnets = context.model_manager.get_available_controlnets()
        return {"controlnets": controlnets}
    except Exception as e:
        logger.error(f"Error listing ControlNets: {e}")
        return {"error": str(e)}

@mcp.tool()
async def upload_workflow(params: str) -> Dict[str, Any]:
    """
    Upload a new workflow to the server.
    
    Args:
        params: JSON string with parameters:
            - workflow_id: ID for the new workflow
            - workflow_data: Workflow data in ComfyUI API format
            
    Returns:
        Dictionary with success status
    """
    context = await mcp.context_getter()
    logger.info("Received upload_workflow request")
    try:
        param_dict = json.loads(params)
        workflow_id = param_dict.get("workflow_id")
        workflow_data = param_dict.get("workflow_data")
        
        if not workflow_id:
            raise ValueError("workflow_id is required")
        if not workflow_data:
            raise ValueError("workflow_data is required")
        
        success = context.comfyui_client.upload_workflow(
            workflow_id=workflow_id,
            workflow_data=workflow_data
        )
        
        if success:
            return {"status": "success", "message": f"Workflow '{workflow_id}' uploaded successfully"}
        else:
            return {"status": "error", "message": "Failed to upload workflow"}
    except Exception as e:
        logger.error(f"Error uploading workflow: {e}")
        return {"error": str(e)}

@mcp.tool()
async def get_queue_status() -> Dict[str, Any]:
    """
    Get status of the job queue.
    
    Returns:
        Dictionary with queue status
    """
    context = await mcp.context_getter()
    try:
        status = await context.job_queue.get_queue_status()
        return status
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return {"error": str(e)}

@mcp.tool()
async def cancel_job(params: str) -> Dict[str, Any]:
    """
    Cancel a pending job.
    
    Args:
        params: JSON string with parameters:
            - job_id: ID of the job to cancel
            
    Returns:
        Dictionary with success status
    """
    context = await mcp.context_getter()
    try:
        param_dict = json.loads(params)
        job_id = param_dict.get("job_id")
        
        if not job_id:
            raise ValueError("job_id is required")
        
        success = await context.job_queue.cancel_job(job_id)
        
        if success:
            return {"status": "success", "message": f"Job {job_id} cancelled successfully"}
        else:
            return {"status": "error", "message": f"Failed to cancel job {job_id}"}
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return {"error": str(e)}

# WebSocket server with enhanced functionality
async def handle_websocket(websocket, path):
    """Handle WebSocket connections with authentication and progress streaming"""
    client_info = {
        "authenticated": False,
        "client_id": None,
        "api_key": None
    }
    
    # Get current app context
    app_context = await mcp.context_getter()
    
    # Add to connected clients
    app_context.ws_clients.add(websocket)
    
    # Register job status callbacks
    job_callbacks = {}
    
    async def job_status_callback(job_id, status_data):
        """Send job status updates to the client"""
        try:
            await websocket.send(json.dumps({
                "type": "job_status",
                "job_id": job_id,
                "data": status_data
            }))
        except Exception as e:
            logger.error(f"Error sending job status update: {e}")
    
    async def cleanup_client():
        """Clean up client resources on disconnect"""
        # Remove from connected clients
        app_context.ws_clients.discard(websocket)
        
        # Unregister job callbacks
        for job_id, callback in job_callbacks.items():
            await app_context.job_queue.unregister_callback(job_id, callback)
    
    logger.info("WebSocket client connected")
    try:
        async for message in websocket:
            try:
                request = json.loads(message)
                msg_type = request.get("type", "unknown")
                logger.info(f"Received WebSocket message type: {msg_type}")
                
                # Handle authentication if enabled
                if app_context.auth_manager.enable_auth and not client_info["authenticated"]:
                    if msg_type == "auth":
                        api_key = request.get("api_key")
                        client_data = app_context.auth_manager.authenticate(api_key)
                        
                        if client_data:
                            client_info["authenticated"] = True
                            client_info["client_id"] = client_data["client_id"]
                            client_info["api_key"] = api_key
                            
                            await websocket.send(json.dumps({
                                "type": "auth_response",
                                "status": "success",
                                "client_id": client_data["client_id"]
                            }))
                            continue
                        else:
                            await websocket.send(json.dumps({
                                "type": "auth_response",
                                "status": "error",
                                "message": "Invalid API key"
                            }))
                            continue
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Authentication required"
                        }))
                        continue
                
                # Handle different message types
                if msg_type == "subscribe_job":
                    # Subscribe to job status updates
                    job_id = request.get("job_id")
                    if job_id:
                        callback = lambda data, jid=job_id: job_status_callback(jid, data)
                        success = await app_context.job_queue.register_callback(job_id, callback)
                        job_callbacks[job_id] = callback
                        
                        await websocket.send(json.dumps({
                            "type": "subscribe_response",
                            "job_id": job_id,
                            "status": "success" if success else "error",
                            "message": f"Subscribed to job {job_id}" if success else f"Job {job_id} not found"
                        }))
                        
                elif msg_type == "unsubscribe_job":
                    # Unsubscribe from job status updates
                    job_id = request.get("job_id")
                    if job_id and job_id in job_callbacks:
                        await app_context.job_queue.unregister_callback(job_id, job_callbacks[job_id])
                        del job_callbacks[job_id]
                        
                        await websocket.send(json.dumps({
                            "type": "unsubscribe_response",
                            "job_id": job_id,
                            "status": "success"
                        }))
                        
                elif msg_type == "get_job_status":
                    # Get status of a job
                    job_id = request.get("job_id")
                    if job_id:
                        status = await app_context.job_queue.get_job_status(job_id)
                        
                        if not status:
                            # Try to get from database
                            status = await app_context.db_manager.get_job(job_id)
                            
                        await websocket.send(json.dumps({
                            "type": "job_status_response",
                            "job_id": job_id,
                            "status": "success" if status else "error",
                            "data": status if status else {"message": f"Job {job_id} not found"}
                        }))
                        
                elif msg_type == "get_queue_status":
                    # Get status of the job queue
                    status = await app_context.job_queue.get_queue_status()
                    
                    await websocket.send(json.dumps({
                        "type": "queue_status_response",
                        "status": "success",
                        "data": status
                    }))
                    
                elif msg_type == "ping":
                    # Simple ping to keep connection alive
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
                    
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
                
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                
    except websockets.ConnectionClosed:
        logger.info("WebSocket client disconnected")
    finally:
        await cleanup_client()

# Main server loop
async def main():
    """Start the MCP server and WebSocket server"""
    host = config["mcp_server_host"]
    port = config["mcp_server_port"]
    
    logger.info(f"Starting enhanced MCP server on ws://{host}:{port}...")
    
    # Start the WebSocket server
    async with websockets.serve(handle_websocket, host, port):
        # Make the server available globally for the MCP context
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
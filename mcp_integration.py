import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import websockets

# Import our MCP implementation
from mcp_protocol import MCP, MCPParameter, MCPTool, MCPContext

# Import existing components
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
logger = logging.getLogger("MCP_Integration")

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

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
        "max_concurrent_jobs": int(os.environ.get("MAX_CONCURRENT_JOBS", "2")),
        "context_save_path": os.environ.get("CONTEXT_SAVE_PATH", "data/mcp_context.json")
    }
    
    # Try to load from config file if specified
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
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

# Define application context with our components
class AppContext:
    """Application context with all service dependencies"""
    
    def __init__(self, 
                 comfyui_client: AdvancedComfyUIClient,
                 progress_tracker: ProgressTracker,
                 auth_manager: AuthManager,
                 workflow_analyzer: WorkflowAnalyzer,
                 model_manager: ModelManager,
                 job_queue: JobQueue,
                 db_manager: SQLiteDatabaseManager,
                 mcp: MCP):
        self.comfyui_client = comfyui_client
        self.progress_tracker = progress_tracker
        self.auth_manager = auth_manager
        self.workflow_analyzer = workflow_analyzer
        self.model_manager = model_manager
        self.job_queue = job_queue
        self.db_manager = db_manager
        self.mcp = mcp
        
        # WebSocket clients for broadcasting events
        self.ws_clients = set()

# Initialize our main MCP instance
mcp = MCP(
    name="ComfyUI MCP Server",
    description="A Model Context Protocol server for ComfyUI image generation"
)

# Initialize our components
comfyui_client = AdvancedComfyUIClient(
    base_url=config["comfyui_url"],
    workflows_dir=config["workflows_dir"]
)
workflow_analyzer = WorkflowAnalyzer()
model_manager = ModelManager(config["comfyui_url"])
auth_manager = AuthManager(enable_auth=config["enable_auth"])
if config["enable_auth"]:
    auth_manager.load_api_keys(config["auth_config_path"])

# These components need async initialization, we'll do that in start_server
progress_tracker = None
db_manager = None
job_queue = None

# Create application context
app_context = AppContext(
    comfyui_client=comfyui_client,
    progress_tracker=progress_tracker,
    auth_manager=auth_manager,
    workflow_analyzer=workflow_analyzer,
    model_manager=model_manager,
    job_queue=job_queue,
    db_manager=db_manager,
    mcp=mcp
)

# Define MCP tools with rich metadata
@mcp.tool(
    name="generate_image",
    description="Generate an image from a text prompt using AI models",
    metadata={
        "capability": "image_generation",
        "models": ["stable-diffusion", "sdxl"],
        "category": "creation",
        "tags": ["image", "ai", "generation"]
    },
    examples=[
        {
            "parameters": {
                "prompt": "a cat in space",
                "width": 512,
                "height": 512,
                "workflow_id": "basic_api_test"
            },
            "result": {
                "job_id": "job_123",
                "status": "queued",
                "message": "Image generation job queued"
            }
        }
    ]
)
async def generate_image(prompt: str, width: int = 512, height: int = 512,
                        workflow_id: str = "basic_api_test", model: Optional[str] = None,
                        negative_prompt: str = "", seed: int = -1, steps: int = 20,
                        cfg_scale: float = 7.0, scheduler: str = "normal",
                        sampler: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate an image using a text prompt and optional parameters.
    
    Args:
        prompt: Text description of the desired image
        width: Width of the output image
        height: Height of the output image
        workflow_id: ID of the workflow to use
        model: Model checkpoint to use
        negative_prompt: Negative prompt to guide generation
        seed: Seed for reproducibility (-1 for random)
        steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
        scheduler: Sampler scheduler
        sampler: Sampler algorithm
    
    Returns:
        Dictionary with job information
    """
    # Check that job_queue is initialized
    if app_context.job_queue is None:
        return {"error": "Server is still initializing, please try again shortly"}
    
    logger.info(f"Received generate_image request with prompt: {prompt}")
    try:
        # Build parameters dictionary
        params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "workflow_id": workflow_id,
            "model": model,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "scheduler": scheduler,
            "sampler": sampler
        }
        
        # Add job to queue
        job_id = await app_context.job_queue.add_job(
            job_type="txt2img",
            params=params
        )
        
        # Return job ID for tracking
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Image generation job queued",
            "submitted_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in generate_image: {e}")
        return {"error": str(e)}

@mcp.tool(
    name="img2img",
    description="Transform an existing image based on a text prompt",
    metadata={
        "capability": "image_transformation",
        "models": ["stable-diffusion", "sdxl"],
        "category": "modification",
        "tags": ["image", "ai", "transformation"]
    }
)
async def img2img(prompt: str, image: str, strength: float = 0.75, 
                 width: int = 512, height: int = 512, workflow_id: str = "img2img",
                 model: Optional[str] = None, negative_prompt: str = "",
                 seed: int = -1, steps: int = 20, cfg_scale: float = 7.0) -> Dict[str, Any]:
    """
    Transform an existing image using a text prompt.
    
    Args:
        prompt: Text description of the desired transformation
        image: Base64 encoded image data
        strength: How much to transform the input image (0-1)
        width: Width of the output image
        height: Height of the output image
        workflow_id: ID of the workflow to use
        model: Model checkpoint to use
        negative_prompt: Negative prompt to guide generation
        seed: Seed for reproducibility (-1 for random)
        steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
    
    Returns:
        Dictionary with job information
    """
    # Check that job_queue is initialized
    if app_context.job_queue is None:
        return {"error": "Server is still initializing, please try again shortly"}
    
    logger.info(f"Received img2img request with prompt: {prompt}")
    try:
        # Validate image data exists
        if not image:
            raise ValueError("Image data is required for img2img")
        
        # Build parameters dictionary
        params = {
            "prompt": prompt,
            "image": image,
            "strength": strength,
            "width": width,
            "height": height,
            "workflow_id": workflow_id,
            "model": model,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale
        }
        
        # Add job to queue
        job_id = await app_context.job_queue.add_job(
            job_type="img2img",
            params=params
        )
        
        # Return job ID for tracking
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Image-to-image job queued",
            "submitted_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in img2img: {e}")
        return {"error": str(e)}

@mcp.tool(
    name="get_job_status",
    description="Get the status of a previously submitted job",
    metadata={
        "capability": "job_tracking",
        "category": "utility",
        "tags": ["job", "status", "tracking"]
    }
)
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a job by its ID.
    
    Args:
        job_id: ID of the job to check
    
    Returns:
        Dictionary with job status information
    """
    # Check that job_queue is initialized
    if app_context.job_queue is None:
        return {"error": "Server is still initializing, please try again shortly"}
    
    try:
        # Get job status from queue
        job_status = await app_context.job_queue.get_job_status(job_id)
        
        if job_status:
            return job_status
        
        # If not in queue, check database
        if app_context.db_manager:
            db_job = await app_context.db_manager.get_job(job_id)
            if db_job:
                return db_job
            
        return {"error": f"Job {job_id} not found"}
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return {"error": str(e)}

@mcp.tool(
    name="list_models",
    description="List available checkpoint models",
    metadata={
        "capability": "model_discovery",
        "category": "utility",
        "tags": ["models", "list", "discovery"]
    }
)
async def list_models() -> Dict[str, Any]:
    """
    List all available checkpoint models.
    
    Returns:
        Dictionary with list of available models
    """
    try:
        models = app_context.model_manager.get_available_checkpoints()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"error": str(e)}

@mcp.tool(
    name="get_tools_manifest",
    description="Get the manifest of all available tools",
    metadata={
        "capability": "self_description",
        "category": "system",
        "tags": ["tools", "manifest", "discovery"]
    }
)
async def get_tools_manifest() -> Dict[str, Any]:
    """
    Get the manifest of all available tools.
    
    Returns:
        Tools manifest
    """
    try:
        return mcp.get_manifest()
    except Exception as e:
        logger.error(f"Error getting tools manifest: {e}")
        return {"error": str(e)}

# WebSocket server with MCP protocol support
async def handle_websocket(websocket, path):
    """Handle WebSocket connections with MCP protocol support"""
    client_info = {
        "authenticated": False,
        "client_id": None,
        "api_key": None,
        "session_id": None
    }
    
    # Add to connected clients
    app_context.ws_clients.add(websocket)
    
    # Job status callbacks
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
        if app_context.job_queue:
            for job_id, callback in job_callbacks.items():
                await app_context.job_queue.unregister_callback(job_id, callback)
    
    # Create a new MCP session for this client
    session_id = mcp.create_session()
    client_info["session_id"] = session_id
    
    logger.info(f"WebSocket client connected, session: {session_id}")
    
    try:
        await websocket.send(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "service": "ComfyUI MCP Server",
            "auth_required": app_context.auth_manager.enable_auth
        }))
        
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
                
                # Handle MCP protocol messages
                if msg_type == "mcp_tool_request":
                    # Process MCP tool request
                    tool_name = request.get("tool")
                    parameters = request.get("parameters", {})
                    request_id = request.get("request_id", str(uuid.uuid4()))
                    
                    # Apply rate limiting if enabled
                    if app_context.auth_manager.enable_auth:
                        client_id = client_info["client_id"]
                        can_proceed, wait_time = app_context.auth_manager.rate_limit(client_id, tool_name)
                        
                        if not can_proceed:
                            await websocket.send(json.dumps({
                                "type": "rate_limit",
                                "request_id": request_id,
                                "wait_time": wait_time,
                                "message": f"Rate limit exceeded. Try again in {wait_time:.1f} seconds."
                            }))
                            continue
                    
                    # Execute tool through MCP
                    result = await mcp.execute_tool(tool_name, parameters, session_id)
                    
                    # Add request ID to response
                    response = {
                        "type": "mcp_tool_response",
                        "request_id": request_id,
                        "tool": tool_name,
                        "result": result
                    }
                    
                    # Check if we need to register for job status updates
                    if tool_name in ["generate_image", "img2img", "inpaint", "controlnet"] and "job_id" in result and "error" not in result:
                        job_id = result["job_id"]
                        
                        if app_context.job_queue:
                            callback = lambda data, jid=job_id: job_status_callback(jid, data)
                            await app_context.job_queue.register_callback(job_id, callback)
                            job_callbacks[job_id] = callback
                    
                    await websocket.send(json.dumps(response))
                
                elif msg_type == "mcp_manifest_request":
                    # Return tools manifest
                    manifest = mcp.get_manifest()
                    await websocket.send(json.dumps({
                        "type": "mcp_manifest_response",
                        "manifest": manifest
                    }))
                
                elif msg_type == "subscribe_job":
                    # Subscribe to job status updates
                    job_id = request.get("job_id")
                    if job_id and app_context.job_queue:
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
                    if job_id and job_id in job_callbacks and app_context.job_queue:
                        await app_context.job_queue.unregister_callback(job_id, job_callbacks[job_id])
                        del job_callbacks[job_id]
                        
                        await websocket.send(json.dumps({
                            "type": "unsubscribe_response",
                            "job_id": job_id,
                            "status": "success"
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
        logger.info(f"WebSocket client disconnected, session: {session_id}")
    finally:
        await cleanup_client()

# Job executor function for the job queue
async def job_executor(job: Job) -> Dict[str, Any]:
    """Execute a generation job"""
    job_type = job.job_type
    params = job.params
    
    try:
        # Check which job type to execute
        if job_type == "txt2img":
            image_url = app_context.comfyui_client.generate_image(**params)
        elif job_type == "img2img":
            image_url = app_context.comfyui_client.img2img(**params)
        elif job_type == "inpaint":
            image_url = app_context.comfyui_client.inpaint(**params)
        elif job_type == "controlnet":
            image_url = app_context.comfyui_client.controlnet(**params)
        elif job_type == "lora":
            image_url = app_context.comfyui_client.lora_generation(**params)
        elif job_type == "upscale":
            image_url = app_context.comfyui_client.upscale(**params)
        else:
            raise ValueError(f"Unknown job type: {job_type}")
        
        # Save job to database
        if app_context.db_manager:
            await app_context.db_manager.save_job({
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

# Function to save MCP context
async def save_context():
    """Save MCP context to a file"""
    if not mcp:
        return
        
    try:
        context_data = mcp.export_context()
        with open(config["context_save_path"], 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2)
        logger.info(f"Saved MCP context to {config['context_save_path']}")
    except Exception as e:
        logger.error(f"Error saving MCP context: {e}")

# Function to load MCP context
async def load_context():
    """Load MCP context from a file"""
    if not mcp or not os.path.exists(config["context_save_path"]):
        return
        
    try:
        with open(config["context_save_path"], 'r', encoding='utf-8') as f:
            context_data = json.load(f)
        mcp.import_context(context_data)
        logger.info(f"Loaded MCP context from {config['context_save_path']}")
    except Exception as e:
        logger.error(f"Error loading MCP context: {e}")

# Main server initialization and startup
async def start_server():
    """Initialize and start the MCP server"""
    global progress_tracker, db_manager, job_queue
    
    logger.info("Starting MCP server initialization...")
    
    try:
        # Initialize async components
        progress_tracker = ProgressTracker(comfyui_ws_url=config["comfyui_ws_url"])
        await progress_tracker.start()
        app_context.progress_tracker = progress_tracker
        
        db_manager = SQLiteDatabaseManager(config["db_path"])
        await db_manager.initialize()
        app_context.db_manager = db_manager
        
        job_queue = JobQueue(
            max_concurrent_jobs=config["max_concurrent_jobs"],
            save_path=config["queue_save_path"]
        )
        app_context.job_queue = job_queue
        
        # Set up job executor
        job_queue.set_job_executor(job_executor)
        
        # Start job queue
        await job_queue.start()
        
        # Try to load MCP context
        await load_context()
        
        # Start the WebSocket server
        host = config["mcp_server_host"]
        port = config["mcp_server_port"]
        
        logger.info(f"Starting MCP WebSocket server on ws://{host}:{port}...")
        async with websockets.serve(handle_websocket, host, port):
            # Run periodic tasks
            while True:
                # Save MCP context periodically
                await save_context()
                
                # Wait before next iteration
                await asyncio.sleep(300)  # Every 5 minutes
    
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")
        # Cleanup
        if progress_tracker:
            await progress_tracker.stop()
        if job_queue:
            await job_queue.stop()
        if db_manager:
            await db_manager.close()
        raise

# Main entry point
def main():
    """Main entry point"""
    try:
        # Run the server
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Unhandled error: {e}")

if __name__ == "__main__":
    main()

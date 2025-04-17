import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
import base64
from datetime import datetime

from aiohttp import web
import aiohttp_cors
from aiohttp_session import setup as setup_session
from aiohttp_session.cookie_storage import EncryptedCookieStorage
import cryptography.fernet

# Import our MCP components 
from enhanced_comfyui_client import ComfyUIClient
from progress_tracker import ProgressTracker
from auth import AuthManager
from workflow_analyzer import WorkflowAnalyzer
from db_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")

class WebDashboard:
    """Web dashboard for managing ComfyUI MCP Server"""
    
    def __init__(self, config: Dict[str, Any], db_manager: DatabaseManager):
        """Initialize dashboard with configuration"""
        self.config = config
        self.db_manager = db_manager
        self.app = web.Application()
        self.comfyui_client = ComfyUIClient(
            base_url=config.get("comfyui_url", "http://localhost:8188"),
            workflows_dir=config.get("workflows_dir", "workflows")
        )
        self.progress_tracker = ProgressTracker(
            comfyui_ws_url=config.get("comfyui_ws_url", "ws://localhost:8188/ws")
        )
        self.auth_manager = AuthManager(
            enable_auth=config.get("enable_auth", False)
        )
        self.workflow_analyzer = WorkflowAnalyzer()
        
        # Client connections for progress updates
        self.ws_clients = set()
        
    async def start(self, host: str = "localhost", port: int = 8080):
        """Start the web dashboard"""
        # Setup session
        fernet_key = cryptography.fernet.Fernet.generate_key()
        secret_key = base64.urlsafe_b64decode(fernet_key)
        setup_session(self.app, EncryptedCookieStorage(secret_key))
        
        # Setup routes
        self._setup_routes()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        # Apply CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
        
        # Start progress tracker
        await self.progress_tracker.start()
        
        # If authentication is enabled, load API keys
        if self.auth_manager.enable_auth:
            self.auth_manager.load_api_keys(
                self.config.get("auth_config_path")
            )
        
        # Start the web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Web dashboard running at http://{host}:{port}")
        
        # Keep the server running
        while True:
            await asyncio.sleep(3600)  # Just keep alive
    
    def _setup_routes(self):
        """Setup web routes"""
        # Static files
        self.app.router.add_static('/static/', path=os.path.join(os.path.dirname(__file__), 'static'), name='static')
        
        # API routes
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/api/workflows', self.handle_list_workflows)
        self.app.router.add_get('/api/models', self.handle_list_models)
        self.app.router.add_get('/api/loras', self.handle_list_loras)
        self.app.router.add_get('/api/controlnets', self.handle_list_controlnets)
        self.app.router.add_get('/api/jobs', self.handle_list_jobs)
        self.app.router.add_post('/api/generate', self.handle_generate)
        self.app.router.add_post('/api/upload_workflow', self.handle_upload_workflow)
        self.app.router.add_get('/api/gallery', self.handle_get_gallery)
        
        # WebSocket for real-time updates
        self.app.router.add_get('/ws', self.handle_websocket)
        
        # Admin routes (protected)
        self.app.router.add_get('/api/admin/api_keys', self.handle_list_api_keys)
        self.app.router.add_post('/api/admin/api_keys', self.handle_create_api_key)
        self.app.router.add_delete('/api/admin/api_keys/{key}', self.handle_revoke_api_key)
    
    # Route handlers
    async def handle_index(self, request):
        """Serve the main dashboard page"""
        with open(os.path.join(os.path.dirname(__file__), 'static', 'index.html')) as f:
            return web.Response(text=f.read(), content_type='text/html')
    
    async def handle_list_workflows(self, request):
        """List available workflows"""
        try:
            workflows = self.comfyui_client.list_available_workflows()
            return web.json_response({"workflows": workflows})
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_list_models(self, request):
        """List available models"""
        try:
            models = self.comfyui_client.list_available_models()
            return web.json_response({"models": models})
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_list_loras(self, request):
        """List available LoRA models"""
        try:
            # This would need to be implemented in the ComfyUIClient
            loras = self.comfyui_client.list_available_loras() if hasattr(self.comfyui_client, 'list_available_loras') else []
            return web.json_response({"loras": loras})
        except Exception as e:
            logger.error(f"Error listing LoRAs: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_list_controlnets(self, request):
        """List available ControlNet models"""
        try:
            # This would need to be implemented in the ComfyUIClient
            controlnets = self.comfyui_client.list_available_controlnets() if hasattr(self.comfyui_client, 'list_available_controlnets') else []
            return web.json_response({"controlnets": controlnets})
        except Exception as e:
            logger.error(f"Error listing ControlNets: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_list_jobs(self, request):
        """List recent jobs"""
        try:
            # Get recent jobs from database
            jobs = await self.db_manager.get_recent_jobs(limit=20)
            return web.json_response({"jobs": jobs})
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_generate(self, request):
        """Handle image generation request"""
        try:
            data = await request.json()
            
            # Check authentication if enabled
            if self.auth_manager.enable_auth:
                api_key = request.headers.get('X-API-Key')
                client_data = self.auth_manager.authenticate(api_key)
                if not client_data:
                    return web.json_response({"error": "Invalid API key"}, status=401)
                
                # Check authorization
                if not self.auth_manager.authorize(client_data, "generate_image"):
                    return web.json_response({"error": "Not authorized"}, status=403)
                
                # Apply rate limiting
                client_id = client_data["client_id"]
                can_proceed, wait_time = self.auth_manager.rate_limit(client_id, "generate_image")
                if not can_proceed:
                    return web.json_response({
                        "error": f"Rate limit exceeded. Try again in {wait_time:.1f} seconds."
                    }, status=429)
            
            # Determine generation mode
            mode = data.get("mode", "txt2img")
            
            # Extract common parameters
            prompt = data.get("prompt", "")
            width = int(data.get("width", 512))
            height = int(data.get("height", 512))
            workflow_id = data.get("workflow_id", "basic_api_test" if mode == "txt2img" else mode)
            model = data.get("model")
            
            # Extract additional parameters based on mode
            if mode == "txt2img":
                # Regular text-to-image generation
                image_url = self.comfyui_client.generate_image(
                    prompt=prompt,
                    width=width,
                    height=height,
                    workflow_id=workflow_id,
                    model=model,
                    **{k: v for k, v in data.items() if k not in ["mode", "prompt", "width", "height", "workflow_id", "model"]}
                )
                
            elif mode == "img2img":
                # Image-to-image generation
                image_data = data.get("image", "")
                strength = float(data.get("strength", 0.75))
                
                if not image_data:
                    return web.json_response({"error": "Image data is required for img2img"}, status=400)
                
                image_url = self.comfyui_client.img2img(
                    prompt=prompt,
                    image_data=image_data,
                    strength=strength,
                    width=width,
                    height=height,
                    workflow_id=workflow_id,
                    model=model,
                    **{k: v for k, v in data.items() if k not in ["mode", "prompt", "image", "strength", "width", "height", "workflow_id", "model"]}
                )
                
            elif mode == "inpaint":
                # Inpainting
                image_data = data.get("image", "")
                mask_data = data.get("mask", "")
                
                if not image_data:
                    return web.json_response({"error": "Image data is required for inpainting"}, status=400)
                if not mask_data:
                    return web.json_response({"error": "Mask data is required for inpainting"}, status=400)
                
                image_url = self.comfyui_client.inpaint(
                    prompt=prompt,
                    image_data=image_data,
                    mask_data=mask_data,
                    width=width,
                    height=height,
                    workflow_id=workflow_id,
                    model=model,
                    **{k: v for k, v in data.items() if k not in ["mode", "prompt", "image", "mask", "width", "height", "workflow_id", "model"]}
                )
                
            else:
                return web.json_response({"error": f"Unknown generation mode: {mode}"}, status=400)
            
            # Save job to database
            job_id = await self.db_manager.save_job({
                "mode": mode,
                "prompt": prompt,
                "width": width,
                "height": height,
                "workflow_id": workflow_id,
                "model": model,
                "image_url": image_url,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "parameters": {k: v for k, v in data.items() if k not in ["mode", "prompt", "width", "height", "workflow_id", "model", "image", "mask"]}
            })
            
            return web.json_response({
                "image_url": image_url,
                "job_id": job_id
            })
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_upload_workflow(self, request):
        """Handle workflow upload"""
        try:
            data = await request.json()
            
            workflow_id = data.get("workflow_id")
            workflow_data = data.get("workflow_data")
            
            if not workflow_id:
                return web.json_response({"error": "workflow_id is required"}, status=400)
            if not workflow_data:
                return web.json_response({"error": "workflow_data is required"}, status=400)
            
            success = self.comfyui_client.upload_workflow(
                workflow_id=workflow_id,
                workflow_data=workflow_data
            )
            
            if success:
                return web.json_response({
                    "status": "success", 
                    "message": f"Workflow '{workflow_id}' uploaded successfully"
                })
            else:
                return web.json_response({
                    "status": "error", 
                    "message": "Failed to upload workflow"
                }, status=500)
                
        except Exception as e:
            logger.error(f"Error uploading workflow: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_get_gallery(self, request):
        """Get gallery of generated images"""
        try:
            page = int(request.query.get("page", "1"))
            limit = int(request.query.get("limit", "20"))
            
            # Get images from database
            images = await self.db_manager.get_gallery_images(page=page, limit=limit)
            total = await self.db_manager.get_gallery_count()
            
            return web.json_response({
                "images": images,
                "total": total,
                "page": page,
                "limit": limit
            })
            
        except Exception as e:
            logger.error(f"Error getting gallery: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_websocket(self, request):
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Add to connected clients
        self.ws_clients.add(ws)
        
        try:
            # Process messages from client
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        cmd = data.get("cmd")
                        
                        if cmd == "subscribe":
                            # Subscribe to progress updates for a job
                            prompt_id = data.get("prompt_id")
                            if prompt_id:
                                # Setup callback to send progress updates to this client
                                async def progress_callback(progress_data):
                                    await ws.send_json({
                                        "type": "progress",
                                        "prompt_id": prompt_id,
                                        "data": progress_data
                                    })
                                
                                # Register callback with progress tracker
                                self.progress_tracker.register_client(prompt_id, progress_callback)
                                
                                await ws.send_json({
                                    "type": "subscribed",
                                    "prompt_id": prompt_id
                                })
                                
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "Invalid JSON"})
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                        await ws.send_json({"error": str(e)})
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception {ws.exception()}")
                    
        finally:
            # Remove from connected clients
            self.ws_clients.remove(ws)
            
        return ws
    
    # Admin handlers
    async def handle_list_api_keys(self, request):
        """List API keys (admin only)"""
        try:
            # Check admin authentication
            api_key = request.headers.get('X-API-Key')
            client_data = self.auth_manager.authenticate(api_key)
            if not client_data or not self.auth_manager.authorize(client_data, "admin"):
                return web.json_response({"error": "Not authorized"}, status=403)
            
            # Get all API keys (excluding the actual key values)
            api_keys = []
            for key, data in self.auth_manager.api_keys.items():
                api_keys.append({
                    "client_id": data["client_id"],
                    "permissions": data["permissions"],
                    "key_prefix": key[:4] + "..." + key[-4:],
                })
            
            return web.json_response({"api_keys": api_keys})
            
        except Exception as e:
            logger.error(f"Error listing API keys: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_create_api_key(self, request):
        """Create a new API key (admin only)"""
        try:
            # Check admin authentication
            api_key = request.headers.get('X-API-Key')
            client_data = self.auth_manager.authenticate(api_key)
            if not client_data or not self.auth_manager.authorize(client_data, "admin"):
                return web.json_response({"error": "Not authorized"}, status=403)
            
            # Get client ID and permissions from request
            data = await request.json()
            client_id = data.get("client_id")
            permissions = data.get("permissions", ["*"])
            
            if not client_id:
                return web.json_response({"error": "client_id is required"}, status=400)
            
            # Create new API key
            new_key = self.auth_manager.create_api_key(client_id, permissions)
            
            return web.json_response({
                "api_key": new_key,
                "client_id": client_id,
                "permissions": permissions
            })
            
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_revoke_api_key(self, request):
        """Revoke an API key (admin only)"""
        try:
            # Check admin authentication
            api_key = request.headers.get('X-API-Key')
            client_data = self.auth_manager.authenticate(api_key)
            if not client_data or not self.auth_manager.authorize(client_data, "admin"):
                return web.json_response({"error": "Not authorized"}, status=403)
            
            # Get key to revoke
            key_to_revoke = request.match_info["key"]
            
            # Revoke key
            success = self.auth_manager.revoke_api_key(key_to_revoke)
            
            if success:
                return web.json_response({
                    "status": "success",
                    "message": "API key revoked"
                })
            else:
                return web.json_response({
                    "status": "error",
                    "message": "API key not found"
                }, status=404)
                
        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast an event to all connected WebSocket clients"""
        if not self.ws_clients:
            return
            
        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        for ws in list(self.ws_clients):
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                # Client probably disconnected - remove from list
                self.ws_clients.discard(ws)

# Run dashboard if executed directly
if __name__ == "__main__":
    from db_manager import SQLiteDatabaseManager
    
    # Load config
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "comfyui_url": "http://localhost:8188",
            "comfyui_ws_url": "ws://localhost:8188/ws",
            "enable_auth": False,
            "db_path": "mcp_server.db",
            "dashboard_host": "localhost",
            "dashboard_port": 8080
        }
    
    # Initialize database
    db_manager = SQLiteDatabaseManager(config.get("db_path", "mcp_server.db"))
    
    # Run the dashboard
    dashboard = WebDashboard(config, db_manager)
    
    asyncio.run(dashboard.start(
        host=config.get("dashboard_host", "localhost"),
        port=config.get("dashboard_port", 8080)
    ))

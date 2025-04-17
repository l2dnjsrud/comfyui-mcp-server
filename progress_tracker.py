import asyncio
import json
import logging
import time
from typing import Dict, Optional, Set, Callable, Awaitable, Any
import websockets

logger = logging.getLogger("ProgressTracker")

class ProgressTracker:
    """
    Tracks the progress of image generation jobs and provides real-time updates.
    Connects to ComfyUI's websocket API to monitor job progress.
    """
    
    def __init__(self, comfyui_ws_url: str = "ws://localhost:8188/ws"):
        self.comfyui_ws_url = comfyui_ws_url
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.client_callbacks: Dict[str, Set[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
        self.ws_connection = None
        self.running = False
        self.listener_task = None

    async def start(self):
        """Start the progress tracking service"""
        if self.running:
            return
            
        self.running = True
        self.listener_task = asyncio.create_task(self._listen_for_updates())
        logger.info("Progress tracker started")

    async def stop(self):
        """Stop the progress tracking service"""
        self.running = False
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
            self.listener_task = None
            
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
            
        logger.info("Progress tracker stopped")

    async def _listen_for_updates(self):
        """Listen for WebSocket updates from ComfyUI"""
        reconnect_delay = 1
        
        while self.running:
            try:
                logger.info(f"Connecting to ComfyUI WebSocket at {self.comfyui_ws_url}")
                async with websockets.connect(self.comfyui_ws_url) as websocket:
                    self.ws_connection = websocket
                    reconnect_delay = 1  # Reset backoff on successful connection
                    
                    logger.info("Connected to ComfyUI WebSocket")
                    
                    while self.running:
                        message = await websocket.recv()
                        try:
                            data = json.loads(message)
                            await self._process_update(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Received invalid JSON: {message[:100]}...")
                            
            except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
                logger.warning(f"ComfyUI WebSocket connection error: {e}")
                self.ws_connection = None
                
                if self.running:
                    # Implement exponential backoff up to 30 seconds
                    await asyncio.sleep(min(reconnect_delay, 30))
                    reconnect_delay *= 2
            except Exception as e:
                logger.error(f"Unexpected error in WebSocket listener: {e}")
                self.ws_connection = None
                
                if self.running:
                    await asyncio.sleep(5)  # Wait before trying to reconnect

    async def _process_update(self, data: Dict[str, Any]):
        """Process update messages from ComfyUI"""
        try:
            if "type" not in data:
                return
                
            msg_type = data["type"]
            
            # Handle different message types
            if msg_type == "execution_start":
                prompt_id = data.get("prompt_id")
                if prompt_id:
                    self.active_jobs[prompt_id] = {
                        "status": "started",
                        "progress": 0,
                        "timestamp": time.time(),
                        "details": {}
                    }
                    await self._notify_clients(prompt_id)
                    
            elif msg_type == "execution_cached":
                prompt_id = data.get("prompt_id")
                if prompt_id:
                    self.active_jobs[prompt_id] = {
                        "status": "completed",
                        "progress": 100,  # Cached results are ready immediately
                        "timestamp": time.time(),
                        "details": {"cached": True}
                    }
                    await self._notify_clients(prompt_id)
                    
            elif msg_type == "executing":
                prompt_id = data.get("prompt_id")
                node_id = data.get("node")
                if prompt_id and prompt_id in self.active_jobs:
                    # Track which node is currently executing
                    self.active_jobs[prompt_id]["details"]["current_node"] = node_id
                    await self._notify_clients(prompt_id)
                    
            elif msg_type == "progress":
                prompt_id = data.get("prompt_id")
                progress = data.get("value", 0)
                max_value = data.get("max", 100)
                
                if prompt_id and prompt_id in self.active_jobs:
                    # Calculate percentage progress
                    percentage = int((progress / max_value) * 100) if max_value else 0
                    self.active_jobs[prompt_id]["progress"] = percentage
                    self.active_jobs[prompt_id]["status"] = "processing"
                    self.active_jobs[prompt_id]["details"]["step"] = progress
                    self.active_jobs[prompt_id]["details"]["total_steps"] = max_value
                    await self._notify_clients(prompt_id)
                    
            elif msg_type == "executed":
                prompt_id = data.get("prompt_id")
                node_id = data.get("node")
                output = data.get("output", {})
                
                if prompt_id and prompt_id in self.active_jobs:
                    # Store node outputs if they contain images
                    if "images" in output:
                        # Just store the fact that images were generated by this node
                        if "image_nodes" not in self.active_jobs[prompt_id]["details"]:
                            self.active_jobs[prompt_id]["details"]["image_nodes"] = []
                        self.active_jobs[prompt_id]["details"]["image_nodes"].append(node_id)
                    await self._notify_clients(prompt_id)
                    
            elif msg_type == "execution_complete":
                prompt_id = data.get("prompt_id")
                if prompt_id and prompt_id in self.active_jobs:
                    self.active_jobs[prompt_id]["status"] = "completed"
                    self.active_jobs[prompt_id]["progress"] = 100
                    await self._notify_clients(prompt_id)
                    
            elif msg_type == "execution_error":
                prompt_id = data.get("prompt_id")
                error_msg = data.get("exception_message", "Unknown error")
                if prompt_id and prompt_id in self.active_jobs:
                    self.active_jobs[prompt_id]["status"] = "error"
                    self.active_jobs[prompt_id]["details"]["error"] = error_msg
                    await self._notify_clients(prompt_id)
                    
        except Exception as e:
            logger.error(f"Error processing update: {e}")

    async def _notify_clients(self, prompt_id: str):
        """Notify registered clients about updates to a job"""
        if prompt_id not in self.client_callbacks:
            return
            
        job_info = self.active_jobs[prompt_id]
        
        # Notify all registered callbacks for this prompt ID
        for callback in list(self.client_callbacks.get(prompt_id, set())):
            try:
                await callback(job_info)
            except Exception as e:
                logger.error(f"Error in client callback: {e}")
                # If a callback raises an exception, remove it
                self.client_callbacks[prompt_id].discard(callback)

    def get_job_status(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a job"""
        return self.active_jobs.get(prompt_id)

    def register_client(self, prompt_id: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Register a client callback for job updates"""
        if prompt_id not in self.client_callbacks:
            self.client_callbacks[prompt_id] = set()
        self.client_callbacks[prompt_id].add(callback)
        
        # If we already have information about this job, send it immediately
        if prompt_id in self.active_jobs:
            asyncio.create_task(callback(self.active_jobs[prompt_id]))

    def unregister_client(self, prompt_id: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Unregister a client callback"""
        if prompt_id in self.client_callbacks:
            self.client_callbacks[prompt_id].discard(callback)
            if not self.client_callbacks[prompt_id]:
                del self.client_callbacks[prompt_id]
                
    def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """Clean up old job data to prevent memory leaks"""
        current_time = time.time()
        expired_jobs = []
        
        for prompt_id, job_info in self.active_jobs.items():
            if current_time - job_info["timestamp"] > max_age_seconds:
                expired_jobs.append(prompt_id)
                
        for prompt_id in expired_jobs:
            if prompt_id in self.active_jobs:
                del self.active_jobs[prompt_id]
            if prompt_id in self.client_callbacks:
                del self.client_callbacks[prompt_id]

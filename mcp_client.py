import asyncio
import json
import time
import base64
import uuid
import argparse
import os
from typing import Dict, Any, Optional, List, Union

import websockets

class MCPClient:
    """
    MCP protocol client for ComfyUI MCP Server.
    Supports the Model Context Protocol for interacting with image generation tools.
    """
    
    def __init__(self, server_uri: str = "ws://localhost:9000", api_key: Optional[str] = None):
        self.server_uri = server_uri
        self.api_key = api_key
        self.websocket = None
        self.connected = False
        self.authenticated = False
        self.request_counter = 0
        self.pending_requests = {}
        self.session_id = None
        self.manifest = None
    
    async def connect(self) -> bool:
        """
        Connect to the MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.websocket = await websockets.connect(self.server_uri)
            
            # Wait for connection established message
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "connection_established":
                self.connected = True
                self.session_id = data.get("session_id")
                print(f"Connected to MCP server at {self.server_uri}, session: {self.session_id}")
                
                # Start the message handler
                asyncio.create_task(self._message_handler())
                
                # Check if authentication is required
                if data.get("auth_required", False):
                    # Authenticate if API key is provided
                    if self.api_key:
                        return await self.authenticate(self.api_key)
                    else:
                        print("Authentication required but no API key provided")
                        return False
                
                return True
            else:
                print(f"Unexpected connection response: {data}")
                return False
                
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.connected = False
            self.authenticated = False
            print("Disconnected from MCP server")
    
    async def authenticate(self, api_key: str) -> bool:
        """
        Authenticate with the server
        
        Args:
            api_key: API key for authentication
            
        Returns:
            True if authentication successful, False otherwise
        """
        if not self.connected:
            print("Not connected to server")
            return False
        
        try:
            auth_request = {
                "type": "auth",
                "api_key": api_key
            }
            await self.websocket.send(json.dumps(auth_request))
            
            # Authentication response will be handled by _message_handler
            # We'll wait for up to 5 seconds
            for _ in range(10):
                if self.authenticated:
                    return True
                await asyncio.sleep(0.5)
                
            print("Authentication timed out")
            return False
            
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    async def _message_handler(self):
        """Handle incoming messages from the server"""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                try:
                    response = json.loads(message)
                    message_type = response.get("type", "unknown")
                    
                    # Handle authentication response
                    if message_type == "auth_response":
                        if response.get("status") == "success":
                            self.authenticated = True
                            print(f"Authenticated as {response.get('client_id')}")
                        else:
                            self.authenticated = False
                            print(f"Authentication failed: {response.get('message')}")
                    
                    # Handle job status updates
                    elif message_type == "job_status":
                        job_id = response.get("job_id")
                        data = response.get("data", {})
                        
                        status = data.get("status", "unknown")
                        progress = data.get("progress", 0)
                        print(f"Job {job_id} progress: {status} - {progress}%")
                        
                        # If completed and has image URL, show it
                        if status == "completed" and "image_url" in data.get("result", {}):
                            print(f"Job {job_id} completed: {data['result']['image_url']}")
                    
                    # Handle tool responses
                    elif message_type == "mcp_tool_response":
                        request_id = response.get("request_id")
                        
                        if request_id in self.pending_requests:
                            # Get and call the callback
                            callback = self.pending_requests[request_id]
                            if callback:
                                callback(response)
                            
                            # Remove from pending requests
                            del self.pending_requests[request_id]
                        else:
                            print(f"Received response for unknown request: {request_id}")
                            print(response)
                    
                    # Handle manifest responses
                    elif message_type == "mcp_manifest_response":
                        manifest = response.get("manifest")
                        self.manifest = manifest
                        print(f"Received manifest with {len(manifest.get('tools', []))} tools")
                    
                    # Handle errors
                    elif message_type == "error":
                        print(f"Server error: {response.get('message')}")
                    
                    # Handle rate limiting
                    elif message_type == "rate_limit":
                        request_id = response.get("request_id")
                        wait_time = response.get("wait_time", 0)
                        print(f"Rate limit hit for request {request_id}. Wait {wait_time:.1f} seconds.")
                    
                    # Other message types
                    else:
                        print(f"Received message type: {message_type}")
                        print(json.dumps(response, indent=2))
                
                except json.JSONDecodeError:
                    print(f"Received invalid JSON: {message[:100]}...")
                
        except websockets.ConnectionClosed:
            print("Connection to server closed")
            self.connected = False
            self.authenticated = False
        except Exception as e:
            print(f"Error in message handler: {e}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call an MCP tool by name
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool result
        """
        if not self.connected:
            raise ConnectionError("Not connected to server")
        
        result_future = asyncio.Future()
        
        # Generate a unique request ID
        self.request_counter += 1
        request_id = f"req_{self.request_counter}"
        
        # Store callback for when we get a response
        def callback(response):
            result = response.get("result", {})
            if "error" in result:
                result_future.set_exception(Exception(result["error"]))
            else:
                result_future.set_result(result)
        
        self.pending_requests[request_id] = callback
        
        # Create and send the request
        request = {
            "type": "mcp_tool_request",
            "tool": tool_name,
            "parameters": parameters or {},
            "request_id": request_id
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            return await result_future
        except Exception as e:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            raise e
    
    async def get_manifest(self) -> Dict[str, Any]:
        """
        Get the tools manifest from the server
        
        Returns:
            Tools manifest
        """
        if not self.connected:
            raise ConnectionError("Not connected to server")
        
        manifest_future = asyncio.Future()
        
        # Store the future to be resolved when we get the manifest
        def on_manifest(manifest):
            self.manifest = manifest
            manifest_future.set_result(manifest)
        
        # Set up to receive the manifest
        self._on_manifest_received = on_manifest
        
        # Send manifest request
        request = {
            "type": "mcp_manifest_request"
        }
        
        await self.websocket.send(json.dumps(request))
        
        # Wait for the manifest with timeout
        try:
            return await asyncio.wait_for(manifest_future, timeout=10.0)
        except asyncio.TimeoutError:
            # If we already have a cached manifest, return it
            if self.manifest:
                return self.manifest
            raise TimeoutError("Timed out waiting for manifest")
    
    async def ping(self) -> float:
        """
        Ping the server to check latency
        
        Returns:
            Round-trip time in seconds
        """
        if not self.connected:
            raise ConnectionError("Not connected to server")
        
        pong_future = asyncio.Future()
        
        # Generate a unique request ID for this ping
        ping_id = str(uuid.uuid4())
        
        # Store callback for when we get a pong
        def on_pong(response):
            if response.get("type") == "pong":
                end_time = time.time()
                rtt = end_time - start_time
                pong_future.set_result(rtt)
        
        # This is a special case not using the normal request system
        self._on_pong_received = on_pong
        
        # Send ping
        request = {
            "type": "ping",
            "id": ping_id,
            "timestamp": time.time()
        }
        
        start_time = time.time()
        await self.websocket.send(json.dumps(request))
        
        # Wait for pong with timeout
        try:
            return await asyncio.wait_for(pong_future, timeout=5.0)
        except asyncio.TimeoutError:
            raise TimeoutError("Ping timed out")
    
    # High-level convenience methods for specific tools
    
    async def generate_image(self, prompt: str, width: int = 512, height: int = 512,
                            negative_prompt: str = "", model: Optional[str] = None,
                            workflow_id: str = "basic_api_test", **kwargs) -> Dict[str, Any]:
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Text description of the desired image
            width: Width of the output image
            height: Height of the output image
            negative_prompt: Negative prompt to guide generation
            model: Model checkpoint to use
            workflow_id: ID of the workflow to use
            **kwargs: Additional parameters
            
        Returns:
            Response with job information
        """
        parameters = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "negative_prompt": negative_prompt,
            "workflow_id": workflow_id
        }
        
        if model:
            parameters["model"] = model
            
        # Add any additional parameters
        parameters.update(kwargs)
        
        return await self.call_tool("generate_image", parameters)
    
    async def img2img(self, prompt: str, image_path: str, strength: float = 0.75,
                     width: int = 512, height: int = 512, negative_prompt: str = "",
                     model: Optional[str] = None, workflow_id: str = "img2img", **kwargs) -> Dict[str, Any]:
        """
        Generate an image based on another image and a text prompt
        
        Args:
            prompt: Text description of the desired transformation
            image_path: Path to the input image file
            strength: How much to transform the input image (0-1)
            width: Width of the output image
            height: Height of the output image
            negative_prompt: Negative prompt to guide generation
            model: Model checkpoint to use
            workflow_id: ID of the workflow to use
            **kwargs: Additional parameters
            
        Returns:
            Response with job information
        """
        # Read and encode the image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        parameters = {
            "prompt": prompt,
            "image": image_base64,
            "strength": strength,
            "width": width,
            "height": height,
            "negative_prompt": negative_prompt,
            "workflow_id": workflow_id
        }
        
        if model:
            parameters["model"] = model
            
        # Add any additional parameters
        parameters.update(kwargs)
        
        return await self.call_tool("img2img", parameters)
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a job
        
        Args:
            job_id: ID of the job to check
            
        Returns:
            Job status information
        """
        return await self.call_tool("get_job_status", {"job_id": job_id})
    
    async def list_models(self) -> Dict[str, Any]:
        """
        List available model checkpoints
        
        Returns:
            List of available models
        """
        return await self.call_tool("list_models", {})
    
    async def wait_for_job_completion(self, job_id: str, polling_interval: float = 2.0, 
                                     timeout: float = 300.0) -> Dict[str, Any]:
        """
        Wait for a job to complete
        
        Args:
            job_id: ID of the job to wait for
            polling_interval: How often to check job status in seconds
            timeout: Maximum time to wait in seconds
            
        Returns:
            Final job status
        """
        start_time = time.time()
        
        while True:
            # Check if we've exceeded timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for job {job_id} to complete")
            
            # Get job status
            status = await self.get_job_status(job_id)
            
            # If job is completed or has error, return status
            if status.get("status") == "completed" or "error" in status:
                return status
            
            # Wait before next check
            await asyncio.sleep(polling_interval)
    
    async def subscribe_to_job(self, job_id: str) -> bool:
        """
        Subscribe to real-time updates for a job
        
        Args:
            job_id: ID of the job to subscribe to
            
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to server")
        
        try:
            request = {
                "type": "subscribe_job",
                "job_id": job_id
            }
            await self.websocket.send(json.dumps(request))
            
            # We could wait for a subscription confirmation, but for now just return True
            return True
        except Exception as e:
            print(f"Error subscribing to job: {e}")
            return False


async def main():
    """Example usage of the MCP client"""
    parser = argparse.ArgumentParser(description="MCP Client for ComfyUI")
    parser.add_argument("--server", default="ws://localhost:9000", help="MCP server WebSocket URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--action", choices=["generate", "img2img", "list-models", "manifest"], 
                        default="generate", help="Action to perform")
    parser.add_argument("--prompt", default="a dog wearing sunglasses", help="Text prompt")
    parser.add_argument("--image", help="Input image path for img2img")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--model", help="Model checkpoint name")
    parser.add_argument("--workflow", default="basic_api_test", help="Workflow ID")
    args = parser.parse_args()
    
    # Create and connect client
    client = MCPClient(server_uri=args.server, api_key=args.api_key)
    if not await client.connect():
        print("Failed to connect to server")
        return
    
    try:
        # Execute requested action
        if args.action == "generate":
            print(f"Generating image with prompt: {args.prompt}")
            result = await client.generate_image(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                model=args.model,
                workflow_id=args.workflow
            )
            
            job_id = result.get("job_id")
            print(f"Job submitted with ID: {job_id}")
            
            # Subscribe to job updates
            await client.subscribe_to_job(job_id)
            
            # Wait for job completion
            try:
                final_status = await client.wait_for_job_completion(job_id)
                if "error" in final_status:
                    print(f"Job failed: {final_status['error']}")
                else:
                    image_url = final_status.get("result", {}).get("image_url")
                    if image_url:
                        print(f"Generated image: {image_url}")
                    else:
                        print("Job completed but no image URL found")
            except TimeoutError:
                print("Timed out waiting for job completion")
            
        elif args.action == "img2img":
            if not args.image or not os.path.exists(args.image):
                print("Please specify a valid image path")
                return
                
            print(f"Running img2img with prompt: {args.prompt}")
            result = await client.img2img(
                prompt=args.prompt,
                image_path=args.image,
                width=args.width,
                height=args.height,
                model=args.model,
                workflow_id=args.workflow
            )
            
            job_id = result.get("job_id")
            print(f"Job submitted with ID: {job_id}")
            
            # Subscribe to job updates
            await client.subscribe_to_job(job_id)
            
            # Wait for job completion
            try:
                final_status = await client.wait_for_job_completion(job_id)
                if "error" in final_status:
                    print(f"Job failed: {final_status['error']}")
                else:
                    image_url = final_status.get("result", {}).get("image_url")
                    if image_url:
                        print(f"Generated image: {image_url}")
                    else:
                        print("Job completed but no image URL found")
            except TimeoutError:
                print("Timed out waiting for job completion")
            
        elif args.action == "list-models":
            print("Listing available models...")
            models = await client.list_models()
            if "models" in models:
                print("\nAvailable models:")
                for model in models["models"]:
                    print(f"- {model}")
            else:
                print(f"Error listing models: {models.get('error', 'Unknown error')}")
            
        elif args.action == "manifest":
            print("Fetching tools manifest...")
            manifest = await client.get_manifest()
            
            if "tools" in manifest:
                tools = manifest["tools"]
                print(f"\nAvailable tools ({len(tools)}):")
                
                for tool in tools:
                    name = tool.get("name", "unnamed")
                    desc = tool.get("description", "No description")
                    print(f"\n- {name}: {desc}")
                    
                    # Show parameters
                    params = tool.get("parameters", [])
                    if params:
                        print("  Parameters:")
                        for param in params:
                            param_name = param.get("name", "unnamed")
                            param_type = param.get("type", "unknown")
                            required = " (required)" if param.get("required", False) else ""
                            param_desc = param.get("description", "No description")
                            print(f"    - {param_name} ({param_type}){required}: {param_desc}")
            else:
                print("Error fetching manifest or no tools available")
    
    finally:
        # Disconnect cleanly
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

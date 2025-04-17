import asyncio
import websockets
import json
import base64
import argparse
import os
from typing import Dict, Any, Optional, List, Union

class EnhancedMCPClient:
    """
    Enhanced MCP client for the ComfyUI MCP Server.
    Supports all available tools with progress tracking.
    """
    
    def __init__(self, server_uri: str = "ws://localhost:9000", api_key: Optional[str] = None):
        self.server_uri = server_uri
        self.api_key = api_key
        self.websocket = None
        self.connected = False
        self.authenticated = False
        self.request_counter = 0
        self.pending_requests = {}
        self.progress_callbacks = {}
    
    async def connect(self):
        """Connect to the MCP server"""
        try:
            self.websocket = await websockets.connect(self.server_uri)
            self.connected = True
            print(f"Connected to MCP server at {self.server_uri}")
            
            # Start the message handler
            asyncio.create_task(self._message_handler())
            
            # Authenticate if API key is provided
            if self.api_key:
                await self.authenticate(self.api_key)
                
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.websocket and self.connected:
            await self.websocket.close()
            self.connected = False
            self.authenticated = False
            print("Disconnected from MCP server")
    
    async def authenticate(self, api_key: str) -> bool:
        """Authenticate with the server"""
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
                    
                    # Handle progress updates
                    elif message_type == "progress_update":
                        prompt_id = response.get("prompt_id")
                        data = response.get("data", {})
                        
                        # Call the registered progress callback if any
                        if prompt_id in self.progress_callbacks:
                            self.progress_callbacks[prompt_id](data)
                        
                        # Print progress
                        status = data.get("status", "unknown")
                        progress = data.get("progress", 0)
                        print(f"Progress for {prompt_id}: {status} - {progress}%")
                    
                    # Handle tool responses
                    elif message_type == "tool_response":
                        request_id = response.get("request_id")
                        
                        if request_id in self.pending_requests:
                            # Get and call the callback
                            callback = self.pending_requests[request_id]
                            if callback:
                                callback(response)
                            
                            # Remove from pending requests
                            del self.pending_requests[request_id]
                    
                    # Handle errors
                    elif message_type == "error":
                        print(f"Server error: {response.get('message')}")
                    
                    # Handle rate limiting
                    elif message_type == "rate_limit":
                        request_id = response.get("request_id")
                        wait_time = response.get("wait_time", 0)
                        print(f"Rate limit hit for request {request_id}. Wait {wait_time:.1f} seconds.")
                    
                except json.JSONDecodeError:
                    print(f"Received invalid JSON: {message[:100]}...")
                
        except websockets.ConnectionClosed:
            print("Connection to server closed")
            self.connected = False
            self.authenticated = False
        except Exception as e:
            print(f"Error in message handler: {e}")
    
    async def _send_request(self, tool: str, params: Dict[str, Any], callback=None):
        """Send a request to the server and register callback"""
        if not self.connected:
            print("Not connected to server")
            return None
        
        # Generate a unique request ID
        self.request_counter += 1
        request_id = f"req_{self.request_counter}"
        
        # Store callback for when we get a response
        self.pending_requests[request_id] = callback
        
        # Create and send the request
        request = {
            "type": "tool_request",
            "tool": tool,
            "params": json.dumps(params),
            "request_id": request_id
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            return request_id
        except Exception as e:
            print(f"Error sending request: {e}")
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            return None
    
    async def generate_image(self, prompt: str, width: int = 512, height: int = 512, 
                            workflow_id: str = "basic_api_test", model: Optional[str] = None, 
                            **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate an image using ComfyUI.
        
        Args:
            prompt: Text prompt for image generation
            width: Image width
            height: Image height
            workflow_id: Workflow ID to use
            model: Model checkpoint name
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary or None if error
        """
        result = asyncio.Future()
        
        params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "workflow_id": workflow_id,
            "model": model,
            **kwargs
        }
        
        def callback(response):
            if "error" in response:
                result.set_exception(Exception(response["error"]))
            else:
                result.set_result(response)
        
        await self._send_request("generate_image", params, callback)
        
        try:
            return await result
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    async def img2img(self, prompt: str, image_path: str, strength: float = 0.75,
                     width: int = 512, height: int = 512, workflow_id: str = "img2img",
                     model: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate an image based on another image and a text prompt.
        
        Args:
            prompt: Text prompt
            image_path: Path to the input image
            strength: How much to transform the input (0-1)
            width: Output width
            height: Output height
            workflow_id: Workflow ID to use
            model: Model checkpoint name
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary or None if error
        """
        # Read and encode the image
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error reading image file: {e}")
            return None
            
        result = asyncio.Future()
        
        params = {
            "prompt": prompt,
            "image": image_base64,
            "strength": strength,
            "width": width,
            "height": height,
            "workflow_id": workflow_id,
            "model": model,
            **kwargs
        }
        
        def callback(response):
            if "error" in response:
                result.set_exception(Exception(response["error"]))
            else:
                result.set_result(response)
        
        await self._send_request("img2img", params, callback)
        
        try:
            return await result
        except Exception as e:
            print(f"Error in img2img: {e}")
            return None
    
    async def inpaint(self, prompt: str, image_path: str, mask_path: str,
                     width: int = 512, height: int = 512, workflow_id: str = "inpaint",
                     model: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Inpaint parts of an image based on a mask.
        
        Args:
            prompt: Text prompt
            image_path: Path to the input image
            mask_path: Path to the mask image
            width: Output width
            height: Output height
            workflow_id: Workflow ID to use
            model: Model checkpoint name
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary or None if error
        """
        # Read and encode the image and mask
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
            with open(mask_path, "rb") as f:
                mask_bytes = f.read()
                mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error reading image or mask file: {e}")
            return None
            
        result = asyncio.Future()
        
        params = {
            "prompt": prompt,
            "image": image_base64,
            "mask": mask_base64,
            "width": width,
            "height": height,
            "workflow_id": workflow_id,
            "model": model,
            **kwargs
        }
        
        def callback(response):
            if "error" in response:
                result.set_exception(Exception(response["error"]))
            else:
                result.set_result(response)
        
        await self._send_request("inpaint", params, callback)
        
        try:
            return await result
        except Exception as e:
            print(f"Error in inpaint: {e}")
            return None
    
    async def list_workflows(self) -> Optional[Dict[str, Any]]:
        """
        List available workflows and their parameters.
        
        Returns:
            Dictionary with workflows info or None if error
        """
        result = asyncio.Future()
        
        def callback(response):
            if "error" in response:
                result.set_exception(Exception(response["error"]))
            else:
                result.set_result(response)
        
        await self._send_request("list_workflows", {}, callback)
        
        try:
            return await result
        except Exception as e:
            print(f"Error listing workflows: {e}")
            return None
    
    async def list_models(self) -> Optional[Dict[str, Any]]:
        """
        List available model checkpoints.
        
        Returns:
            Dictionary with models info or None if error
        """
        result = asyncio.Future()
        
        def callback(response):
            if "error" in response:
                result.set_exception(Exception(response["error"]))
            else:
                result.set_result(response)
        
        await self._send_request("list_models", {}, callback)
        
        try:
            return await result
        except Exception as e:
            print(f"Error listing models: {e}")
            return None
    
    async def upload_workflow(self, workflow_id: str, workflow_path: str) -> Optional[Dict[str, Any]]:
        """
        Upload a new workflow to the server.
        
        Args:
            workflow_id: ID for the new workflow
            workflow_path: Path to the workflow JSON file
            
        Returns:
            Response dictionary or None if error
        """
        # Read the workflow file
        try:
            with open(workflow_path, "r") as f:
                workflow_data = json.load(f)
        except Exception as e:
            print(f"Error reading workflow file: {e}")
            return None
            
        result = asyncio.Future()
        
        params = {
            "workflow_id": workflow_id,
            "workflow_data": workflow_data
        }
        
        def callback(response):
            if "error" in response:
                result.set_exception(Exception(response["error"]))
            else:
                result.set_result(response)
        
        await self._send_request("upload_workflow", params, callback)
        
        try:
            return await result
        except Exception as e:
            print(f"Error uploading workflow: {e}")
            return None

async def main():
    """Example usage of the enhanced client"""
    parser = argparse.ArgumentParser(description="ComfyUI MCP Client")
    parser.add_argument("--server", default="ws://localhost:9000", help="MCP server WebSocket URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--mode", choices=["txt2img", "img2img", "inpaint", "list"], default="txt2img", help="Operation mode")
    parser.add_argument("--prompt", default="a dog wearing sunglasses", help="Text prompt")
    parser.add_argument("--image", help="Input image path for img2img or inpaint")
    parser.add_argument("--mask", help="Mask image path for inpaint")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--model", default="v1-5-pruned-emaonly.ckpt", help="Model checkpoint name")
    parser.add_argument("--workflow", default="basic_api_test", help="Workflow ID")
    
    args = parser.parse_args()
    
    # Create client
    client = EnhancedMCPClient(server_uri=args.server, api_key=args.api_key)
    
    # Connect to server
    if not await client.connect():
        print("Failed to connect to server")
        return
    
    try:
        # Choose operation based on mode
        if args.mode == "list":
            # List available workflows
            workflows = await client.list_workflows()
            if workflows:
                print("\nAvailable workflows:")
                for workflow_id, params in workflows.get("workflows", {}).items():
                    print(f"- {workflow_id}: {', '.join(params)}")
            
            # List available models
            models = await client.list_models()
            if models:
                print("\nAvailable models:")
                for model in models.get("models", []):
                    print(f"- {model}")
                    
        elif args.mode == "txt2img":
            # Generate image from text
            result = await client.generate_image(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                workflow_id=args.workflow,
                model=args.model
            )
            
            if result:
                print(f"\nGenerated image: {result.get('image_url')}")
            
        elif args.mode == "img2img":
            # Check if image is provided
            if not args.image or not os.path.exists(args.image):
                print("Image file not found. Please provide a valid image path.")
                return
                
            # Generate image from image
            result = await client.img2img(
                prompt=args.prompt,
                image_path=args.image,
                width=args.width,
                height=args.height,
                workflow_id=args.workflow or "img2img",
                model=args.model
            )
            
            if result:
                print(f"\nGenerated image: {result.get('image_url')}")
            
        elif args.mode == "inpaint":
            # Check if image and mask are provided
            if not args.image or not os.path.exists(args.image):
                print("Image file not found. Please provide a valid image path.")
                return
            if not args.mask or not os.path.exists(args.mask):
                print("Mask file not found. Please provide a valid mask path.")
                return
                
            # Inpaint image
            result = await client.inpaint(
                prompt=args.prompt,
                image_path=args.image,
                mask_path=args.mask,
                width=args.width,
                height=args.height,
                workflow_id=args.workflow or "inpaint",
                model=args.model
            )
            
            if result:
                print(f"\nGenerated image: {result.get('image_url')}")
    
    finally:
        # Disconnect from server
        await client.disconnect()

if __name__ == "__main__":
    print("Testing enhanced MCP client...")
    asyncio.run(main())
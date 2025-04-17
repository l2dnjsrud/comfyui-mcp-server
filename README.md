# ComfyUI MCP Server


A full-featured implementation of the Model Context Protocol (MCP) for ComfyUI, enabling AI agents to generate and manipulate images using a standardized API.


## 📋 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Advanced Usage](#-advanced-usage)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Introduction

**ComfyUI MCP Server** bridges the gap between AI agents and image generation by implementing the Model Context Protocol (MCP) for ComfyUI. 

The Model Context Protocol is a standardized way for AI agents to discover and interact with tools through rich metadata and schema information. This implementation allows any AI assistant to easily generate and manipulate images using Stable Diffusion models through ComfyUI.

> 🔍 **What is MCP?** The Model Context Protocol (MCP) standardizes how AI agents communicate with external tools, enabling consistent interfaces with rich metadata and capabilities descriptions.

## 🚀 Features

- **Full MCP Protocol Implementation**
  - Tool discovery with metadata and schemas
  - Session and context management
  - Real-time communication

- **Rich Image Generation Capabilities**
  - Text-to-image (txt2img)
  - Image-to-image transformation (img2img)
  - Inpainting
  - ControlNet support
  - LoRA model integration

- **Advanced Infrastructure**
  - Job queue with priority support
  - Progress tracking and real-time updates
  - Authentication and rate limiting
  - Session persistence
  - Database backend for job history

## 💻 Installation

### Prerequisites

- Python 3.10+
- ComfyUI installed and running
- 4GB+ RAM recommended
- CUDA-compatible GPU recommended (but not required)

### Option 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/comfyui-mcp-server.git
cd comfyui-mcp-server

# Install dependencies
pip install -r requirements.txt

# Run the installation script
python install.py --create-venv
```

### Option 2: Manual Setup

```bash
# Create and prepare project directory
mkdir comfyui-mcp-server
cd comfyui-mcp-server

# Copy the source files into this directory

# Install required packages
pip install websockets requests aiohttp aiohttp_cors aiohttp_session cryptography aiosqlite mcp
```

## 🏃‍♂️ Quick Start

### Step 1: Start ComfyUI

Make sure to start ComfyUI first before running the MCP server:

```bash
cd /path/to/ComfyUI
python main.py --port 8188
```

Verify ComfyUI is running by accessing http://localhost:8188 in your browser.

### Step 2: Start the MCP Server

```bash
cd /path/to/comfyui-mcp-server

# Run with default settings
python mcp_integration.py

# Or run with custom config
python mcp_integration.py --config my_config.json
```

### Step 3: Test the Connection

Run the MCP client to verify everything is working:

```bash
# Check available tools
python mcp_client.py --action manifest

# Generate a test image
python mcp_client.py --action generate --prompt "a dog wearing sunglasses"
```

## 📝 Usage Examples

### Basic Image Generation

Generate an image from a text prompt:

```bash
python mcp_client.py --action generate --prompt "a cat in space" --width 768 --height 512
```

### Image-to-Image Transformation

Transform an existing image based on a prompt:

```bash
python mcp_client.py --action img2img --prompt "watercolor painting" --image input.jpg --strength 0.75
```

### List Available Models

See what models are available to use:

```bash
python mcp_client.py --action list-models
```

### Using in Python Scripts

```python
import asyncio
from mcp_client import MCPClient

async def generate_images():
    client = MCPClient()
    await client.connect()
    
    # Generate an image
    result = await client.generate_image(
        prompt="an astronaut riding a horse on mars",
        width=768,
        height=512
    )
    
    # Get job ID and wait for completion
    job_id = result["job_id"]
    await client.subscribe_to_job(job_id)
    final_status = await client.wait_for_job_completion(job_id)
    
    # Print result
    print(f"Generated image: {final_status['result']['image_url']}")
    
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(generate_images())
```

## 📂 Project Structure

```
comfyui-mcp-server/
├── mcp_protocol.py     # Core MCP protocol implementation
├── mcp_integration.py  # MCP server implementation
├── mcp_client.py       # MCP client
├── job_queue.py        # Job queue system
├── db_manager.py       # Database management
├── model_manager.py    # Model management
├── workflow_analyzer.py # Workflow analysis
├── auth.py             # Authentication & authorization
├── progress_tracker.py # Progress tracking
├── config.json         # Configuration file
├── workflows/          # Workflow definition files
│   ├── basic_api_test.json
│   ├── img2img.json
│   ├── inpaint.json
│   └── simple.json
├── data/               # Data storage folder (created on run)
└── logs/               # Log files (created on run)
```

## 📚 API Reference

### Available MCP Tools

| Tool Name | Description | Key Parameters |
|-----------|-------------|----------------|
| `generate_image` | Generate image from text | prompt, width, height, negative_prompt, model |
| `img2img` | Transform an image based on text | prompt, image, strength, width, height |
| `get_job_status` | Check status of a job | job_id |
| `list_models` | List available models | - |
| `get_tools_manifest` | Get complete tools manifest | - |

### Configuration Options

The `config.json` file contains these key settings:

```json
{
  "comfyui_url": "http://localhost:8188",
  "comfyui_ws_url": "ws://localhost:8188/ws",
  "mcp_server_host": "localhost",
  "mcp_server_port": 9000,
  "enable_auth": false,
  "max_concurrent_jobs": 2
}
```

## 🔧 Advanced Usage

### Setting Up Authentication

1. Edit `config.json` to enable authentication:
   ```json
   {
     "enable_auth": true,
     "auth_config_path": "auth_config.json"
   }
   ```

2. Create or modify `auth_config.json`:
   ```json
   {
     "default_client": {
       "api_key": "YOUR_API_KEY_HERE",
       "permissions": ["*"],
       "rate_limit": {
         "rate": 5,
         "per": 60,
         "burst": 10
       }
     }
   }
   ```

3. Use the API key in client requests:
   ```bash
   python mcp_client.py --api-key YOUR_API_KEY_HERE --action generate --prompt "test"
   ```

### Customizing Workflows

1. Export a workflow from ComfyUI in API format (enable dev mode in settings)
2. Save the JSON file to the `workflows/` directory
3. Use it with the client:
   ```bash
   python mcp_client.py --action generate --prompt "test" --workflow my_custom_workflow
   ```

## ❓ Troubleshooting

### Connection Issues

**Problem**: Cannot connect to ComfyUI
**Solution**: Ensure ComfyUI is running and accessible at the configured URL:
```bash
curl http://localhost:8188/system_stats
```

**Problem**: MCP server won't start
**Solution**: Check logs in the `logs` directory and verify port 9000 is available:
```bash
lsof -i :9000    # On Linux/Mac
# or
netstat -ano | findstr :9000    # On Windows
```

### Image Generation Issues

**Problem**: Generation fails with workflow errors
**Solution**: 
- Check that ComfyUI has the required models installed
- Update the workflow JSON to match your ComfyUI installation
- Verify seed values are positive integers
- Check JSON encoding in workflow files

**Problem**: Timeout during image generation
**Solution**: Increase the timeout value in `advanced_comfyui_client.py`:
```python
max_attempts = 300  # 5 minutes
```

## 🤝 Contributing

Contributions are welcome! Here are some ways to get involved:

1. **Report bugs and request features** by opening Issues
2. **Submit Pull Requests** for bug fixes or enhancements
3. **Create new workflows** for different generation tasks
4. **Improve documentation** to help new users

Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for AI and image generation
</p>

<p align="center">
  <a href="https://github.com/l2dnjsrud/comfyui-mcp-server/issues">Report Bug</a> •
  <a href="https://github.com/l2dnjsrud/comfyui-mcp-server/issues">Request Feature</a>
</p>

#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import json
import argparse
from pathlib import Path
import platform

# Setup command line arguments
parser = argparse.ArgumentParser(description='Install Enhanced ComfyUI MCP Server')
parser.add_argument('--comfyui-path', help='Path to ComfyUI installation', required=False)
parser.add_argument('--port', type=int, default=9000, help='Port for MCP server (default: 9000)')
parser.add_argument('--dashboard-port', type=int, default=8080, help='Port for dashboard (default: 8080)')
parser.add_argument('--enable-auth', action='store_true', help='Enable authentication')
parser.add_argument('--create-venv', action='store_true', help='Create a virtual environment')
parser.add_argument('--install-dir', default='./comfyui-mcp-server', help='Installation directory')
args = parser.parse_args()

print("Enhanced ComfyUI MCP Server Installer")
print("=====================================")

# Determine installation directory
install_dir = Path(args.install_dir).absolute()
print(f"Installing to: {install_dir}")

# Create installation directory if it doesn't exist
if not install_dir.exists():
    install_dir.mkdir(parents=True)
    print(f"Created installation directory: {install_dir}")

# Change to installation directory
os.chdir(install_dir)

# Create required directories
for dir_name in ["workflows", "data", "logs", "static"]:
    os.makedirs(dir_name, exist_ok=True)
    print(f"Created directory: {dir_name}")

# Function to check if a command exists
def command_exists(command):
    try:
        subprocess.run([command, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

# Check Python version
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"Python version: {python_version}")

if sys.version_info < (3, 8):
    print("Error: Python 3.8 or newer is required")
    sys.exit(1)

# Create virtual environment if requested
venv_path = None
if args.create_venv:
    import venv
    venv_path = install_dir / "venv"
    print(f"Creating virtual environment at {venv_path}")
    venv.create(venv_path, with_pip=True)
    
    # Determine path to Python in venv
    if platform.system() == "Windows":
        python_exec = venv_path / "Scripts" / "python.exe"
    else:
        python_exec = venv_path / "bin" / "python"
        
    print(f"Using Python from virtual environment: {python_exec}")
else:
    python_exec = sys.executable
    print(f"Using system Python: {python_exec}")

# Install required packages
print("Installing required packages...")
requirements = [
    "websockets", 
    "requests", 
    "aiohttp", 
    "aiohttp_cors", 
    "aiosqlite", 
    "aiohttp_session", 
    "cryptography",
    "mcp"
]

try:
    subprocess.run([str(python_exec), "-m", "pip", "install", "--upgrade"] + requirements, check=True)
    print("Successfully installed required packages")
except subprocess.CalledProcessError as e:
    print(f"Error installing packages: {e}")
    sys.exit(1)

# Detect ComfyUI path
comfyui_path = args.comfyui_path
if not comfyui_path:
    print("ComfyUI path not provided, attempting to detect...")
    potential_paths = [
        "../ComfyUI",
        "../../ComfyUI",
        "/opt/ComfyUI",
        "C:\\ComfyUI",
        os.path.expanduser("~/ComfyUI")
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "main.py")):
            comfyui_path = path
            print(f"Found ComfyUI at: {comfyui_path}")
            break
    
    if not comfyui_path:
        print("Warning: Could not detect ComfyUI path. You'll need to set it manually in config.json")

# Create configuration
config = {
    "comfyui_url": f"http://localhost:8188" if comfyui_path else "http://localhost:8188",
    "comfyui_ws_url": f"ws://localhost:8188/ws" if comfyui_path else "ws://localhost:8188/ws",
    "mcp_server_host": "0.0.0.0",
    "mcp_server_port": args.port,
    "dashboard_host": "0.0.0.0",
    "dashboard_port": args.dashboard_port,
    "enable_auth": args.enable_auth,
    "auth_config_path": "auth_config.json",
    "workflows_dir": "workflows",
    "db_path": "data/mcp_server.db",
    "queue_save_path": "data/queue_state.json",
    "max_concurrent_jobs": 2
}

# Write configuration to file
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Created configuration file: config.json")

# Create basic auth config if authentication is enabled
if args.enable_auth:
    import secrets
    import base64
    
    # Generate a random API key
    api_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    auth_config = {
        "default_client": {
            "api_key": api_key,
            "permissions": ["*"],
            "rate_limit": {
                "rate": 5,
                "per": 60,
                "burst": 10
            }
        }
    }
    
    with open("auth_config.json", "w") as f:
        json.dump(auth_config, f, indent=2)
    print(f"Created authentication configuration with default API key: {api_key}")

# Copy sample workflows
sample_workflows = {
    "basic_api_test.json": """{
  "3": {
    "inputs": {
      "seed": 156680208700286,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "v1-5-pruned-emaonly.ckpt"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "5": {
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "6": {
    "inputs": {
      "text": "beautiful scenery nature glass bottle landscape, , purple galaxy bottle,",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "text, watermark",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  }
}""",
    "img2img.json": """{
  "3": {
    "inputs": {
      "seed": 156680208700286,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 0.75,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "11",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "v1-5-pruned-emaonly.ckpt"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "6": {
    "inputs": {
      "text": "beautiful scenery nature glass bottle landscape",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "text, watermark, blurry, distorted",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Negative Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "10": {
    "inputs": {
      "image": "base64_encoded_image_placeholder",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image (Base64)"
    }
  },
  "11": {
    "inputs": {
      "pixels": [
        "10",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  }
}""",
    "inpaint.json": """{
  "3": {
    "inputs": {
      "seed": 156680208700286,
      "steps": 20,
      "cfg": 8,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "12",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "v1-5-pruned-emaonly.ckpt"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "6": {
    "inputs": {
      "text": "a beautiful landscape",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "text, watermark, blurry, distorted",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Negative Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI_inpaint",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "10": {
    "inputs": {
      "image": "base64_encoded_image_placeholder",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image (Base64)"
    }
  },
  "11": {
    "inputs": {
      "image": "base64_encoded_mask_placeholder",
      "upload": "mask"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Mask (Base64)"
    }
  },
  "12": {
    "inputs": {
      "samples": [
        "10",
        0
      ],
      "mask": [
        "11",
        0
      ],
      "grow_mask_by": 6,
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEEncodeForInpaint",
    "_meta": {
      "title": "VAE Encode for Inpainting"
    }
  }
}"""
}

# Write sample workflows
for filename, content in sample_workflows.items():
    with open(os.path.join("workflows", filename), "w") as f:
        f.write(content)
    print(f"Created sample workflow: {filename}")

# Create startup scripts
if platform.system() == "Windows":
    # Windows batch file
    with open("start_server.bat", "w") as f:
        if venv_path:
            f.write(f'@echo off\n"{venv_path}\\Scripts\\python.exe" integration_mcp_server.py\n')
        else:
            f.write('@echo off\npython integration_mcp_server.py\n')
            
    with open("start_dashboard.bat", "w") as f:
        if venv_path:
            f.write(f'@echo off\n"{venv_path}\\Scripts\\python.exe" web_dashboard.py\n')
        else:
            f.write('@echo off\npython web_dashboard.py\n')
else:
    # Unix shell script
    with open("start_server.sh", "w") as f:
        if venv_path:
            f.write(f'#!/bin/bash\n"{venv_path}/bin/python" integration_mcp_server.py\n')
        else:
            f.write('#!/bin/bash\npython3 integration_mcp_server.py\n')
    os.chmod("start_server.sh", 0o755)
    
    with open("start_dashboard.sh", "w") as f:
        if venv_path:
            f.write(f'#!/bin/bash\n"{venv_path}/bin/python" web_dashboard.py\n')
        else:
            f.write('#!/bin/bash\npython3 web_dashboard.py\n')
    os.chmod("start_dashboard.sh", 0o755)

print("Created startup scripts")

# Download required Python files
required_files = [
    "workflow_analyzer.py",
    "enhanced_comfyui_client.py",
    "advanced_comfyui_client.py",
    "progress_tracker.py",
    "auth.py",
    "model_manager.py",
    "job_queue.py",
    "db_manager.py",
    "web_dashboard.py",
    "integration_mcp_server.py"
]

print("Installation completed successfully!")
print("\nNext steps:")
print("1. Start ComfyUI: python <comfyui_path>/main.py --port 8188")
print(f"2. Start the MCP server: {'start_server.bat' if platform.system() == 'Windows' else './start_server.sh'}")
print(f"3. Start the web dashboard: {'start_dashboard.bat' if platform.system() == 'Windows' else './start_dashboard.sh'}")
print(f"4. Connect to the dashboard at: http://localhost:{args.dashboard_port}")
print(f"5. Access the MCP server at: ws://localhost:{args.port}")

if args.enable_auth:
    with open("auth_config.json", "r") as f:
        auth_data = json.load(f)
        api_key = auth_data["default_client"]["api_key"]
    print(f"\nYour API key is: {api_key}")
    print("Keep this key secure as it provides full access to your MCP server.")

print("\nEnjoy your enhanced ComfyUI MCP Server!")

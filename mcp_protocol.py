import json
import inspect
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type, get_type_hints, get_origin, get_args

logger = logging.getLogger("MCP_Protocol")

class MCPParameter:
    """Define a parameter for an MCP tool"""
    
    def __init__(self, name: str, parameter_type: Type, description: str, required: bool = True, 
                 default: Any = None, enum: List[Any] = None, format: str = None, example: Any = None):
        self.name = name
        self.parameter_type = parameter_type  
        self.description = description
        self.required = required
        self.default = default
        self.enum = enum
        self.format = format
        self.example = example
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter definition to dictionary"""
        result = {
            "name": self.name,
            "description": self.description,
            "required": self.required
        }
        
        # Handle Python type to JSON schema type conversion
        if self.parameter_type == str:
            result["type"] = "string"
            if self.format:
                result["format"] = self.format
        elif self.parameter_type == int:
            result["type"] = "integer"
        elif self.parameter_type == float:
            result["type"] = "number"
        elif self.parameter_type == bool:
            result["type"] = "boolean"
        elif self.parameter_type == list or get_origin(self.parameter_type) == list:
            result["type"] = "array"
            # If we have list type arguments, add item type
            if get_origin(self.parameter_type) == list:
                item_type = get_args(self.parameter_type)[0]
                if item_type == str:
                    result["items"] = {"type": "string"}
                elif item_type == int:
                    result["items"] = {"type": "integer"}
                elif item_type == float:
                    result["items"] = {"type": "number"}
                elif item_type == bool:
                    result["items"] = {"type": "boolean"}
                else:
                    result["items"] = {"type": "object"}
        elif self.parameter_type == dict or get_origin(self.parameter_type) == dict:
            result["type"] = "object"
        else:
            # Default to string for complex types
            result["type"] = "string"
            
        if self.default is not None:
            result["default"] = self.default
            
        if self.enum:
            result["enum"] = self.enum
            
        if self.example:
            result["example"] = self.example
            
        return result


class MCPTool:
    """Represents an MCP tool with metadata and execution capability"""
    
    def __init__(self, name: str, description: str, function: Callable, 
                 parameters: List[MCPParameter] = None, return_schema: Dict[str, Any] = None,
                 examples: List[Dict[str, Any]] = None, metadata: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters or []
        self.return_schema = return_schema or {"type": "object"}
        self.examples = examples or []
        self.metadata = metadata or {}
        
        # Auto-derive parameters from function signature if not provided
        if not parameters:
            self._derive_parameters_from_function()
    
    def _derive_parameters_from_function(self):
        """Automatically derive parameters from function signature"""
        signature = inspect.signature(self.function)
        type_hints = get_type_hints(self.function)
        
        for name, param in signature.parameters.items():
            # Skip self, context, etc.
            if name in ('self', 'cls', 'context'):
                continue
                
            param_type = type_hints.get(name, str)
            default = param.default if param.default is not inspect.Parameter.empty else None
            required = param.default is inspect.Parameter.empty
            
            # Create parameter from function signature
            self.parameters.append(MCPParameter(
                name=name,
                parameter_type=param_type,
                description=f"Parameter {name}",  # Default description
                required=required,
                default=default
            ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.to_dict() for param in self.parameters],
            "return_schema": self.return_schema,
            "examples": self.examples,
            "metadata": self.metadata
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        # Validate parameters
        for param in self.parameters:
            if param.required and param.name not in parameters:
                raise ValueError(f"Required parameter '{param.name}' missing")
        
        # Add default values for missing optional parameters
        for param in self.parameters:
            if param.name not in parameters and param.default is not None:
                parameters[param.name] = param.default
        
        # Call the function
        try:
            result = await self.function(**parameters)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return {"error": str(e)}
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI spec for this tool"""
        parameters = {}
        required_params = []
        
        for param in self.parameters:
            param_dict = param.to_dict()
            param_name = param_dict.pop("name")
            if param_dict.pop("required", False):
                required_params.append(param_name)
            parameters[param_name] = param_dict
        
        return {
            "openapi": "3.0.0",
            "info": {
                "title": f"MCP Tool: {self.name}",
                "description": self.description,
                "version": "1.0.0"
            },
            "paths": {
                f"/tools/{self.name}": {
                    "post": {
                        "summary": self.description,
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": parameters,
                                        "required": required_params
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful response",
                                "content": {
                                    "application/json": {
                                        "schema": self.return_schema
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


class MCPToolRegistry:
    """Registry for MCP tools"""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
    
    def register_tool(self, tool: MCPTool):
        """Register a tool in the registry"""
        self.tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def tool_exists(self, name: str) -> bool:
        """Check if a tool exists"""
        return name in self.tools
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary"""
        return {
            "tools": [tool.to_dict() for tool in self.tools.values()]
        }
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate a manifest of all tools"""
        return {
            "manifest_version": "1.0",
            "tools": [tool.to_dict() for tool in self.tools.values()]
        }


class MCPContext:
    """Context for MCP execution with conversation history and session management"""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.tool_states: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self) -> str:
        """Create a new session and return the session ID"""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "tool_calls": [],
            "metadata": {}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID"""
        return self.active_sessions.get(session_id)
    
    def update_session(self, session_id: str, data: Dict[str, Any]):
        """Update session data"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].update(data)
            self.active_sessions[session_id]["last_activity"] = time.time()
    
    def add_to_history(self, message: Dict[str, Any], session_id: Optional[str] = None):
        """Add a message to conversation history"""
        entry = {
            "timestamp": time.time(),
            "message": message
        }
        
        if session_id:
            entry["session_id"] = session_id
            
            # Also update session history
            if session_id in self.active_sessions:
                if "history" not in self.active_sessions[session_id]:
                    self.active_sessions[session_id]["history"] = []
                self.active_sessions[session_id]["history"].append(entry)
                self.active_sessions[session_id]["last_activity"] = time.time()
        
        self.conversation_history.append(entry)
    
    def get_tool_state(self, tool_name: str) -> Dict[str, Any]:
        """Get state for a specific tool"""
        return self.tool_states.get(tool_name, {})
    
    def update_tool_state(self, tool_name: str, state: Dict[str, Any]):
        """Update state for a specific tool"""
        if tool_name not in self.tool_states:
            self.tool_states[tool_name] = {}
        self.tool_states[tool_name].update(state)
    
    def clear_old_sessions(self, max_age_seconds: int = 3600):
        """Clear sessions older than max_age_seconds"""
        current_time = time.time()
        to_remove = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session["last_activity"] > max_age_seconds:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.active_sessions[session_id]
    
    def export_context(self) -> Dict[str, Any]:
        """Export the full context (for persistence)"""
        return {
            "conversation_history": self.conversation_history,
            "active_sessions": self.active_sessions,
            "tool_states": self.tool_states
        }
    
    def import_context(self, data: Dict[str, Any]):
        """Import context data (for restoration)"""
        if "conversation_history" in data:
            self.conversation_history = data["conversation_history"]
        if "active_sessions" in data:
            self.active_sessions = data["active_sessions"]
        if "tool_states" in data:
            self.tool_states = data["tool_states"]


class MCP:
    """Main MCP protocol implementation"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.registry = MCPToolRegistry()
        self.context = MCPContext()
    
    def register_tool(self, tool: MCPTool):
        """Register a tool"""
        self.registry.register_tool(tool)
    
    def tool(self, name: Optional[str] = None, description: Optional[str] = None, 
             parameters: List[MCPParameter] = None, return_schema: Dict[str, Any] = None,
             examples: List[Dict[str, Any]] = None, metadata: Dict[str, Any] = None):
        """Decorator to register a function as a tool"""
        def decorator(func):
            nonlocal name, description
            
            # Use function name/docstring if not provided
            if name is None:
                name = func.__name__
            if description is None and func.__doc__:
                description = func.__doc__.strip()
            elif description is None:
                description = f"Tool: {name}"
            
            tool = MCPTool(
                name=name,
                description=description,
                function=func,
                parameters=parameters,
                return_schema=return_schema,
                examples=examples,
                metadata=metadata
            )
            
            self.register_tool(tool)
            return func
        
        return decorator
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a tool by name with parameters"""
        if not self.registry.tool_exists(tool_name):
            return {"error": f"Tool '{tool_name}' not found"}
        
        tool = self.registry.get_tool(tool_name)
        
        # Record the request in history
        self.context.add_to_history({
            "type": "tool_request",
            "tool": tool_name,
            "parameters": parameters
        }, session_id)
        
        # Execute tool
        try:
            start_time = time.time()
            result = await tool.execute(parameters)
            execution_time = time.time() - start_time
            
            # Record result in history
            self.context.add_to_history({
                "type": "tool_response",
                "tool": tool_name,
                "result": result,
                "execution_time": execution_time
            }, session_id)
            
            # If we have a session, record the tool call
            if session_id and session_id in self.context.active_sessions:
                tool_calls = self.context.active_sessions[session_id].get("tool_calls", [])
                tool_calls.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result,
                    "timestamp": time.time(),
                    "execution_time": execution_time
                })
                self.context.update_session(session_id, {"tool_calls": tool_calls})
            
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            error_response = {"error": str(e)}
            
            # Record error in history
            self.context.add_to_history({
                "type": "tool_error",
                "tool": tool_name,
                "error": str(e)
            }, session_id)
            
            return error_response
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        return self.context.create_session()
    
    def get_manifest(self) -> Dict[str, Any]:
        """Get the tools manifest"""
        manifest = self.registry.generate_manifest()
        manifest["service"] = {
            "name": self.name,
            "description": self.description,
            "version": "1.0.0"
        }
        return manifest
    
    def export_context(self) -> Dict[str, Any]:
        """Export the current context"""
        return self.context.export_context()
    
    def import_context(self, data: Dict[str, Any]):
        """Import context data"""
        self.context.import_context(data)

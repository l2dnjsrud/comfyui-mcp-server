import logging
import json
import time
import secrets
import hashlib
import base64
from typing import Dict, List, Optional, Callable, Any, Tuple, Set

logger = logging.getLogger("Auth")

class RateLimiter:
    """
    Implements rate limiting for API requests.
    Uses a simple token bucket algorithm for limiting requests.
    """
    
    def __init__(self, rate: int = 5, per: int = 60, burst: int = 10):
        """
        Initialize rate limiter
        
        Args:
            rate: How many tokens to add per time period
            per: Time period in seconds
            burst: Maximum number of tokens that can be accumulated
        """
        self.rate = rate 
        self.per = per
        self.burst = burst
        self.tokens: Dict[str, float] = {}  # client_id -> token count
        self.last_update: Dict[str, float] = {}  # client_id -> last update time
    
    def _update_tokens(self, client_id: str):
        """Update token count based on elapsed time"""
        current_time = time.time()
        if client_id not in self.last_update:
            self.tokens[client_id] = self.burst
            self.last_update[client_id] = current_time
            return
            
        elapsed = current_time - self.last_update[client_id]
        new_tokens = elapsed * (self.rate / self.per)
        
        self.tokens[client_id] = min(self.burst, self.tokens.get(client_id, 0) + new_tokens)
        self.last_update[client_id] = current_time
    
    def can_process(self, client_id: str, cost: float = 1.0) -> bool:
        """Check if client can process a request with given cost"""
        self._update_tokens(client_id)
        return self.tokens.get(client_id, 0) >= cost
    
    def consume(self, client_id: str, cost: float = 1.0) -> bool:
        """Consume tokens for a request if possible"""
        if self.can_process(client_id, cost):
            self.tokens[client_id] -= cost
            return True
        return False
    
    def wait_time(self, client_id: str, cost: float = 1.0) -> float:
        """Calculate wait time until client can make a request"""
        self._update_tokens(client_id)
        
        current_tokens = self.tokens.get(client_id, 0)
        if current_tokens >= cost:
            return 0
            
        missing_tokens = cost - current_tokens
        return missing_tokens * (self.per / self.rate)


class AuthManager:
    """
    Manages authentication and authorization for the MCP server.
    Supports API key authentication and per-client rate limiting.
    """
    
    def __init__(self, enable_auth: bool = True):
        self.enable_auth = enable_auth
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limiter = RateLimiter()
        
        # Different rate limiting for different operations
        self.operation_costs = {
            "generate_image": 1.0,
            "img2img": 1.5,
            "inpaint": 2.0
        }
        
        # Default cost for unspecified operations
        self.default_cost = 1.0
    
    def load_api_keys(self, config_path: Optional[str] = None):
        """Load API keys from a JSON config file"""
        if not config_path:
            # Create a default API key if none provided
            api_key = self._generate_api_key()
            self.api_keys[api_key] = {
                "client_id": "default_client",
                "permissions": ["*"],
                "rate_limit": {"rate": 5, "per": 60, "burst": 10}
            }
            logger.info(f"Created default API key: {api_key}")
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            for client_id, client_data in config.items():
                if "api_key" in client_data:
                    self.api_keys[client_data["api_key"]] = {
                        "client_id": client_id,
                        "permissions": client_data.get("permissions", ["*"]),
                        "rate_limit": client_data.get("rate_limit", {"rate": 5, "per": 60, "burst": 10})
                    }
                    
            logger.info(f"Loaded {len(self.api_keys)} API keys from {config_path}")
                
        except Exception as e:
            logger.error(f"Error loading API keys from {config_path}: {e}")
            # Create a default API key as fallback
            if not self.api_keys:
                api_key = self._generate_api_key()
                self.api_keys[api_key] = {
                    "client_id": "default_client",
                    "permissions": ["*"],
                    "rate_limit": {"rate": 5, "per": 60, "burst": 10}
                }
                logger.info(f"Created default API key: {api_key}")
    
    def _generate_api_key(self, length: int = 32) -> str:
        """Generate a secure random API key"""
        random_bytes = secrets.token_bytes(length)
        return base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')
    
    def authenticate(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a client using an API key.
        Returns client data if authentication successful, None otherwise.
        """
        if not self.enable_auth:
            # If auth is disabled, create a default client for the request
            return {
                "client_id": "anonymous",
                "permissions": ["*"]
            }
            
        return self.api_keys.get(api_key)
    
    def authorize(self, client_data: Dict[str, Any], operation: str) -> bool:
        """Check if a client is authorized to perform an operation"""
        if not self.enable_auth:
            return True
            
        if not client_data:
            return False
            
        permissions = client_data.get("permissions", [])
        return "*" in permissions or operation in permissions
    
    def rate_limit(self, client_id: str, operation: str) -> Tuple[bool, float]:
        """
        Apply rate limiting to a client request.
        Returns (can_proceed, wait_time_seconds)
        """
        if not self.enable_auth:
            return True, 0
            
        # Get the cost for this operation
        cost = self.operation_costs.get(operation, self.default_cost)
        
        if self.rate_limiter.can_process(client_id, cost):
            self.rate_limiter.consume(client_id, cost)
            return True, 0
        else:
            wait_time = self.rate_limiter.wait_time(client_id, cost)
            return False, wait_time
    
    def configure_client_rate_limit(self, client_id: str, rate: int, per: int, burst: int):
        """Configure custom rate limiting for a client"""
        # Find all API keys for this client
        for api_key, data in self.api_keys.items():
            if data.get("client_id") == client_id:
                data["rate_limit"] = {"rate": rate, "per": per, "burst": burst}
                
        # Update rate limiter if client already has token bucket
        if client_id in self.rate_limiter.tokens:
            # Keep current tokens but update rate parameters
            current_tokens = self.rate_limiter.tokens.get(client_id, 0)
            self.rate_limiter.tokens[client_id] = min(burst, current_tokens)

    def create_api_key(self, client_id: str, permissions: List[str] = None) -> str:
        """Create a new API key for a client"""
        if permissions is None:
            permissions = ["*"]
            
        api_key = self._generate_api_key()
        self.api_keys[api_key] = {
            "client_id": client_id,
            "permissions": permissions,
            "rate_limit": {"rate": 5, "per": 60, "burst": 10}
        }
        
        logger.info(f"Created new API key for client {client_id}")
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            logger.info(f"Revoked API key: {api_key}")
            return True
        return False
        
    def get_client_api_keys(self, client_id: str) -> List[str]:
        """Get all API keys for a client"""
        return [
            key for key, data in self.api_keys.items()
            if data.get("client_id") == client_id
        ]
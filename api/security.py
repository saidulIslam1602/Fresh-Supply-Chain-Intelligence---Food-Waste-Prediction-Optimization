"""
Enterprise-Grade Security Components for Fresh Supply Chain Intelligence API
"""

import jwt
import bcrypt
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from pydantic import BaseModel, validator
import redis
import json
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from passlib.context import CryptContext
from passlib.hash import pbkdf2_sha256
import structlog
from enum import Enum
import ipaddress
from functools import wraps
import asyncio
import time

logger = structlog.get_logger()

# Security configuration
class SecurityConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    PASSWORD_MIN_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    SESSION_TIMEOUT_MINUTES = 60
    
    # API Key settings
    API_KEY_LENGTH = 32
    API_KEY_PREFIX = "fsc_"
    
    # Encryption settings
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    
    # IP whitelist (for admin endpoints)
    ADMIN_IP_WHITELIST = [
        "127.0.0.1",
        "::1",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16"
    ]

config = SecurityConfig()

# Password hashing
pwd_context = CryptContext(
    schemes=["bcrypt", "pbkdf2_sha256"],
    deprecated="auto",
    bcrypt__rounds=12,
    pbkdf2_sha256__rounds=100000
)

# Encryption utilities
class EncryptionManager:
    """Handle data encryption/decryption for sensitive data"""
    
    def __init__(self, key: str = None):
        if key:
            self.key = key.encode() if isinstance(key, str) else key
        else:
            self.key = config.ENCRYPTION_KEY.encode()
        
        # Derive key for Fernet
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt_for_api',  # In production, use random salt per encryption
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.key))
        self.fernet = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise HTTPException(status_code=500, detail="Encryption failed")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise HTTPException(status_code=500, detail="Decryption failed")

# User roles and permissions
class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"

class Permission(str, Enum):
    READ_PRODUCTS = "read:products"
    WRITE_PRODUCTS = "write:products"
    READ_FORECASTS = "read:forecasts"
    WRITE_FORECASTS = "write:forecasts"
    READ_ANALYTICS = "read:analytics"
    ADMIN_ACCESS = "admin:access"
    API_ACCESS = "api:access"
    BATCH_OPERATIONS = "batch:operations"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ_PRODUCTS, Permission.WRITE_PRODUCTS,
        Permission.READ_FORECASTS, Permission.WRITE_FORECASTS,
        Permission.READ_ANALYTICS, Permission.ADMIN_ACCESS,
        Permission.API_ACCESS, Permission.BATCH_OPERATIONS
    ],
    UserRole.MANAGER: [
        Permission.READ_PRODUCTS, Permission.WRITE_PRODUCTS,
        Permission.READ_FORECASTS, Permission.WRITE_FORECASTS,
        Permission.READ_ANALYTICS, Permission.API_ACCESS
    ],
    UserRole.ANALYST: [
        Permission.READ_PRODUCTS, Permission.READ_FORECASTS,
        Permission.READ_ANALYTICS, Permission.API_ACCESS
    ],
    UserRole.VIEWER: [
        Permission.READ_PRODUCTS, Permission.READ_ANALYTICS
    ],
    UserRole.API_USER: [
        Permission.API_ACCESS, Permission.READ_PRODUCTS,
        Permission.READ_FORECASTS
    ]
}

# Enhanced user models
class User(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: str
    roles: List[UserRole]
    permissions: List[Permission]
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    class Config:
        use_enum_values = True

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    roles: List[UserRole] = [UserRole.VIEWER]
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < config.PASSWORD_MIN_LENGTH:
            raise ValueError(f'Password must be at least {config.PASSWORD_MIN_LENGTH} characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v or '.' not in v.split('@')[1]:
            raise ValueError('Invalid email format')
        return v.lower()

class APIKey(BaseModel):
    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[Permission]
    is_active: bool = True
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0

# Security utilities
class SecurityUtils:
    """Utility functions for security operations"""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with salt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_api_key() -> tuple[str, str]:
        """Generate API key and its hash"""
        key = f"{config.API_KEY_PREFIX}{secrets.token_urlsafe(config.API_KEY_LENGTH)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash
    
    @staticmethod
    def verify_api_key(key: str, key_hash: str) -> bool:
        """Verify API key against hash"""
        return hashlib.sha256(key.encode()).hexdigest() == key_hash
    
    @staticmethod
    def is_ip_allowed(ip: str, whitelist: List[str]) -> bool:
        """Check if IP is in whitelist"""
        try:
            client_ip = ipaddress.ip_address(ip)
            for allowed in whitelist:
                if '/' in allowed:  # CIDR notation
                    if client_ip in ipaddress.ip_network(allowed, strict=False):
                        return True
                else:  # Single IP
                    if client_ip == ipaddress.ip_address(allowed):
                        return True
            return False
        except ValueError:
            return False

# JWT token management
class JWTManager:
    """Enhanced JWT token management with refresh tokens and blacklisting"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.encryption_manager = EncryptionManager()
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode = {
            "sub": user.username,
            "user_id": user.user_id,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode = {
            "sub": user.username,
            "user_id": user.user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": SecurityUtils.generate_secure_token(16)  # JWT ID for blacklisting
        }
        
        encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
        
        # Store refresh token in Redis for blacklisting
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"refresh_token:{to_encode['jti']}",
                    config.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
                    json.dumps({"user_id": user.user_id, "username": user.username})
                )
            except Exception as e:
                logger.error(f"Failed to store refresh token: {e}")
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type:
                raise HTTPException(status_code=401, detail="Invalid token type")
            
            # Check if token is blacklisted (for refresh tokens)
            if token_type == "refresh" and self.redis_client:
                jti = payload.get("jti")
                if jti and not self.redis_client.exists(f"refresh_token:{jti}"):
                    raise HTTPException(status_code=401, detail="Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def revoke_refresh_token(self, token: str):
        """Revoke refresh token by removing from Redis"""
        try:
            payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
            jti = payload.get("jti")
            if jti and self.redis_client:
                self.redis_client.delete(f"refresh_token:{jti}")
                logger.info(f"Refresh token revoked: {jti}")
        except Exception as e:
            logger.error(f"Failed to revoke refresh token: {e}")

# Authentication manager
class AuthenticationManager:
    """Comprehensive authentication management"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.jwt_manager = JWTManager(redis_client)
        self.encryption_manager = EncryptionManager()
    
    async def authenticate_user(self, username: str, password: str, client_ip: str) -> tuple[User, str, str]:
        """Authenticate user with enhanced security checks"""
        
        # Check if user is locked out
        if await self._is_user_locked(username):
            raise HTTPException(status_code=423, detail="Account temporarily locked due to multiple failed attempts")
        
        # Get user from database (mock implementation)
        user = await self._get_user_by_username(username)
        if not user:
            await self._record_failed_attempt(username, client_ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        if not SecurityUtils.verify_password(password, user.hashed_password):
            await self._record_failed_attempt(username, client_ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(status_code=401, detail="Account is disabled")
        
        # Reset failed attempts on successful login
        await self._reset_failed_attempts(username)
        
        # Update last login
        user.last_login = datetime.utcnow()
        
        # Create tokens
        access_token = self.jwt_manager.create_access_token(user)
        refresh_token = self.jwt_manager.create_refresh_token(user)
        
        # Log successful login
        logger.info(f"User {username} logged in successfully from {client_ip}")
        
        return user, access_token, refresh_token
    
    async def authenticate_api_key(self, api_key: str, client_ip: str) -> User:
        """Authenticate using API key"""
        
        # Get API key from database (mock implementation)
        api_key_obj = await self._get_api_key(api_key)
        if not api_key_obj or not api_key_obj.is_active:
            logger.warning(f"Invalid API key used from {client_ip}")
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Check expiration
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="API key has expired")
        
        # Update usage statistics
        await self._update_api_key_usage(api_key_obj.key_id)
        
        # Get associated user
        user = await self._get_user_by_id(api_key_obj.user_id)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="Associated user account is disabled")
        
        # Override user permissions with API key permissions
        user.permissions = api_key_obj.permissions
        
        logger.info(f"API key authentication successful for user {user.username} from {client_ip}")
        
        return user
    
    async def _is_user_locked(self, username: str) -> bool:
        """Check if user account is locked"""
        if not self.redis_client:
            return False
        
        try:
            lock_data = self.redis_client.get(f"user_lock:{username}")
            if lock_data:
                lock_info = json.loads(lock_data)
                locked_until = datetime.fromisoformat(lock_info["locked_until"])
                return datetime.utcnow() < locked_until
        except Exception as e:
            logger.error(f"Error checking user lock status: {e}")
        
        return False
    
    async def _record_failed_attempt(self, username: str, client_ip: str):
        """Record failed login attempt"""
        if not self.redis_client:
            return
        
        try:
            key = f"failed_attempts:{username}"
            attempts = self.redis_client.incr(key)
            self.redis_client.expire(key, config.LOCKOUT_DURATION_MINUTES * 60)
            
            logger.warning(f"Failed login attempt {attempts} for {username} from {client_ip}")
            
            # Lock account if too many attempts
            if attempts >= config.MAX_LOGIN_ATTEMPTS:
                locked_until = datetime.utcnow() + timedelta(minutes=config.LOCKOUT_DURATION_MINUTES)
                lock_data = {
                    "locked_until": locked_until.isoformat(),
                    "attempts": attempts,
                    "client_ip": client_ip
                }
                self.redis_client.setex(
                    f"user_lock:{username}",
                    config.LOCKOUT_DURATION_MINUTES * 60,
                    json.dumps(lock_data)
                )
                logger.warning(f"User {username} locked until {locked_until}")
                
        except Exception as e:
            logger.error(f"Error recording failed attempt: {e}")
    
    async def _reset_failed_attempts(self, username: str):
        """Reset failed login attempts counter"""
        if self.redis_client:
            try:
                self.redis_client.delete(f"failed_attempts:{username}")
                self.redis_client.delete(f"user_lock:{username}")
            except Exception as e:
                logger.error(f"Error resetting failed attempts: {e}")
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username (mock implementation)"""
        # In production, this would query your user database
        mock_users = {
            "admin": User(
                user_id="1",
                username="admin",
                email="admin@freshsupply.com",
                full_name="System Administrator",
                roles=[UserRole.ADMIN],
                permissions=ROLE_PERMISSIONS[UserRole.ADMIN],
                created_at=datetime.utcnow(),
                hashed_password=SecurityUtils.hash_password("admin123")
            ),
            "analyst": User(
                user_id="2",
                username="analyst",
                email="analyst@freshsupply.com",
                full_name="Supply Chain Analyst",
                roles=[UserRole.ANALYST],
                permissions=ROLE_PERMISSIONS[UserRole.ANALYST],
                created_at=datetime.utcnow(),
                hashed_password=SecurityUtils.hash_password("analyst123")
            )
        }
        
        user_data = mock_users.get(username)
        if user_data:
            # Add hashed_password attribute for verification
            user_data.hashed_password = mock_users[username].hashed_password
        
        return user_data
    
    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID (mock implementation)"""
        # Mock implementation - in production, query database
        return None
    
    async def _get_api_key(self, api_key: str) -> Optional[APIKey]:
        """Get API key object (mock implementation)"""
        # Mock implementation - in production, query database
        return None
    
    async def _update_api_key_usage(self, key_id: str):
        """Update API key usage statistics"""
        # Mock implementation - in production, update database
        pass

# Security decorators and dependencies
class SecurityDependencies:
    """FastAPI security dependencies"""
    
    def __init__(self, redis_client=None):
        self.auth_manager = AuthenticationManager(redis_client)
        self.jwt_manager = JWTManager(redis_client)
        self.bearer_scheme = HTTPBearer()
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> User:
        """Get current authenticated user from JWT token"""
        token = credentials.credentials
        
        try:
            payload = self.jwt_manager.verify_token(token, "access")
            username = payload.get("sub")
            
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Get user from database
            user = await self.auth_manager._get_user_by_username(username)
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(status_code=401, detail="Token verification failed")
    
    async def get_current_active_user(self, current_user: User = Depends(get_current_user)) -> User:
        """Get current active user"""
        if not current_user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
    
    def require_permissions(self, required_permissions: List[Permission]):
        """Decorator to require specific permissions"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                current_user = kwargs.get('current_user')
                if not current_user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                # Check permissions
                user_permissions = set(current_user.permissions)
                required_perms = set(required_permissions)
                
                if not required_perms.issubset(user_permissions):
                    missing_perms = required_perms - user_permissions
                    raise HTTPException(
                        status_code=403,
                        detail=f"Insufficient permissions. Missing: {[p.value for p in missing_perms]}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_roles(self, required_roles: List[UserRole]):
        """Decorator to require specific roles"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                current_user = kwargs.get('current_user')
                if not current_user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                # Check roles
                user_roles = set(current_user.roles)
                required_role_set = set(required_roles)
                
                if not user_roles.intersection(required_role_set):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Insufficient privileges. Required roles: {[r.value for r in required_roles]}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_ip_whitelist(self, whitelist: List[str] = None):
        """Decorator to require IP whitelist"""
        if whitelist is None:
            whitelist = config.ADMIN_IP_WHITELIST
        
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                client_ip = request.client.host
                
                if not SecurityUtils.is_ip_allowed(client_ip, whitelist):
                    logger.warning(f"Access denied for IP {client_ip}")
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: IP not in whitelist"
                    )
                
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator

# Session management
class SessionManager:
    """Manage user sessions with Redis"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def create_session(self, user: User, session_data: Dict[str, Any] = None) -> str:
        """Create user session"""
        session_id = SecurityUtils.generate_secure_token(32)
        
        session_info = {
            "user_id": user.user_id,
            "username": user.username,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "data": session_data or {}
        }
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    config.SESSION_TIMEOUT_MINUTES * 60,
                    json.dumps(session_info, default=str)
                )
            except Exception as e:
                logger.error(f"Failed to create session: {e}")
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if not self.redis_client:
            return None
        
        try:
            session_data = self.redis_client.get(f"session:{session_id}")
            if session_data:
                return json.loads(session_data)
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
        
        return None
    
    async def update_session_activity(self, session_id: str):
        """Update session last activity"""
        session_data = await self.get_session(session_id)
        if session_data:
            session_data["last_activity"] = datetime.utcnow().isoformat()
            
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"session:{session_id}",
                        config.SESSION_TIMEOUT_MINUTES * 60,
                        json.dumps(session_data, default=str)
                    )
                except Exception as e:
                    logger.error(f"Failed to update session: {e}")
    
    async def delete_session(self, session_id: str):
        """Delete session"""
        if self.redis_client:
            try:
                self.redis_client.delete(f"session:{session_id}")
            except Exception as e:
                logger.error(f"Failed to delete session: {e}")

# Security audit logging
class SecurityAuditLogger:
    """Log security events for compliance and monitoring"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.logger = structlog.get_logger("security_audit")
    
    async def log_authentication_event(self, event_type: str, username: str, client_ip: str, 
                                     success: bool, details: Dict[str, Any] = None):
        """Log authentication events"""
        event = {
            "event_type": event_type,
            "username": username,
            "client_ip": client_ip,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        self.logger.info(
            f"Authentication event: {event_type}",
            **event
        )
        
        # Store in Redis for analysis
        if self.redis_client:
            try:
                event_key = f"audit:auth:{int(time.time())}"
                self.redis_client.setex(event_key, 86400 * 30, json.dumps(event))  # Keep for 30 days
            except Exception as e:
                logger.error(f"Failed to store audit event: {e}")
    
    async def log_authorization_event(self, username: str, resource: str, action: str, 
                                    allowed: bool, client_ip: str):
        """Log authorization events"""
        event = {
            "event_type": "authorization",
            "username": username,
            "resource": resource,
            "action": action,
            "allowed": allowed,
            "client_ip": client_ip,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(
            f"Authorization event: {action} on {resource}",
            **event
        )
    
    async def log_security_violation(self, violation_type: str, client_ip: str, 
                                   details: Dict[str, Any]):
        """Log security violations"""
        event = {
            "event_type": "security_violation",
            "violation_type": violation_type,
            "client_ip": client_ip,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        
        self.logger.warning(
            f"Security violation: {violation_type}",
            **event
        )
        
        # Store high-priority security events
        if self.redis_client:
            try:
                event_key = f"audit:violation:{int(time.time())}"
                self.redis_client.setex(event_key, 86400 * 90, json.dumps(event))  # Keep for 90 days
            except Exception as e:
                logger.error(f"Failed to store security violation: {e}")
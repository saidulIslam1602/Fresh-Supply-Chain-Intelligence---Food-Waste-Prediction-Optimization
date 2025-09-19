"""
Advanced Authentication and Authorization System
Enterprise-grade security with multi-factor authentication, OAuth2, and fine-grained permissions
"""

import os
import secrets
import hashlib
import hmac
import time
import qrcode
import pyotp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import redis
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = structlog.get_logger()

class AuthMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"

class UserRole(Enum):
    """User roles with hierarchical permissions"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"
    GUEST = "guest"

class Permission(Enum):
    """Fine-grained permissions"""
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    
    # ML model permissions
    MODEL_TRAIN = "model:train"
    MODEL_PREDICT = "model:predict"
    MODEL_DEPLOY = "model:deploy"
    MODEL_DELETE = "model:delete"
    
    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    
    # Dashboard permissions
    DASHBOARD_VIEW = "dashboard:view"
    DASHBOARD_EDIT = "dashboard:edit"
    DASHBOARD_ADMIN = "dashboard:admin"

@dataclass
class SecurityConfig:
    """Security configuration"""
    # JWT settings
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Password settings
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digits: bool = True
    password_require_special: bool = True
    password_history_count: int = 5
    
    # MFA settings
    mfa_required_roles: List[str] = field(default_factory=lambda: ["admin", "super_admin"])
    totp_issuer: str = "Fresh Supply Chain Intelligence"
    
    # Session settings
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 3
    
    # Security settings
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    enable_audit_logging: bool = True
    enable_encryption_at_rest: bool = True
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/1"

@dataclass
class User:
    """Enhanced user model with security features"""
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[UserRole]
    permissions: List[Permission] = field(default_factory=list)
    
    # Security fields
    is_active: bool = True
    is_verified: bool = False
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    last_login: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=datetime.utcnow)
    password_history: List[str] = field(default_factory=list)
    
    # MFA fields
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)
    
    # Session tracking
    active_sessions: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_password_change: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AuthSession:
    """Authentication session"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    mfa_verified: bool = False

class PasswordValidator:
    """Advanced password validation"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.common_passwords = self._load_common_passwords()
    
    def _load_common_passwords(self) -> set:
        """Load common passwords list"""
        # In production, load from a comprehensive list
        return {
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "dragon", "master"
        }
    
    def validate_password(self, password: str, username: str = None) -> Tuple[bool, List[str]]:
        """Validate password against security requirements"""
        errors = []
        
        # Length check
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        # Character requirements
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.password_require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        # Common password check
        if password.lower() in self.common_passwords:
            errors.append("Password is too common")
        
        # Username similarity check
        if username and username.lower() in password.lower():
            errors.append("Password cannot contain username")
        
        # Entropy check (simplified)
        if self._calculate_entropy(password) < 3.0:
            errors.append("Password is not complex enough")
        
        return len(errors) == 0, errors
    
    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy"""
        if not password:
            return 0.0
        
        # Count unique characters
        unique_chars = len(set(password))
        
        # Estimate character space
        char_space = 0
        if any(c.islower() for c in password):
            char_space += 26
        if any(c.isupper() for c in password):
            char_space += 26
        if any(c.isdigit() for c in password):
            char_space += 10
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            char_space += 32
        
        # Calculate entropy
        import math
        if char_space > 0:
            return len(password) * math.log2(char_space) / len(password)
        return 0.0

class MFAManager:
    """Multi-Factor Authentication manager"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def setup_totp(self, user: User) -> Tuple[str, str]:
        """Setup TOTP for user"""
        secret = pyotp.random_base32()
        user.mfa_secret = secret
        user.mfa_enabled = True
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        user.backup_codes = backup_codes
        
        # Generate QR code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user.email,
            issuer_name=self.config.totp_issuer
        )
        
        return secret, totp_uri
    
    def verify_totp(self, user: User, token: str) -> bool:
        """Verify TOTP token"""
        if not user.mfa_secret:
            return False
        
        totp = pyotp.TOTP(user.mfa_secret)
        
        # Check current token and previous/next tokens for clock skew
        for time_offset in [-1, 0, 1]:
            if totp.verify(token, time.time() + time_offset * 30):
                return True
        
        return False
    
    def verify_backup_code(self, user: User, code: str) -> bool:
        """Verify backup code"""
        code = code.upper().strip()
        if code in user.backup_codes:
            user.backup_codes.remove(code)
            return True
        return False
    
    def generate_backup_codes(self, user: User) -> List[str]:
        """Generate new backup codes"""
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        user.backup_codes = backup_codes
        return backup_codes

class AdvancedAuthManager:
    """Advanced authentication and authorization manager"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.password_validator = PasswordValidator(self.config)
        self.mfa_manager = MFAManager(self.config)
        
        # Initialize Redis for session management
        self.redis_client = None
        self._initialize_redis()
        
        # Initialize encryption
        self.cipher_suite = self._initialize_encryption()
        
        # Role-permission mapping
        self.role_permissions = self._setup_role_permissions()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.config.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis session store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption for sensitive data"""
        # In production, use proper key management (HSM, KMS, etc.)
        key = base64.urlsafe_b64encode(self.config.jwt_secret_key.encode()[:32].ljust(32, b'0'))
        return Fernet(key)
    
    def _setup_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Setup role-based permissions"""
        return {
            UserRole.SUPER_ADMIN: list(Permission),  # All permissions
            
            UserRole.ADMIN: [
                Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR,
                Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
                Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_DELETE, Permission.DATA_EXPORT,
                Permission.MODEL_TRAIN, Permission.MODEL_PREDICT, Permission.MODEL_DEPLOY,
                Permission.API_READ, Permission.API_WRITE, Permission.API_ADMIN,
                Permission.DASHBOARD_VIEW, Permission.DASHBOARD_EDIT, Permission.DASHBOARD_ADMIN
            ],
            
            UserRole.MANAGER: [
                Permission.SYSTEM_MONITOR,
                Permission.USER_READ, Permission.USER_UPDATE,
                Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_EXPORT,
                Permission.MODEL_PREDICT, Permission.MODEL_DEPLOY,
                Permission.API_READ, Permission.API_WRITE,
                Permission.DASHBOARD_VIEW, Permission.DASHBOARD_EDIT
            ],
            
            UserRole.ANALYST: [
                Permission.DATA_READ, Permission.DATA_EXPORT,
                Permission.MODEL_PREDICT,
                Permission.API_READ,
                Permission.DASHBOARD_VIEW
            ],
            
            UserRole.VIEWER: [
                Permission.DATA_READ,
                Permission.API_READ,
                Permission.DASHBOARD_VIEW
            ],
            
            UserRole.API_USER: [
                Permission.API_READ, Permission.API_WRITE,
                Permission.MODEL_PREDICT
            ],
            
            UserRole.GUEST: [
                Permission.DASHBOARD_VIEW
            ]
        }
    
    def create_user(self, username: str, email: str, password: str, roles: List[UserRole]) -> Tuple[User, List[str]]:
        """Create new user with validation"""
        # Validate password
        is_valid, errors = self.password_validator.validate_password(password, username)
        if not is_valid:
            return None, errors
        
        # Hash password
        password_hash = self.pwd_context.hash(password)
        
        # Create user
        user = User(
            id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles,
            permissions=self._get_permissions_for_roles(roles),
            password_history=[password_hash]
        )
        
        # Setup MFA for privileged roles
        if any(role.value in self.config.mfa_required_roles for role in roles):
            self.mfa_manager.setup_totp(user)
        
        return user, []
    
    def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Tuple[Optional[User], Optional[str]]:
        """Authenticate user with comprehensive security checks"""
        # Get user (in production, from database)
        user = self._get_user_by_username(username)
        if not user:
            return None, "Invalid credentials"
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            return None, f"Account locked until {user.locked_until}"
        
        # Check if account is active
        if not user.is_active:
            return None, "Account is disabled"
        
        # Verify password
        if not self.pwd_context.verify(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.config.lockout_duration_minutes)
                logger.warning(f"Account {username} locked due to failed login attempts")
            
            return None, "Invalid credentials"
        
        # Reset failed attempts on successful authentication
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        return user, None
    
    def verify_mfa(self, user: User, token: str) -> bool:
        """Verify MFA token"""
        if not user.mfa_enabled:
            return True
        
        # Try TOTP first
        if self.mfa_manager.verify_totp(user, token):
            return True
        
        # Try backup code
        if self.mfa_manager.verify_backup_code(user, token):
            return True
        
        return False
    
    def create_session(self, user: User, ip_address: str, user_agent: str, mfa_verified: bool = False) -> AuthSession:
        """Create authentication session"""
        session_id = secrets.token_urlsafe(32)
        
        session = AuthSession(
            session_id=session_id,
            user_id=user.id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            mfa_verified=mfa_verified
        )
        
        # Store session in Redis
        if self.redis_client:
            session_data = {
                "user_id": user.id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "ip_address": ip_address,
                "user_agent": user_agent,
                "mfa_verified": mfa_verified
            }
            
            self.redis_client.setex(
                f"session:{session_id}",
                self.config.session_timeout_minutes * 60,
                json.dumps(session_data)
            )
        
        # Add to user's active sessions
        user.active_sessions.append(session_id)
        
        # Limit concurrent sessions
        if len(user.active_sessions) > self.config.max_concurrent_sessions:
            oldest_session = user.active_sessions.pop(0)
            self._invalidate_session(oldest_session)
        
        return session
    
    def validate_session(self, session_id: str) -> Optional[AuthSession]:
        """Validate and refresh session"""
        if not self.redis_client:
            return None
        
        session_data = self.redis_client.get(f"session:{session_id}")
        if not session_data:
            return None
        
        try:
            data = json.loads(session_data)
            session = AuthSession(
                session_id=session_id,
                user_id=data["user_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                last_activity=datetime.fromisoformat(data["last_activity"]),
                ip_address=data["ip_address"],
                user_agent=data["user_agent"],
                mfa_verified=data["mfa_verified"]
            )
            
            # Update last activity
            session.last_activity = datetime.utcnow()
            data["last_activity"] = session.last_activity.isoformat()
            
            # Refresh session in Redis
            self.redis_client.setex(
                f"session:{session_id}",
                self.config.session_timeout_minutes * 60,
                json.dumps(data)
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to validate session: {e}")
            return None
    
    def _invalidate_session(self, session_id: str):
        """Invalidate session"""
        if self.redis_client:
            self.redis_client.delete(f"session:{session_id}")
    
    def create_access_token(self, user: User, session: AuthSession) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.config.access_token_expire_minutes)
        
        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "session_id": session.session_id,
            "mfa_verified": session.mfa_verified,
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def create_refresh_token(self, user: User, session: AuthSession) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.config.refresh_token_expire_days)
        
        payload = {
            "sub": user.id,
            "session_id": session.session_id,
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            
            # Verify session is still valid
            if "session_id" in payload:
                session = self.validate_session(payload["session_id"])
                if not session:
                    return None
            
            return payload
            
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def check_permission(self, user: User, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        return required_permission in user.permissions
    
    def check_role(self, user: User, required_roles: List[UserRole]) -> bool:
        """Check if user has any of the required roles"""
        return any(role in user.roles for role in required_roles)
    
    def _get_permissions_for_roles(self, roles: List[UserRole]) -> List[Permission]:
        """Get all permissions for given roles"""
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, []))
        return list(permissions)
    
    def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username (placeholder - implement with your user store)"""
        # In production, implement with your user database
        return None
    
    def change_password(self, user: User, old_password: str, new_password: str) -> Tuple[bool, List[str]]:
        """Change user password with validation"""
        # Verify old password
        if not self.pwd_context.verify(old_password, user.password_hash):
            return False, ["Current password is incorrect"]
        
        # Validate new password
        is_valid, errors = self.password_validator.validate_password(new_password, user.username)
        if not is_valid:
            return False, errors
        
        # Check password history
        new_hash = self.pwd_context.hash(new_password)
        for old_hash in user.password_history:
            if self.pwd_context.verify(new_password, old_hash):
                return False, ["Cannot reuse recent passwords"]
        
        # Update password
        user.password_hash = new_hash
        user.password_history.append(new_hash)
        user.password_changed_at = datetime.utcnow()
        
        # Keep only recent passwords
        if len(user.password_history) > self.config.password_history_count:
            user.password_history = user.password_history[-self.config.password_history_count:]
        
        # Invalidate all sessions (force re-login)
        for session_id in user.active_sessions:
            self._invalidate_session(session_id)
        user.active_sessions.clear()
        
        return True, []
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

# Global auth manager instance
auth_manager = None

def initialize_auth_manager(config: SecurityConfig = None) -> AdvancedAuthManager:
    """Initialize global auth manager"""
    global auth_manager
    auth_manager = AdvancedAuthManager(config)
    return auth_manager

def get_auth_manager() -> AdvancedAuthManager:
    """Get global auth manager instance"""
    return auth_manager
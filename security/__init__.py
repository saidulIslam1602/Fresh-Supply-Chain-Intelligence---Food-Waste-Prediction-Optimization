"""
Enterprise Security Package for Fresh Supply Chain Intelligence System
Comprehensive security features including authentication, authorization, audit logging, and threat detection
"""

from .advanced_auth import (
    AdvancedAuthManager, SecurityConfig, User, AuthSession,
    AuthMethod, UserRole, Permission, PasswordValidator, MFAManager,
    initialize_auth_manager, get_auth_manager
)

from .audit_logger import (
    AdvancedAuditLogger, AuditConfig, AuditEvent, AuditEventType, AuditSeverity,
    ComplianceStandard, ComplianceReporter,
    initialize_audit_logger, get_audit_logger,
    log_auth_event, log_data_event, log_security_event
)

from .threat_detector import (
    AdvancedThreatDetector, ThreatEvent, ThreatLevel, ThreatType,
    DetectionRule, UserBehaviorProfile,
    initialize_threat_detector, get_threat_detector
)

__version__ = "2.0.0"
__author__ = "Fresh Supply Chain Intelligence Security Team"

# Security features overview
SECURITY_FEATURES = {
    "advanced_authentication": {
        "description": "Multi-factor authentication with TOTP, backup codes, and session management",
        "features": [
            "Password complexity validation",
            "Multi-factor authentication (TOTP)",
            "Session management with Redis",
            "Account lockout protection",
            "Password history tracking",
            "Secure password hashing (bcrypt)"
        ],
        "compliance": ["GDPR", "SOX", "ISO 27001"]
    },
    
    "fine_grained_authorization": {
        "description": "Role-based access control with fine-grained permissions",
        "features": [
            "Hierarchical role system",
            "Fine-grained permissions",
            "Resource-based access control",
            "Dynamic permission checking",
            "Role inheritance",
            "Permission auditing"
        ],
        "roles": ["Super Admin", "Admin", "Manager", "Analyst", "Viewer", "API User", "Guest"]
    },
    
    "comprehensive_audit_logging": {
        "description": "Complete audit trail for compliance and forensics",
        "features": [
            "Tamper-evident logging",
            "Real-time audit events",
            "Compliance reporting (GDPR, HIPAA, SOX)",
            "Event integrity verification",
            "Automated log rotation",
            "Structured log format (JSON)"
        ],
        "retention": "7 years default",
        "compliance": ["GDPR", "HIPAA", "SOX", "PCI DSS", "ISO 27001"]
    },
    
    "advanced_threat_detection": {
        "description": "AI-powered threat detection and response",
        "features": [
            "Real-time threat analysis",
            "Machine learning anomaly detection",
            "Pattern-based attack detection",
            "Geolocation analysis",
            "Rate limiting and DDoS protection",
            "Behavioral analysis",
            "Automated threat response"
        ],
        "detection_types": [
            "SQL Injection", "XSS", "CSRF", "DDoS", "Brute Force",
            "Credential Stuffing", "Anomalous Behavior", "Data Exfiltration"
        ]
    }
}

# Security metrics and KPIs
SECURITY_METRICS = {
    "authentication": [
        "Login success rate",
        "MFA adoption rate",
        "Account lockout incidents",
        "Password policy compliance",
        "Session security metrics"
    ],
    
    "authorization": [
        "Access denied events",
        "Permission escalation attempts",
        "Role assignment changes",
        "Unauthorized access attempts"
    ],
    
    "audit_compliance": [
        "Audit log completeness",
        "Compliance report generation time",
        "Data retention compliance",
        "Log integrity verification rate"
    ],
    
    "threat_detection": [
        "Threats detected per day",
        "False positive rate",
        "Response time to threats",
        "Blocked attack attempts",
        "Anomaly detection accuracy"
    ]
}

def get_security_summary() -> dict:
    """Get comprehensive security system summary"""
    return {
        "version": __version__,
        "features": SECURITY_FEATURES,
        "metrics": SECURITY_METRICS,
        "compliance_standards": [
            "GDPR - General Data Protection Regulation",
            "HIPAA - Health Insurance Portability and Accountability Act",
            "SOX - Sarbanes-Oxley Act",
            "PCI DSS - Payment Card Industry Data Security Standard",
            "ISO 27001 - Information Security Management",
            "NIST - National Institute of Standards and Technology"
        ],
        "security_levels": {
            "authentication": "Enterprise-grade with MFA",
            "authorization": "Fine-grained RBAC",
            "audit_logging": "Comprehensive with 7-year retention",
            "threat_detection": "AI-powered real-time protection"
        }
    }

def initialize_security_system(
    auth_config: SecurityConfig = None,
    audit_config: AuditConfig = None,
    enable_threat_detection: bool = True
) -> dict:
    """Initialize complete security system"""
    
    components = {}
    
    # Initialize authentication system
    auth_manager = initialize_auth_manager(auth_config)
    components["auth_manager"] = auth_manager
    
    # Initialize audit logging
    audit_logger = initialize_audit_logger(audit_config)
    components["audit_logger"] = audit_logger
    
    # Initialize threat detection
    if enable_threat_detection:
        threat_detector = initialize_threat_detector()
        components["threat_detector"] = threat_detector
    
    # Start services
    if audit_logger:
        audit_logger.start()
    
    if enable_threat_detection and threat_detector:
        threat_detector.start()
    
    return components

# Security best practices
SECURITY_BEST_PRACTICES = {
    "password_policy": {
        "minimum_length": 12,
        "require_complexity": True,
        "password_history": 5,
        "max_age_days": 90,
        "lockout_threshold": 5,
        "lockout_duration_minutes": 30
    },
    
    "session_management": {
        "session_timeout_minutes": 60,
        "max_concurrent_sessions": 3,
        "secure_cookies": True,
        "session_rotation": True
    },
    
    "mfa_requirements": {
        "required_for_admin": True,
        "required_for_privileged": True,
        "backup_codes": 10,
        "totp_window": 30
    },
    
    "audit_requirements": {
        "log_all_access": True,
        "log_all_changes": True,
        "tamper_protection": True,
        "real_time_monitoring": True,
        "compliance_reporting": True
    },
    
    "threat_detection": {
        "real_time_analysis": True,
        "ml_anomaly_detection": True,
        "geolocation_monitoring": True,
        "rate_limiting": True,
        "automated_response": True
    }
}

def validate_security_configuration(config: dict) -> dict:
    """Validate security configuration against best practices"""
    validation_results = {
        "compliant": True,
        "warnings": [],
        "recommendations": []
    }
    
    # Validate password policy
    password_config = config.get("password_policy", {})
    if password_config.get("minimum_length", 0) < 12:
        validation_results["warnings"].append("Password minimum length should be at least 12 characters")
        validation_results["compliant"] = False
    
    # Validate MFA configuration
    mfa_config = config.get("mfa_policy", {})
    if not mfa_config.get("required_for_admin", False):
        validation_results["warnings"].append("MFA should be required for admin accounts")
        validation_results["compliant"] = False
    
    # Validate audit configuration
    audit_config = config.get("audit_policy", {})
    if not audit_config.get("tamper_protection", False):
        validation_results["recommendations"].append("Enable audit log tamper protection")
    
    return validation_results

__all__ = [
    # Authentication & Authorization
    "AdvancedAuthManager", "SecurityConfig", "User", "AuthSession",
    "AuthMethod", "UserRole", "Permission", "PasswordValidator", "MFAManager",
    "initialize_auth_manager", "get_auth_manager",
    
    # Audit Logging
    "AdvancedAuditLogger", "AuditConfig", "AuditEvent", "AuditEventType", "AuditSeverity",
    "ComplianceStandard", "ComplianceReporter",
    "initialize_audit_logger", "get_audit_logger",
    "log_auth_event", "log_data_event", "log_security_event",
    
    # Threat Detection
    "AdvancedThreatDetector", "ThreatEvent", "ThreatLevel", "ThreatType",
    "DetectionRule", "UserBehaviorProfile",
    "initialize_threat_detector", "get_threat_detector",
    
    # System Management
    "get_security_summary", "initialize_security_system",
    "validate_security_configuration",
    
    # Constants
    "SECURITY_FEATURES", "SECURITY_METRICS", "SECURITY_BEST_PRACTICES"
]
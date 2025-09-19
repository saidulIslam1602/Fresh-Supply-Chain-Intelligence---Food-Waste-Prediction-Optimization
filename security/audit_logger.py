"""
Advanced Audit Logging System for Fresh Supply Chain Intelligence
Comprehensive audit trails for compliance, security monitoring, and forensics
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
from collections import deque
import asyncio
import aiofiles
import structlog
from pathlib import Path

logger = structlog.get_logger()

class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    MFA_SETUP = "auth.mfa.setup"
    MFA_DISABLE = "auth.mfa.disable"
    ACCOUNT_LOCKED = "auth.account.locked"
    ACCOUNT_UNLOCKED = "auth.account.unlocked"
    
    # Authorization events
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_GRANTED = "authz.permission.granted"
    PERMISSION_REVOKED = "authz.permission.revoked"
    ROLE_ASSIGNED = "authz.role.assigned"
    ROLE_REMOVED = "authz.role.removed"
    
    # Data events
    DATA_READ = "data.read"
    DATA_CREATE = "data.create"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGE = "system.config.change"
    BACKUP_CREATE = "system.backup.create"
    BACKUP_RESTORE = "system.backup.restore"
    
    # API events
    API_REQUEST = "api.request"
    API_ERROR = "api.error"
    API_RATE_LIMIT = "api.rate_limit"
    
    # ML model events
    MODEL_TRAIN = "ml.model.train"
    MODEL_DEPLOY = "ml.model.deploy"
    MODEL_PREDICT = "ml.model.predict"
    MODEL_DELETE = "ml.model.delete"
    
    # Security events
    SECURITY_VIOLATION = "security.violation"
    INTRUSION_ATTEMPT = "security.intrusion"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    
    # Compliance events
    GDPR_REQUEST = "compliance.gdpr.request"
    DATA_RETENTION = "compliance.data.retention"
    PRIVACY_BREACH = "compliance.privacy.breach"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStandard(Enum):
    """Compliance standards"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"

@dataclass
class AuditEvent:
    """Comprehensive audit event"""
    # Core event information
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    
    # Actor information
    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    
    # Context information
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    
    # Event details
    action: Optional[str] = None
    outcome: str = "success"  # success, failure, error
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Technical information
    application: str = "fresh-supply-chain"
    component: Optional[str] = None
    version: Optional[str] = None
    
    # Compliance and risk
    compliance_tags: List[ComplianceStandard] = field(default_factory=list)
    risk_score: int = 0  # 0-100
    
    # Integrity
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum for integrity"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum for event integrity"""
        # Create a copy without checksum for hashing
        event_dict = asdict(self)
        event_dict.pop('checksum', None)
        
        # Convert to JSON string (sorted for consistency)
        event_json = json.dumps(event_dict, sort_keys=True, default=str)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(event_json.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity"""
        current_checksum = self.checksum
        self.checksum = None
        calculated_checksum = self._calculate_checksum()
        self.checksum = current_checksum
        
        return current_checksum == calculated_checksum

@dataclass
class AuditConfig:
    """Audit logging configuration"""
    # Storage settings
    log_directory: str = "audit_logs"
    max_file_size_mb: int = 100
    max_files: int = 1000
    compression_enabled: bool = True
    
    # Retention settings
    retention_days: int = 2555  # 7 years for compliance
    archive_after_days: int = 365
    
    # Performance settings
    batch_size: int = 100
    flush_interval_seconds: int = 30
    async_logging: bool = True
    
    # Security settings
    encrypt_logs: bool = True
    sign_logs: bool = True
    tamper_detection: bool = True
    
    # Compliance settings
    gdpr_compliance: bool = True
    anonymize_pii: bool = True
    
    # Filtering
    min_severity: AuditSeverity = AuditSeverity.LOW
    excluded_event_types: List[AuditEventType] = field(default_factory=list)
    
    # External integration
    siem_integration: bool = False
    siem_endpoint: Optional[str] = None
    webhook_url: Optional[str] = None

class AuditBuffer:
    """Thread-safe audit event buffer"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.overflow_count = 0
    
    def add(self, event: AuditEvent):
        """Add event to buffer"""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.overflow_count += 1
                logger.warning(f"Audit buffer overflow: {self.overflow_count} events lost")
            
            self.buffer.append(event)
    
    def get_batch(self, batch_size: int) -> List[AuditEvent]:
        """Get batch of events from buffer"""
        with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
            return batch
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

class AuditStorage:
    """Audit log storage manager"""
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.log_directory = Path(config.log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.current_file = None
        self.current_file_size = 0
        self.file_count = 0
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage system"""
        # Create new log file
        self._rotate_file()
        
        # Start cleanup task
        if self.config.retention_days > 0:
            self._cleanup_old_files()
    
    def _rotate_file(self):
        """Rotate to new log file"""
        if self.current_file:
            self.current_file.close()
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{timestamp}_{self.file_count:04d}.jsonl"
        filepath = self.log_directory / filename
        
        self.current_file = open(filepath, 'w', encoding='utf-8')
        self.current_file_size = 0
        self.file_count += 1
        
        logger.info(f"Created new audit log file: {filepath}")
    
    def write_events(self, events: List[AuditEvent]):
        """Write events to storage"""
        if not events:
            return
        
        for event in events:
            # Convert to JSON
            event_json = json.dumps(asdict(event), default=str)
            
            # Write to file
            self.current_file.write(event_json + '\n')
            self.current_file_size += len(event_json) + 1
            
            # Check if file rotation is needed
            if self.current_file_size > self.config.max_file_size_mb * 1024 * 1024:
                self._rotate_file()
        
        # Flush to disk
        self.current_file.flush()
    
    async def write_events_async(self, events: List[AuditEvent]):
        """Write events asynchronously"""
        if not events:
            return
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{timestamp}_{int(time.time())}.jsonl"
        filepath = self.log_directory / filename
        
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            for event in events:
                event_json = json.dumps(asdict(event), default=str)
                await f.write(event_json + '\n')
    
    def _cleanup_old_files(self):
        """Clean up old audit files"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
        
        for file_path in self.log_directory.glob("audit_*.jsonl*"):
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    logger.info(f"Deleted old audit file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete old audit file {file_path}: {e}")

class ComplianceReporter:
    """Generate compliance reports from audit logs"""
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.log_directory = Path(config.log_directory)
    
    def generate_gdpr_report(self, user_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate GDPR compliance report for a user"""
        events = self._read_events_for_user(user_id, start_date, end_date)
        
        # Categorize events
        data_access_events = [e for e in events if e.event_type == AuditEventType.DATA_READ]
        data_modification_events = [e for e in events if e.event_type in [
            AuditEventType.DATA_CREATE, AuditEventType.DATA_UPDATE, AuditEventType.DATA_DELETE
        ]]
        export_events = [e for e in events if e.event_type == AuditEventType.DATA_EXPORT]
        
        return {
            "user_id": user_id,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_events": len(events),
                "data_access_count": len(data_access_events),
                "data_modification_count": len(data_modification_events),
                "data_export_count": len(export_events)
            },
            "data_access_log": [self._event_to_dict(e) for e in data_access_events],
            "data_modifications": [self._event_to_dict(e) for e in data_modification_events],
            "data_exports": [self._event_to_dict(e) for e in export_events],
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def generate_security_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate security incident report"""
        events = self._read_events_by_date_range(start_date, end_date)
        
        # Filter security events
        security_events = [e for e in events if e.event_type in [
            AuditEventType.LOGIN_FAILURE, AuditEventType.ACCESS_DENIED,
            AuditEventType.SECURITY_VIOLATION, AuditEventType.INTRUSION_ATTEMPT,
            AuditEventType.SUSPICIOUS_ACTIVITY
        ]]
        
        # Group by severity
        by_severity = {}
        for severity in AuditSeverity:
            by_severity[severity.value] = [e for e in security_events if e.severity == severity]
        
        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_security_events": len(security_events),
                "by_severity": {k: len(v) for k, v in by_severity.items()},
                "unique_users_affected": len(set(e.user_id for e in security_events if e.user_id)),
                "unique_ip_addresses": len(set(e.ip_address for e in security_events if e.ip_address))
            },
            "events_by_severity": {k: [self._event_to_dict(e) for e in v] for k, v in by_severity.items()},
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _read_events_for_user(self, user_id: str, start_date: datetime, end_date: datetime) -> List[AuditEvent]:
        """Read audit events for specific user"""
        events = []
        
        for file_path in self.log_directory.glob("audit_*.jsonl"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        event_data = json.loads(line.strip())
                        event = AuditEvent(**event_data)
                        
                        if (event.user_id == user_id and 
                            start_date <= event.timestamp <= end_date):
                            events.append(event)
            except Exception as e:
                logger.error(f"Failed to read audit file {file_path}: {e}")
        
        return sorted(events, key=lambda x: x.timestamp)
    
    def _read_events_by_date_range(self, start_date: datetime, end_date: datetime) -> List[AuditEvent]:
        """Read audit events within date range"""
        events = []
        
        for file_path in self.log_directory.glob("audit_*.jsonl"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        event_data = json.loads(line.strip())
                        event = AuditEvent(**event_data)
                        
                        if start_date <= event.timestamp <= end_date:
                            events.append(event)
            except Exception as e:
                logger.error(f"Failed to read audit file {file_path}: {e}")
        
        return sorted(events, key=lambda x: x.timestamp)
    
    def _event_to_dict(self, event: AuditEvent) -> Dict[str, Any]:
        """Convert event to dictionary for reporting"""
        return {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "action": event.action,
            "outcome": event.outcome,
            "resource": {
                "type": event.resource_type,
                "id": event.resource_id,
                "name": event.resource_name
            },
            "details": event.details
        }

class AdvancedAuditLogger:
    """Advanced audit logging system"""
    
    def __init__(self, config: AuditConfig = None):
        self.config = config or AuditConfig()
        self.buffer = AuditBuffer()
        self.storage = AuditStorage(self.config)
        self.compliance_reporter = ComplianceReporter(self.config)
        
        self.is_running = False
        self.flush_task = None
        
        # Statistics
        self.stats = {
            "events_logged": 0,
            "events_dropped": 0,
            "last_flush": None,
            "buffer_overflows": 0
        }
    
    def start(self):
        """Start audit logging system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        if self.config.async_logging:
            self.flush_task = asyncio.create_task(self._flush_loop())
        
        logger.info("Audit logging system started")
    
    async def stop(self):
        """Stop audit logging system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining events
        await self._flush_events()
        
        logger.info("Audit logging system stopped")
    
    def log_event(self, event_type: AuditEventType, **kwargs):
        """Log audit event"""
        # Check if event type is excluded
        if event_type in self.config.excluded_event_types:
            return
        
        # Determine severity
        severity = kwargs.get('severity', self._determine_severity(event_type))
        
        # Check minimum severity
        if severity.value < self.config.min_severity.value:
            return
        
        # Create event
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            severity=severity,
            **kwargs
        )
        
        # Add to buffer
        self.buffer.add(event)
        self.stats["events_logged"] += 1
        
        # Immediate flush for critical events
        if severity == AuditSeverity.CRITICAL:
            asyncio.create_task(self._flush_events())
    
    def log_authentication_event(self, event_type: AuditEventType, user_id: str = None, 
                                username: str = None, ip_address: str = None, 
                                outcome: str = "success", **kwargs):
        """Log authentication-related event"""
        self.log_event(
            event_type=event_type,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            outcome=outcome,
            component="authentication",
            **kwargs
        )
    
    def log_data_access_event(self, event_type: AuditEventType, user_id: str, 
                            resource_type: str, resource_id: str, 
                            action: str, **kwargs):
        """Log data access event"""
        self.log_event(
            event_type=event_type,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            component="data_access",
            compliance_tags=[ComplianceStandard.GDPR],
            **kwargs
        )
    
    def log_security_event(self, event_type: AuditEventType, severity: AuditSeverity,
                          ip_address: str = None, user_id: str = None, 
                          details: Dict[str, Any] = None, **kwargs):
        """Log security event"""
        self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
            component="security",
            risk_score=self._calculate_risk_score(event_type, severity),
            **kwargs
        )
    
    async def _flush_loop(self):
        """Periodic flush loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audit flush loop: {e}")
    
    async def _flush_events(self):
        """Flush events from buffer to storage"""
        events = self.buffer.get_batch(self.config.batch_size)
        
        if events:
            try:
                if self.config.async_logging:
                    await self.storage.write_events_async(events)
                else:
                    self.storage.write_events(events)
                
                self.stats["last_flush"] = datetime.utcnow()
                logger.debug(f"Flushed {len(events)} audit events")
                
            except Exception as e:
                logger.error(f"Failed to flush audit events: {e}")
                self.stats["events_dropped"] += len(events)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _determine_severity(self, event_type: AuditEventType) -> AuditSeverity:
        """Determine event severity based on type"""
        high_severity_events = {
            AuditEventType.LOGIN_FAILURE,
            AuditEventType.ACCESS_DENIED,
            AuditEventType.ACCOUNT_LOCKED,
            AuditEventType.DATA_DELETE,
            AuditEventType.PERMISSION_REVOKED,
            AuditEventType.CONFIG_CHANGE
        }
        
        critical_severity_events = {
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.INTRUSION_ATTEMPT,
            AuditEventType.PRIVACY_BREACH,
            AuditEventType.SYSTEM_STOP
        }
        
        if event_type in critical_severity_events:
            return AuditSeverity.CRITICAL
        elif event_type in high_severity_events:
            return AuditSeverity.HIGH
        else:
            return AuditSeverity.MEDIUM
    
    def _calculate_risk_score(self, event_type: AuditEventType, severity: AuditSeverity) -> int:
        """Calculate risk score for security events"""
        base_scores = {
            AuditSeverity.LOW: 10,
            AuditSeverity.MEDIUM: 30,
            AuditSeverity.HIGH: 60,
            AuditSeverity.CRITICAL: 90
        }
        
        event_multipliers = {
            AuditEventType.INTRUSION_ATTEMPT: 1.5,
            AuditEventType.SECURITY_VIOLATION: 1.3,
            AuditEventType.PRIVACY_BREACH: 1.4,
            AuditEventType.LOGIN_FAILURE: 1.1
        }
        
        base_score = base_scores.get(severity, 30)
        multiplier = event_multipliers.get(event_type, 1.0)
        
        return min(int(base_score * multiplier), 100)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        return {
            **self.stats,
            "buffer_size": self.buffer.size(),
            "buffer_overflows": self.buffer.overflow_count,
            "is_running": self.is_running
        }

# Global audit logger instance
audit_logger = None

def initialize_audit_logger(config: AuditConfig = None) -> AdvancedAuditLogger:
    """Initialize global audit logger"""
    global audit_logger
    audit_logger = AdvancedAuditLogger(config)
    return audit_logger

def get_audit_logger() -> AdvancedAuditLogger:
    """Get global audit logger instance"""
    return audit_logger

# Convenience functions
def log_auth_event(event_type: AuditEventType, **kwargs):
    """Log authentication event"""
    if audit_logger:
        audit_logger.log_authentication_event(event_type, **kwargs)

def log_data_event(event_type: AuditEventType, **kwargs):
    """Log data access event"""
    if audit_logger:
        audit_logger.log_data_access_event(event_type, **kwargs)

def log_security_event(event_type: AuditEventType, severity: AuditSeverity, **kwargs):
    """Log security event"""
    if audit_logger:
        audit_logger.log_security_event(event_type, severity, **kwargs)
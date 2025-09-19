"""
Advanced Error Handling and Data Recovery System for Fresh Supply Chain Intelligence System
Provides robust error handling, automatic recovery, and system resilience
"""

import logging
import traceback
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import redis
import pickle
import os
from pathlib import Path
import threading
from queue import Queue
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories"""
    DATA_QUALITY = "data_quality"
    SYSTEM_FAILURE = "system_failure"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    AUTHENTICATION_ERROR = "authentication_error"
    CONFIGURATION_ERROR = "configuration_error"

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    MANUAL_INTERVENTION = "manual_intervention"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class ErrorEvent:
    """Represents an error event"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any]
    stack_trace: str
    component: str
    user: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
            'message': self.message,
            'details': self.details,
            'stack_trace': self.stack_trace,
            'component': self.component,
            'user': self.user,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_strategy': self.recovery_strategy.value if self.recovery_strategy else None
        }

@dataclass
class RecoveryAction:
    """Represents a recovery action"""
    action_id: str
    strategy: RecoveryStrategy
    description: str
    handler: Callable
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    conditions: Dict[str, Any] = field(default_factory=dict)

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class AdvancedErrorHandler:
    """Advanced error handling and recovery system"""
    
    def __init__(self, 
                 connection_string: str = None,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 backup_path: str = "./data/backups"):
        
        self.connection_string = connection_string
        self.engine = create_engine(connection_string) if connection_string else None
        
        # Redis for caching and coordination
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
        except:
            logger.warning("Redis not available, using local error handling only")
            self.redis_client = None
        
        # Error storage and recovery
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        self.error_events: List[ErrorEvent] = []
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error handling configuration
        self.max_error_history = 10000
        self.error_notification_threshold = ErrorSeverity.HIGH
        self.auto_recovery_enabled = True
        
        # Notification settings
        self.notification_handlers: List[Callable] = []
        self.email_config = {}
        
        # Statistics
        self.stats = {
            'total_errors': 0,
            'errors_by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'errors_by_category': {category.value: 0 for category in ErrorCategory},
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'start_time': datetime.now()
        }
        
        # Setup default recovery actions
        self._setup_default_recovery_actions()
        
        # Background error processing
        self.error_queue = Queue()
        self.processing_thread = None
        self.is_running = False
    
    def start_error_processing(self):
        """Start background error processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_error_queue, daemon=True)
        self.processing_thread.start()
        logger.info("Started error processing thread")
    
    def stop_error_processing(self):
        """Stop background error processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Stopped error processing thread")
    
    def handle_error(self, 
                    error: Exception,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.PROCESSING_ERROR,
                    component: str = "unknown",
                    user: str = None,
                    context: Dict[str, Any] = None,
                    auto_recover: bool = True) -> ErrorEvent:
        """Main error handling entry point"""
        
        # Create error event
        error_event = ErrorEvent(
            error_id=f"err_{int(time.time() * 1000)}_{len(self.error_events)}",
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=str(error),
            details=context or {},
            stack_trace=traceback.format_exc(),
            component=component,
            user=user
        )
        
        # Add to processing queue
        self.error_queue.put((error_event, auto_recover))
        
        # Update statistics
        self.stats['total_errors'] += 1
        self.stats['errors_by_severity'][severity.value] += 1
        self.stats['errors_by_category'][category.value] += 1
        
        logger.error(f"Error handled: {error_event.error_id} - {error_event.message}")
        
        return error_event
    
    def register_recovery_action(self, 
                                error_category: ErrorCategory,
                                strategy: RecoveryStrategy,
                                handler: Callable,
                                description: str = "",
                                max_retries: int = 3,
                                retry_delay: float = 1.0,
                                conditions: Dict[str, Any] = None) -> str:
        """Register a recovery action for specific error types"""
        
        action_id = f"{error_category.value}_{strategy.value}_{len(self.recovery_actions)}"
        
        recovery_action = RecoveryAction(
            action_id=action_id,
            strategy=strategy,
            description=description,
            handler=handler,
            max_retries=max_retries,
            retry_delay=retry_delay,
            conditions=conditions or {}
        )
        
        self.recovery_actions[action_id] = recovery_action
        logger.info(f"Registered recovery action: {action_id}")
        
        return action_id
    
    def add_notification_handler(self, handler: Callable):
        """Add notification handler for error events"""
        self.notification_handlers.append(handler)
    
    def configure_email_notifications(self, 
                                    smtp_server: str,
                                    smtp_port: int,
                                    username: str,
                                    password: str,
                                    recipients: List[str]):
        """Configure email notifications for critical errors"""
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'recipients': recipients
        }
    
    def create_data_backup(self, 
                          data: Union[pd.DataFrame, Dict, Any],
                          backup_name: str,
                          metadata: Dict[str, Any] = None) -> str:
        """Create backup of critical data"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_name}_{timestamp}.pkl"
        backup_filepath = self.backup_path / backup_filename
        
        backup_data = {
            'data': data,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'backup_name': backup_name
        }
        
        try:
            with open(backup_filepath, 'wb') as f:
                pickle.dump(backup_data, f)
            
            logger.info(f"Created data backup: {backup_filepath}")
            return str(backup_filepath)
            
        except Exception as e:
            logger.error(f"Failed to create backup {backup_name}: {e}")
            raise
    
    def restore_data_backup(self, backup_filepath: str) -> Any:
        """Restore data from backup"""
        
        try:
            with open(backup_filepath, 'rb') as f:
                backup_data = pickle.load(f)
            
            logger.info(f"Restored data backup: {backup_filepath}")
            return backup_data['data']
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_filepath}: {e}")
            raise
    
    def get_circuit_breaker(self, component: str, 
                           failure_threshold: int = 5,
                           recovery_timeout: int = 60) -> CircuitBreaker:
        """Get or create circuit breaker for component"""
        
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        
        return self.circuit_breakers[component]
    
    def execute_with_circuit_breaker(self, 
                                   component: str,
                                   func: Callable,
                                   *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        circuit_breaker = self.get_circuit_breaker(component)
        
        try:
            return circuit_breaker.call(func, *args, **kwargs)
        except Exception as e:
            self.handle_error(
                error=e,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SYSTEM_FAILURE,
                component=component,
                context={'circuit_breaker_state': circuit_breaker.state}
            )
            raise
    
    def retry_with_backoff(self, 
                          func: Callable,
                          max_retries: int = 3,
                          base_delay: float = 1.0,
                          max_delay: float = 60.0,
                          backoff_factor: float = 2.0,
                          exceptions: tuple = (Exception,)) -> Any:
        """Retry function with exponential backoff"""
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        # All retries failed
        self.handle_error(
            error=last_exception,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PROCESSING_ERROR,
            context={
                'max_retries': max_retries,
                'attempts_made': max_retries + 1,
                'function': func.__name__ if hasattr(func, '__name__') else str(func)
            }
        )
        
        raise last_exception
    
    def validate_data_integrity(self, 
                              df: pd.DataFrame,
                              expected_columns: List[str] = None,
                              min_rows: int = 0,
                              max_null_percentage: float = 0.5) -> bool:
        """Validate data integrity and handle errors"""
        
        try:
            # Check if DataFrame is empty
            if df.empty and min_rows > 0:
                raise ValueError(f"DataFrame is empty, expected at least {min_rows} rows")
            
            # Check minimum rows
            if len(df) < min_rows:
                raise ValueError(f"DataFrame has {len(df)} rows, expected at least {min_rows}")
            
            # Check expected columns
            if expected_columns:
                missing_columns = set(expected_columns) - set(df.columns)
                if missing_columns:
                    raise ValueError(f"Missing expected columns: {missing_columns}")
            
            # Check null percentage
            for column in df.columns:
                null_percentage = df[column].isnull().sum() / len(df)
                if null_percentage > max_null_percentage:
                    raise ValueError(f"Column {column} has {null_percentage:.2%} null values, "
                                   f"exceeds threshold of {max_null_percentage:.2%}")
            
            return True
            
        except Exception as e:
            self.handle_error(
                error=e,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.DATA_QUALITY,
                component="data_validation",
                context={
                    'dataframe_shape': df.shape,
                    'expected_columns': expected_columns,
                    'min_rows': min_rows,
                    'max_null_percentage': max_null_percentage
                }
            )
            return False
    
    def handle_database_error(self, error: Exception, query: str = None) -> bool:
        """Handle database-specific errors with recovery"""
        
        error_event = self.handle_error(
            error=error,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATABASE_ERROR,
            component="database",
            context={'query': query}
        )
        
        # Attempt database recovery
        if self.auto_recovery_enabled:
            return self._attempt_database_recovery(error_event)
        
        return False
    
    def handle_network_error(self, error: Exception, endpoint: str = None) -> bool:
        """Handle network-specific errors with recovery"""
        
        error_event = self.handle_error(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK_ERROR,
            component="network",
            context={'endpoint': endpoint}
        )
        
        # Attempt network recovery
        if self.auto_recovery_enabled:
            return self._attempt_network_recovery(error_event)
        
        return False
    
    def generate_error_report(self, 
                            start_date: datetime = None,
                            end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter errors by date range
        filtered_errors = [
            error for error in self.error_events
            if start_date <= error.timestamp <= end_date
        ]
        
        # Calculate statistics
        total_errors = len(filtered_errors)
        errors_by_severity = {}
        errors_by_category = {}
        errors_by_component = {}
        recovery_stats = {'attempted': 0, 'successful': 0}
        
        for error in filtered_errors:
            # Group by severity
            severity = error.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
            
            # Group by category
            category = error.category.value
            errors_by_category[category] = errors_by_category.get(category, 0) + 1
            
            # Group by component
            component = error.component
            errors_by_component[component] = errors_by_component.get(component, 0) + 1
            
            # Recovery statistics
            if error.recovery_attempted:
                recovery_stats['attempted'] += 1
                if error.recovery_successful:
                    recovery_stats['successful'] += 1
        
        # Calculate error trends
        error_trends = self._calculate_error_trends(filtered_errors)
        
        # Top error patterns
        top_error_patterns = self._identify_error_patterns(filtered_errors)
        
        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_errors': total_errors,
                'errors_by_severity': errors_by_severity,
                'errors_by_category': errors_by_category,
                'errors_by_component': errors_by_component,
                'recovery_success_rate': (recovery_stats['successful'] / recovery_stats['attempted'] * 100) 
                                       if recovery_stats['attempted'] > 0 else 0
            },
            'trends': error_trends,
            'top_error_patterns': top_error_patterns,
            'circuit_breaker_status': {
                component: {'state': cb.state, 'failure_count': cb.failure_count}
                for component, cb in self.circuit_breakers.items()
            },
            'recommendations': self._generate_error_recommendations(filtered_errors)
        }
        
        return report
    
    def _process_error_queue(self):
        """Background thread for processing error events"""
        
        while self.is_running:
            try:
                # Get error event from queue
                error_event, auto_recover = self.error_queue.get(timeout=1)
                
                # Store error event
                self.error_events.append(error_event)
                
                # Limit error history
                if len(self.error_events) > self.max_error_history:
                    self.error_events = self.error_events[-self.max_error_history:]
                
                # Persist error to database
                if self.engine:
                    self._persist_error_event(error_event)
                
                # Cache in Redis
                if self.redis_client:
                    self._cache_error_event(error_event)
                
                # Attempt recovery if enabled
                if auto_recover and self.auto_recovery_enabled:
                    self._attempt_error_recovery(error_event)
                
                # Send notifications if severity is high enough
                if error_event.severity.value in [ErrorSeverity.HIGH.value, ErrorSeverity.CRITICAL.value]:
                    self._send_error_notifications(error_event)
                
            except Exception as e:
                if self.is_running:  # Only log if we're supposed to be running
                    logger.error(f"Error in error processing thread: {e}")
                time.sleep(1)
    
    def _setup_default_recovery_actions(self):
        """Setup default recovery actions for common error types"""
        
        # Database connection recovery
        self.register_recovery_action(
            error_category=ErrorCategory.DATABASE_ERROR,
            strategy=RecoveryStrategy.RETRY,
            handler=self._recover_database_connection,
            description="Retry database connection with exponential backoff",
            max_retries=3,
            retry_delay=2.0
        )
        
        # Network error recovery
        self.register_recovery_action(
            error_category=ErrorCategory.NETWORK_ERROR,
            strategy=RecoveryStrategy.RETRY,
            handler=self._recover_network_connection,
            description="Retry network request with backoff",
            max_retries=5,
            retry_delay=1.0
        )
        
        # Data quality fallback
        self.register_recovery_action(
            error_category=ErrorCategory.DATA_QUALITY,
            strategy=RecoveryStrategy.FALLBACK,
            handler=self._recover_data_quality,
            description="Use cached or backup data when quality issues occur",
            max_retries=1
        )
        
        # System failure circuit breaker
        self.register_recovery_action(
            error_category=ErrorCategory.SYSTEM_FAILURE,
            strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            handler=self._recover_system_failure,
            description="Activate circuit breaker for system failures",
            max_retries=1
        )
    
    def _attempt_error_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover from an error"""
        
        # Find applicable recovery actions
        applicable_actions = [
            action for action in self.recovery_actions.values()
            if action.strategy.value in [RecoveryStrategy.RETRY.value, RecoveryStrategy.FALLBACK.value]
            and self._matches_recovery_conditions(error_event, action)
        ]
        
        if not applicable_actions:
            logger.info(f"No recovery actions available for error {error_event.error_id}")
            return False
        
        # Try recovery actions in order of preference
        for action in applicable_actions:
            try:
                error_event.recovery_attempted = True
                error_event.recovery_strategy = action.strategy
                
                self.stats['recovery_attempts'] += 1
                
                # Execute recovery action
                success = self._execute_recovery_action(action, error_event)
                
                if success:
                    error_event.recovery_successful = True
                    self.stats['successful_recoveries'] += 1
                    logger.info(f"Successfully recovered from error {error_event.error_id} using {action.strategy.value}")
                    return True
                
            except Exception as recovery_error:
                logger.error(f"Recovery action {action.action_id} failed: {recovery_error}")
        
        logger.warning(f"All recovery attempts failed for error {error_event.error_id}")
        return False
    
    def _execute_recovery_action(self, action: RecoveryAction, error_event: ErrorEvent) -> bool:
        """Execute a specific recovery action"""
        
        for attempt in range(action.max_retries):
            try:
                # Call the recovery handler
                result = action.handler(error_event)
                
                if result:
                    return True
                
                if attempt < action.max_retries - 1:
                    time.sleep(action.retry_delay * (2 ** attempt))  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Recovery action attempt {attempt + 1} failed: {e}")
                
                if attempt < action.max_retries - 1:
                    time.sleep(action.retry_delay * (2 ** attempt))
        
        return False
    
    def _matches_recovery_conditions(self, error_event: ErrorEvent, action: RecoveryAction) -> bool:
        """Check if error event matches recovery action conditions"""
        
        # Check if action applies to this error category
        if action.action_id.startswith(error_event.category.value):
            return True
        
        # Check custom conditions
        for condition_key, condition_value in action.conditions.items():
            if condition_key in error_event.details:
                if error_event.details[condition_key] != condition_value:
                    return False
        
        return True
    
    def _recover_database_connection(self, error_event: ErrorEvent) -> bool:
        """Recover from database connection errors"""
        
        try:
            if self.engine:
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
        
        return False
    
    def _recover_network_connection(self, error_event: ErrorEvent) -> bool:
        """Recover from network connection errors"""
        
        try:
            import requests
            # Test network connectivity
            response = requests.get("https://httpbin.org/status/200", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Network recovery failed: {e}")
        
        return False
    
    def _recover_data_quality(self, error_event: ErrorEvent) -> bool:
        """Recover from data quality issues using backup data"""
        
        try:
            # Look for recent backups
            backup_files = list(self.backup_path.glob("*.pkl"))
            if backup_files:
                # Use most recent backup
                latest_backup = max(backup_files, key=os.path.getctime)
                backup_data = self.restore_data_backup(str(latest_backup))
                logger.info(f"Using backup data for recovery: {latest_backup}")
                return True
        except Exception as e:
            logger.error(f"Data quality recovery failed: {e}")
        
        return False
    
    def _recover_system_failure(self, error_event: ErrorEvent) -> bool:
        """Recover from system failures using circuit breaker"""
        
        component = error_event.component
        circuit_breaker = self.get_circuit_breaker(component)
        
        # Force circuit breaker to open state
        circuit_breaker.state = "OPEN"
        circuit_breaker.last_failure_time = time.time()
        
        logger.info(f"Activated circuit breaker for component {component}")
        return True
    
    def _persist_error_event(self, error_event: ErrorEvent):
        """Persist error event to database"""
        
        try:
            error_data = {
                'error_id': error_event.error_id,
                'timestamp': error_event.timestamp,
                'severity': error_event.severity.value,
                'category': error_event.category.value,
                'message': error_event.message,
                'details': json.dumps(error_event.details),
                'stack_trace': error_event.stack_trace,
                'component': error_event.component,
                'user_name': error_event.user,
                'recovery_attempted': error_event.recovery_attempted,
                'recovery_successful': error_event.recovery_successful,
                'recovery_strategy': error_event.recovery_strategy.value if error_event.recovery_strategy else None
            }
            
            df = pd.DataFrame([error_data])
            df.to_sql('ErrorEvents', self.engine, if_exists='append', index=False)
            
        except Exception as e:
            logger.error(f"Failed to persist error event: {e}")
    
    def _cache_error_event(self, error_event: ErrorEvent):
        """Cache error event in Redis"""
        
        try:
            cache_key = f"error:{error_event.error_id}"
            error_data = json.dumps(error_event.to_dict(), default=str)
            self.redis_client.setex(cache_key, 3600, error_data)  # Cache for 1 hour
            
            # Add to recent errors list
            self.redis_client.lpush("recent_errors", error_event.error_id)
            self.redis_client.ltrim("recent_errors", 0, 99)  # Keep last 100 errors
            
        except Exception as e:
            logger.error(f"Failed to cache error event: {e}")
    
    def _send_error_notifications(self, error_event: ErrorEvent):
        """Send notifications for critical errors"""
        
        # Call custom notification handlers
        for handler in self.notification_handlers:
            try:
                handler(error_event)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
        
        # Send email notifications if configured
        if self.email_config and error_event.severity == ErrorSeverity.CRITICAL:
            self._send_email_notification(error_event)
    
    def _send_email_notification(self, error_event: ErrorEvent):
        """Send email notification for critical errors"""
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"CRITICAL ERROR: {error_event.component} - {error_event.message[:50]}..."
            
            body = f"""
            Critical Error Alert
            
            Error ID: {error_event.error_id}
            Timestamp: {error_event.timestamp}
            Component: {error_event.component}
            Category: {error_event.category.value}
            Severity: {error_event.severity.value}
            
            Message: {error_event.message}
            
            Details: {json.dumps(error_event.details, indent=2)}
            
            Recovery Attempted: {error_event.recovery_attempted}
            Recovery Successful: {error_event.recovery_successful}
            
            Please investigate immediately.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for error {error_event.error_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _calculate_error_trends(self, errors: List[ErrorEvent]) -> Dict[str, Any]:
        """Calculate error trends over time"""
        
        if not errors:
            return {}
        
        # Group errors by hour
        hourly_counts = {}
        for error in errors:
            hour_key = error.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
        
        # Calculate trend
        hours = sorted(hourly_counts.keys())
        counts = [hourly_counts[hour] for hour in hours]
        
        if len(counts) > 1:
            # Simple linear trend calculation
            x = list(range(len(counts)))
            trend_slope = np.polyfit(x, counts, 1)[0] if len(counts) > 1 else 0
        else:
            trend_slope = 0
        
        return {
            'hourly_distribution': hourly_counts,
            'trend_slope': trend_slope,
            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable'
        }
    
    def _identify_error_patterns(self, errors: List[ErrorEvent]) -> List[Dict[str, Any]]:
        """Identify common error patterns"""
        
        patterns = {}
        
        for error in errors:
            # Create pattern key based on category, component, and first few words of message
            message_words = error.message.split()[:3]
            pattern_key = f"{error.category.value}_{error.component}_{' '.join(message_words)}"
            
            if pattern_key not in patterns:
                patterns[pattern_key] = {
                    'pattern': pattern_key,
                    'category': error.category.value,
                    'component': error.component,
                    'sample_message': error.message,
                    'count': 0,
                    'first_seen': error.timestamp,
                    'last_seen': error.timestamp,
                    'severity_distribution': {}
                }
            
            pattern_data = patterns[pattern_key]
            pattern_data['count'] += 1
            pattern_data['last_seen'] = max(pattern_data['last_seen'], error.timestamp)
            
            severity = error.severity.value
            pattern_data['severity_distribution'][severity] = pattern_data['severity_distribution'].get(severity, 0) + 1
        
        # Sort by count and return top patterns
        sorted_patterns = sorted(patterns.values(), key=lambda x: x['count'], reverse=True)
        return sorted_patterns[:10]
    
    def _generate_error_recommendations(self, errors: List[ErrorEvent]) -> List[str]:
        """Generate recommendations based on error patterns"""
        
        recommendations = []
        
        if not errors:
            return recommendations
        
        # Analyze error categories
        category_counts = {}
        for error in errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        # Generate specific recommendations
        if category_counts.get(ErrorCategory.DATABASE_ERROR.value, 0) > 5:
            recommendations.append("Consider implementing database connection pooling and health checks")
        
        if category_counts.get(ErrorCategory.NETWORK_ERROR.value, 0) > 10:
            recommendations.append("Implement more robust network retry mechanisms and circuit breakers")
        
        if category_counts.get(ErrorCategory.DATA_QUALITY.value, 0) > 3:
            recommendations.append("Enhance data validation and implement automated data quality monitoring")
        
        # Check recovery success rate
        recovery_attempted = sum(1 for error in errors if error.recovery_attempted)
        recovery_successful = sum(1 for error in errors if error.recovery_successful)
        
        if recovery_attempted > 0:
            success_rate = recovery_successful / recovery_attempted
            if success_rate < 0.5:
                recommendations.append("Review and improve error recovery strategies")
        
        return recommendations

# Usage example and testing
def setup_error_handling_system(connection_string: str = None) -> AdvancedErrorHandler:
    """Setup comprehensive error handling system"""
    
    error_handler = AdvancedErrorHandler(connection_string)
    
    # Configure email notifications (example)
    # error_handler.configure_email_notifications(
    #     smtp_server="smtp.gmail.com",
    #     smtp_port=587,
    #     username="alerts@company.com",
    #     password="password",
    #     recipients=["admin@company.com", "ops@company.com"]
    # )
    
    # Add custom notification handler
    def log_critical_errors(error_event: ErrorEvent):
        if error_event.severity == ErrorSeverity.CRITICAL:
            print(f"🚨 CRITICAL ERROR: {error_event.message}")
    
    error_handler.add_notification_handler(log_critical_errors)
    
    # Start error processing
    error_handler.start_error_processing()
    
    logger.info("Error handling system setup completed")
    return error_handler

if __name__ == "__main__":
    # Example usage
    error_handler = setup_error_handling_system()
    
    try:
        # Simulate various types of errors
        
        # Database error
        try:
            raise ConnectionError("Database connection failed")
        except Exception as e:
            error_handler.handle_error(
                error=e,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.DATABASE_ERROR,
                component="database_connector"
            )
        
        # Data quality error
        try:
            df = pd.DataFrame({'col1': [1, 2, None, None, None]})
            error_handler.validate_data_integrity(df, min_rows=3, max_null_percentage=0.3)
        except Exception as e:
            pass  # Already handled in validate_data_integrity
        
        # Network error with retry
        def failing_network_call():
            raise requests.exceptions.ConnectionError("Network unreachable")
        
        try:
            error_handler.retry_with_backoff(failing_network_call, max_retries=2)
        except Exception as e:
            pass  # Already handled in retry_with_backoff
        
        # Generate error report
        time.sleep(2)  # Allow processing
        report = error_handler.generate_error_report()
        print("Error Report:")
        print(json.dumps(report, indent=2, default=str))
        
    finally:
        error_handler.stop_error_processing()
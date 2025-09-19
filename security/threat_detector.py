"""
Advanced Threat Detection System for Fresh Supply Chain Intelligence
Real-time threat detection, anomaly detection, and security monitoring
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import defaultdict, deque
import statistics
import ipaddress
import re
import hashlib
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

logger = structlog.get_logger()

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats"""
    BRUTE_FORCE = "brute_force"
    CREDENTIAL_STUFFING = "credential_stuffing"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DDoS = "ddos"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_PAYLOAD = "malicious_payload"
    SUSPICIOUS_LOCATION = "suspicious_location"
    UNUSUAL_TIME_ACCESS = "unusual_time_access"
    RAPID_API_CALLS = "rapid_api_calls"
    LARGE_DATA_DOWNLOAD = "large_data_download"

class DetectionRule(Enum):
    """Detection rule types"""
    RATE_LIMITING = "rate_limiting"
    PATTERN_MATCHING = "pattern_matching"
    ANOMALY_DETECTION = "anomaly_detection"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    GEOLOCATION = "geolocation"
    TIME_BASED = "time_based"

@dataclass
class ThreatEvent:
    """Security threat event"""
    threat_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    timestamp: datetime
    
    # Source information
    source_ip: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Context
    resource: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Detection information
    detection_rule: DetectionRule
    confidence_score: float = 0.0  # 0.0 to 1.0
    
    # Response
    blocked: bool = False
    response_action: Optional[str] = None

@dataclass
class UserBehaviorProfile:
    """User behavior profile for anomaly detection"""
    user_id: str
    
    # Access patterns
    typical_login_times: List[int] = field(default_factory=list)  # Hours of day
    typical_locations: Set[str] = field(default_factory=set)  # IP ranges/countries
    typical_user_agents: Set[str] = field(default_factory=set)
    
    # Activity patterns
    avg_session_duration: float = 0.0
    avg_requests_per_session: float = 0.0
    typical_endpoints: Set[str] = field(default_factory=set)
    
    # Data access patterns
    typical_data_volume: float = 0.0
    typical_export_frequency: float = 0.0
    
    # Update tracking
    last_updated: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0

class RateLimitTracker:
    """Track request rates for rate limiting"""
    
    def __init__(self, window_size: int = 300, max_requests: int = 100):
        self.window_size = window_size  # Time window in seconds
        self.max_requests = max_requests
        self.requests = defaultdict(deque)  # IP -> deque of timestamps
        self.lock = threading.Lock()
    
    def is_rate_limited(self, ip_address: str) -> bool:
        """Check if IP is rate limited"""
        with self.lock:
            now = time.time()
            request_times = self.requests[ip_address]
            
            # Remove old requests outside the window
            while request_times and now - request_times[0] > self.window_size:
                request_times.popleft()
            
            # Check if rate limit exceeded
            if len(request_times) >= self.max_requests:
                return True
            
            # Add current request
            request_times.append(now)
            return False
    
    def get_request_rate(self, ip_address: str) -> float:
        """Get current request rate for IP"""
        with self.lock:
            now = time.time()
            request_times = self.requests[ip_address]
            
            # Count requests in current window
            recent_requests = sum(1 for t in request_times if now - t <= self.window_size)
            return recent_requests / self.window_size

class AnomalyDetector:
    """Machine learning-based anomaly detection"""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'hour_of_day', 'day_of_week', 'session_duration',
            'requests_per_minute', 'data_volume', 'unique_endpoints',
            'failed_requests_ratio', 'geographic_distance'
        ]
    
    def extract_features(self, session_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from session data"""
        features = []
        
        # Time-based features
        timestamp = session_data.get('timestamp', datetime.utcnow())
        features.append(timestamp.hour)
        features.append(timestamp.weekday())
        
        # Activity features
        features.append(session_data.get('session_duration', 0))
        features.append(session_data.get('requests_per_minute', 0))
        features.append(session_data.get('data_volume', 0))
        features.append(session_data.get('unique_endpoints', 0))
        features.append(session_data.get('failed_requests_ratio', 0))
        features.append(session_data.get('geographic_distance', 0))
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train anomaly detection model"""
        if len(training_data) < 100:
            logger.warning("Insufficient training data for anomaly detection")
            return
        
        # Extract features
        features = []
        for session in training_data:
            feature_vector = self.extract_features(session)
            features.append(feature_vector.flatten())
        
        X = np.array(features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        logger.info(f"Anomaly detection model trained on {len(training_data)} samples")
    
    def detect_anomaly(self, session_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Detect if session data is anomalous"""
        if not self.is_trained:
            return False, 0.0
        
        # Extract and scale features
        features = self.extract_features(session_data)
        features_scaled = self.scaler.transform(features)
        
        # Predict anomaly
        anomaly_score = self.model.decision_function(features_scaled)[0]
        is_anomaly = self.model.predict(features_scaled)[0] == -1
        
        # Convert score to confidence (0-1)
        confidence = max(0, min(1, (0.5 - anomaly_score) * 2))
        
        return is_anomaly, confidence

class GeolocationAnalyzer:
    """Analyze geographic patterns for threat detection"""
    
    def __init__(self):
        # Simplified IP geolocation (in production, use proper GeoIP database)
        self.known_locations = {}
        self.suspicious_countries = {'CN', 'RU', 'KP', 'IR'}  # Example suspicious countries
    
    def analyze_location(self, ip_address: str, user_id: str = None) -> Dict[str, Any]:
        """Analyze IP location for threats"""
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            
            # Skip private IPs
            if ip_obj.is_private:
                return {"is_suspicious": False, "reason": "private_ip"}
            
            # Simplified geolocation (replace with real GeoIP service)
            location_info = self._get_location_info(ip_address)
            
            analysis = {
                "country": location_info.get("country"),
                "is_suspicious": False,
                "reasons": []
            }
            
            # Check for suspicious countries
            if location_info.get("country") in self.suspicious_countries:
                analysis["is_suspicious"] = True
                analysis["reasons"].append("suspicious_country")
            
            # Check for location changes (if user_id provided)
            if user_id:
                previous_locations = self.known_locations.get(user_id, set())
                current_country = location_info.get("country")
                
                if previous_locations and current_country not in previous_locations:
                    analysis["is_suspicious"] = True
                    analysis["reasons"].append("location_change")
                
                # Update known locations
                if user_id not in self.known_locations:
                    self.known_locations[user_id] = set()
                self.known_locations[user_id].add(current_country)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Geolocation analysis failed for {ip_address}: {e}")
            return {"is_suspicious": False, "error": str(e)}
    
    def _get_location_info(self, ip_address: str) -> Dict[str, str]:
        """Get location information for IP (placeholder)"""
        # In production, integrate with MaxMind GeoIP2 or similar service
        return {
            "country": "US",  # Default
            "city": "Unknown",
            "region": "Unknown"
        }

class PatternMatcher:
    """Pattern matching for known attack signatures"""
    
    def __init__(self):
        self.sql_injection_patterns = [
            r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
            r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
            r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
            r"((\%27)|(\'))union",
            r"exec(\s|\+)+(s|x)p\w+",
            r"UNION.*SELECT",
            r"SELECT.*FROM.*WHERE"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>"
        ]
        
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"\%2e\%2e\%2f",
            r"\%2e\%2e\\",
            r"\.\.%2f",
            r"\.\.%5c"
        ]
    
    def detect_sql_injection(self, payload: str) -> bool:
        """Detect SQL injection patterns"""
        payload_lower = payload.lower()
        return any(re.search(pattern, payload_lower, re.IGNORECASE) for pattern in self.sql_injection_patterns)
    
    def detect_xss(self, payload: str) -> bool:
        """Detect XSS patterns"""
        return any(re.search(pattern, payload, re.IGNORECASE) for pattern in self.xss_patterns)
    
    def detect_path_traversal(self, payload: str) -> bool:
        """Detect path traversal patterns"""
        return any(re.search(pattern, payload, re.IGNORECASE) for pattern in self.path_traversal_patterns)
    
    def analyze_payload(self, payload: str) -> Dict[str, Any]:
        """Analyze payload for multiple attack patterns"""
        threats = []
        
        if self.detect_sql_injection(payload):
            threats.append(ThreatType.SQL_INJECTION)
        
        if self.detect_xss(payload):
            threats.append(ThreatType.XSS)
        
        if self.detect_path_traversal(payload):
            threats.append(ThreatType.MALICIOUS_PAYLOAD)
        
        return {
            "threats_detected": threats,
            "is_malicious": len(threats) > 0,
            "payload_hash": hashlib.sha256(payload.encode()).hexdigest()[:16]
        }

class AdvancedThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.rate_limiter = RateLimitTracker()
        self.anomaly_detector = AnomalyDetector()
        self.geolocation_analyzer = GeolocationAnalyzer()
        self.pattern_matcher = PatternMatcher()
        
        # Threat tracking
        self.active_threats = {}
        self.user_profiles = {}
        self.blocked_ips = set()
        
        # Statistics
        self.stats = {
            "threats_detected": 0,
            "threats_blocked": 0,
            "false_positives": 0,
            "last_model_update": None
        }
        
        # Background tasks
        self.is_running = False
        self.cleanup_task = None
    
    def start(self):
        """Start threat detection system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Advanced threat detection system started")
    
    async def stop(self):
        """Stop threat detection system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced threat detection system stopped")
    
    def analyze_request(self, request_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Analyze incoming request for threats"""
        threats = []
        
        ip_address = request_data.get('ip_address')
        user_id = request_data.get('user_id')
        payload = request_data.get('payload', '')
        
        # Skip analysis for blocked IPs
        if ip_address in self.blocked_ips:
            return []
        
        # Rate limiting check
        if self.rate_limiter.is_rate_limited(ip_address):
            threat = ThreatEvent(
                threat_id=self._generate_threat_id(),
                threat_type=ThreatType.DDoS,
                threat_level=ThreatLevel.HIGH,
                timestamp=datetime.utcnow(),
                source_ip=ip_address,
                user_id=user_id,
                detection_rule=DetectionRule.RATE_LIMITING,
                confidence_score=0.9,
                details={"request_rate": self.rate_limiter.get_request_rate(ip_address)}
            )
            threats.append(threat)
        
        # Pattern matching
        pattern_analysis = self.pattern_matcher.analyze_payload(payload)
        if pattern_analysis["is_malicious"]:
            for threat_type in pattern_analysis["threats_detected"]:
                threat = ThreatEvent(
                    threat_id=self._generate_threat_id(),
                    threat_type=threat_type,
                    threat_level=ThreatLevel.HIGH,
                    timestamp=datetime.utcnow(),
                    source_ip=ip_address,
                    user_id=user_id,
                    detection_rule=DetectionRule.PATTERN_MATCHING,
                    confidence_score=0.8,
                    details=pattern_analysis
                )
                threats.append(threat)
        
        # Geolocation analysis
        if ip_address:
            geo_analysis = self.geolocation_analyzer.analyze_location(ip_address, user_id)
            if geo_analysis.get("is_suspicious"):
                threat = ThreatEvent(
                    threat_id=self._generate_threat_id(),
                    threat_type=ThreatType.SUSPICIOUS_LOCATION,
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=datetime.utcnow(),
                    source_ip=ip_address,
                    user_id=user_id,
                    detection_rule=DetectionRule.GEOLOCATION,
                    confidence_score=0.6,
                    details=geo_analysis
                )
                threats.append(threat)
        
        # Anomaly detection
        if user_id and self.anomaly_detector.is_trained:
            session_data = self._extract_session_data(request_data)
            is_anomaly, confidence = self.anomaly_detector.detect_anomaly(session_data)
            
            if is_anomaly:
                threat = ThreatEvent(
                    threat_id=self._generate_threat_id(),
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.MEDIUM,
                    timestamp=datetime.utcnow(),
                    source_ip=ip_address,
                    user_id=user_id,
                    detection_rule=DetectionRule.ANOMALY_DETECTION,
                    confidence_score=confidence,
                    details=session_data
                )
                threats.append(threat)
        
        # Update statistics
        if threats:
            self.stats["threats_detected"] += len(threats)
        
        return threats
    
    def analyze_user_behavior(self, user_id: str, session_data: Dict[str, Any]) -> List[ThreatEvent]:
        """Analyze user behavior for anomalies"""
        threats = []
        
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # Analyze login time
        current_hour = datetime.utcnow().hour
        if profile.typical_login_times:
            avg_hour = statistics.mean(profile.typical_login_times)
            if abs(current_hour - avg_hour) > 6:  # More than 6 hours difference
                threat = ThreatEvent(
                    threat_id=self._generate_threat_id(),
                    threat_type=ThreatType.UNUSUAL_TIME_ACCESS,
                    threat_level=ThreatLevel.LOW,
                    timestamp=datetime.utcnow(),
                    user_id=user_id,
                    detection_rule=DetectionRule.TIME_BASED,
                    confidence_score=0.4,
                    details={"current_hour": current_hour, "typical_hours": profile.typical_login_times}
                )
                threats.append(threat)
        
        # Update profile
        profile.typical_login_times.append(current_hour)
        if len(profile.typical_login_times) > 100:  # Keep last 100 login times
            profile.typical_login_times = profile.typical_login_times[-100:]
        
        profile.last_updated = datetime.utcnow()
        profile.sample_count += 1
        
        return threats
    
    def block_ip(self, ip_address: str, duration_hours: int = 24):
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        
        # Schedule unblock (simplified - in production, use proper scheduler)
        asyncio.create_task(self._schedule_unblock(ip_address, duration_hours))
        
        logger.warning(f"Blocked IP address: {ip_address} for {duration_hours} hours")
    
    async def _schedule_unblock(self, ip_address: str, duration_hours: int):
        """Schedule IP unblock"""
        await asyncio.sleep(duration_hours * 3600)
        self.blocked_ips.discard(ip_address)
        logger.info(f"Unblocked IP address: {ip_address}")
    
    def respond_to_threat(self, threat: ThreatEvent) -> str:
        """Respond to detected threat"""
        response_action = "logged"
        
        # Determine response based on threat level and type
        if threat.threat_level == ThreatLevel.CRITICAL:
            if threat.source_ip:
                self.block_ip(threat.source_ip, 48)  # Block for 48 hours
                response_action = "ip_blocked"
                threat.blocked = True
        
        elif threat.threat_level == ThreatLevel.HIGH:
            if threat.threat_type in [ThreatType.SQL_INJECTION, ThreatType.XSS]:
                if threat.source_ip:
                    self.block_ip(threat.source_ip, 24)  # Block for 24 hours
                    response_action = "ip_blocked"
                    threat.blocked = True
            elif threat.threat_type == ThreatType.DDoS:
                if threat.source_ip:
                    self.block_ip(threat.source_ip, 1)  # Block for 1 hour
                    response_action = "rate_limited"
                    threat.blocked = True
        
        threat.response_action = response_action
        self.active_threats[threat.threat_id] = threat
        
        if threat.blocked:
            self.stats["threats_blocked"] += 1
        
        return response_action
    
    def train_anomaly_model(self, training_data: List[Dict[str, Any]]):
        """Train anomaly detection model with historical data"""
        self.anomaly_detector.train(training_data)
        self.stats["last_model_update"] = datetime.utcnow()
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old threats (keep for 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                old_threats = [
                    threat_id for threat_id, threat in self.active_threats.items()
                    if threat.timestamp < cutoff_time
                ]
                
                for threat_id in old_threats:
                    del self.active_threats[threat_id]
                
                logger.debug(f"Cleaned up {len(old_threats)} old threat records")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in threat detector cleanup: {e}")
    
    def _extract_session_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract session data for anomaly detection"""
        return {
            "timestamp": request_data.get("timestamp", datetime.utcnow()),
            "session_duration": request_data.get("session_duration", 0),
            "requests_per_minute": request_data.get("requests_per_minute", 0),
            "data_volume": request_data.get("data_volume", 0),
            "unique_endpoints": request_data.get("unique_endpoints", 0),
            "failed_requests_ratio": request_data.get("failed_requests_ratio", 0),
            "geographic_distance": request_data.get("geographic_distance", 0)
        }
    
    def _generate_threat_id(self) -> str:
        """Generate unique threat ID"""
        import uuid
        return str(uuid.uuid4())
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics"""
        return {
            **self.stats,
            "active_threats": len(self.active_threats),
            "blocked_ips": len(self.blocked_ips),
            "user_profiles": len(self.user_profiles),
            "model_trained": self.anomaly_detector.is_trained
        }
    
    def get_active_threats(self, threat_level: ThreatLevel = None) -> List[ThreatEvent]:
        """Get active threats, optionally filtered by level"""
        threats = list(self.active_threats.values())
        
        if threat_level:
            threats = [t for t in threats if t.threat_level == threat_level]
        
        return sorted(threats, key=lambda x: x.timestamp, reverse=True)

# Global threat detector instance
threat_detector = None

def initialize_threat_detector() -> AdvancedThreatDetector:
    """Initialize global threat detector"""
    global threat_detector
    threat_detector = AdvancedThreatDetector()
    return threat_detector

def get_threat_detector() -> AdvancedThreatDetector:
    """Get global threat detector instance"""
    return threat_detector
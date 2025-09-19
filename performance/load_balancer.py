"""
Advanced Load Balancer for Fresh Supply Chain Intelligence System
Intelligent load balancing with health checks, circuit breakers, and auto-scaling
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import deque
import aiohttp
import structlog

logger = structlog.get_logger()

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"

class ServerStatus(Enum):
    """Server status states"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    health_check_url: str = "/health"
    health_check_interval: int = 30
    health_check_timeout: int = 5
    failure_threshold: int = 3
    recovery_threshold: int = 2
    max_response_time: float = 2.0

@dataclass
class ServerMetrics:
    """Server performance metrics"""
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_response_time: float = 0.0
    health_check_failures: int = 0
    health_check_successes: int = 0
    last_health_check: Optional[datetime] = None
    status: ServerStatus = ServerStatus.HEALTHY
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

class CircuitBreaker:
    """Circuit breaker for server protection"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time > self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class Server:
    """Server instance with health monitoring"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.metrics = ServerMetrics()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=60
        )
        self.response_times = deque(maxlen=100)  # Keep last 100 response times
        self.lock = threading.Lock()
    
    @property
    def url(self) -> str:
        """Get server URL"""
        return f"http://{self.config.host}:{self.config.port}"
    
    @property
    def health_check_url(self) -> str:
        """Get health check URL"""
        return f"{self.url}{self.config.health_check_url}"
    
    def is_available(self) -> bool:
        """Check if server is available for requests"""
        return (
            self.metrics.status == ServerStatus.HEALTHY and
            self.metrics.active_connections < self.config.max_connections and
            self.circuit_breaker.state != "open"
        )
    
    def get_load_score(self) -> float:
        """Get server load score (lower is better)"""
        if not self.is_available():
            return float('inf')
        
        # Combine multiple factors for load scoring
        connection_load = self.metrics.active_connections / self.config.max_connections
        response_time_load = min(self.metrics.avg_response_time / self.config.max_response_time, 1.0)
        cpu_load = self.metrics.cpu_usage / 100.0
        memory_load = self.metrics.memory_usage / 100.0
        
        # Weighted average
        load_score = (
            connection_load * 0.3 +
            response_time_load * 0.3 +
            cpu_load * 0.2 +
            memory_load * 0.2
        )
        
        return load_score
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics"""
        with self.lock:
            self.metrics.total_requests += 1
            
            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
            
            # Update response time metrics
            self.response_times.append(response_time)
            self.metrics.last_response_time = response_time
            
            if self.response_times:
                self.metrics.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def increment_connections(self):
        """Increment active connections"""
        with self.lock:
            self.metrics.active_connections += 1
    
    def decrement_connections(self):
        """Decrement active connections"""
        with self.lock:
            self.metrics.active_connections = max(0, self.metrics.active_connections - 1)
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)) as session:
                async with session.get(self.health_check_url) as response:
                    is_healthy = response.status == 200
                    
                    with self.lock:
                        self.metrics.last_health_check = datetime.now()
                        
                        if is_healthy:
                            self.metrics.health_check_successes += 1
                            
                            # Reset failure count if we have enough successes
                            if self.metrics.health_check_successes >= self.config.recovery_threshold:
                                self.metrics.health_check_failures = 0
                                if self.metrics.status == ServerStatus.UNHEALTHY:
                                    self.metrics.status = ServerStatus.HEALTHY
                                    logger.info(f"Server {self.url} recovered")
                        else:
                            self.metrics.health_check_failures += 1
                            self.metrics.health_check_successes = 0
                            
                            # Mark as unhealthy if too many failures
                            if self.metrics.health_check_failures >= self.config.failure_threshold:
                                self.metrics.status = ServerStatus.UNHEALTHY
                                logger.warning(f"Server {self.url} marked as unhealthy")
                    
                    return is_healthy
                    
        except Exception as e:
            logger.error(f"Health check failed for {self.url}: {e}")
            
            with self.lock:
                self.metrics.health_check_failures += 1
                self.metrics.health_check_successes = 0
                self.metrics.last_health_check = datetime.now()
                
                if self.metrics.health_check_failures >= self.config.failure_threshold:
                    self.metrics.status = ServerStatus.UNHEALTHY
            
            return False

class LoadBalancer:
    """Advanced load balancer with multiple strategies"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.servers: List[Server] = []
        self.current_index = 0
        self.health_check_task = None
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "requests_per_second": 0.0
        }
        self.request_history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def add_server(self, config: ServerConfig):
        """Add server to load balancer"""
        server = Server(config)
        self.servers.append(server)
        logger.info(f"Added server: {server.url}")
    
    def remove_server(self, host: str, port: int):
        """Remove server from load balancer"""
        self.servers = [s for s in self.servers if not (s.config.host == host and s.config.port == port)]
        logger.info(f"Removed server: {host}:{port}")
    
    def get_server(self, client_ip: str = None) -> Optional[Server]:
        """Get server based on load balancing strategy"""
        available_servers = [s for s in self.servers if s.is_available()]
        
        if not available_servers:
            logger.warning("No available servers")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(available_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(available_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(available_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(available_servers)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash(available_servers, client_ip)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random(available_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random(available_servers)
        else:
            return self._least_connections(available_servers)
    
    def _round_robin(self, servers: List[Server]) -> Server:
        """Round robin selection"""
        with self.lock:
            server = servers[self.current_index % len(servers)]
            self.current_index += 1
            return server
    
    def _weighted_round_robin(self, servers: List[Server]) -> Server:
        """Weighted round robin selection"""
        # Create weighted list
        weighted_servers = []
        for server in servers:
            weighted_servers.extend([server] * server.config.weight)
        
        with self.lock:
            server = weighted_servers[self.current_index % len(weighted_servers)]
            self.current_index += 1
            return server
    
    def _least_connections(self, servers: List[Server]) -> Server:
        """Least connections selection"""
        return min(servers, key=lambda s: s.metrics.active_connections)
    
    def _least_response_time(self, servers: List[Server]) -> Server:
        """Least response time selection"""
        return min(servers, key=lambda s: s.metrics.avg_response_time)
    
    def _ip_hash(self, servers: List[Server], client_ip: str) -> Server:
        """IP hash selection for session affinity"""
        if not client_ip:
            return self._least_connections(servers)
        
        hash_value = hash(client_ip)
        return servers[hash_value % len(servers)]
    
    def _random(self, servers: List[Server]) -> Server:
        """Random selection"""
        return random.choice(servers)
    
    def _weighted_random(self, servers: List[Server]) -> Server:
        """Weighted random selection"""
        total_weight = sum(s.config.weight for s in servers)
        random_weight = random.uniform(0, total_weight)
        
        current_weight = 0
        for server in servers:
            current_weight += server.config.weight
            if random_weight <= current_weight:
                return server
        
        return servers[-1]  # Fallback
    
    async def forward_request(self, method: str, path: str, headers: Dict[str, str] = None, 
                            data: Any = None, client_ip: str = None) -> Tuple[int, Dict[str, str], Any]:
        """Forward request to selected server"""
        server = self.get_server(client_ip)
        
        if not server:
            raise Exception("No available servers")
        
        server.increment_connections()
        start_time = time.time()
        
        try:
            url = f"{server.url}{path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, data=data) as response:
                    response_data = await response.read()
                    response_headers = dict(response.headers)
                    
                    response_time = time.time() - start_time
                    server.record_request(response_time, response.status < 400)
                    
                    self._record_global_metrics(response_time, response.status < 400)
                    
                    return response.status, response_headers, response_data
        
        except Exception as e:
            response_time = time.time() - start_time
            server.record_request(response_time, False)
            self._record_global_metrics(response_time, False)
            raise e
        
        finally:
            server.decrement_connections()
    
    def _record_global_metrics(self, response_time: float, success: bool):
        """Record global load balancer metrics"""
        with self.lock:
            self.metrics["total_requests"] += 1
            
            if success:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
            
            # Update request history for RPS calculation
            self.request_history.append(time.time())
            
            # Calculate requests per second (last minute)
            now = time.time()
            recent_requests = [t for t in self.request_history if now - t <= 60]
            self.metrics["requests_per_second"] = len(recent_requests) / 60.0
            
            # Update average response time
            if self.metrics["total_requests"] > 0:
                current_avg = self.metrics["avg_response_time"]
                total_requests = self.metrics["total_requests"]
                self.metrics["avg_response_time"] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )
    
    async def start_health_checks(self):
        """Start periodic health checks"""
        if self.health_check_task:
            return
        
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started health check monitoring")
    
    async def stop_health_checks(self):
        """Stop health checks"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            logger.info("Stopped health check monitoring")
    
    async def _health_check_loop(self):
        """Health check loop"""
        while True:
            try:
                # Perform health checks for all servers
                health_check_tasks = [server.health_check() for server in self.servers]
                await asyncio.gather(*health_check_tasks, return_exceptions=True)
                
                # Wait for next health check interval
                min_interval = min(s.config.health_check_interval for s in self.servers) if self.servers else 30
                await asyncio.sleep(min_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(30)
    
    def get_server_status(self) -> List[Dict[str, Any]]:
        """Get status of all servers"""
        return [
            {
                "url": server.url,
                "status": server.metrics.status.value,
                "active_connections": server.metrics.active_connections,
                "total_requests": server.metrics.total_requests,
                "success_rate": (
                    server.metrics.successful_requests / max(server.metrics.total_requests, 1)
                ),
                "avg_response_time": server.metrics.avg_response_time,
                "last_health_check": server.metrics.last_health_check.isoformat() if server.metrics.last_health_check else None,
                "health_check_failures": server.metrics.health_check_failures,
                "circuit_breaker_state": server.circuit_breaker.state,
                "load_score": server.get_load_score()
            }
            for server in self.servers
        ]
    
    def get_load_balancer_metrics(self) -> Dict[str, Any]:
        """Get load balancer metrics"""
        healthy_servers = len([s for s in self.servers if s.metrics.status == ServerStatus.HEALTHY])
        total_servers = len(self.servers)
        
        return {
            "total_servers": total_servers,
            "healthy_servers": healthy_servers,
            "unhealthy_servers": total_servers - healthy_servers,
            "strategy": self.strategy.value,
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": (
                self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1)
            ),
            "avg_response_time": self.metrics["avg_response_time"],
            "requests_per_second": self.metrics["requests_per_second"]
        }
    
    def auto_scale(self, target_cpu_threshold: float = 70.0, target_response_time: float = 1.0):
        """Auto-scaling logic based on metrics"""
        # Calculate average metrics across healthy servers
        healthy_servers = [s for s in self.servers if s.metrics.status == ServerStatus.HEALTHY]
        
        if not healthy_servers:
            return
        
        avg_cpu = sum(s.metrics.cpu_usage for s in healthy_servers) / len(healthy_servers)
        avg_response_time = sum(s.metrics.avg_response_time for s in healthy_servers) / len(healthy_servers)
        
        # Scale up conditions
        should_scale_up = (
            avg_cpu > target_cpu_threshold or
            avg_response_time > target_response_time or
            len(healthy_servers) < 2  # Minimum 2 servers for redundancy
        )
        
        # Scale down conditions
        should_scale_down = (
            avg_cpu < target_cpu_threshold * 0.5 and
            avg_response_time < target_response_time * 0.5 and
            len(healthy_servers) > 2  # Keep at least 2 servers
        )
        
        if should_scale_up:
            logger.info(f"Auto-scaling up triggered: CPU={avg_cpu:.1f}%, Response Time={avg_response_time:.3f}s")
            # In a real implementation, this would trigger server provisioning
        elif should_scale_down:
            logger.info(f"Auto-scaling down triggered: CPU={avg_cpu:.1f}%, Response Time={avg_response_time:.3f}s")
            # In a real implementation, this would trigger server deprovisioning

# Global load balancer instance
load_balancer = None

def initialize_load_balancer(strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS) -> LoadBalancer:
    """Initialize global load balancer"""
    global load_balancer
    load_balancer = LoadBalancer(strategy)
    return load_balancer

def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance"""
    return load_balancer
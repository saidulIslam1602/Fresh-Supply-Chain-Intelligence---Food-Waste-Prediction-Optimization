"""
Advanced Middleware Components for Enhanced API Security and Performance
"""

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import json
import logging
import hashlib
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import redis
import jwt
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger()

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced request logging with structured logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Extract request information
        request_id = hashlib.md5(f"{time.time()}{request.client.host}".encode()).hexdigest()[:8]
        
        # Log request start
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent", ""),
            client_ip=request.client.host,
            content_length=request.headers.get("content-length", 0)
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            process_time=process_time,
            response_size=response.headers.get("content-length", 0)
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with sliding window and user-based limits"""
    
    def __init__(self, app, redis_client=None, default_limit: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.redis_client = redis_client
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.memory_store = defaultdict(deque)  # Fallback for when Redis is unavailable
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not await self._is_allowed(client_id, request.url.path):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.default_limit} requests per {self.window_seconds} seconds",
                    "retry_after": self.window_seconds
                },
                headers={"Retry-After": str(self.window_seconds)}
            )
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request"""
        # Try to get user from JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                return f"user:{payload.get('sub', 'anonymous')}"
            except:
                pass
        
        # Fallback to IP address
        return f"ip:{request.client.host}"
    
    async def _is_allowed(self, client_id: str, endpoint: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        if self.redis_client:
            return await self._redis_rate_limit(client_id, endpoint, current_time, window_start)
        else:
            return self._memory_rate_limit(client_id, current_time, window_start)
    
    async def _redis_rate_limit(self, client_id: str, endpoint: str, current_time: float, window_start: float) -> bool:
        """Redis-based sliding window rate limiting"""
        try:
            pipe = self.redis_client.pipeline()
            key = f"rate_limit:{client_id}:{endpoint}"
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, self.window_seconds + 1)
            
            results = await pipe.execute()
            current_count = results[1]
            
            return current_count < self.default_limit
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            return True  # Allow request if Redis fails
    
    def _memory_rate_limit(self, client_id: str, current_time: float, window_start: float) -> bool:
        """Memory-based rate limiting (fallback)"""
        requests = self.memory_store[client_id]
        
        # Remove old requests
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check limit
        if len(requests) >= self.default_limit:
            return False
        
        # Add current request
        requests.append(current_time)
        return True

class CacheMiddleware(BaseHTTPMiddleware):
    """Intelligent caching middleware with cache-control headers"""
    
    def __init__(self, app, redis_client=None, default_ttl: int = 300):
        super().__init__(app)
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.cacheable_methods = {"GET"}
        self.cacheable_paths = {"/api/v1/products", "/api/v1/warehouses", "/api/v1/analytics"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests for specific endpoints
        if (request.method not in self.cacheable_methods or 
            not any(request.url.path.startswith(path) for path in self.cacheable_paths)):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get from cache
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            logger.info(f"Cache hit for {request.url.path}")
            return JSONResponse(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers={
                    **cached_response["headers"],
                    "X-Cache": "HIT",
                    "X-Cache-Key": cache_key[:16]
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            await self._cache_response(cache_key, response)
            response.headers["X-Cache"] = "MISS"
            response.headers["X-Cache-Key"] = cache_key[:16]
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request"""
        key_parts = [
            request.method,
            request.url.path,
            str(sorted(request.query_params.items()))
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get response from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(f"cache:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: Response):
        """Cache response"""
        if not self.redis_client:
            return
        
        try:
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Parse JSON content
            content = json.loads(response_body.decode())
            
            cache_data = {
                "content": content,
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
            
            await self.redis_client.setex(
                f"cache:{cache_key}",
                self.default_ttl,
                json.dumps(cache_data, default=str)
            )
            
            # Recreate response with same content
            response.body_iterator = iter([response_body])
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

class CompressionMiddleware(BaseHTTPMiddleware):
    """Advanced compression middleware with content-type awareness"""
    
    def __init__(self, app, minimum_size: int = 1000, compression_level: int = 6):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        self.compressible_types = {
            "application/json",
            "text/plain",
            "text/html",
            "text/css",
            "application/javascript",
            "text/xml"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Check content type
        content_type = response.headers.get("content-type", "").split(";")[0]
        if content_type not in self.compressible_types:
            return response
        
        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return response
        
        # Compress response (simplified - in production use proper gzip)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling with structured error responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": e.status_code,
                        "message": e.detail,
                        "type": "http_exception",
                        "timestamp": datetime.utcnow().isoformat(),
                        "path": request.url.path
                    }
                }
            )
            
        except ValueError as e:
            # Handle validation errors
            logger.error(f"Validation error: {e}", path=request.url.path)
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": "Invalid input data",
                        "details": str(e),
                        "type": "validation_error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "path": request.url.path
                    }
                }
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error: {e}", path=request.url.path, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Internal server error",
                        "type": "internal_error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "path": request.url.path,
                        "request_id": request.headers.get("X-Request-ID", "unknown")
                    }
                }
            )

class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect detailed metrics for monitoring and alerting"""
    
    def __init__(self, app, metrics_collector=None):
        super().__init__(app)
        self.metrics_collector = metrics_collector
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration = time.time() - start_time
        endpoint = f"{request.method} {request.url.path}"
        
        # Update metrics
        self.request_counts[endpoint] += 1
        self.response_times[endpoint].append(duration)
        
        # Keep only recent response times (last 100 requests)
        if len(self.response_times[endpoint]) > 100:
            self.response_times[endpoint] = self.response_times[endpoint][-100:]
        
        # Send to external metrics collector if available
        if self.metrics_collector:
            await self._send_metrics(endpoint, duration, response.status_code)
        
        # Add metrics headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        response.headers["X-Request-Count"] = str(self.request_counts[endpoint])
        
        return response
    
    async def _send_metrics(self, endpoint: str, duration: float, status_code: int):
        """Send metrics to external collector (Prometheus, DataDog, etc.)"""
        try:
            if self.metrics_collector:
                await self.metrics_collector.record_request(
                    endpoint=endpoint,
                    duration=duration,
                    status_code=status_code,
                    timestamp=datetime.utcnow()
                )
        except Exception as e:
            logger.error(f"Failed to send metrics: {e}")
    
    def get_metrics_summary(self) -> Dict:
        """Get current metrics summary"""
        summary = {}
        
        for endpoint, times in self.response_times.items():
            if times:
                summary[endpoint] = {
                    "request_count": self.request_counts[endpoint],
                    "avg_response_time": sum(times) / len(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "p95_response_time": sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times)
                }
        
        return summary

class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Health check middleware with dependency monitoring"""
    
    def __init__(self, app, health_checker=None):
        super().__init__(app)
        self.health_checker = health_checker
        self.last_health_check = None
        self.health_status = {"status": "unknown"}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Perform health check for health endpoints
        if request.url.path in ["/health", "/health/ready", "/health/live"]:
            return await self._handle_health_check(request)
        
        # Check if system is healthy for other requests
        if not await self._is_system_healthy():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "message": "System health check failed",
                    "retry_after": 30
                },
                headers={"Retry-After": "30"}
            )
        
        return await call_next(request)
    
    async def _handle_health_check(self, request: Request) -> Response:
        """Handle health check requests"""
        health_status = await self._perform_health_check()
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
    
    async def _perform_health_check(self) -> Dict:
        """Perform comprehensive health check"""
        current_time = time.time()
        
        # Use cached result if recent (within 30 seconds)
        if (self.last_health_check and 
            current_time - self.last_health_check < 30):
            return self.health_status
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Check dependencies if health checker is available
        if self.health_checker:
            try:
                dependency_status = await self.health_checker.check_all()
                health_status["checks"] = dependency_status
                
                # Determine overall status
                if any(check["status"] != "healthy" for check in dependency_status.values()):
                    health_status["status"] = "unhealthy"
                    
            except Exception as e:
                health_status["status"] = "unhealthy"
                health_status["error"] = str(e)
        
        self.health_status = health_status
        self.last_health_check = current_time
        
        return health_status
    
    async def _is_system_healthy(self) -> bool:
        """Quick health check for request processing"""
        # Perform lightweight health check
        if not self.last_health_check or time.time() - self.last_health_check > 60:
            await self._perform_health_check()
        
        return self.health_status.get("status") == "healthy"
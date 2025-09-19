"""
Advanced Cache Manager for Fresh Supply Chain Intelligence System
Multi-layer caching with Redis, in-memory, and CDN integration
"""

import asyncio
import json
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import redis
import redis.asyncio as aioredis
from cachetools import TTLCache, LRUCache
import structlog

logger = structlog.get_logger()

class CacheLevel(Enum):
    """Cache levels for multi-tier caching"""
    MEMORY = "memory"
    REDIS = "redis"
    CDN = "cdn"
    DATABASE = "database"

class CacheStrategy(Enum):
    """Cache strategies"""
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"

@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_url: str = "redis://localhost:6379/0"
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 300  # 5 minutes
    redis_ttl: int = 3600  # 1 hour
    compression_enabled: bool = True
    serialization_format: str = "json"  # json, pickle, msgpack
    max_key_length: int = 250
    enable_metrics: bool = True

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    redis_connections: int = 0

class AdvancedCacheManager:
    """Advanced multi-tier cache manager with intelligent caching strategies"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.metrics = CacheMetrics()
        
        # Initialize memory cache
        self.memory_cache = TTLCache(
            maxsize=self.config.memory_cache_size,
            ttl=self.config.memory_cache_ttl
        )
        
        # Initialize Redis clients
        self.redis_client = None
        self.async_redis_client = None
        self._initialize_redis()
        
        # Cache strategies
        self.strategies = {}
        self._setup_default_strategies()
        
        # Performance monitoring
        self.performance_data = []
        
    def _initialize_redis(self):
        """Initialize Redis connections"""
        try:
            # Synchronous Redis client
            self.redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            
            # Asynchronous Redis client
            self.async_redis_client = aioredis.from_url(
                self.config.redis_url,
                decode_responses=True
            )
            
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
            self.async_redis_client = None
    
    def _setup_default_strategies(self):
        """Setup default caching strategies for different data types"""
        self.strategies = {
            # Fast-changing data - memory only
            "user_sessions": {
                "levels": [CacheLevel.MEMORY],
                "ttl": 300,  # 5 minutes
                "strategy": CacheStrategy.CACHE_ASIDE
            },
            
            # API responses - memory + Redis
            "api_responses": {
                "levels": [CacheLevel.MEMORY, CacheLevel.REDIS],
                "ttl": 900,  # 15 minutes
                "strategy": CacheStrategy.READ_THROUGH
            },
            
            # ML predictions - Redis with longer TTL
            "ml_predictions": {
                "levels": [CacheLevel.REDIS],
                "ttl": 3600,  # 1 hour
                "strategy": CacheStrategy.WRITE_THROUGH
            },
            
            # Static data - all levels
            "static_data": {
                "levels": [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.CDN],
                "ttl": 86400,  # 24 hours
                "strategy": CacheStrategy.WRITE_BEHIND
            },
            
            # Database queries - memory + Redis
            "database_queries": {
                "levels": [CacheLevel.MEMORY, CacheLevel.REDIS],
                "ttl": 1800,  # 30 minutes
                "strategy": CacheStrategy.CACHE_ASIDE
            },
            
            # Business KPIs - Redis only
            "business_kpis": {
                "levels": [CacheLevel.REDIS],
                "ttl": 600,  # 10 minutes
                "strategy": CacheStrategy.WRITE_THROUGH
            }
        }
    
    def _generate_cache_key(self, namespace: str, key: str, **kwargs) -> str:
        """Generate standardized cache key"""
        # Include kwargs in key for parameter-based caching
        if kwargs:
            key_parts = [str(v) for v in sorted(kwargs.items())]
            key = f"{key}:{'_'.join(key_parts)}"
        
        cache_key = f"{namespace}:{key}"
        
        # Hash long keys
        if len(cache_key) > self.config.max_key_length:
            cache_key = f"{namespace}:{hashlib.md5(key.encode()).hexdigest()}"
        
        return cache_key
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for caching"""
        try:
            if self.config.serialization_format == "json":
                return json.dumps(data, default=str).encode()
            elif self.config.serialization_format == "pickle":
                return pickle.dumps(data)
            else:
                return json.dumps(data, default=str).encode()
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return json.dumps({"error": "serialization_failed"}).encode()
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize cached data"""
        try:
            if self.config.serialization_format == "json":
                return json.loads(data.decode())
            elif self.config.serialization_format == "pickle":
                return pickle.loads(data)
            else:
                return json.loads(data.decode())
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    def get(self, namespace: str, key: str, **kwargs) -> Optional[Any]:
        """Get value from cache with multi-tier lookup"""
        start_time = time.time()
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        
        try:
            strategy_config = self.strategies.get(namespace, self.strategies["api_responses"])
            levels = strategy_config["levels"]
            
            # Try memory cache first
            if CacheLevel.MEMORY in levels:
                if cache_key in self.memory_cache:
                    self._update_metrics("hit", time.time() - start_time)
                    logger.debug(f"Memory cache hit: {cache_key}")
                    return self.memory_cache[cache_key]
            
            # Try Redis cache
            if CacheLevel.REDIS in levels and self.redis_client:
                try:
                    cached_data = self.redis_client.get(cache_key)
                    if cached_data:
                        deserialized_data = self._deserialize_data(cached_data.encode())
                        
                        # Populate memory cache for faster future access
                        if CacheLevel.MEMORY in levels:
                            self.memory_cache[cache_key] = deserialized_data
                        
                        self._update_metrics("hit", time.time() - start_time)
                        logger.debug(f"Redis cache hit: {cache_key}")
                        return deserialized_data
                except Exception as e:
                    logger.error(f"Redis get error: {e}")
                    self.metrics.errors += 1
            
            # Cache miss
            self._update_metrics("miss", time.time() - start_time)
            logger.debug(f"Cache miss: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.metrics.errors += 1
            return None
    
    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """Set value in cache with multi-tier storage"""
        start_time = time.time()
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        
        try:
            strategy_config = self.strategies.get(namespace, self.strategies["api_responses"])
            levels = strategy_config["levels"]
            cache_ttl = ttl or strategy_config["ttl"]
            
            success = True
            
            # Store in memory cache
            if CacheLevel.MEMORY in levels:
                self.memory_cache[cache_key] = value
                logger.debug(f"Stored in memory cache: {cache_key}")
            
            # Store in Redis cache
            if CacheLevel.REDIS in levels and self.redis_client:
                try:
                    serialized_data = self._serialize_data(value)
                    self.redis_client.setex(cache_key, cache_ttl, serialized_data)
                    logger.debug(f"Stored in Redis cache: {cache_key}")
                except Exception as e:
                    logger.error(f"Redis set error: {e}")
                    self.metrics.errors += 1
                    success = False
            
            self._update_metrics("set", time.time() - start_time)
            return success
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.metrics.errors += 1
            return False
    
    def delete(self, namespace: str, key: str, **kwargs) -> bool:
        """Delete value from all cache levels"""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        
        try:
            success = True
            
            # Delete from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            
            # Delete from Redis cache
            if self.redis_client:
                try:
                    self.redis_client.delete(cache_key)
                except Exception as e:
                    logger.error(f"Redis delete error: {e}")
                    success = False
            
            self.metrics.deletes += 1
            return success
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self.metrics.errors += 1
            return False
    
    def invalidate_pattern(self, namespace: str, pattern: str) -> int:
        """Invalidate cache entries matching a pattern"""
        try:
            count = 0
            
            # Invalidate memory cache entries
            keys_to_delete = []
            for key in self.memory_cache.keys():
                if key.startswith(f"{namespace}:") and pattern in key:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.memory_cache[key]
                count += 1
            
            # Invalidate Redis cache entries
            if self.redis_client:
                try:
                    redis_pattern = f"{namespace}:*{pattern}*"
                    redis_keys = self.redis_client.keys(redis_pattern)
                    if redis_keys:
                        self.redis_client.delete(*redis_keys)
                        count += len(redis_keys)
                except Exception as e:
                    logger.error(f"Redis pattern invalidation error: {e}")
            
            logger.info(f"Invalidated {count} cache entries matching pattern: {pattern}")
            return count
            
        except Exception as e:
            logger.error(f"Pattern invalidation error: {e}")
            return 0
    
    async def async_get(self, namespace: str, key: str, **kwargs) -> Optional[Any]:
        """Async version of cache get"""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        
        try:
            # Try memory cache first (synchronous)
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
            
            # Try Redis cache (asynchronous)
            if self.async_redis_client:
                try:
                    cached_data = await self.async_redis_client.get(cache_key)
                    if cached_data:
                        deserialized_data = self._deserialize_data(cached_data.encode())
                        # Populate memory cache
                        self.memory_cache[cache_key] = deserialized_data
                        return deserialized_data
                except Exception as e:
                    logger.error(f"Async Redis get error: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Async cache get error: {e}")
            return None
    
    async def async_set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """Async version of cache set"""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        
        try:
            strategy_config = self.strategies.get(namespace, self.strategies["api_responses"])
            cache_ttl = ttl or strategy_config["ttl"]
            
            # Store in memory cache (synchronous)
            self.memory_cache[cache_key] = value
            
            # Store in Redis cache (asynchronous)
            if self.async_redis_client:
                try:
                    serialized_data = self._serialize_data(value)
                    await self.async_redis_client.setex(cache_key, cache_ttl, serialized_data)
                    return True
                except Exception as e:
                    logger.error(f"Async Redis set error: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Async cache set error: {e}")
            return False
    
    def _update_metrics(self, operation: str, response_time: float):
        """Update cache performance metrics"""
        if not self.config.enable_metrics:
            return
        
        self.metrics.total_requests += 1
        
        if operation == "hit":
            self.metrics.hits += 1
        elif operation == "miss":
            self.metrics.misses += 1
        elif operation == "set":
            self.metrics.sets += 1
        
        # Update average response time
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + response_time) /
            self.metrics.total_requests
        )
        
        # Store performance data for analysis
        self.performance_data.append({
            "timestamp": datetime.now(),
            "operation": operation,
            "response_time": response_time
        })
        
        # Keep only last 1000 entries
        if len(self.performance_data) > 1000:
            self.performance_data = self.performance_data[-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        hit_rate = self.metrics.hits / max(self.metrics.total_requests, 1)
        miss_rate = self.metrics.misses / max(self.metrics.total_requests, 1)
        
        return {
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "sets": self.metrics.sets,
            "deletes": self.metrics.deletes,
            "errors": self.metrics.errors,
            "total_requests": self.metrics.total_requests,
            "avg_response_time": self.metrics.avg_response_time,
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_maxsize": self.memory_cache.maxsize,
            "redis_connected": self.redis_client is not None and self._test_redis_connection()
        }
    
    def _test_redis_connection(self) -> bool:
        """Test Redis connection"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
        except:
            pass
        return False
    
    def clear_all(self):
        """Clear all cache levels"""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear Redis cache
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"Redis clear error: {e}")
            
            logger.info("All caches cleared")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def warm_up(self, warm_up_data: Dict[str, Dict[str, Any]]):
        """Warm up cache with frequently accessed data"""
        logger.info("Starting cache warm-up...")
        
        for namespace, data_items in warm_up_data.items():
            for key, value in data_items.items():
                self.set(namespace, key, value)
        
        logger.info(f"Cache warm-up completed: {sum(len(items) for items in warm_up_data.values())} items")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        stats = self.get_metrics()
        
        # Add performance analysis
        if self.performance_data:
            recent_data = [p for p in self.performance_data if 
                          (datetime.now() - p["timestamp"]).seconds < 300]  # Last 5 minutes
            
            if recent_data:
                response_times = [p["response_time"] for p in recent_data]
                stats.update({
                    "recent_avg_response_time": sum(response_times) / len(response_times),
                    "recent_max_response_time": max(response_times),
                    "recent_min_response_time": min(response_times),
                    "recent_requests": len(recent_data)
                })
        
        return stats

# Cache decorators for easy integration
def cached(namespace: str, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache_manager.get(namespace, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(namespace, cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def async_cached(namespace: str, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Async decorator for caching function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = await cache_manager.async_get(namespace, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.async_set(namespace, cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Global cache manager instance
cache_manager = AdvancedCacheManager()

def initialize_cache_manager(config: CacheConfig = None) -> AdvancedCacheManager:
    """Initialize global cache manager"""
    global cache_manager
    cache_manager = AdvancedCacheManager(config)
    return cache_manager

def get_cache_manager() -> AdvancedCacheManager:
    """Get global cache manager instance"""
    return cache_manager
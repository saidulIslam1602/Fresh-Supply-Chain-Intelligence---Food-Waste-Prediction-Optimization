"""
Database Performance Optimizer for Fresh Supply Chain Intelligence System
Advanced database optimization with connection pooling, query optimization, and indexing
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging
import sqlalchemy as sa
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.engine import Engine
import pandas as pd
import structlog

logger = structlog.get_logger()

@dataclass
class DatabaseConfig:
    """Database configuration for optimization"""
    connection_string: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True
    echo: bool = False
    query_timeout: int = 30
    enable_query_logging: bool = True
    enable_slow_query_logging: bool = True
    slow_query_threshold: float = 1.0  # seconds
    connection_retry_attempts: int = 3
    connection_retry_delay: float = 1.0

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_count: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    slow_queries: int = 0
    failed_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

class QueryOptimizer:
    """Query optimization and analysis"""
    
    def __init__(self):
        self.query_cache = {}
        self.query_stats = {}
        self.optimization_rules = self._setup_optimization_rules()
    
    def _setup_optimization_rules(self) -> Dict[str, Any]:
        """Setup query optimization rules"""
        return {
            "select_optimization": {
                "avoid_select_star": True,
                "use_specific_columns": True,
                "limit_result_sets": True
            },
            "join_optimization": {
                "prefer_inner_joins": True,
                "use_proper_indexes": True,
                "avoid_cartesian_products": True
            },
            "where_clause_optimization": {
                "use_indexed_columns": True,
                "avoid_functions_in_where": True,
                "use_parameterized_queries": True
            },
            "index_recommendations": {
                "analyze_query_patterns": True,
                "suggest_composite_indexes": True,
                "identify_unused_indexes": True
            }
        }
    
    def analyze_query(self, query: str, execution_time: float, result_count: int) -> Dict[str, Any]:
        """Analyze query performance and suggest optimizations"""
        analysis = {
            "query": query,
            "execution_time": execution_time,
            "result_count": result_count,
            "performance_rating": self._rate_performance(execution_time, result_count),
            "optimizations": [],
            "index_suggestions": []
        }
        
        # Analyze query structure
        query_lower = query.lower().strip()
        
        # Check for SELECT * usage
        if "select *" in query_lower:
            analysis["optimizations"].append({
                "type": "select_optimization",
                "issue": "Using SELECT * can be inefficient",
                "suggestion": "Specify only needed columns",
                "priority": "medium"
            })
        
        # Check for missing WHERE clause in large tables
        if "select" in query_lower and "where" not in query_lower and result_count > 1000:
            analysis["optimizations"].append({
                "type": "where_clause",
                "issue": "Large result set without WHERE clause",
                "suggestion": "Add appropriate WHERE conditions to limit results",
                "priority": "high"
            })
        
        # Check for potential N+1 query problems
        if execution_time > 0.1 and result_count < 10:
            analysis["optimizations"].append({
                "type": "query_efficiency",
                "issue": "High execution time for small result set",
                "suggestion": "Check for N+1 query pattern or missing indexes",
                "priority": "high"
            })
        
        # Suggest indexes based on WHERE clauses
        if "where" in query_lower:
            where_conditions = self._extract_where_conditions(query)
            for condition in where_conditions:
                analysis["index_suggestions"].append({
                    "column": condition,
                    "type": "single_column_index",
                    "reason": "Used in WHERE clause"
                })
        
        return analysis
    
    def _rate_performance(self, execution_time: float, result_count: int) -> str:
        """Rate query performance"""
        if execution_time < 0.1:
            return "excellent"
        elif execution_time < 0.5:
            return "good"
        elif execution_time < 1.0:
            return "fair"
        elif execution_time < 2.0:
            return "poor"
        else:
            return "critical"
    
    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract column names from WHERE conditions"""
        # Simplified extraction - in production, use proper SQL parser
        conditions = []
        query_lower = query.lower()
        
        if "where" in query_lower:
            where_part = query_lower.split("where")[1].split("order by")[0].split("group by")[0]
            # Extract column names (simplified)
            import re
            columns = re.findall(r'(\w+)\s*[=<>!]', where_part)
            conditions.extend(columns)
        
        return conditions

class ConnectionPoolManager:
    """Advanced connection pool management"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self.pool_stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "connections_active": 0,
            "pool_overflows": 0,
            "connection_errors": 0
        }
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine with optimized settings"""
        try:
            self.engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
                connect_args={
                    "connect_timeout": self.config.query_timeout,
                    "command_timeout": self.config.query_timeout
                }
            )
            
            # Setup event listeners for monitoring
            self._setup_event_listeners()
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            logger.info("Database engine initialized with optimized settings")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring"""
        
        @event.listens_for(self.engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            self.pool_stats["connections_created"] += 1
            logger.debug("Database connection created")
        
        @event.listens_for(self.engine, "close")
        def on_close(dbapi_connection, connection_record):
            self.pool_stats["connections_closed"] += 1
            logger.debug("Database connection closed")
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            execution_time = time.time() - context._query_start_time
            
            if self.config.enable_slow_query_logging and execution_time > self.config.slow_query_threshold:
                logger.warning(f"Slow query detected: {execution_time:.3f}s - {statement[:200]}...")
    
    @contextmanager
    def get_session(self):
        """Get database session with proper cleanup"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status"""
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "stats": self.pool_stats
        }

class DatabaseOptimizer:
    """Main database optimizer class"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool_manager = ConnectionPoolManager(config)
        self.query_optimizer = QueryOptimizer()
        self.metrics = QueryMetrics()
        self.query_cache = {}
        self.index_recommendations = []
        
    def execute_query(self, query: str, params: Dict[str, Any] = None, use_cache: bool = True) -> pd.DataFrame:
        """Execute query with optimization and caching"""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(query, params)
            
            # Check cache first
            if use_cache and cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if not self._is_cache_expired(cache_entry):
                    self.metrics.cache_hits += 1
                    logger.debug(f"Query cache hit: {cache_key}")
                    return cache_entry["data"]
                else:
                    del self.query_cache[cache_key]
            
            # Execute query
            with self.pool_manager.get_session() as session:
                if params:
                    result = session.execute(text(query), params)
                else:
                    result = session.execute(text(query))
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            execution_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(execution_time, len(df))
            
            # Analyze query performance
            analysis = self.query_optimizer.analyze_query(query, execution_time, len(df))
            if analysis["optimizations"]:
                logger.info(f"Query optimization suggestions available for: {query[:50]}...")
            
            # Cache result if appropriate
            if use_cache and self._should_cache_query(query, execution_time, len(df)):
                self.query_cache[cache_key] = {
                    "data": df,
                    "timestamp": datetime.now(),
                    "ttl": self._get_cache_ttl(query)
                }
                self.metrics.cache_misses += 1
            
            return df
            
        except Exception as e:
            self.metrics.failed_queries += 1
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def execute_query_async(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute query asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_query, query, params)
    
    def _generate_cache_key(self, query: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key for query"""
        import hashlib
        key_data = query
        if params:
            key_data += str(sorted(params.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - cache_entry["timestamp"] > timedelta(seconds=cache_entry["ttl"])
    
    def _should_cache_query(self, query: str, execution_time: float, result_count: int) -> bool:
        """Determine if query result should be cached"""
        # Cache expensive queries with reasonable result sizes
        return (
            execution_time > 0.5 and  # Expensive queries
            result_count < 10000 and  # Reasonable result size
            "select" in query.lower() and  # Only SELECT queries
            "now()" not in query.lower() and  # Avoid time-dependent queries
            "current_timestamp" not in query.lower()
        )
    
    def _get_cache_ttl(self, query: str) -> int:
        """Get cache TTL based on query type"""
        query_lower = query.lower()
        
        if any(table in query_lower for table in ["temperaturelogs", "wasteevents"]):
            return 300  # 5 minutes for frequently changing data
        elif any(table in query_lower for table in ["products", "warehouses"]):
            return 3600  # 1 hour for relatively static data
        else:
            return 900  # 15 minutes default
    
    def _update_metrics(self, execution_time: float, result_count: int):
        """Update query metrics"""
        self.metrics.query_count += 1
        self.metrics.total_execution_time += execution_time
        self.metrics.avg_execution_time = self.metrics.total_execution_time / self.metrics.query_count
        
        if execution_time > self.config.slow_query_threshold:
            self.metrics.slow_queries += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        pool_status = self.pool_manager.get_pool_status()
        
        return {
            "query_metrics": {
                "total_queries": self.metrics.query_count,
                "avg_execution_time": self.metrics.avg_execution_time,
                "slow_queries": self.metrics.slow_queries,
                "failed_queries": self.metrics.failed_queries,
                "cache_hit_rate": self.metrics.cache_hits / max(self.metrics.query_count, 1),
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses
            },
            "connection_pool": pool_status,
            "cache_status": {
                "cached_queries": len(self.query_cache),
                "cache_memory_usage": self._estimate_cache_memory()
            }
        }
    
    def _estimate_cache_memory(self) -> int:
        """Estimate cache memory usage in bytes"""
        import sys
        total_size = 0
        for cache_entry in self.query_cache.values():
            total_size += sys.getsizeof(cache_entry["data"])
        return total_size
    
    def optimize_database_schema(self) -> List[Dict[str, Any]]:
        """Analyze and suggest database schema optimizations"""
        recommendations = []
        
        try:
            with self.pool_manager.get_session() as session:
                # Get table information
                tables_query = """
                    SELECT TABLE_NAME, TABLE_ROWS, DATA_LENGTH, INDEX_LENGTH
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                """
                
                try:
                    tables_df = pd.read_sql(tables_query, session.bind)
                    
                    for _, table in tables_df.iterrows():
                        table_name = table['TABLE_NAME']
                        
                        # Check for missing indexes
                        missing_indexes = self._analyze_missing_indexes(session, table_name)
                        recommendations.extend(missing_indexes)
                        
                        # Check for unused indexes
                        unused_indexes = self._analyze_unused_indexes(session, table_name)
                        recommendations.extend(unused_indexes)
                        
                        # Check table size and suggest partitioning
                        if table['TABLE_ROWS'] > 1000000:  # 1M rows
                            recommendations.append({
                                "type": "partitioning",
                                "table": table_name,
                                "issue": f"Large table with {table['TABLE_ROWS']} rows",
                                "suggestion": "Consider table partitioning for better performance",
                                "priority": "medium"
                            })
                
                except Exception as e:
                    logger.warning(f"Could not analyze schema (might be SQLite): {e}")
                    # Provide general recommendations for SQLite or other databases
                    recommendations.extend(self._get_general_recommendations())
        
        except Exception as e:
            logger.error(f"Schema optimization analysis failed: {e}")
        
        return recommendations
    
    def _analyze_missing_indexes(self, session: Session, table_name: str) -> List[Dict[str, Any]]:
        """Analyze missing indexes for a table"""
        recommendations = []
        
        # This is a simplified analysis - in production, use query plan analysis
        common_index_columns = {
            "temperaturelogs": ["WarehouseID", "LogTime"],
            "wasteevents": ["ProductID", "WarehouseID", "EventDate"],
            "products": ["Category"],
            "warehouses": ["Location"]
        }
        
        if table_name.lower() in common_index_columns:
            for column in common_index_columns[table_name.lower()]:
                recommendations.append({
                    "type": "missing_index",
                    "table": table_name,
                    "column": column,
                    "suggestion": f"Consider adding index on {column} for better query performance",
                    "priority": "medium"
                })
        
        return recommendations
    
    def _analyze_unused_indexes(self, session: Session, table_name: str) -> List[Dict[str, Any]]:
        """Analyze unused indexes for a table"""
        # This would require database-specific queries to check index usage statistics
        # Placeholder implementation
        return []
    
    def _get_general_recommendations(self) -> List[Dict[str, Any]]:
        """Get general database optimization recommendations"""
        return [
            {
                "type": "general",
                "suggestion": "Ensure frequently queried columns have appropriate indexes",
                "priority": "high"
            },
            {
                "type": "general",
                "suggestion": "Use EXPLAIN/ANALYZE to identify slow queries",
                "priority": "medium"
            },
            {
                "type": "general",
                "suggestion": "Consider query result caching for expensive operations",
                "priority": "medium"
            },
            {
                "type": "general",
                "suggestion": "Monitor connection pool usage and adjust settings as needed",
                "priority": "low"
            }
        ]
    
    def clear_query_cache(self):
        """Clear query result cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of slow queries for analysis"""
        # This would typically come from database logs or monitoring
        # Placeholder implementation
        return [
            {
                "query": "SELECT * FROM large_table WHERE condition",
                "avg_execution_time": 2.5,
                "execution_count": 150,
                "last_executed": datetime.now() - timedelta(minutes=30)
            }
        ]

# Global database optimizer instance
db_optimizer = None

def initialize_database_optimizer(config: DatabaseConfig) -> DatabaseOptimizer:
    """Initialize global database optimizer"""
    global db_optimizer
    db_optimizer = DatabaseOptimizer(config)
    return db_optimizer

def get_database_optimizer() -> DatabaseOptimizer:
    """Get global database optimizer instance"""
    return db_optimizer
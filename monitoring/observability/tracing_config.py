"""
Enhanced Distributed Tracing Configuration for Fresh Supply Chain Intelligence System
OpenTelemetry integration with Jaeger for comprehensive observability
"""

import os
import logging
from typing import Dict, Any, Optional
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.propagators.composite import CompositeHTTPPropagator
import structlog

logger = structlog.get_logger()

class TracingConfig:
    """Configuration for distributed tracing"""
    
    def __init__(self):
        self.service_name = os.getenv("SERVICE_NAME", "fresh-supply-chain-api")
        self.service_version = os.getenv("SERVICE_VERSION", "2.0.0")
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
        self.jaeger_agent_host = os.getenv("JAEGER_AGENT_HOST", "localhost")
        self.jaeger_agent_port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
        self.sampling_rate = float(os.getenv("TRACING_SAMPLING_RATE", "0.1"))
        self.enable_console_export = os.getenv("ENABLE_CONSOLE_TRACING", "false").lower() == "true"
        
    def get_resource(self) -> Resource:
        """Create resource with service information"""
        return Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
            "service.environment": self.environment,
            "service.instance.id": os.getenv("HOSTNAME", "unknown"),
            "deployment.environment": self.environment
        })

class TracingManager:
    """Manages distributed tracing setup and configuration"""
    
    def __init__(self, config: TracingConfig = None):
        self.config = config or TracingConfig()
        self.tracer_provider = None
        self.tracer = None
        
    def setup_tracing(self):
        """Initialize distributed tracing"""
        try:
            # Create resource
            resource = self.config.get_resource()
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self.tracer_provider)
            
            # Setup Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_agent_host,
                agent_port=self.config.jaeger_agent_port,
                collector_endpoint=self.config.jaeger_endpoint
            )
            
            # Add batch span processor
            span_processor = BatchSpanProcessor(
                jaeger_exporter,
                max_queue_size=2048,
                schedule_delay_millis=5000,
                export_timeout_millis=30000,
                max_export_batch_size=512
            )
            self.tracer_provider.add_span_processor(span_processor)
            
            # Add console exporter for debugging
            if self.config.enable_console_export:
                console_exporter = ConsoleSpanExporter()
                console_processor = BatchSpanProcessor(console_exporter)
                self.tracer_provider.add_span_processor(console_processor)
            
            # Setup propagators
            set_global_textmap(
                CompositeHTTPPropagator([
                    JaegerPropagator(),
                    B3MultiFormat()
                ])
            )
            
            # Get tracer
            self.tracer = trace.get_tracer(
                __name__,
                version=self.config.service_version
            )
            
            logger.info(
                "Distributed tracing initialized",
                service_name=self.config.service_name,
                jaeger_endpoint=self.config.jaeger_endpoint,
                sampling_rate=self.config.sampling_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            raise
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application"""
        try:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=self.tracer_provider,
                excluded_urls="health,metrics,docs,openapi.json"
            )
            logger.info("FastAPI instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
    
    def instrument_requests(self):
        """Instrument requests library"""
        try:
            RequestsInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
            logger.info("Requests instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument requests: {e}")
    
    def instrument_sqlalchemy(self, engine):
        """Instrument SQLAlchemy"""
        try:
            SQLAlchemyInstrumentor().instrument(
                engine=engine,
                tracer_provider=self.tracer_provider,
                enable_commenter=True,
                commenter_options={"db_driver": True, "dbapi_level": True}
            )
            logger.info("SQLAlchemy instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument SQLAlchemy: {e}")
    
    def instrument_redis(self):
        """Instrument Redis"""
        try:
            RedisInstrumentor().instrument(
                tracer_provider=self.tracer_provider
            )
            logger.info("Redis instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument Redis: {e}")
    
    def create_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a new span with optional attributes"""
        if not self.tracer:
            return trace.NoOpSpan()
        
        span = self.tracer.start_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return span
    
    def add_span_attributes(self, span, attributes: Dict[str, Any]):
        """Add attributes to an existing span"""
        if span and attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
    
    def record_exception(self, span, exception: Exception):
        """Record an exception in a span"""
        if span:
            span.record_exception(exception)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))

class BusinessMetricsTracer:
    """Custom tracer for business-specific metrics and spans"""
    
    def __init__(self, tracing_manager: TracingManager):
        self.tracing_manager = tracing_manager
        self.tracer = tracing_manager.tracer
    
    def trace_prediction_request(self, model_type: str, input_data: Dict[str, Any]):
        """Trace ML model prediction requests"""
        span_name = f"ml.prediction.{model_type}"
        attributes = {
            "ml.model.type": model_type,
            "ml.input.size": len(str(input_data)),
            "ml.request.timestamp": trace.time_ns()
        }
        
        return self.tracing_manager.create_span(span_name, attributes)
    
    def trace_quality_assessment(self, image_path: str, product_id: Optional[int] = None):
        """Trace quality assessment operations"""
        attributes = {
            "quality.image.path": image_path,
            "quality.product.id": product_id or "unknown",
            "quality.assessment.type": "computer_vision"
        }
        
        return self.tracing_manager.create_span("quality.assessment", attributes)
    
    def trace_demand_forecast(self, product_id: int, warehouse_id: int, horizon_days: int):
        """Trace demand forecasting operations"""
        attributes = {
            "forecast.product.id": product_id,
            "forecast.warehouse.id": warehouse_id,
            "forecast.horizon.days": horizon_days,
            "forecast.type": "temporal_fusion_transformer"
        }
        
        return self.tracing_manager.create_span("demand.forecast", attributes)
    
    def trace_route_optimization(self, warehouse_count: int, product_count: int):
        """Trace route optimization operations"""
        attributes = {
            "optimization.warehouses": warehouse_count,
            "optimization.products": product_count,
            "optimization.algorithm": "gnn_based"
        }
        
        return self.tracing_manager.create_span("route.optimization", attributes)
    
    def trace_data_processing(self, operation: str, record_count: int, processing_time: float):
        """Trace data processing operations"""
        attributes = {
            "data.operation": operation,
            "data.record.count": record_count,
            "data.processing.time.ms": processing_time * 1000,
            "data.pipeline.stage": "processing"
        }
        
        return self.tracing_manager.create_span(f"data.{operation}", attributes)
    
    def trace_cache_operation(self, operation: str, key: str, hit: bool = None):
        """Trace cache operations"""
        attributes = {
            "cache.operation": operation,
            "cache.key": key[:50],  # Truncate long keys
            "cache.backend": "redis"
        }
        
        if hit is not None:
            attributes["cache.hit"] = hit
        
        return self.tracing_manager.create_span(f"cache.{operation}", attributes)
    
    def trace_database_query(self, query_type: str, table: str, duration: float = None):
        """Trace database operations"""
        attributes = {
            "db.operation": query_type,
            "db.table": table,
            "db.system": "mssql"
        }
        
        if duration:
            attributes["db.duration.ms"] = duration * 1000
        
        return self.tracing_manager.create_span(f"db.{query_type}", attributes)

class PerformanceTracer:
    """Tracer for performance monitoring and optimization"""
    
    def __init__(self, tracing_manager: TracingManager):
        self.tracing_manager = tracing_manager
        self.tracer = tracing_manager.tracer
    
    def trace_api_endpoint(self, method: str, endpoint: str, user_id: str = None):
        """Trace API endpoint calls"""
        attributes = {
            "http.method": method,
            "http.route": endpoint,
            "api.version": "v2"
        }
        
        if user_id:
            attributes["user.id"] = user_id
        
        return self.tracing_manager.create_span(f"api.{method.lower()}.{endpoint}", attributes)
    
    def trace_model_training(self, model_type: str, dataset_size: int, epochs: int):
        """Trace model training operations"""
        attributes = {
            "ml.training.model": model_type,
            "ml.training.dataset.size": dataset_size,
            "ml.training.epochs": epochs,
            "ml.training.stage": "training"
        }
        
        return self.tracing_manager.create_span(f"ml.training.{model_type}", attributes)
    
    def trace_feature_engineering(self, feature_count: int, processing_time: float):
        """Trace feature engineering operations"""
        attributes = {
            "feature.count": feature_count,
            "feature.processing.time.ms": processing_time * 1000,
            "feature.pipeline.stage": "engineering"
        }
        
        return self.tracing_manager.create_span("feature.engineering", attributes)

# Global tracing manager instance
tracing_manager = TracingManager()
business_tracer = BusinessMetricsTracer(tracing_manager)
performance_tracer = PerformanceTracer(tracing_manager)

def initialize_tracing(app=None, engine=None):
    """Initialize tracing for the application"""
    try:
        # Setup basic tracing
        tracing_manager.setup_tracing()
        
        # Instrument FastAPI if provided
        if app:
            tracing_manager.instrument_fastapi(app)
        
        # Instrument SQLAlchemy if provided
        if engine:
            tracing_manager.instrument_sqlalchemy(engine)
        
        # Instrument other libraries
        tracing_manager.instrument_requests()
        tracing_manager.instrument_redis()
        
        logger.info("Tracing initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        raise

def get_tracer(name: str = __name__):
    """Get a tracer instance"""
    return trace.get_tracer(name)

def get_current_span():
    """Get the current active span"""
    return trace.get_current_span()

def add_span_event(name: str, attributes: Dict[str, Any] = None):
    """Add an event to the current span"""
    span = get_current_span()
    if span:
        span.add_event(name, attributes or {})

def set_span_attribute(key: str, value: Any):
    """Set an attribute on the current span"""
    span = get_current_span()
    if span:
        span.set_attribute(key, value)

# Decorators for automatic tracing
def trace_function(operation_name: str = None, attributes: Dict[str, Any] = None):
    """Decorator to automatically trace function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracing_manager.create_span(span_name, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    tracing_manager.record_exception(span, e)
                    raise
        
        return wrapper
    return decorator

def trace_async_function(operation_name: str = None, attributes: Dict[str, Any] = None):
    """Decorator to automatically trace async function calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracing_manager.create_span(span_name, attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    tracing_manager.record_exception(span, e)
                    raise
        
        return wrapper
    return decorator
"""
Enhanced Metrics Collection System for Fresh Supply Chain Intelligence
Custom metrics for business KPIs, ML model performance, and operational insights
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import json
import redis
import pandas as pd
from sqlalchemy import create_engine, text
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
import structlog

logger = structlog.get_logger()

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"

@dataclass
class MetricDefinition:
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None

class EnhancedMetricsCollector:
    """Enhanced metrics collector for comprehensive monitoring"""
    
    def __init__(self, registry: CollectorRegistry = None, redis_client=None, database_engine=None):
        self.registry = registry or CollectorRegistry()
        self.redis_client = redis_client
        self.database_engine = database_engine
        
        # Metric storage
        self.metrics = {}
        self.business_metrics = {}
        self.ml_metrics = {}
        self.operational_metrics = {}
        
        # Background collection
        self.collection_interval = 30  # seconds
        self.collection_thread = None
        self.running = False
        
        # Initialize metrics
        self._initialize_api_metrics()
        self._initialize_business_metrics()
        self._initialize_ml_metrics()
        self._initialize_operational_metrics()
        
    def _initialize_api_metrics(self):
        """Initialize API-related metrics"""
        
        # Request metrics
        self.metrics['api_requests_total'] = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['api_request_duration_seconds'] = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Authentication metrics
        self.metrics['api_authentication_attempts_total'] = Counter(
            'api_authentication_attempts_total',
            'Total authentication attempts',
            ['method', 'result'],
            registry=self.registry
        )
        
        self.metrics['api_authentication_failures_total'] = Counter(
            'api_authentication_failures_total',
            'Total authentication failures',
            ['reason'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.metrics['api_rate_limit_exceeded_total'] = Counter(
            'api_rate_limit_exceeded_total',
            'Total rate limit exceeded events',
            ['client_type', 'endpoint'],
            registry=self.registry
        )
        
        # Cache metrics
        self.metrics['api_cache_requests_total'] = Counter(
            'api_cache_requests_total',
            'Total cache requests',
            ['cache_type', 'operation'],
            registry=self.registry
        )
        
        self.metrics['api_cache_hits_total'] = Counter(
            'api_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.metrics['api_cache_misses_total'] = Counter(
            'api_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Database metrics
        self.metrics['api_database_connections_active'] = Gauge(
            'api_database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.metrics['api_database_connections_max'] = Gauge(
            'api_database_connections_max',
            'Maximum database connections',
            registry=self.registry
        )
        
        self.metrics['api_database_query_duration_seconds'] = Histogram(
            'api_database_query_duration_seconds',
            'Database query duration',
            ['operation', 'table'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            registry=self.registry
        )
    
    def _initialize_business_metrics(self):
        """Initialize business KPI metrics"""
        
        # Supply chain performance
        self.business_metrics['business_otif_rate'] = Gauge(
            'business_otif_rate',
            'On-Time In-Full delivery rate',
            ['warehouse', 'product_category'],
            registry=self.registry
        )
        
        self.business_metrics['business_temperature_compliance_rate'] = Gauge(
            'business_temperature_compliance_rate',
            'Temperature compliance rate',
            ['warehouse'],
            registry=self.registry
        )
        
        self.business_metrics['business_waste_reduction_percentage'] = Gauge(
            'business_waste_reduction_percentage',
            'Waste reduction percentage',
            ['warehouse', 'product_category'],
            registry=self.registry
        )
        
        # Financial metrics
        self.business_metrics['business_monthly_cost_savings'] = Gauge(
            'business_monthly_cost_savings',
            'Monthly cost savings in NOK',
            ['category'],
            registry=self.registry
        )
        
        self.business_metrics['business_roi_percentage'] = Gauge(
            'business_roi_percentage',
            'Return on Investment percentage',
            ['initiative'],
            registry=self.registry
        )
        
        self.business_metrics['business_monthly_revenue'] = Gauge(
            'business_monthly_revenue',
            'Monthly revenue in NOK',
            ['warehouse', 'product_category'],
            registry=self.registry
        )
        
        # Quality metrics
        self.business_metrics['business_average_quality_score'] = Gauge(
            'business_average_quality_score',
            'Average quality score',
            ['warehouse', 'product_category'],
            registry=self.registry
        )
        
        self.business_metrics['business_quality_rejections_total'] = Counter(
            'business_quality_rejections_total',
            'Total quality rejections',
            ['warehouse', 'product_category', 'reason'],
            registry=self.registry
        )
        
        # Inventory metrics
        self.business_metrics['business_inventory_turnover_ratio'] = Gauge(
            'business_inventory_turnover_ratio',
            'Inventory turnover ratio',
            ['warehouse', 'product_category'],
            registry=self.registry
        )
        
        self.business_metrics['business_stockout_rate'] = Gauge(
            'business_stockout_rate',
            'Stockout rate',
            ['warehouse', 'product_category'],
            registry=self.registry
        )
        
        # Customer satisfaction
        self.business_metrics['business_customer_satisfaction_score'] = Gauge(
            'business_customer_satisfaction_score',
            'Customer satisfaction score (1-5)',
            ['region'],
            registry=self.registry
        )
        
        self.business_metrics['business_customer_complaints_total'] = Counter(
            'business_customer_complaints_total',
            'Total customer complaints',
            ['category', 'severity'],
            registry=self.registry
        )
        
        # Sustainability metrics
        self.business_metrics['business_carbon_footprint_kg'] = Counter(
            'business_carbon_footprint_kg',
            'Carbon footprint in kg CO2',
            ['source'],
            registry=self.registry
        )
        
        self.business_metrics['business_energy_efficiency_score'] = Gauge(
            'business_energy_efficiency_score',
            'Energy efficiency score (0-1)',
            ['facility'],
            registry=self.registry
        )
    
    def _initialize_ml_metrics(self):
        """Initialize ML model performance metrics"""
        
        # Model accuracy metrics
        self.ml_metrics['model_prediction_accuracy'] = Gauge(
            'model_prediction_accuracy',
            'Model prediction accuracy',
            ['model_type', 'model_version'],
            registry=self.registry
        )
        
        self.ml_metrics['model_prediction_uncertainty'] = Histogram(
            'model_prediction_uncertainty',
            'Model prediction uncertainty',
            ['model_type'],
            buckets=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0],
            registry=self.registry
        )
        
        # Model performance metrics
        self.ml_metrics['model_prediction_duration_seconds'] = Histogram(
            'model_prediction_duration_seconds',
            'Model prediction duration',
            ['model_type', 'batch_size'],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.ml_metrics['model_predictions_total'] = Counter(
            'model_predictions_total',
            'Total model predictions',
            ['model_type', 'prediction_class'],
            registry=self.registry
        )
        
        # Forecasting specific metrics
        self.ml_metrics['model_forecast_mape'] = Gauge(
            'model_forecast_mape',
            'Model forecast Mean Absolute Percentage Error',
            ['model_type', 'horizon_days'],
            registry=self.registry
        )
        
        self.ml_metrics['model_forecast_bias'] = Gauge(
            'model_forecast_bias',
            'Model forecast bias',
            ['model_type', 'product_category'],
            registry=self.registry
        )
        
        # Data drift metrics
        self.ml_metrics['model_data_drift_score'] = Gauge(
            'model_data_drift_score',
            'Model data drift score (0-1)',
            ['model_type', 'feature_group'],
            registry=self.registry
        )
        
        # Training metrics
        self.ml_metrics['model_training_duration_seconds'] = Histogram(
            'model_training_duration_seconds',
            'Model training duration',
            ['model_type'],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800],
            registry=self.registry
        )
        
        self.ml_metrics['model_training_failures_total'] = Counter(
            'model_training_failures_total',
            'Total model training failures',
            ['model_type', 'failure_reason'],
            registry=self.registry
        )
        
        self.ml_metrics['model_last_updated_timestamp'] = Gauge(
            'model_last_updated_timestamp',
            'Timestamp of last model update',
            ['model_type'],
            registry=self.registry
        )
        
        # Feature store metrics
        self.ml_metrics['feature_store_request_duration_seconds'] = Histogram(
            'feature_store_request_duration_seconds',
            'Feature store request duration',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.ml_metrics['feature_quality_score'] = Gauge(
            'feature_quality_score',
            'Feature quality score (0-1)',
            ['feature_name', 'feature_group'],
            registry=self.registry
        )
        
        # Model memory and resource usage
        self.ml_metrics['model_memory_usage_bytes'] = Gauge(
            'model_memory_usage_bytes',
            'Model memory usage in bytes',
            ['model_type'],
            registry=self.registry
        )
        
        self.ml_metrics['model_business_impact_score'] = Gauge(
            'model_business_impact_score',
            'Model business impact score (0-1)',
            ['model_type', 'impact_category'],
            registry=self.registry
        )
    
    def _initialize_operational_metrics(self):
        """Initialize operational metrics"""
        
        # System health
        self.operational_metrics['system_health_score'] = Gauge(
            'system_health_score',
            'Overall system health score (0-1)',
            ['component'],
            registry=self.registry
        )
        
        # Data processing metrics
        self.operational_metrics['data_processing_records_total'] = Counter(
            'data_processing_records_total',
            'Total records processed',
            ['pipeline', 'stage', 'status'],
            registry=self.registry
        )
        
        self.operational_metrics['data_processing_duration_seconds'] = Histogram(
            'data_processing_duration_seconds',
            'Data processing duration',
            ['pipeline', 'stage'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        # Queue metrics
        self.operational_metrics['queue_size'] = Gauge(
            'queue_size',
            'Queue size',
            ['queue_name'],
            registry=self.registry
        )
        
        self.operational_metrics['queue_processing_time_seconds'] = Histogram(
            'queue_processing_time_seconds',
            'Queue processing time',
            ['queue_name'],
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Alert metrics
        self.operational_metrics['alerts_fired_total'] = Counter(
            'alerts_fired_total',
            'Total alerts fired',
            ['alert_name', 'severity', 'component'],
            registry=self.registry
        )
        
        self.operational_metrics['alerts_resolved_total'] = Counter(
            'alerts_resolved_total',
            'Total alerts resolved',
            ['alert_name', 'resolution_method'],
            registry=self.registry
        )
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        self.metrics['api_requests_total'].labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status_code)
        ).inc()
        
        self.metrics['api_request_duration_seconds'].labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def record_authentication_attempt(self, method: str, success: bool, failure_reason: str = None):
        """Record authentication metrics"""
        result = "success" if success else "failure"
        self.metrics['api_authentication_attempts_total'].labels(
            method=method, 
            result=result
        ).inc()
        
        if not success and failure_reason:
            self.metrics['api_authentication_failures_total'].labels(
                reason=failure_reason
            ).inc()
    
    def record_cache_operation(self, cache_type: str, operation: str, hit: bool = None):
        """Record cache operation metrics"""
        self.metrics['api_cache_requests_total'].labels(
            cache_type=cache_type, 
            operation=operation
        ).inc()
        
        if hit is not None:
            if hit:
                self.metrics['api_cache_hits_total'].labels(cache_type=cache_type).inc()
            else:
                self.metrics['api_cache_misses_total'].labels(cache_type=cache_type).inc()
    
    def record_model_prediction(self, model_type: str, duration: float, accuracy: float = None, 
                              uncertainty: float = None, prediction_class: str = None):
        """Record ML model prediction metrics"""
        
        # Record prediction count
        if prediction_class:
            self.ml_metrics['model_predictions_total'].labels(
                model_type=model_type,
                prediction_class=prediction_class
            ).inc()
        
        # Record duration
        self.ml_metrics['model_prediction_duration_seconds'].labels(
            model_type=model_type,
            batch_size="1"  # Default for single predictions
        ).observe(duration)
        
        # Record accuracy if provided
        if accuracy is not None:
            self.ml_metrics['model_prediction_accuracy'].labels(
                model_type=model_type,
                model_version="latest"
            ).set(accuracy)
        
        # Record uncertainty if provided
        if uncertainty is not None:
            self.ml_metrics['model_prediction_uncertainty'].labels(
                model_type=model_type
            ).observe(uncertainty)
    
    def record_business_kpi(self, kpi_name: str, value: float, labels: Dict[str, str] = None):
        """Record business KPI metrics"""
        if kpi_name in self.business_metrics:
            metric = self.business_metrics[kpi_name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def record_data_processing(self, pipeline: str, stage: str, record_count: int, 
                             duration: float, status: str = "success"):
        """Record data processing metrics"""
        self.operational_metrics['data_processing_records_total'].labels(
            pipeline=pipeline,
            stage=stage,
            status=status
        ).inc(record_count)
        
        self.operational_metrics['data_processing_duration_seconds'].labels(
            pipeline=pipeline,
            stage=stage
        ).observe(duration)
    
    def update_system_health(self, component: str, health_score: float):
        """Update system health metrics"""
        self.operational_metrics['system_health_score'].labels(
            component=component
        ).set(health_score)
    
    def start_background_collection(self):
        """Start background metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._background_collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        logger.info("Background metrics collection started")
    
    def stop_background_collection(self):
        """Stop background metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        logger.info("Background metrics collection stopped")
    
    def _background_collection_loop(self):
        """Background collection loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_business_metrics()
                self._collect_ml_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in background metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # Database connection metrics
            if self.database_engine:
                pool = self.database_engine.pool
                self.metrics['api_database_connections_active'].set(pool.checkedout())
                self.metrics['api_database_connections_max'].set(pool.size())
            
            # Redis metrics
            if self.redis_client:
                try:
                    info = self.redis_client.info()
                    # Add Redis-specific metrics here
                except Exception as e:
                    logger.warning(f"Failed to collect Redis metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_business_metrics(self):
        """Collect business KPI metrics from database"""
        if not self.database_engine:
            return
        
        try:
            with self.database_engine.connect() as conn:
                # OTIF Rate calculation
                otif_query = """
                    SELECT 
                        w.WarehouseName,
                        p.Category,
                        AVG(CASE WHEN tl.QualityScore >= 0.8 THEN 1.0 ELSE 0.0 END) as otif_rate
                    FROM TemperatureLogs tl
                    JOIN Warehouses w ON tl.WarehouseID = w.WarehouseID
                    CROSS JOIN Products p
                    WHERE tl.LogTime >= DATEADD(day, -1, GETDATE())
                    GROUP BY w.WarehouseName, p.Category
                """
                
                otif_results = pd.read_sql(otif_query, conn)
                for _, row in otif_results.iterrows():
                    self.business_metrics['business_otif_rate'].labels(
                        warehouse=row['WarehouseName'],
                        product_category=row['Category']
                    ).set(row['otif_rate'])
                
                # Temperature compliance
                temp_query = """
                    SELECT 
                        w.WarehouseName,
                        AVG(CASE WHEN tl.Temperature BETWEEN 2 AND 6 THEN 1.0 ELSE 0.0 END) as compliance_rate
                    FROM TemperatureLogs tl
                    JOIN Warehouses w ON tl.WarehouseID = w.WarehouseID
                    WHERE tl.LogTime >= DATEADD(hour, -1, GETDATE())
                    GROUP BY w.WarehouseName
                """
                
                temp_results = pd.read_sql(temp_query, conn)
                for _, row in temp_results.iterrows():
                    self.business_metrics['business_temperature_compliance_rate'].labels(
                        warehouse=row['WarehouseName']
                    ).set(row['compliance_rate'])
                
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
    
    def _collect_ml_metrics(self):
        """Collect ML model performance metrics"""
        try:
            # Model accuracy tracking (would be updated by model serving)
            current_time = time.time()
            
            # Update model last updated timestamp
            for model_type in ['vision', 'forecasting', 'gnn']:
                self.ml_metrics['model_last_updated_timestamp'].labels(
                    model_type=model_type
                ).set(current_time)
            
        except Exception as e:
            logger.error(f"Error collecting ML metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_count": {
                "api_metrics": len(self.metrics),
                "business_metrics": len(self.business_metrics),
                "ml_metrics": len(self.ml_metrics),
                "operational_metrics": len(self.operational_metrics)
            },
            "collection_status": {
                "background_running": self.running,
                "collection_interval": self.collection_interval
            }
        }
        
        return summary
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get content type for metrics export"""
        return CONTENT_TYPE_LATEST

# Global metrics collector instance
metrics_collector = EnhancedMetricsCollector()

def initialize_metrics(redis_client=None, database_engine=None):
    """Initialize the global metrics collector"""
    global metrics_collector
    metrics_collector = EnhancedMetricsCollector(
        redis_client=redis_client,
        database_engine=database_engine
    )
    metrics_collector.start_background_collection()
    
    logger.info("Enhanced metrics collector initialized")
    
    return metrics_collector

def get_metrics_collector() -> EnhancedMetricsCollector:
    """Get the global metrics collector instance"""
    return metrics_collector
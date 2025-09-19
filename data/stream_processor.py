"""
Real-time Streaming Data Processor for Fresh Supply Chain Intelligence System
Handles real-time IoT data streams, inventory updates, and quality alerts
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import redis
from concurrent.futures import ThreadPoolExecutor
import websockets
from queue import Queue
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    """Base class for streaming events"""
    event_id: str
    timestamp: datetime
    event_type: str
    source: str
    data: Dict[str, Any]

@dataclass
class IoTSensorReading(StreamEvent):
    """IoT sensor reading event"""
    device_id: str
    warehouse_id: int
    temperature: float
    humidity: float
    co2_level: float
    ethylene_level: float
    quality_score: float
    
    def __post_init__(self):
        self.event_type = "iot_reading"

@dataclass
class QualityAlert(StreamEvent):
    """Quality alert event"""
    alert_level: str  # 'warning', 'critical'
    product_id: int
    warehouse_id: int
    lot_number: str
    alert_reason: str
    recommended_action: str
    
    def __post_init__(self):
        self.event_type = "quality_alert"

@dataclass
class InventoryUpdate(StreamEvent):
    """Inventory update event"""
    product_id: int
    warehouse_id: int
    lot_number: str
    quantity_change: int
    new_quantity: int
    update_reason: str
    
    def __post_init__(self):
        self.event_type = "inventory_update"

class StreamProcessor:
    """Real-time stream processing engine"""
    
    def __init__(self, connection_string: str, redis_host: str = 'localhost', redis_port: int = 6379):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        
        # Redis for real-time caching and pub/sub
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
        except:
            logger.warning("Redis not available, using in-memory processing only")
            self.redis_client = None
        
        # Event processing queues
        self.iot_queue = Queue(maxsize=10000)
        self.alert_queue = Queue(maxsize=1000)
        self.inventory_queue = Queue(maxsize=5000)
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'iot_reading': [],
            'quality_alert': [],
            'inventory_update': []
        }
        
        # Processing statistics
        self.stats = {
            'events_processed': 0,
            'alerts_generated': 0,
            'processing_errors': 0,
            'start_time': datetime.now()
        }
        
        # Background processing threads
        self.processing_threads = []
        self.is_running = False
        
        # Quality thresholds
        self.quality_thresholds = {
            'temperature_min': 0,
            'temperature_max': 8,
            'humidity_min': 80,
            'humidity_max': 98,
            'quality_score_min': 0.5,
            'co2_max': 1000,
            'ethylene_max': 0.1
        }
    
    def start_processing(self):
        """Start background processing threads"""
        if self.is_running:
            logger.warning("Stream processor is already running")
            return
        
        self.is_running = True
        logger.info("Starting stream processor...")
        
        # Start processing threads
        threads = [
            threading.Thread(target=self._process_iot_stream, daemon=True),
            threading.Thread(target=self._process_alert_stream, daemon=True),
            threading.Thread(target=self._process_inventory_stream, daemon=True),
            threading.Thread(target=self._monitor_system_health, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"Started {len(threads)} processing threads")
    
    def stop_processing(self):
        """Stop background processing"""
        logger.info("Stopping stream processor...")
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5)
        
        logger.info("Stream processor stopped")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add custom event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added handler for {event_type} events")
    
    def process_iot_reading(self, reading_data: Dict[str, Any]) -> IoTSensorReading:
        """Process incoming IoT sensor reading"""
        try:
            # Create IoT reading event
            reading = IoTSensorReading(
                event_id=f"iot_{reading_data['device_id']}_{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                source=reading_data['device_id'],
                data=reading_data,
                device_id=reading_data['device_id'],
                warehouse_id=reading_data['warehouse_id'],
                temperature=reading_data['temperature'],
                humidity=reading_data['humidity'],
                co2_level=reading_data.get('co2_level', 400),
                ethylene_level=reading_data.get('ethylene_level', 0.01),
                quality_score=reading_data.get('quality_score', 0.8)
            )
            
            # Add to processing queue
            if not self.iot_queue.full():
                self.iot_queue.put(reading)
            else:
                logger.warning("IoT queue is full, dropping reading")
            
            # Real-time quality assessment
            alerts = self._assess_quality_real_time(reading)
            for alert in alerts:
                self.process_quality_alert(alert)
            
            # Cache latest reading
            if self.redis_client:
                cache_key = f"latest_reading:{reading.device_id}"
                self.redis_client.setex(cache_key, 300, json.dumps(asdict(reading), default=str))
            
            return reading
            
        except Exception as e:
            logger.error(f"Error processing IoT reading: {e}")
            self.stats['processing_errors'] += 1
            raise
    
    def process_quality_alert(self, alert_data: Dict[str, Any]) -> QualityAlert:
        """Process quality alert"""
        try:
            alert = QualityAlert(
                event_id=f"alert_{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                source="quality_monitor",
                data=alert_data,
                alert_level=alert_data['alert_level'],
                product_id=alert_data.get('product_id', 0),
                warehouse_id=alert_data['warehouse_id'],
                lot_number=alert_data.get('lot_number', ''),
                alert_reason=alert_data['alert_reason'],
                recommended_action=alert_data['recommended_action']
            )
            
            # Add to alert queue
            if not self.alert_queue.full():
                self.alert_queue.put(alert)
            else:
                logger.warning("Alert queue is full, dropping alert")
            
            # Publish alert to subscribers
            if self.redis_client:
                self.redis_client.publish('quality_alerts', json.dumps(asdict(alert), default=str))
            
            self.stats['alerts_generated'] += 1
            return alert
            
        except Exception as e:
            logger.error(f"Error processing quality alert: {e}")
            self.stats['processing_errors'] += 1
            raise
    
    def process_inventory_update(self, update_data: Dict[str, Any]) -> InventoryUpdate:
        """Process inventory update"""
        try:
            update = InventoryUpdate(
                event_id=f"inv_{update_data['lot_number']}_{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                source="inventory_system",
                data=update_data,
                product_id=update_data['product_id'],
                warehouse_id=update_data['warehouse_id'],
                lot_number=update_data['lot_number'],
                quantity_change=update_data['quantity_change'],
                new_quantity=update_data['new_quantity'],
                update_reason=update_data.get('update_reason', 'manual_update')
            )
            
            # Add to inventory queue
            if not self.inventory_queue.full():
                self.inventory_queue.put(update)
            else:
                logger.warning("Inventory queue is full, dropping update")
            
            # Check for low inventory alerts
            if update.new_quantity <= 10:  # Low inventory threshold
                low_inventory_alert = {
                    'alert_level': 'warning',
                    'warehouse_id': update.warehouse_id,
                    'product_id': update.product_id,
                    'lot_number': update.lot_number,
                    'alert_reason': f'Low inventory: {update.new_quantity} units remaining',
                    'recommended_action': 'Reorder product or transfer from other warehouses'
                }
                self.process_quality_alert(low_inventory_alert)
            
            return update
            
        except Exception as e:
            logger.error(f"Error processing inventory update: {e}")
            self.stats['processing_errors'] += 1
            raise
    
    def _process_iot_stream(self):
        """Background thread for processing IoT readings"""
        logger.info("Started IoT stream processing thread")
        
        batch_size = 100
        batch_timeout = 5  # seconds
        batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                # Get reading from queue with timeout
                try:
                    reading = self.iot_queue.get(timeout=1)
                    batch.append(reading)
                except:
                    reading = None
                
                # Process batch if full or timeout reached
                current_time = time.time()
                should_process_batch = (
                    len(batch) >= batch_size or 
                    (batch and (current_time - last_batch_time) >= batch_timeout)
                )
                
                if should_process_batch:
                    self._process_iot_batch(batch)
                    batch = []
                    last_batch_time = current_time
                
                if reading:
                    # Execute event handlers
                    for handler in self.event_handlers.get('iot_reading', []):
                        try:
                            handler(reading)
                        except Exception as e:
                            logger.error(f"Error in IoT event handler: {e}")
                    
                    self.stats['events_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error in IoT stream processing: {e}")
                time.sleep(1)
        
        # Process remaining batch
        if batch:
            self._process_iot_batch(batch)
        
        logger.info("IoT stream processing thread stopped")
    
    def _process_iot_batch(self, batch: List[IoTSensorReading]):
        """Process batch of IoT readings"""
        if not batch:
            return
        
        try:
            # Convert to DataFrame for batch processing
            batch_data = []
            for reading in batch:
                batch_data.append({
                    'LogTime': reading.timestamp,
                    'DeviceID': reading.device_id,
                    'WarehouseID': reading.warehouse_id,
                    'Temperature': reading.temperature,
                    'Humidity': reading.humidity,
                    'CO2Level': reading.co2_level,
                    'EthyleneLevel': reading.ethylene_level,
                    'QualityScore': reading.quality_score,
                    'Zone': f"Zone_{np.random.randint(1, 4)}"  # Simulate zone assignment
                })
            
            df = pd.DataFrame(batch_data)
            
            # Batch insert to database
            df.to_sql('TemperatureLogs', self.engine, if_exists='append', index=False)
            
            # Update real-time aggregations
            self._update_real_time_aggregations(df)
            
            logger.debug(f"Processed IoT batch of {len(batch)} readings")
            
        except Exception as e:
            logger.error(f"Error processing IoT batch: {e}")
    
    def _process_alert_stream(self):
        """Background thread for processing alerts"""
        logger.info("Started alert stream processing thread")
        
        while self.is_running:
            try:
                # Get alert from queue with timeout
                try:
                    alert = self.alert_queue.get(timeout=1)
                except:
                    continue
                
                # Store alert in database
                alert_data = {
                    'AlertTime': alert.timestamp,
                    'AlertLevel': alert.alert_level,
                    'WarehouseID': alert.warehouse_id,
                    'ProductID': alert.product_id or None,
                    'LotNumber': alert.lot_number or None,
                    'AlertReason': alert.alert_reason,
                    'RecommendedAction': alert.recommended_action,
                    'Status': 'ACTIVE'
                }
                
                df = pd.DataFrame([alert_data])
                df.to_sql('QualityAlerts', self.engine, if_exists='append', index=False)
                
                # Execute event handlers
                for handler in self.event_handlers.get('quality_alert', []):
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in alert event handler: {e}")
                
                logger.info(f"Processed {alert.alert_level} alert: {alert.alert_reason}")
                
            except Exception as e:
                logger.error(f"Error in alert stream processing: {e}")
                time.sleep(1)
        
        logger.info("Alert stream processing thread stopped")
    
    def _process_inventory_stream(self):
        """Background thread for processing inventory updates"""
        logger.info("Started inventory stream processing thread")
        
        while self.is_running:
            try:
                # Get update from queue with timeout
                try:
                    update = self.inventory_queue.get(timeout=1)
                except:
                    continue
                
                # Update inventory in database
                update_query = text("""
                    UPDATE Inventory 
                    SET Quantity = :new_quantity, 
                        LastUpdated = :timestamp,
                        Status = CASE 
                            WHEN :new_quantity <= 0 THEN 'OUT_OF_STOCK'
                            WHEN :new_quantity <= 10 THEN 'LOW_STOCK'
                            ELSE 'IN_STOCK'
                        END
                    WHERE LotNumber = :lot_number 
                    AND ProductID = :product_id 
                    AND WarehouseID = :warehouse_id
                """)
                
                with self.engine.connect() as conn:
                    conn.execute(update_query, {
                        'new_quantity': update.new_quantity,
                        'timestamp': update.timestamp,
                        'lot_number': update.lot_number,
                        'product_id': update.product_id,
                        'warehouse_id': update.warehouse_id
                    })
                    conn.commit()
                
                # Execute event handlers
                for handler in self.event_handlers.get('inventory_update', []):
                    try:
                        handler(update)
                    except Exception as e:
                        logger.error(f"Error in inventory event handler: {e}")
                
                logger.debug(f"Processed inventory update for lot {update.lot_number}")
                
            except Exception as e:
                logger.error(f"Error in inventory stream processing: {e}")
                time.sleep(1)
        
        logger.info("Inventory stream processing thread stopped")
    
    def _assess_quality_real_time(self, reading: IoTSensorReading) -> List[Dict[str, Any]]:
        """Real-time quality assessment based on IoT reading"""
        alerts = []
        
        # Temperature alerts
        if reading.temperature < self.quality_thresholds['temperature_min']:
            alerts.append({
                'alert_level': 'critical',
                'warehouse_id': reading.warehouse_id,
                'alert_reason': f'Temperature too low: {reading.temperature}°C (min: {self.quality_thresholds["temperature_min"]}°C)',
                'recommended_action': 'Check refrigeration system immediately'
            })
        elif reading.temperature > self.quality_thresholds['temperature_max']:
            alerts.append({
                'alert_level': 'critical',
                'warehouse_id': reading.warehouse_id,
                'alert_reason': f'Temperature too high: {reading.temperature}°C (max: {self.quality_thresholds["temperature_max"]}°C)',
                'recommended_action': 'Check cooling system and move products to backup storage'
            })
        
        # Humidity alerts
        if reading.humidity < self.quality_thresholds['humidity_min']:
            alerts.append({
                'alert_level': 'warning',
                'warehouse_id': reading.warehouse_id,
                'alert_reason': f'Humidity too low: {reading.humidity}% (min: {self.quality_thresholds["humidity_min"]}%)',
                'recommended_action': 'Increase humidity to prevent product dehydration'
            })
        elif reading.humidity > self.quality_thresholds['humidity_max']:
            alerts.append({
                'alert_level': 'warning',
                'warehouse_id': reading.warehouse_id,
                'alert_reason': f'Humidity too high: {reading.humidity}% (max: {self.quality_thresholds["humidity_max"]}%)',
                'recommended_action': 'Reduce humidity to prevent mold growth'
            })
        
        # Quality score alerts
        if reading.quality_score < self.quality_thresholds['quality_score_min']:
            alerts.append({
                'alert_level': 'warning',
                'warehouse_id': reading.warehouse_id,
                'alert_reason': f'Low quality score: {reading.quality_score} (min: {self.quality_thresholds["quality_score_min"]})',
                'recommended_action': 'Inspect products and consider early distribution'
            })
        
        # CO2 alerts
        if reading.co2_level > self.quality_thresholds['co2_max']:
            alerts.append({
                'alert_level': 'warning',
                'warehouse_id': reading.warehouse_id,
                'alert_reason': f'High CO2 level: {reading.co2_level}ppm (max: {self.quality_thresholds["co2_max"]}ppm)',
                'recommended_action': 'Improve ventilation system'
            })
        
        # Ethylene alerts
        if reading.ethylene_level > self.quality_thresholds['ethylene_max']:
            alerts.append({
                'alert_level': 'warning',
                'warehouse_id': reading.warehouse_id,
                'alert_reason': f'High ethylene level: {reading.ethylene_level}ppm (max: {self.quality_thresholds["ethylene_max"]}ppm)',
                'recommended_action': 'Separate ethylene-producing fruits from sensitive products'
            })
        
        return alerts
    
    def _update_real_time_aggregations(self, df: pd.DataFrame):
        """Update real-time aggregations and metrics"""
        if self.redis_client is None:
            return
        
        try:
            # Calculate warehouse-level aggregations
            warehouse_stats = df.groupby('WarehouseID').agg({
                'Temperature': ['mean', 'min', 'max'],
                'Humidity': ['mean', 'min', 'max'],
                'QualityScore': 'mean',
                'CO2Level': 'mean',
                'EthyleneLevel': 'mean'
            }).round(2)
            
            # Store in Redis with expiration
            for warehouse_id in warehouse_stats.index:
                stats = warehouse_stats.loc[warehouse_id].to_dict()
                cache_key = f"warehouse_stats:{warehouse_id}"
                self.redis_client.setex(cache_key, 300, json.dumps(stats))
            
            # Update system-wide metrics
            system_stats = {
                'avg_temperature': df['Temperature'].mean(),
                'avg_humidity': df['Humidity'].mean(),
                'avg_quality': df['QualityScore'].mean(),
                'total_readings': len(df),
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.setex('system_stats', 300, json.dumps(system_stats, default=str))
            
        except Exception as e:
            logger.error(f"Error updating real-time aggregations: {e}")
    
    def _monitor_system_health(self):
        """Monitor system health and performance"""
        logger.info("Started system health monitoring thread")
        
        while self.is_running:
            try:
                # Calculate processing rates
                uptime = (datetime.now() - self.stats['start_time']).total_seconds()
                events_per_second = self.stats['events_processed'] / max(uptime, 1)
                
                # Queue health
                queue_health = {
                    'iot_queue_size': self.iot_queue.qsize(),
                    'alert_queue_size': self.alert_queue.qsize(),
                    'inventory_queue_size': self.inventory_queue.qsize(),
                    'events_per_second': round(events_per_second, 2),
                    'total_events': self.stats['events_processed'],
                    'total_alerts': self.stats['alerts_generated'],
                    'processing_errors': self.stats['processing_errors'],
                    'uptime_seconds': uptime
                }
                
                # Store health metrics
                if self.redis_client:
                    self.redis_client.setex('system_health', 60, json.dumps(queue_health))
                
                # Log health status periodically
                if int(uptime) % 300 == 0:  # Every 5 minutes
                    logger.info(f"System health: {queue_health}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")
                time.sleep(10)
        
        logger.info("System health monitoring thread stopped")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time processing statistics"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'events_processed': self.stats['events_processed'],
            'alerts_generated': self.stats['alerts_generated'],
            'processing_errors': self.stats['processing_errors'],
            'events_per_second': self.stats['events_processed'] / max(uptime, 1),
            'queue_sizes': {
                'iot_queue': self.iot_queue.qsize(),
                'alert_queue': self.alert_queue.qsize(),
                'inventory_queue': self.inventory_queue.qsize()
            },
            'is_running': self.is_running
        }
    
    async def websocket_handler(self, websocket, path):
        """WebSocket handler for real-time data streaming"""
        logger.info(f"New WebSocket connection: {path}")
        
        try:
            # Send initial connection message
            await websocket.send(json.dumps({
                'type': 'connection',
                'message': 'Connected to Fresh Supply Chain stream',
                'timestamp': datetime.now().isoformat()
            }))
            
            # Stream real-time data
            while True:
                # Get latest system stats
                stats = self.get_real_time_stats()
                
                # Get latest warehouse data from Redis
                warehouse_data = {}
                if self.redis_client:
                    for key in self.redis_client.scan_iter(match="warehouse_stats:*"):
                        warehouse_id = key.split(':')[1]
                        data = self.redis_client.get(key)
                        if data:
                            warehouse_data[warehouse_id] = json.loads(data)
                
                # Send update
                update = {
                    'type': 'real_time_update',
                    'timestamp': datetime.now().isoformat(),
                    'system_stats': stats,
                    'warehouse_data': warehouse_data
                }
                
                await websocket.send(json.dumps(update, default=str))
                await asyncio.sleep(2)  # Update every 2 seconds
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")

# Usage example and testing
def simulate_iot_data_stream(processor: StreamProcessor, duration_seconds: int = 60):
    """Simulate IoT data stream for testing"""
    logger.info(f"Starting IoT data simulation for {duration_seconds} seconds")
    
    warehouses = [1, 2, 3, 4, 5]
    devices_per_warehouse = 3
    
    start_time = time.time()
    
    while (time.time() - start_time) < duration_seconds:
        for warehouse_id in warehouses:
            for device_num in range(1, devices_per_warehouse + 1):
                # Generate realistic sensor data
                base_temp = 4.0 + np.random.normal(0, 0.5)
                
                # Occasionally simulate temperature violations
                if np.random.random() < 0.05:  # 5% chance
                    base_temp += np.random.uniform(5, 10)
                
                reading_data = {
                    'device_id': f'SENSOR_WH{warehouse_id:02d}_{device_num:02d}',
                    'warehouse_id': warehouse_id,
                    'temperature': round(base_temp, 2),
                    'humidity': round(np.random.uniform(85, 95), 2),
                    'co2_level': round(np.random.uniform(400, 500), 2),
                    'ethylene_level': round(np.random.uniform(0.01, 0.05), 4),
                    'quality_score': round(max(0, min(1, 0.9 - abs(base_temp - 4) * 0.1)), 2)
                }
                
                processor.process_iot_reading(reading_data)
        
        time.sleep(1)  # Generate data every second
    
    logger.info("IoT data simulation completed")

if __name__ == "__main__":
    # Example usage
    from config.database_config import get_connection_string
    
    # Create stream processor
    processor = StreamProcessor(get_connection_string())
    
    # Add custom event handlers
    def log_critical_alerts(alert: QualityAlert):
        if alert.alert_level == 'critical':
            print(f"🚨 CRITICAL ALERT: {alert.alert_reason}")
    
    processor.add_event_handler('quality_alert', log_critical_alerts)
    
    # Start processing
    processor.start_processing()
    
    try:
        # Simulate data stream
        simulate_iot_data_stream(processor, 30)
        
        # Print statistics
        stats = processor.get_real_time_stats()
        print(f"Processing Statistics: {stats}")
        
    finally:
        processor.stop_processing()
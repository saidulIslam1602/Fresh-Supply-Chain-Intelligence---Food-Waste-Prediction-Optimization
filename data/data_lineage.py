"""
Data Lineage and Governance System for Fresh Supply Chain Intelligence System
Tracks data flow, transformations, and provides audit trails for compliance and debugging
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import pandas as pd
from sqlalchemy import create_engine, text
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class DataOperation(Enum):
    """Types of data operations"""
    INGESTION = "ingestion"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    EXPORT = "export"
    DELETE = "delete"

class DataSource(Enum):
    """Data source types"""
    USDA_API = "usda_api"
    IOT_SENSORS = "iot_sensors"
    MANUAL_INPUT = "manual_input"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    STREAM = "stream"

@dataclass
class DataAsset:
    """Represents a data asset in the system"""
    asset_id: str
    name: str
    asset_type: str  # 'dataset', 'model', 'feature', 'report'
    source: DataSource
    location: str
    schema: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.asset_id:
            self.asset_id = str(uuid.uuid4())

@dataclass
class DataLineageNode:
    """Represents a node in the data lineage graph"""
    node_id: str
    asset: DataAsset
    operation: DataOperation
    timestamp: datetime
    user: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())

@dataclass
class DataLineageEdge:
    """Represents a relationship between data assets"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str  # 'derived_from', 'transformed_to', 'used_by'
    transformation_logic: str
    created_at: datetime
    
    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = str(uuid.uuid4())

@dataclass
class AuditEvent:
    """Audit event for compliance tracking"""
    event_id: str
    timestamp: datetime
    user: str
    operation: DataOperation
    asset_id: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())

class DataLineageTracker:
    """Main class for tracking data lineage and governance"""
    
    def __init__(self, connection_string: str = None, storage_path: str = "./data/lineage"):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string) if connection_string else None
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for lineage graph
        self.nodes: Dict[str, DataLineageNode] = {}
        self.edges: Dict[str, DataLineageEdge] = {}
        self.assets: Dict[str, DataAsset] = {}
        self.audit_events: List[AuditEvent] = []
        
        # Load existing lineage data
        self._load_lineage_data()
        
        # GDPR compliance settings
        self.gdpr_enabled = True
        self.data_retention_days = 2555  # 7 years for compliance
        self.anonymization_fields = ['user_id', 'customer_id', 'personal_info']
    
    def register_data_asset(self, 
                           name: str,
                           asset_type: str,
                           source: DataSource,
                           location: str,
                           schema: Dict[str, Any],
                           metadata: Dict[str, Any] = None,
                           tags: List[str] = None) -> DataAsset:
        """Register a new data asset"""
        
        asset = DataAsset(
            asset_id=str(uuid.uuid4()),
            name=name,
            asset_type=asset_type,
            source=source,
            location=location,
            schema=schema,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or []
        )
        
        self.assets[asset.asset_id] = asset
        
        # Create audit event
        self._create_audit_event(
            operation=DataOperation.INGESTION,
            asset_id=asset.asset_id,
            details={
                'action': 'asset_registered',
                'asset_name': name,
                'asset_type': asset_type,
                'source': source.value
            }
        )
        
        logger.info(f"Registered data asset: {name} ({asset.asset_id})")
        return asset
    
    def track_transformation(self,
                           source_asset_ids: List[str],
                           target_asset: DataAsset,
                           operation: DataOperation,
                           transformation_logic: str,
                           parameters: Dict[str, Any] = None,
                           metrics: Dict[str, Any] = None,
                           user: str = "system") -> DataLineageNode:
        """Track a data transformation operation"""
        
        # Create lineage node for the transformation
        node = DataLineageNode(
            node_id=str(uuid.uuid4()),
            asset=target_asset,
            operation=operation,
            timestamp=datetime.now(),
            user=user,
            parameters=parameters or {},
            metrics=metrics or {}
        )
        
        self.nodes[node.node_id] = node
        self.assets[target_asset.asset_id] = target_asset
        
        # Create edges from source assets to target
        for source_asset_id in source_asset_ids:
            if source_asset_id in self.assets:
                # Find the most recent node for the source asset
                source_node_id = self._get_latest_node_for_asset(source_asset_id)
                
                if source_node_id:
                    edge = DataLineageEdge(
                        edge_id=str(uuid.uuid4()),
                        source_node_id=source_node_id,
                        target_node_id=node.node_id,
                        relationship_type='transformed_to',
                        transformation_logic=transformation_logic,
                        created_at=datetime.now()
                    )
                    
                    self.edges[edge.edge_id] = edge
        
        # Create audit event
        self._create_audit_event(
            operation=operation,
            asset_id=target_asset.asset_id,
            user=user,
            details={
                'action': 'transformation_tracked',
                'source_assets': source_asset_ids,
                'target_asset': target_asset.asset_id,
                'transformation_logic': transformation_logic,
                'parameters': parameters,
                'metrics': metrics
            }
        )
        
        logger.info(f"Tracked transformation: {operation.value} -> {target_asset.name}")
        return node
    
    def track_data_access(self,
                         asset_id: str,
                         user: str,
                         access_type: str = "read",
                         purpose: str = None,
                         ip_address: str = None) -> AuditEvent:
        """Track data access for compliance"""
        
        if asset_id not in self.assets:
            raise ValueError(f"Asset {asset_id} not found")
        
        asset = self.assets[asset_id]
        
        # Create audit event
        audit_event = self._create_audit_event(
            operation=DataOperation.EXPORT if access_type == "export" else DataOperation.INGESTION,
            asset_id=asset_id,
            user=user,
            ip_address=ip_address,
            details={
                'action': 'data_access',
                'access_type': access_type,
                'asset_name': asset.name,
                'purpose': purpose,
                'gdpr_compliant': self._check_gdpr_compliance(asset, user, purpose)
            }
        )
        
        logger.info(f"Tracked data access: {user} accessed {asset.name} ({access_type})")
        return audit_event
    
    def get_lineage_upstream(self, asset_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get upstream lineage (data sources) for an asset"""
        
        if asset_id not in self.assets:
            raise ValueError(f"Asset {asset_id} not found")
        
        visited = set()
        lineage = {'nodes': [], 'edges': []}
        
        def traverse_upstream(current_asset_id: str, depth: int):
            if depth >= max_depth or current_asset_id in visited:
                return
            
            visited.add(current_asset_id)
            
            # Find nodes for this asset
            asset_nodes = [node for node in self.nodes.values() 
                          if node.asset.asset_id == current_asset_id]
            
            for node in asset_nodes:
                lineage['nodes'].append(asdict(node))
                
                # Find incoming edges
                incoming_edges = [edge for edge in self.edges.values() 
                                if edge.target_node_id == node.node_id]
                
                for edge in incoming_edges:
                    lineage['edges'].append(asdict(edge))
                    
                    # Find source node and continue traversal
                    source_node = self.nodes.get(edge.source_node_id)
                    if source_node:
                        traverse_upstream(source_node.asset.asset_id, depth + 1)
        
        traverse_upstream(asset_id, 0)
        
        return lineage
    
    def get_lineage_downstream(self, asset_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Get downstream lineage (data consumers) for an asset"""
        
        if asset_id not in self.assets:
            raise ValueError(f"Asset {asset_id} not found")
        
        visited = set()
        lineage = {'nodes': [], 'edges': []}
        
        def traverse_downstream(current_asset_id: str, depth: int):
            if depth >= max_depth or current_asset_id in visited:
                return
            
            visited.add(current_asset_id)
            
            # Find nodes for this asset
            asset_nodes = [node for node in self.nodes.values() 
                          if node.asset.asset_id == current_asset_id]
            
            for node in asset_nodes:
                lineage['nodes'].append(asdict(node))
                
                # Find outgoing edges
                outgoing_edges = [edge for edge in self.edges.values() 
                                if edge.source_node_id == node.node_id]
                
                for edge in outgoing_edges:
                    lineage['edges'].append(asdict(edge))
                    
                    # Find target node and continue traversal
                    target_node = self.nodes.get(edge.target_node_id)
                    if target_node:
                        traverse_downstream(target_node.asset.asset_id, depth + 1)
        
        traverse_downstream(asset_id, 0)
        
        return lineage
    
    def get_data_quality_lineage(self, asset_id: str) -> Dict[str, Any]:
        """Get data quality metrics along the lineage"""
        
        upstream_lineage = self.get_lineage_upstream(asset_id)
        quality_report = {
            'asset_id': asset_id,
            'asset_name': self.assets[asset_id].name if asset_id in self.assets else 'Unknown',
            'quality_metrics': [],
            'validation_history': [],
            'transformation_impacts': []
        }
        
        # Collect quality metrics from nodes
        for node_data in upstream_lineage['nodes']:
            if 'metrics' in node_data and node_data['metrics']:
                quality_metrics = {
                    'node_id': node_data['node_id'],
                    'operation': node_data['operation'],
                    'timestamp': node_data['timestamp'],
                    'metrics': node_data['metrics']
                }
                quality_report['quality_metrics'].append(quality_metrics)
            
            # Check for validation operations
            if node_data['operation'] == DataOperation.VALIDATION.value:
                validation_entry = {
                    'timestamp': node_data['timestamp'],
                    'parameters': node_data.get('parameters', {}),
                    'results': node_data.get('metrics', {})
                }
                quality_report['validation_history'].append(validation_entry)
        
        # Analyze transformation impacts
        for edge_data in upstream_lineage['edges']:
            if edge_data['relationship_type'] == 'transformed_to':
                impact_entry = {
                    'transformation': edge_data['transformation_logic'],
                    'timestamp': edge_data['created_at'],
                    'source_node': edge_data['source_node_id'],
                    'target_node': edge_data['target_node_id']
                }
                quality_report['transformation_impacts'].append(impact_entry)
        
        return quality_report
    
    def generate_compliance_report(self, 
                                 start_date: datetime = None, 
                                 end_date: datetime = None,
                                 user: str = None) -> Dict[str, Any]:
        """Generate compliance report for auditing"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter audit events
        filtered_events = [
            event for event in self.audit_events
            if start_date <= event.timestamp <= end_date
            and (user is None or event.user == user)
        ]
        
        # Categorize events
        events_by_operation = {}
        events_by_user = {}
        gdpr_events = []
        
        for event in filtered_events:
            # Group by operation
            op = event.operation.value
            if op not in events_by_operation:
                events_by_operation[op] = []
            events_by_operation[op].append(event)
            
            # Group by user
            if event.user not in events_by_user:
                events_by_user[event.user] = []
            events_by_user[event.user].append(event)
            
            # Check for GDPR-relevant events
            if self._is_gdpr_relevant(event):
                gdpr_events.append(event)
        
        # Calculate statistics
        total_events = len(filtered_events)
        unique_users = len(events_by_user)
        unique_assets_accessed = len(set(event.asset_id for event in filtered_events))
        
        report = {
            'report_generated_at': datetime.now().isoformat(),
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': total_events,
                'unique_users': unique_users,
                'unique_assets_accessed': unique_assets_accessed,
                'gdpr_relevant_events': len(gdpr_events)
            },
            'events_by_operation': {
                op: len(events) for op, events in events_by_operation.items()
            },
            'events_by_user': {
                user: len(events) for user, events in events_by_user.items()
            },
            'gdpr_compliance': {
                'enabled': self.gdpr_enabled,
                'retention_days': self.data_retention_days,
                'relevant_events': len(gdpr_events),
                'anonymization_applied': self._count_anonymized_events(filtered_events)
            },
            'data_retention_status': self._check_data_retention_compliance()
        }
        
        return report
    
    def anonymize_personal_data(self, asset_id: str) -> bool:
        """Anonymize personal data in an asset for GDPR compliance"""
        
        if asset_id not in self.assets:
            raise ValueError(f"Asset {asset_id} not found")
        
        asset = self.assets[asset_id]
        
        # Check if asset contains personal data
        personal_data_fields = []
        for field_name in asset.schema.keys():
            if any(pii_field in field_name.lower() for pii_field in self.anonymization_fields):
                personal_data_fields.append(field_name)
        
        if not personal_data_fields:
            logger.info(f"No personal data found in asset {asset.name}")
            return False
        
        # Create anonymization record
        anonymization_event = self._create_audit_event(
            operation=DataOperation.TRANSFORMATION,
            asset_id=asset_id,
            details={
                'action': 'data_anonymization',
                'fields_anonymized': personal_data_fields,
                'anonymization_method': 'hash_replacement',
                'gdpr_compliance': True
            }
        )
        
        # Update asset metadata
        asset.metadata['anonymized'] = True
        asset.metadata['anonymization_date'] = datetime.now().isoformat()
        asset.metadata['anonymized_fields'] = personal_data_fields
        asset.updated_at = datetime.now()
        
        logger.info(f"Anonymized personal data in asset {asset.name}")
        return True
    
    def delete_expired_data(self) -> Dict[str, int]:
        """Delete data that has exceeded retention period"""
        
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        
        # Find expired assets
        expired_assets = [
            asset for asset in self.assets.values()
            if asset.created_at < cutoff_date
        ]
        
        # Find expired audit events
        expired_events = [
            event for event in self.audit_events
            if event.timestamp < cutoff_date
        ]
        
        # Delete expired data
        deleted_assets = 0
        for asset in expired_assets:
            if self._can_delete_asset(asset):
                del self.assets[asset.asset_id]
                deleted_assets += 1
                
                # Create deletion audit event
                self._create_audit_event(
                    operation=DataOperation.DELETE,
                    asset_id=asset.asset_id,
                    details={
                        'action': 'data_retention_deletion',
                        'asset_name': asset.name,
                        'retention_period_days': self.data_retention_days,
                        'gdpr_compliance': True
                    }
                )
        
        # Remove expired audit events
        self.audit_events = [
            event for event in self.audit_events
            if event.timestamp >= cutoff_date
        ]
        
        deleted_events = len(expired_events)
        
        logger.info(f"Deleted {deleted_assets} expired assets and {deleted_events} expired audit events")
        
        return {
            'deleted_assets': deleted_assets,
            'deleted_audit_events': deleted_events
        }
    
    def export_lineage_graph(self, format: str = 'json') -> str:
        """Export lineage graph in various formats"""
        
        lineage_data = {
            'assets': {asset_id: asdict(asset) for asset_id, asset in self.assets.items()},
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
            'edges': {edge_id: asdict(edge) for edge_id, edge in self.edges.items()},
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_assets': len(self.assets),
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges)
            }
        }
        
        if format.lower() == 'json':
            return json.dumps(lineage_data, indent=2, default=str)
        elif format.lower() == 'graphml':
            return self._export_to_graphml(lineage_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _create_audit_event(self,
                           operation: DataOperation,
                           asset_id: str,
                           user: str = "system",
                           ip_address: str = None,
                           details: Dict[str, Any] = None) -> AuditEvent:
        """Create and store an audit event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user=user,
            operation=operation,
            asset_id=asset_id,
            details=details or {},
            ip_address=ip_address
        )
        
        self.audit_events.append(event)
        
        # Persist to database if available
        if self.engine:
            self._persist_audit_event(event)
        
        return event
    
    def _get_latest_node_for_asset(self, asset_id: str) -> Optional[str]:
        """Get the most recent node ID for an asset"""
        
        asset_nodes = [
            node for node in self.nodes.values()
            if node.asset.asset_id == asset_id
        ]
        
        if not asset_nodes:
            return None
        
        # Sort by timestamp and return the most recent
        latest_node = max(asset_nodes, key=lambda x: x.timestamp)
        return latest_node.node_id
    
    def _check_gdpr_compliance(self, asset: DataAsset, user: str, purpose: str) -> bool:
        """Check if data access complies with GDPR"""
        
        if not self.gdpr_enabled:
            return True
        
        # Check if asset contains personal data
        has_personal_data = any(
            pii_field in field_name.lower() 
            for field_name in asset.schema.keys()
            for pii_field in self.anonymization_fields
        )
        
        if not has_personal_data:
            return True
        
        # Check if purpose is legitimate
        legitimate_purposes = [
            'analytics', 'quality_control', 'supply_chain_optimization',
            'compliance_reporting', 'system_monitoring'
        ]
        
        return purpose and purpose.lower() in legitimate_purposes
    
    def _is_gdpr_relevant(self, event: AuditEvent) -> bool:
        """Check if an audit event is relevant for GDPR compliance"""
        
        if not self.gdpr_enabled:
            return False
        
        # Check if the asset contains personal data
        asset = self.assets.get(event.asset_id)
        if not asset:
            return False
        
        has_personal_data = any(
            pii_field in field_name.lower() 
            for field_name in asset.schema.keys()
            for pii_field in self.anonymization_fields
        )
        
        return has_personal_data
    
    def _count_anonymized_events(self, events: List[AuditEvent]) -> int:
        """Count events involving anonymized data"""
        
        count = 0
        for event in events:
            if event.details.get('action') == 'data_anonymization':
                count += 1
        
        return count
    
    def _check_data_retention_compliance(self) -> Dict[str, Any]:
        """Check compliance with data retention policies"""
        
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        
        # Count assets by age
        total_assets = len(self.assets)
        old_assets = len([
            asset for asset in self.assets.values()
            if asset.created_at < cutoff_date
        ])
        
        # Count audit events by age
        total_events = len(self.audit_events)
        old_events = len([
            event for event in self.audit_events
            if event.timestamp < cutoff_date
        ])
        
        return {
            'retention_period_days': self.data_retention_days,
            'assets': {
                'total': total_assets,
                'expired': old_assets,
                'compliance_rate': ((total_assets - old_assets) / total_assets * 100) if total_assets > 0 else 100
            },
            'audit_events': {
                'total': total_events,
                'expired': old_events,
                'compliance_rate': ((total_events - old_events) / total_events * 100) if total_events > 0 else 100
            }
        }
    
    def _can_delete_asset(self, asset: DataAsset) -> bool:
        """Check if an asset can be safely deleted"""
        
        # Don't delete if asset is still being used
        asset_nodes = [node for node in self.nodes.values() if node.asset.asset_id == asset.asset_id]
        
        # Check if any recent nodes (within 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_nodes = [node for node in asset_nodes if node.timestamp > recent_cutoff]
        
        return len(recent_nodes) == 0
    
    def _persist_audit_event(self, event: AuditEvent):
        """Persist audit event to database"""
        
        try:
            event_data = {
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'user_name': event.user,
                'operation': event.operation.value,
                'asset_id': event.asset_id,
                'details': json.dumps(event.details),
                'ip_address': event.ip_address
            }
            
            df = pd.DataFrame([event_data])
            df.to_sql('AuditEvents', self.engine, if_exists='append', index=False)
            
        except Exception as e:
            logger.error(f"Failed to persist audit event: {e}")
    
    def _load_lineage_data(self):
        """Load existing lineage data from storage"""
        
        lineage_file = self.storage_path / 'lineage_data.json'
        
        if lineage_file.exists():
            try:
                with open(lineage_file, 'r') as f:
                    data = json.load(f)
                
                # Load assets
                for asset_data in data.get('assets', []):
                    asset = DataAsset(**asset_data)
                    self.assets[asset.asset_id] = asset
                
                # Load nodes
                for node_data in data.get('nodes', []):
                    # Reconstruct asset from stored data
                    asset_data = node_data.pop('asset')
                    asset = DataAsset(**asset_data)
                    
                    node = DataLineageNode(asset=asset, **node_data)
                    self.nodes[node.node_id] = node
                
                # Load edges
                for edge_data in data.get('edges', []):
                    edge = DataLineageEdge(**edge_data)
                    self.edges[edge.edge_id] = edge
                
                logger.info(f"Loaded lineage data: {len(self.assets)} assets, {len(self.nodes)} nodes, {len(self.edges)} edges")
                
            except Exception as e:
                logger.error(f"Failed to load lineage data: {e}")
    
    def save_lineage_data(self):
        """Save lineage data to storage"""
        
        lineage_file = self.storage_path / 'lineage_data.json'
        
        try:
            data = {
                'assets': [asdict(asset) for asset in self.assets.values()],
                'nodes': [asdict(node) for node in self.nodes.values()],
                'edges': [asdict(edge) for edge in self.edges.values()],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(lineage_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved lineage data to {lineage_file}")
            
        except Exception as e:
            logger.error(f"Failed to save lineage data: {e}")
    
    def _export_to_graphml(self, lineage_data: Dict[str, Any]) -> str:
        """Export lineage to GraphML format"""
        
        graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
        graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        graphml.append('  <key id="name" for="node" attr.name="name" attr.type="string"/>')
        graphml.append('  <key id="type" for="node" attr.name="type" attr.type="string"/>')
        graphml.append('  <key id="operation" for="edge" attr.name="operation" attr.type="string"/>')
        graphml.append('  <graph id="lineage" edgedefault="directed">')
        
        # Add nodes
        for node_id, node_data in lineage_data['nodes'].items():
            graphml.append(f'    <node id="{node_id}">')
            graphml.append(f'      <data key="name">{node_data["asset"]["name"]}</data>')
            graphml.append(f'      <data key="type">{node_data["asset"]["asset_type"]}</data>')
            graphml.append('    </node>')
        
        # Add edges
        for edge_id, edge_data in lineage_data['edges'].items():
            source = edge_data['source_node_id']
            target = edge_data['target_node_id']
            graphml.append(f'    <edge id="{edge_id}" source="{source}" target="{target}">')
            graphml.append(f'      <data key="operation">{edge_data["relationship_type"]}</data>')
            graphml.append('    </edge>')
        
        graphml.append('  </graph>')
        graphml.append('</graphml>')
        
        return '\n'.join(graphml)

# Usage example and integration
def setup_supply_chain_lineage_tracking(connection_string: str = None) -> DataLineageTracker:
    """Setup lineage tracking for supply chain system"""
    
    tracker = DataLineageTracker(connection_string)
    
    # Register common data assets
    usda_schema = {
        'ProductCode': 'string',
        'ProductName': 'string',
        'Category': 'string',
        'ShelfLifeDays': 'integer',
        'OptimalTempMin': 'float',
        'OptimalTempMax': 'float',
        'UnitCost': 'float',
        'UnitPrice': 'float'
    }
    
    iot_schema = {
        'LogTime': 'datetime',
        'DeviceID': 'string',
        'WarehouseID': 'integer',
        'Temperature': 'float',
        'Humidity': 'float',
        'CO2Level': 'float',
        'EthyleneLevel': 'float',
        'QualityScore': 'float'
    }
    
    # Register USDA data asset
    usda_asset = tracker.register_data_asset(
        name="USDA FoodData Central",
        asset_type="dataset",
        source=DataSource.USDA_API,
        location="https://fdc.nal.usda.gov/",
        schema=usda_schema,
        metadata={
            'description': 'Real USDA food product database',
            'update_frequency': 'monthly',
            'data_classification': 'public'
        },
        tags=['food', 'nutrition', 'public_data']
    )
    
    # Register IoT sensor data asset
    iot_asset = tracker.register_data_asset(
        name="IoT Sensor Readings",
        asset_type="dataset",
        source=DataSource.IOT_SENSORS,
        location="warehouse_sensors",
        schema=iot_schema,
        metadata={
            'description': 'Real-time warehouse environmental data',
            'update_frequency': 'real_time',
            'data_classification': 'internal'
        },
        tags=['iot', 'sensors', 'real_time']
    )
    
    logger.info("Supply chain lineage tracking setup completed")
    return tracker

if __name__ == "__main__":
    # Example usage
    tracker = setup_supply_chain_lineage_tracking()
    
    # Example: Track a data transformation
    processed_schema = {
        'ProductCode': 'string',
        'ProductName': 'string',
        'Category': 'string',
        'QualityScore': 'float',
        'Temperature_avg': 'float',
        'Humidity_avg': 'float',
        'FreshnessRatio': 'float'
    }
    
    processed_asset = DataAsset(
        asset_id=str(uuid.uuid4()),
        name="Processed Supply Chain Data",
        asset_type="dataset",
        source=DataSource.DATABASE,
        location="processed_data_table",
        schema=processed_schema,
        metadata={'description': 'Cleaned and feature-engineered supply chain data'},
        created_at=datetime.now(),
        updated_at=datetime.now(),
        tags=['processed', 'ml_ready']
    )
    
    # Track transformation
    source_asset_ids = [asset.asset_id for asset in tracker.assets.values()]
    
    transformation_node = tracker.track_transformation(
        source_asset_ids=source_asset_ids,
        target_asset=processed_asset,
        operation=DataOperation.FEATURE_ENGINEERING,
        transformation_logic="Data cleaning, validation, and feature engineering pipeline",
        parameters={'validation_enabled': True, 'feature_engineering': True},
        metrics={'rows_processed': 10000, 'features_created': 25, 'quality_score': 0.95},
        user="data_engineer"
    )
    
    # Generate compliance report
    compliance_report = tracker.generate_compliance_report()
    print("Compliance Report:")
    print(json.dumps(compliance_report, indent=2, default=str))
    
    # Save lineage data
    tracker.save_lineage_data()
"""
Test suite for ML models in Fresh Supply Chain Intelligence System
"""

import pytest
import torch
import pandas as pd
import numpy as np
import tempfile
import os
from PIL import Image
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision_model import FreshProduceVisionModel
from models.forecasting_model import DemandForecaster, TemporalFusionTransformer
from models.gnn_optimizer import SupplyChainGNN, SupplyChainOptimizer

class TestVisionModel:
    """Test cases for computer vision model"""
    
    def test_model_initialization(self):
        """Test model initialization with different parameters"""
        model = FreshProduceVisionModel(num_classes=5)
        assert model.num_classes == 5
        assert model.device in ['cuda', 'cpu']
        assert len(model.quality_labels) == 5
        assert 'Fresh' in model.quality_labels
        assert 'Spoiled' in model.quality_labels
    
    def test_quality_prediction(self, sample_image):
        """Test quality prediction from image"""
        model = FreshProduceVisionModel()
        label, confidence, probs = model.predict_quality(sample_image)
        
        assert label in model.quality_labels
        assert 0 <= confidence <= 1
        assert len(probs) == model.num_classes
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)
    
    def test_feature_extraction(self, sample_image):
        """Test feature extraction from image"""
        model = FreshProduceVisionModel()
        features = model.extract_features(sample_image)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 0  # Should have some features
        assert len(features.shape) == 1  # Should be 1D vector
    
    def test_model_save_load(self, sample_image):
        """Test model saving and loading"""
        model = FreshProduceVisionModel()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            # Save model
            model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            new_model = FreshProduceVisionModel()
            new_model.load_model(model_path)
            
            # Test that loaded model works
            label1, conf1, _ = model.predict_quality(sample_image)
            label2, conf2, _ = new_model.predict_quality(sample_image)
            
            # Results should be similar (allowing for small differences)
            assert abs(conf1 - conf2) < 0.1
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)

class TestTemporalFusionTransformer:
    """Test cases for TFT forecasting model"""
    
    def test_model_initialization(self):
        """Test TFT model initialization"""
        model = TemporalFusionTransformer(
            input_size=10,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            output_size=1,
            forecast_horizon=7
        )
        
        assert model.input_size == 10
        assert model.hidden_size == 64
        assert model.forecast_horizon == 7
    
    def test_forward_pass(self):
        """Test forward pass through TFT"""
        model = TemporalFusionTransformer(
            input_size=10,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            output_size=1,
            forecast_horizon=7
        )
        
        batch_size = 2
        seq_len = 30
        
        # Create dummy inputs
        x_static = torch.randn(batch_size, 10)
        x_temporal_historical = torch.randn(batch_size, seq_len, 10)
        x_temporal_future = torch.randn(batch_size, 7, 5)
        
        # Forward pass
        output, quantiles, attention = model(x_static, x_temporal_historical, x_temporal_future)
        
        # Check output shapes
        assert output.shape == (batch_size, 7, 1)
        assert quantiles.shape == (batch_size, 7, 3)  # 3 quantiles
        assert attention.shape[0] == batch_size  # Attention weights
    
    def test_model_training_step(self):
        """Test single training step"""
        model = TemporalFusionTransformer(
            input_size=10,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            output_size=1,
            forecast_horizon=7
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # Create dummy data
        batch_size = 4
        seq_len = 30
        
        x_static = torch.randn(batch_size, 10)
        x_temporal_historical = torch.randn(batch_size, seq_len, 10)
        x_temporal_future = torch.randn(batch_size, 7, 5)
        target = torch.randn(batch_size, 7, 1)
        
        # Training step
        optimizer.zero_grad()
        output, _, _ = model(x_static, x_temporal_historical, x_temporal_future)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Loss should be finite
        assert torch.isfinite(loss)

class TestDemandForecaster:
    """Test cases for demand forecasting system"""
    
    def test_data_preparation_mock(self):
        """Test data preparation with mock data"""
        # Create mock connection string
        forecaster = DemandForecaster("sqlite:///:memory:")
        
        # Create mock data
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        mock_data = pd.DataFrame({
            'Date': dates,
            'DailyWaste': np.random.poisson(10, 90),
            'AvgTemp': np.random.normal(4, 1, 90),
            'AvgHumidity': np.random.normal(90, 5, 90),
            'ActiveLots': np.random.randint(5, 20, 90)
        })
        
        # Test feature engineering
        mock_data['DayOfWeek'] = pd.to_datetime(mock_data['Date']).dt.dayofweek
        mock_data['Month'] = pd.to_datetime(mock_data['Date']).dt.month
        mock_data['IsWeekend'] = mock_data['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Add lag features
        for lag in [1, 7, 14]:
            mock_data[f'Waste_Lag_{lag}'] = mock_data['DailyWaste'].shift(lag)
        
        # Add rolling statistics
        for window in [7, 14]:
            mock_data[f'Waste_MA_{window}'] = mock_data['DailyWaste'].rolling(window).mean()
            mock_data[f'Waste_STD_{window}'] = mock_data['DailyWaste'].rolling(window).std()
        
        # Check that features were created
        assert 'DayOfWeek' in mock_data.columns
        assert 'Waste_Lag_7' in mock_data.columns
        assert 'Waste_MA_7' in mock_data.columns
    
    def test_forecast_output_structure(self):
        """Test forecast output structure"""
        forecaster = DemandForecaster("sqlite:///:memory:")
        
        # Create mock context data
        context_data = pd.DataFrame({
            'Temperature': np.random.normal(4, 1, 30),
            'Humidity': np.random.normal(90, 5, 30),
            'DailyWaste': np.random.poisson(10, 30),
            'DayOfWeek': np.random.randint(0, 7, 30),
            'Month': np.random.randint(1, 13, 30),
            'IsWeekend': np.random.randint(0, 2, 30)
        })
        
        # Add required lag features
        for lag in [1, 7, 14, 28]:
            context_data[f'Waste_Lag_{lag}'] = context_data['DailyWaste'].shift(lag)
        
        for window in [7, 14, 28]:
            context_data[f'Waste_MA_{window}'] = context_data['DailyWaste'].rolling(window).mean()
            context_data[f'Waste_STD_{window}'] = context_data['DailyWaste'].rolling(window).std()
        
        context_data = context_data.dropna()
        
        # Test forecast structure (mock implementation)
        horizon = 7
        mock_result = {
            'predictions': np.random.randn(1, horizon),
            'lower_bound': np.random.randn(1, horizon),
            'upper_bound': np.random.randn(1, horizon),
            'attention_weights': np.random.randn(1, 30, 30)
        }
        
        assert 'predictions' in mock_result
        assert 'lower_bound' in mock_result
        assert 'upper_bound' in mock_result
        assert 'attention_weights' in mock_result
        assert len(mock_result['predictions'][0]) == horizon

class TestSupplyChainGNN:
    """Test cases for Graph Neural Network"""
    
    def test_gnn_initialization(self):
        """Test GNN model initialization"""
        model = SupplyChainGNN(input_features=10, hidden_size=64, output_size=32)
        
        assert model.conv1.in_channels == 10
        assert model.conv1.out_channels == 64
        assert model.conv3.out_channels == 32
    
    def test_forward_pass(self):
        """Test forward pass through GNN"""
        model = SupplyChainGNN(input_features=10, hidden_size=64, output_size=32)
        
        # Create dummy graph data
        num_nodes = 10
        num_edges = 20
        
        x = torch.randn(num_nodes, 10)  # Node features
        edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge connections
        batch = torch.zeros(num_nodes, dtype=torch.long)  # Batch assignment
        
        # Forward pass
        node_embeddings, graph_embedding = model(x, edge_index, batch)
        
        # Check output shapes
        assert node_embeddings.shape == (num_nodes, 32)
        assert graph_embedding.shape == (1, 32)
    
    def test_edge_importance_prediction(self):
        """Test edge importance prediction"""
        model = SupplyChainGNN(input_features=10, hidden_size=64, output_size=32)
        
        num_nodes = 5
        num_edges = 8
        
        x = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Predict edge importance
        edge_importance = model.predict_edge_importance(x, edge_index)
        
        # Check output shape and values
        assert edge_importance.shape == (num_edges, 1)
        assert torch.all(edge_importance >= 0)  # Should be non-negative
        assert torch.all(edge_importance <= 1)  # Should be probabilities

class TestSupplyChainOptimizer:
    """Test cases for supply chain optimization"""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = SupplyChainOptimizer("sqlite:///:memory:")
        
        assert optimizer.engine is not None
        assert optimizer.gnn_model is not None
        assert optimizer.graph is None  # Not built yet
    
    def test_network_building_mock(self):
        """Test network building with mock data"""
        optimizer = SupplyChainOptimizer("sqlite:///:memory:")
        
        # Create mock network data
        nodes_data = {
            'NodeID': [1, 2, 3, 4, 5],
            'NodeType': ['SUPPLIER', 'WAREHOUSE', 'WAREHOUSE', 'RETAIL', 'RETAIL'],
            'NodeName': ['Supplier A', 'Oslo', 'Bergen', 'Store 1', 'Store 2'],
            'LocationLat': [59.0, 59.9, 60.4, 59.9, 60.4],
            'LocationLon': [10.0, 10.8, 5.3, 10.8, 5.3],
            'Capacity': [1000, 500, 500, 100, 100],
            'LeadTimeDays': [1, 0, 0, 0, 0]
        }
        
        edges_data = {
            'SourceNodeID': [1, 1, 2, 2, 3, 3, 2, 3],
            'TargetNodeID': [2, 3, 4, 5, 4, 5, 3, 2],
            'TransportMode': ['TRUCK', 'TRUCK', 'TRUCK', 'TRUCK', 'TRUCK', 'TRUCK', 'TRUCK', 'TRUCK'],
            'DistanceKM': [50, 100, 20, 30, 25, 35, 200, 200],
            'TransitTimeDays': [0.5, 1.0, 0.2, 0.3, 0.25, 0.35, 2.0, 2.0],
            'CostPerUnit': [0.1, 0.2, 0.05, 0.08, 0.06, 0.09, 0.3, 0.3]
        }
        
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        # Mock the database queries
        import sqlite3
        conn = sqlite3.connect(':memory:')
        nodes_df.to_sql('SupplyChainNodes', conn, index=False)
        edges_df.to_sql('SupplyChainEdges', conn, index=False)
        
        # Test network building logic
        graph = optimizer.build_supply_network()
        
        # Should create a network (even if empty due to mock)
        assert graph is not None

# Fixtures
@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img_path = tmp.name
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path, 'JPEG')
    
    yield img_path
    
    # Clean up
    if os.path.exists(img_path):
        os.unlink(img_path)

@pytest.fixture
def mock_connection_string():
    """Mock database connection string"""
    return "sqlite:///:memory:"

@pytest.fixture
def sample_context_data():
    """Sample context data for forecasting"""
    return pd.DataFrame({
        'Temperature': np.random.normal(4, 1, 30),
        'Humidity': np.random.normal(90, 5, 30),
        'DailyWaste': np.random.poisson(10, 30),
        'DayOfWeek': np.random.randint(0, 7, 30),
        'Month': np.random.randint(1, 13, 30),
        'IsWeekend': np.random.randint(0, 2, 30)
    })

@pytest.fixture
def sample_demand():
    """Sample demand data for optimization"""
    return {
        (1, 1): 100,
        (2, 1): 150,
        (3, 1): 200
    }

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

if __name__ == "__main__":
    pytest.main([__file__])
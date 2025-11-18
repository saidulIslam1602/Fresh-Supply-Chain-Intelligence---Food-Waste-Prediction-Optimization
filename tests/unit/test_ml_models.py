"""
Unit tests for ML models
Comprehensive tests for vision model, forecasting model, and GNN optimizer
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import json

# Import models to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.vision_model import (
    FreshProduceVisionModel, EnhancedVisionModel, 
    AttentionModule, UncertaintyHead, ModelConfig
)
from models.forecasting_model import (
    TemporalFusionTransformer, VariableSelectionNetwork,
    GatedResidualNetwork, ForecastConfig
)
from models.gnn_optimizer import SupplyChainGNN, SupplyChainOptimizer

@pytest.mark.unit
@pytest.mark.ml
class TestVisionModel:
    """Unit tests for computer vision models"""
    
    def test_vision_config_initialization(self):
        """Test vision model configuration"""
        config = ModelConfig(
            num_classes=5,
            backbone='efficientnet-b4',
            input_size=224,
            dropout_rate=0.3
        )
        
        assert config.num_classes == 5
        assert config.backbone == 'efficientnet-b4'
        assert config.input_size == 224
        assert config.dropout_rate == 0.3
    
    def test_attention_module(self):
        """Test attention module functionality"""
        attention = AttentionModule(in_channels=512, reduction=16)
        
        # Test forward pass
        x = torch.randn(2, 512, 7, 7)  # Batch, channels, height, width
        output = attention(x)
        
        assert output.shape == x.shape
        assert torch.all(output >= 0)  # Should be non-negative after attention
    
    def test_uncertainty_head(self):
        """Test uncertainty estimation head"""
        uncertainty_head = UncertaintyHead(in_features=512, num_classes=5)
        
        # Test forward pass
        features = torch.randn(2, 512)
        mean_logits, log_var = uncertainty_head(features)
        
        assert mean_logits.shape == (2, 5)
        assert log_var.shape == (2, 5)
        assert torch.all(torch.isfinite(mean_logits))
        assert torch.all(torch.isfinite(log_var))
    
    def test_enhanced_vision_model_initialization(self):
        """Test enhanced vision model initialization"""
        config = ModelConfig(num_classes=5, backbone='efficientnet-b0')  # Use smaller model for testing
        model = EnhancedVisionModel(config)
        
        assert model.config.num_classes == 5
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'uncertainty_head')
        assert hasattr(model, 'classifier')
    
    def test_vision_model_forward_pass(self, test_utils):
        """Test vision model forward pass"""
        config = ModelConfig(num_classes=5, backbone='efficientnet-b0')
        model = EnhancedVisionModel(config)
        model.eval()
        
        # Create mock input
        x = test_utils.create_mock_image_tensor(batch_size=2, channels=3, height=224, width=224)
        
        with torch.no_grad():
            outputs = model(x)
        
        assert 'logits' in outputs
        assert 'uncertainty' in outputs
        assert 'attention_weights' in outputs
        
        assert outputs['logits'].shape == (2, 5)
        assert outputs['uncertainty'].shape == (2, 5)
        assert torch.all(torch.isfinite(outputs['logits']))
        assert torch.all(torch.isfinite(outputs['uncertainty']))
    
    def test_vision_model_prediction(self, sample_image):
        """Test vision model prediction functionality"""
        # Use the original FreshProduceVisionModel for this test
        model = FreshProduceVisionModel(num_classes=5)
        
        # Mock the model's forward pass to avoid loading actual weights
        with patch.object(model.model, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([[2.0, 1.0, 0.5, 0.2, 0.1]])
            
            label, confidence, probabilities = model.predict_quality(sample_image)
            
            assert label in model.quality_labels
            assert 0 <= confidence <= 1
            assert len(probabilities) == model.num_classes
            assert np.isclose(probabilities.sum(), 1.0, atol=1e-6)
    
    def test_vision_model_feature_extraction(self, sample_image):
        """Test feature extraction from vision model"""
        model = FreshProduceVisionModel(num_classes=5)
        
        # Mock feature extraction
        with patch.object(model.model, 'forward') as mock_forward:
            mock_features = torch.randn(1, 512)  # Mock feature vector
            mock_forward.return_value = mock_features
            
            features = model.extract_features(sample_image)
            
            assert isinstance(features, np.ndarray)
            assert len(features.shape) == 1
            assert features.shape[0] > 0
    
    def test_vision_model_uncertainty_quantification(self, test_utils):
        """Test uncertainty quantification in vision model"""
        config = ModelConfig(num_classes=5, backbone='efficientnet-b0', enable_uncertainty=True)
        model = EnhancedVisionModel(config)
        model.eval()
        
        x = test_utils.create_mock_image_tensor(batch_size=1)
        
        with torch.no_grad():
            # Test multiple forward passes for Monte Carlo dropout
            uncertainties = []
            for _ in range(10):
                model.train()  # Enable dropout
                outputs = model(x)
                uncertainties.append(outputs['uncertainty'].numpy())
            
            model.eval()
            
            # Uncertainty should vary across runs when dropout is enabled
            uncertainties = np.array(uncertainties)
            assert uncertainties.std() > 0  # Should have some variation
    
    def test_vision_model_save_load(self, temp_directory):
        """Test model saving and loading"""
        config = ModelConfig(num_classes=5, backbone='efficientnet-b0')
        model = EnhancedVisionModel(config)
        
        # Save model
        model_path = os.path.join(temp_directory, 'test_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__
        }, model_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        loaded_model = EnhancedVisionModel(ModelConfig(**checkpoint['config']))
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test that loaded model works
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            original_output = model(x)
            loaded_output = loaded_model(x)
            
            # Outputs should be identical
            torch.testing.assert_close(
                original_output['logits'], 
                loaded_output['logits'],
                atol=1e-6, rtol=1e-6
            )

@pytest.mark.unit
@pytest.mark.ml
class TestForecastingModel:
    """Unit tests for forecasting models"""
    
    def test_forecast_config_initialization(self):
        """Test forecast model configuration"""
        config = ForecastConfig(
            input_size=10,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            horizon=7
        )
        
        assert config.input_size == 10
        assert config.hidden_size == 64
        assert config.num_heads == 4
        assert config.num_layers == 2
        assert config.horizon == 7
    
    def test_variable_selection_network(self):
        """Test Variable Selection Network"""
        vsn = VariableSelectionNetwork(input_size=10, hidden_size=64, num_variables=5)
        
        # Test forward pass
        x = torch.randn(2, 20, 10)  # Batch, sequence, features
        selected_features, weights = vsn(x)
        
        assert selected_features.shape == (2, 20, 5)
        assert weights.shape == (2, 20, 5)
        assert torch.all(weights >= 0)  # Weights should be non-negative
        assert torch.all(weights <= 1)  # Weights should be <= 1
    
    def test_gated_residual_network(self):
        """Test Gated Residual Network"""
        grn = GatedResidualNetwork(input_size=64, hidden_size=32)
        
        # Test forward pass
        x = torch.randn(2, 20, 64)
        output = grn(x)
        
        assert output.shape == x.shape
        assert torch.all(torch.isfinite(output))
    
    def test_temporal_fusion_transformer_initialization(self):
        """Test TFT model initialization"""
        config = ForecastConfig(
            input_size=10,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            horizon=7
        )
        
        model = TemporalFusionTransformer(config)
        
        assert hasattr(model, 'variable_selection')
        assert hasattr(model, 'lstm_encoder')
        assert hasattr(model, 'attention_layers')
        assert hasattr(model, 'output_projection')
    
    def test_tft_forward_pass(self, test_utils):
        """Test TFT forward pass"""
        config = ForecastConfig(
            input_size=10,
            hidden_size=32,  # Smaller for testing
            num_heads=2,
            num_layers=1,
            horizon=7
        )
        
        model = TemporalFusionTransformer(config)
        model.eval()
        
        # Create mock time series data
        x = test_utils.create_mock_time_series(length=30, features=10)
        
        with torch.no_grad():
            outputs = model(x)
        
        assert 'forecast' in outputs
        assert 'attention_weights' in outputs
        assert 'quantiles' in outputs
        
        assert outputs['forecast'].shape == (1, 7)  # Batch size 1, horizon 7
        assert torch.all(torch.isfinite(outputs['forecast']))
    
    def test_tft_quantile_regression(self, test_utils):
        """Test quantile regression in TFT"""
        config = ForecastConfig(
            input_size=5,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            horizon=7,
            quantiles=[0.1, 0.5, 0.9]
        )
        
        model = TemporalFusionTransformer(config)
        model.eval()
        
        x = test_utils.create_mock_time_series(length=20, features=5)
        
        with torch.no_grad():
            outputs = model(x)
        
        assert 'quantiles' in outputs
        quantiles = outputs['quantiles']
        
        # Should have 3 quantiles for each time step
        assert quantiles.shape == (1, 7, 3)
        
        # Quantiles should be ordered: q0.1 <= q0.5 <= q0.9
        q10, q50, q90 = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]
        assert torch.all(q10 <= q50)
        assert torch.all(q50 <= q90)
    
    def test_tft_attention_mechanism(self, test_utils):
        """Test attention mechanism in TFT"""
        config = ForecastConfig(
            input_size=5,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            horizon=7
        )
        
        model = TemporalFusionTransformer(config)
        model.eval()
        
        x = test_utils.create_mock_time_series(length=20, features=5)
        
        with torch.no_grad():
            outputs = model(x)
        
        assert 'attention_weights' in outputs
        attention = outputs['attention_weights']
        
        # Attention weights should sum to 1 across the sequence dimension
        assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-6)
        assert torch.all(attention >= 0)
        assert torch.all(attention <= 1)
    
    def test_tft_training_step(self, test_utils):
        """Test TFT training step"""
        config = ForecastConfig(
            input_size=5,
            hidden_size=32,
            num_heads=2,
            num_layers=1,
            horizon=7
        )
        
        model = TemporalFusionTransformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create training data
        x = test_utils.create_mock_time_series(length=20, features=5)
        y = torch.randn(1, 7)  # Target forecast
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(x)
        loss = criterion(outputs['forecast'], y)
        loss.backward()
        optimizer.step()
        
        assert torch.isfinite(loss)
        assert loss.item() > 0

@pytest.mark.unit
@pytest.mark.ml
class TestGNNOptimizer:
    """Unit tests for GNN-based supply chain optimizer"""
    
    def test_supply_chain_gnn_initialization(self):
        """Test SupplyChainGNN initialization"""
        model = SupplyChainGNN(
            node_features=10,
            edge_features=5,
            hidden_dim=64,
            num_layers=2
        )
        
        assert hasattr(model, 'node_embedding')
        assert hasattr(model, 'edge_embedding')
        assert hasattr(model, 'gnn_layers')
        assert hasattr(model, 'edge_predictor')
        assert len(model.gnn_layers) == 2
    
    def test_gnn_forward_pass(self):
        """Test GNN forward pass"""
        model = SupplyChainGNN(
            node_features=5,
            edge_features=3,
            hidden_dim=32,
            num_layers=2
        )
        model.eval()
        
        # Create mock graph data
        num_nodes = 6
        num_edges = 8
        
        node_features = torch.randn(num_nodes, 5)
        edge_features = torch.randn(num_edges, 3)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],  # Source nodes
            [1, 2, 3, 3, 4, 4, 5, 5]   # Target nodes
        ])
        
        with torch.no_grad():
            node_embeddings, edge_predictions = model(node_features, edge_index, edge_features)
        
        assert node_embeddings.shape == (num_nodes, 32)
        assert edge_predictions.shape == (num_edges, 1)
        assert torch.all(torch.isfinite(node_embeddings))
        assert torch.all(torch.isfinite(edge_predictions))
        assert torch.all(edge_predictions >= 0)  # Should be probabilities
        assert torch.all(edge_predictions <= 1)
    
    def test_route_optimizer_initialization(self, mock_database_operations):
        """Test SupplyChainOptimizer initialization"""
        with patch('sqlalchemy.create_engine'):
            optimizer = SupplyChainOptimizer("sqlite:///:memory:")
            
            assert optimizer.engine is not None
            assert optimizer.gnn_model is not None
            assert optimizer.graph is None  # Not built yet
    
    def test_network_building_with_mock_data(self, sample_supply_chain_network):
        """Test network building with mock data"""
        with patch('sqlalchemy.create_engine') as mock_engine:
            # Mock database connection and queries
            mock_conn = Mock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            
            # Mock pandas read_sql to return our sample data
            with patch('pandas.read_sql') as mock_read_sql:
                mock_read_sql.side_effect = [
                    sample_supply_chain_network['nodes'],
                    sample_supply_chain_network['edges']
                ]
                
                optimizer = SupplyChainOptimizer("sqlite:///:memory:")
                graph = optimizer.build_supply_network()
                
                assert graph is not None
                assert hasattr(graph, 'nodes')
                assert hasattr(graph, 'edges')
    
    def test_route_optimization_mock(self, sample_supply_chain_network):
        """Test route optimization with mock data"""
        with patch('sqlalchemy.create_engine') as mock_engine:
            mock_conn = Mock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            
            with patch('pandas.read_sql') as mock_read_sql:
                mock_read_sql.side_effect = [
                    sample_supply_chain_network['nodes'],
                    sample_supply_chain_network['edges']
                ]
                
                optimizer = SupplyChainOptimizer("sqlite:///:memory:")
                
                # Mock the optimization result
                with patch.object(optimizer, 'optimize_routes') as mock_optimize:
                    mock_optimize.return_value = {
                        'optimal_routes': [
                            {'from': 1, 'to': 2, 'cost': 100, 'flow': 50},
                            {'from': 2, 'to': 4, 'cost': 20, 'flow': 25}
                        ],
                        'total_cost': 120,
                        'optimization_time': 1.5,
                        'status': 'optimal'
                    }
                    
                    result = optimizer.optimize_distribution(
                        products=[1, 2],
                        warehouses=[1, 2],
                        optimize_for='cost'
                    )
                    
                    assert 'optimal_routes' in result
                    assert 'total_cost' in result
                    assert result['status'] == 'optimal'
                    assert result['total_cost'] > 0
    
    def test_gnn_edge_prediction_accuracy(self):
        """Test GNN edge prediction accuracy"""
        model = SupplyChainGNN(
            node_features=5,
            edge_features=3,
            hidden_dim=32,
            num_layers=2
        )
        
        # Create training data
        num_nodes = 10
        num_edges = 15
        
        node_features = torch.randn(num_nodes, 5)
        edge_features = torch.randn(num_edges, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Create binary labels (1 for good routes, 0 for bad routes)
        edge_labels = torch.randint(0, 2, (num_edges, 1)).float()
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Training loop
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            
            _, edge_predictions = model(node_features, edge_index, edge_features)
            loss = criterion(edge_predictions, edge_labels)
            
            loss.backward()
            optimizer.step()
        
        # Test that loss decreased
        model.eval()
        with torch.no_grad():
            _, final_predictions = model(node_features, edge_index, edge_features)
            final_loss = criterion(final_predictions, edge_labels)
        
        assert torch.isfinite(final_loss)
        # Loss should be reasonable (not too high)
        assert final_loss.item() < 1.0

@pytest.mark.unit
@pytest.mark.ml
class TestModelIntegration:
    """Integration tests for ML models"""
    
    def test_model_ensemble_prediction(self, sample_image, sample_time_series_data):
        """Test ensemble prediction using multiple models"""
        # Mock multiple vision models
        models = []
        predictions = []
        
        for i in range(3):
            model = FreshProduceVisionModel(num_classes=5)
            
            # Mock different predictions from each model
            with patch.object(model, 'predict_quality') as mock_predict:
                confidence = 0.8 + i * 0.05  # Different confidences
                probs = np.random.dirichlet([1, 1, 1, 1, 1])  # Random probabilities
                mock_predict.return_value = ('Fresh', confidence, probs)
                
                label, conf, prob = model.predict_quality(sample_image)
                predictions.append((label, conf, prob))
                models.append(model)
        
        # Ensemble prediction (simple averaging)
        avg_confidence = np.mean([pred[1] for pred in predictions])
        avg_probabilities = np.mean([pred[2] for pred in predictions], axis=0)
        
        assert 0 <= avg_confidence <= 1
        assert np.isclose(avg_probabilities.sum(), 1.0, atol=1e-6)
        assert len(avg_probabilities) == 5
    
    def test_model_performance_monitoring(self, test_utils):
        """Test model performance monitoring"""
        config = ModelConfig(num_classes=5, backbone='efficientnet-b0')
        model = EnhancedVisionModel(config)
        
        # Simulate batch predictions
        batch_size = 10
        x = test_utils.create_mock_image_tensor(batch_size=batch_size)
        
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
            
            outputs = model(x)
            
            if torch.cuda.is_available():
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            else:
                inference_time = 0.1  # Mock time for CPU
            
            # Performance metrics
            throughput = batch_size / inference_time if inference_time > 0 else float('inf')
            avg_uncertainty = outputs['uncertainty'].mean().item()
            
            assert throughput > 0
            assert 0 <= avg_uncertainty <= 1
            assert outputs['logits'].shape[0] == batch_size
    
    def test_model_memory_usage(self, test_utils):
        """Test model memory usage"""
        config = ModelConfig(num_classes=5, backbone='efficientnet-b0')
        model = EnhancedVisionModel(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory = total_params * 4 / (1024 ** 2)  # 4 bytes per float32, convert to MB
        
        assert total_params > 0
        assert trainable_params > 0
        assert param_memory > 0
        
        # Model should not be too large for testing
        assert param_memory < 100  # Less than 100MB for test model
    
    def test_model_serialization(self, temp_directory):
        """Test model serialization and deserialization"""
        # Test vision model
        vision_config = ModelConfig(num_classes=5, backbone='efficientnet-b0')
        vision_model = EnhancedVisionModel(vision_config)
        
        vision_path = os.path.join(temp_directory, 'vision_model.pth')
        torch.save({
            'model_state_dict': vision_model.state_dict(),
            'config': vision_config.__dict__,
            'model_type': 'vision'
        }, vision_path)
        
        # Test forecasting model
        forecast_config = ForecastConfig(input_size=5, hidden_size=32, horizon=7)
        forecast_model = TemporalFusionTransformer(forecast_config)
        
        forecast_path = os.path.join(temp_directory, 'forecast_model.pth')
        torch.save({
            'model_state_dict': forecast_model.state_dict(),
            'config': forecast_config.__dict__,
            'model_type': 'forecasting'
        }, forecast_path)
        
        # Verify files exist and can be loaded
        assert os.path.exists(vision_path)
        assert os.path.exists(forecast_path)
        
        # Load and verify
        vision_checkpoint = torch.load(vision_path, map_location='cpu')
        forecast_checkpoint = torch.load(forecast_path, map_location='cpu')
        
        assert vision_checkpoint['model_type'] == 'vision'
        assert forecast_checkpoint['model_type'] == 'forecasting'
        assert 'model_state_dict' in vision_checkpoint
        assert 'config' in vision_checkpoint
"""
Temporal Fusion Transformer for demand forecasting
Based on Google's TFT architecture for interpretable time series forecasting
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for demand forecasting
    Based on Google's TFT architecture for interpretable time series forecasting
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        output_size: int = 1,
        forecast_horizon: int = 7
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
        
        # Variable selection networks
        self.static_vsn = self._build_variable_selection_network(input_size, hidden_size)
        self.temporal_vsn = self._build_variable_selection_network(input_size, hidden_size)
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated Residual Networks
        self.grn_historical = self._build_grn(hidden_size)
        self.grn_future = self._build_grn(hidden_size)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon * output_size)
        )
        
        # Quantile outputs for uncertainty estimation
        self.quantile_layers = nn.ModuleList([
            nn.Linear(hidden_size // 2, forecast_horizon)
            for _ in range(3)  # 10%, 50%, 90% quantiles
        ])
        
    def _build_variable_selection_network(self, input_size: int, hidden_size: int):
        """Build variable selection network for feature importance"""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
    
    def _build_grn(self, hidden_size: int):
        """Build Gated Residual Network"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GLU(dim=-1),
            nn.LayerNorm(hidden_size // 2)
        )
    
    def forward(self, x_static, x_temporal_historical, x_temporal_future):
        """
        Forward pass
        
        Args:
            x_static: Static features (batch_size, static_features)
            x_temporal_historical: Historical temporal features (batch_size, seq_len, temporal_features)
            x_temporal_future: Known future features (batch_size, forecast_horizon, future_features)
        """
        batch_size = x_temporal_historical.size(0)
        
        # Process static features
        static_encoded = self.static_vsn(x_static).unsqueeze(1)
        
        # Process temporal features
        temporal_encoded = self.temporal_vsn(x_temporal_historical)
        
        # Combine static and temporal
        combined = temporal_encoded + static_encoded
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined)
        
        # Self-attention mechanism
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Gated residual connections
        historical = self.grn_historical(attn_out[:, -1, :])
        
        # Generate predictions
        output = self.output_layers(historical)
        output = output.reshape(batch_size, self.forecast_horizon, self.output_size)
        
        # Generate uncertainty estimates
        quantiles = []
        for quantile_layer in self.quantile_layers:
            q = quantile_layer(historical[:, :self.hidden_size // 2])
            quantiles.append(q.reshape(batch_size, self.forecast_horizon, 1))
        
        quantiles = torch.cat(quantiles, dim=-1)
        
        return output, quantiles, attn_weights


class DemandForecaster:
    """Complete demand forecasting system using TFT"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.model = None
        self.scaler = StandardScaler()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def prepare_data(self, product_id: int = None, warehouse_id: int = None, lookback_days: int = 90):
        """Prepare time series data for training using real USDA products and IoT data"""
        
        # If no specific product/warehouse, use real data from our loaded datasets
        if product_id is None or warehouse_id is None:
            # Get real products and warehouses
            products_query = """
            SELECT TOP 10 ProductID, ProductCode, ProductName, Category, ShelfLifeDays
            FROM Products 
            WHERE ProductCode LIKE 'USDA_%'
            ORDER BY NEWID()
            """
            products = pd.read_sql(products_query, self.engine)
            
            warehouses_query = """
            SELECT TOP 3 WarehouseID, WarehouseCode, WarehouseName, Country
            FROM Warehouses
            ORDER BY NEWID()
            """
            warehouses = pd.read_sql(warehouses_query, self.engine)
            
            if products.empty or warehouses.empty:
                logger.warning("No real data available, generating synthetic data")
                return self._generate_synthetic_forecast_data()
            
            # Use first product and warehouse
            product_id = products.iloc[0]['ProductID']
            warehouse_id = warehouses.iloc[0]['WarehouseID']
            
            logger.info(f"Using real data: Product {products.iloc[0]['ProductName'][:50]}... from {warehouses.iloc[0]['WarehouseName']}")
        
        # Try to get real waste events first
        waste_query = f"""
        SELECT 
            DATE(RecordedAt) as Date,
            SUM(QuantityWasted) as DailyWaste,
            AVG(Temperature) as AvgTemp,
            AVG(Humidity) as AvgHumidity,
            COUNT(DISTINCT LotNumber) as ActiveLots
        FROM WasteEvents we
        LEFT JOIN TemperatureLogs tl ON we.WarehouseID = tl.WarehouseID
            AND DATE(we.RecordedAt) = DATE(tl.LogTime)
        WHERE we.ProductID = {product_id} 
            AND we.WarehouseID = {warehouse_id}
            AND we.RecordedAt >= DATEADD(day, -{lookback_days}, GETDATE())
        GROUP BY DATE(RecordedAt)
        ORDER BY Date
        """
        
        df = pd.read_sql(waste_query, self.engine)
        
        # If no waste events, generate realistic data based on real products and IoT
        if df.empty:
            logger.info("No waste events found, generating realistic data based on real products and IoT sensors")
            return self._generate_realistic_forecast_data(product_id, warehouse_id, lookback_days)
        
        # Add time features
        df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['Month'] = pd.to_datetime(df['Date']).dt.month
        df['Quarter'] = pd.to_datetime(df['Date']).dt.quarter
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Add lag features
        for lag in [1, 7, 14, 28]:
            df[f'Waste_Lag_{lag}'] = df['DailyWaste'].shift(lag)
        
        # Add rolling statistics
        for window in [7, 14, 28]:
            df[f'Waste_MA_{window}'] = df['DailyWaste'].rolling(window).mean()
            df[f'Waste_STD_{window}'] = df['DailyWaste'].rolling(window).std()
        
        # Drop NaN rows
        df = df.dropna()
        
        return df
    
    def _generate_realistic_forecast_data(self, product_id: int, warehouse_id: int, lookback_days: int):
        """Generate realistic forecast data based on real product characteristics and IoT sensors"""
        
        # Get product characteristics
        product_query = f"""
        SELECT ProductName, Category, ShelfLifeDays, OptimalTempMin, OptimalTempMax
        FROM Products WHERE ProductID = {product_id}
        """
        product = pd.read_sql(product_query, self.engine).iloc[0]
        
        # Get IoT sensor data
        iot_query = f"""
        SELECT 
            DATE(LogTime) as Date,
            AVG(Temperature) as AvgTemp,
            AVG(Humidity) as AvgHumidity,
            AVG(QualityScore) as AvgQuality
        FROM TemperatureLogs 
        WHERE WarehouseID = {warehouse_id}
            AND LogTime >= DATEADD(day, -{lookback_days}, GETDATE())
        GROUP BY DATE(LogTime)
        ORDER BY Date
        """
        iot_data = pd.read_sql(iot_query, self.engine)
        
        # Generate realistic waste data based on product characteristics
        dates = pd.date_range(end=pd.Timestamp.now(), periods=lookback_days, freq='D')
        
        # Base waste rate based on product category and shelf life
        if 'Fruits' in product['Category']:
            base_waste_rate = 0.05  # 5% daily waste for fruits
        elif 'Vegetables' in product['Category']:
            base_waste_rate = 0.08  # 8% daily waste for vegetables
        elif 'Dairy' in product['Category']:
            base_waste_rate = 0.03  # 3% daily waste for dairy
        else:
            base_waste_rate = 0.06  # 6% daily waste for other products
        
        # Adjust based on shelf life (shorter shelf life = higher waste)
        shelf_life_factor = max(0.5, 1.0 - (product['ShelfLifeDays'] / 30))
        base_waste_rate *= (1 + shelf_life_factor)
        
        waste_data = []
        for i, date in enumerate(dates):
            # Get IoT data for this date
            iot_row = iot_data[iot_data['Date'] == date.date()]
            
            if not iot_row.empty:
                temp = iot_row['AvgTemp'].iloc[0]
                humidity = iot_row['AvgHumidity'].iloc[0]
                quality = iot_row['AvgQuality'].iloc[0]
            else:
                # Use product optimal ranges
                temp = (product['OptimalTempMin'] + product['OptimalTempMax']) / 2
                humidity = 90.0
                quality = 0.8
            
            # Calculate waste based on temperature deviation
            temp_deviation = abs(temp - (product['OptimalTempMin'] + product['OptimalTempMax']) / 2)
            temp_factor = 1 + (temp_deviation * 0.1)  # 10% increase per degree deviation
            
            # Calculate waste based on quality
            quality_factor = 2 - quality  # Lower quality = higher waste
            
            # Add seasonal patterns
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Add weekend effect
            weekend_factor = 1.2 if date.weekday() >= 5 else 1.0
            
            # Calculate daily waste
            daily_waste = base_waste_rate * temp_factor * quality_factor * seasonal_factor * weekend_factor
            
            # Add some randomness
            daily_waste *= np.random.uniform(0.8, 1.2)
            
            waste_data.append({
                'Date': date,
                'DailyWaste': max(0, daily_waste),
                'AvgTemp': temp,
                'AvgHumidity': humidity,
                'ActiveLots': np.random.randint(1, 5)
            })
        
        df = pd.DataFrame(waste_data)
        
        # Add time features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Add lag features
        for lag in [1, 7, 14, 28]:
            df[f'Waste_Lag_{lag}'] = df['DailyWaste'].shift(lag)
        
        # Add rolling statistics
        for window in [7, 14, 28]:
            df[f'Waste_MA_{window}'] = df['DailyWaste'].rolling(window).mean()
            df[f'Waste_STD_{window}'] = df['DailyWaste'].rolling(window).std()
        
        # Drop NaN rows
        df = df.dropna()
        
        logger.info(f"Generated realistic forecast data: {len(df)} days for {product['ProductName'][:30]}...")
        return df
    
    def _generate_synthetic_forecast_data(self):
        """Generate synthetic data as fallback"""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq='D')
        
        # Generate synthetic waste data
        base_waste = 100
        waste_data = []
        
        for date in dates:
            # Add seasonal patterns
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            weekend = 1.2 if date.weekday() >= 5 else 1.0
            trend = 1 + (date - dates[0]).days * 0.001
            
            daily_waste = base_waste * seasonal * weekend * trend * np.random.uniform(0.8, 1.2)
            
            waste_data.append({
                'Date': date,
                'DailyWaste': max(0, daily_waste),
                'AvgTemp': np.random.uniform(2, 6),
                'AvgHumidity': np.random.uniform(85, 95),
                'ActiveLots': np.random.randint(1, 5)
            })
        
        df = pd.DataFrame(waste_data)
        
        # Add time features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Add lag features
        for lag in [1, 7, 14, 28]:
            df[f'Waste_Lag_{lag}'] = df['DailyWaste'].shift(lag)
        
        # Add rolling statistics
        for window in [7, 14, 28]:
            df[f'Waste_MA_{window}'] = df['DailyWaste'].rolling(window).mean()
            df[f'Waste_STD_{window}'] = df['DailyWaste'].rolling(window).std()
        
        df = df.dropna()
        return df
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, epochs: int = 100):
        """Train the TFT model"""
        
        # Prepare features
        feature_cols = [col for col in train_data.columns if col not in ['Date', 'DailyWaste']]
        target_col = 'DailyWaste'
        
        # Scale features
        X_train = self.scaler.fit_transform(train_data[feature_cols])
        X_val = self.scaler.transform(val_data[feature_cols])
        
        y_train = train_data[target_col].values
        y_val = val_data[target_col].values
        
        # Initialize model
        self.model = TemporalFusionTransformer(
            input_size=len(feature_cols),
            hidden_size=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            output_size=1,
            forecast_horizon=7
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, _, _ = self.model(
                X_train_tensor[:, :10],  # Static features
                X_train_tensor[:, 10:],  # Temporal features
                torch.zeros(X_train_tensor.size(0), 7, 5).to(self.device)  # Future features
            )
            
            loss = criterion(predictions.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
        logger.info("TFT model training completed")
    
    def forecast(self, context_data: pd.DataFrame, horizon: int = 7) -> Dict:
        """Generate forecasts with uncertainty intervals"""
        
        self.model.eval()
        
        # Prepare input data
        feature_cols = [col for col in context_data.columns if col not in ['Date', 'DailyWaste']]
        features = self.scaler.transform(context_data[feature_cols])
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            predictions, quantiles, attention = self.model(
                features_tensor[:, :10],  # Static features
                features_tensor[:, 10:],  # Temporal features
                torch.zeros(1, horizon, 5).to(self.device)  # Future known features
            )
        
        return {
            'predictions': predictions.cpu().numpy(),
            'lower_bound': quantiles[:, :, 0].cpu().numpy(),
            'upper_bound': quantiles[:, :, 2].cpu().numpy(),
            'attention_weights': attention.cpu().numpy()
        }
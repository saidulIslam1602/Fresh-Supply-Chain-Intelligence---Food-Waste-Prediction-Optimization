"""
Advanced Feature Engineering Pipeline for Fresh Supply Chain Intelligence System
Creates sophisticated features for ML models including time series, domain-specific, and automated features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import warnings
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Time series features
    create_lag_features: bool = True
    lag_periods: List[int] = None
    create_rolling_features: bool = True
    rolling_windows: List[int] = None
    
    # Domain-specific features
    create_supply_chain_features: bool = True
    create_quality_features: bool = True
    create_economic_features: bool = True
    
    # Advanced features
    create_interaction_features: bool = True
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Automated feature generation
    auto_feature_generation: bool = True
    max_auto_features: int = 20
    
    # Feature selection
    feature_selection_method: str = 'mutual_info'  # 'mutual_info', 'f_regression', 'rfe', 'pca'
    max_features: int = 50
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 3, 6, 12, 24]
        if self.rolling_windows is None:
            self.rolling_windows = [3, 6, 12, 24, 48]

class AdvancedFeatureEngineer:
    """Advanced feature engineering pipeline for supply chain data"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_importance_scores = {}
        self.generated_features = []
        self.feature_descriptions = {}
        self.feature_history = []  # Track feature engineering operations
        
    def engineer_features(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        logger.info(f"Starting feature engineering for dataset with {len(df)} rows, {len(df.columns)} columns")
        
        # Create a copy to avoid modifying original data
        engineered_df = df.copy()
        original_features = len(engineered_df.columns)
        
        # 1. Time series features
        if self.config.create_lag_features or self.config.create_rolling_features:
            engineered_df = self._create_time_series_features(engineered_df)
        
        # 2. Domain-specific features
        if self.config.create_supply_chain_features:
            engineered_df = self._create_supply_chain_features(engineered_df)
        
        if self.config.create_quality_features:
            engineered_df = self._create_quality_features(engineered_df)
        
        if self.config.create_economic_features:
            engineered_df = self._create_economic_features(engineered_df)
        
        # 3. Interaction features
        if self.config.create_interaction_features:
            engineered_df = self._create_interaction_features(engineered_df)
        
        # 4. Polynomial features
        if self.config.create_polynomial_features:
            engineered_df = self._create_polynomial_features(engineered_df)
        
        # 5. Automated feature generation
        if self.config.auto_feature_generation:
            engineered_df = self._auto_generate_features(engineered_df)
        
        # 6. Feature selection
        if target_column and target_column in engineered_df.columns:
            engineered_df = self._select_best_features(engineered_df, target_column)
        
        # 7. Final cleanup
        engineered_df = self._cleanup_features(engineered_df)
        
        new_features = len(engineered_df.columns)
        logger.info(f"Feature engineering completed: {original_features} → {new_features} features")
        
        return engineered_df
    
    def _create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time series features"""
        logger.info("Creating time series features...")
        
        # Identify time columns
        time_columns = df.select_dtypes(include=['datetime64']).columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for time_col in time_columns:
            # Sort by time for proper lag calculation
            df = df.sort_values(time_col)
            
            # Basic time features
            base_name = time_col.replace('Time', '').replace('Date', '')
            df[f'{base_name}_Hour'] = df[time_col].dt.hour
            df[f'{base_name}_DayOfWeek'] = df[time_col].dt.dayofweek
            df[f'{base_name}_Month'] = df[time_col].dt.month
            df[f'{base_name}_Quarter'] = df[time_col].dt.quarter
            df[f'{base_name}_IsWeekend'] = (df[time_col].dt.dayofweek >= 5).astype(int)
            df[f'{base_name}_IsBusinessHour'] = ((df[time_col].dt.hour >= 8) & (df[time_col].dt.hour <= 17)).astype(int)
            
            # Seasonal features
            df[f'{base_name}_Season'] = ((df[time_col].dt.month % 12 + 3) // 3).astype(int)
            df[f'{base_name}_DayOfYear'] = df[time_col].dt.dayofyear
            df[f'{base_name}_WeekOfYear'] = df[time_col].dt.isocalendar().week
            
            # Cyclical encoding for periodic features
            df[f'{base_name}_Hour_sin'] = np.sin(2 * np.pi * df[time_col].dt.hour / 24)
            df[f'{base_name}_Hour_cos'] = np.cos(2 * np.pi * df[time_col].dt.hour / 24)
            df[f'{base_name}_DayOfWeek_sin'] = np.sin(2 * np.pi * df[time_col].dt.dayofweek / 7)
            df[f'{base_name}_DayOfWeek_cos'] = np.cos(2 * np.pi * df[time_col].dt.dayofweek / 7)
            df[f'{base_name}_Month_sin'] = np.sin(2 * np.pi * df[time_col].dt.month / 12)
            df[f'{base_name}_Month_cos'] = np.cos(2 * np.pi * df[time_col].dt.month / 12)
            
            self._add_feature_description(f'{base_name}_Hour_sin', 'Cyclical encoding of hour (sine)')
            self._add_feature_description(f'{base_name}_DayOfWeek_sin', 'Cyclical encoding of day of week (sine)')
        
        # Lag features for numeric columns
        if self.config.create_lag_features:
            for col in numeric_columns:
                if col in ['ProductID', 'WarehouseID']:  # Skip ID columns
                    continue
                
                for lag in self.config.lag_periods:
                    lag_col = f'{col}_lag_{lag}'
                    df[lag_col] = df[col].shift(lag)
                    self._add_feature_description(lag_col, f'{col} value {lag} periods ago')
        
        # Rolling window features
        if self.config.create_rolling_features:
            for col in numeric_columns:
                if col in ['ProductID', 'WarehouseID']:  # Skip ID columns
                    continue
                
                for window in self.config.rolling_windows:
                    # Rolling statistics
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                    
                    # Rolling range and coefficient of variation
                    df[f'{col}_rolling_range_{window}'] = (
                        df[f'{col}_rolling_max_{window}'] - df[f'{col}_rolling_min_{window}']
                    )
                    df[f'{col}_rolling_cv_{window}'] = (
                        df[f'{col}_rolling_std_{window}'] / df[f'{col}_rolling_mean_{window}']
                    ).fillna(0)
                    
                    # Trend features
                    df[f'{col}_trend_{window}'] = df[col] - df[f'{col}_rolling_mean_{window}']
                    
                    self._add_feature_description(f'{col}_rolling_mean_{window}', f'{col} rolling mean over {window} periods')
                    self._add_feature_description(f'{col}_trend_{window}', f'{col} deviation from {window}-period trend')
        
        return df
    
    def create_time_features(self, df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Public method to create time-based features from a timestamp column"""
        result_df = df.copy()
        
        if timestamp_column not in result_df.columns:
            return result_df
        
        # Convert to datetime if not already
        result_df[timestamp_column] = pd.to_datetime(result_df[timestamp_column])
        
        # Extract time components
        result_df['hour'] = result_df[timestamp_column].dt.hour
        result_df['day'] = result_df[timestamp_column].dt.day
        result_df['month'] = result_df[timestamp_column].dt.month
        result_df['year'] = result_df[timestamp_column].dt.year
        result_df['dayofweek'] = result_df[timestamp_column].dt.dayofweek
        result_df['quarter'] = result_df[timestamp_column].dt.quarter
        result_df['is_weekend'] = (result_df[timestamp_column].dt.dayofweek >= 5).astype(int)
        
        self.feature_history.append(f"Created time features from {timestamp_column}")
        return result_df
    
    def create_lag_features(self, df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
        """Public method to create lag features for a specified column"""
        result_df = df.copy()
        
        if column not in result_df.columns:
            return result_df
        
        for lag in lags:
            lag_col = f'{column}_lag_{lag}'
            result_df[lag_col] = result_df[column].shift(lag)
            self.feature_history.append(f"Created lag feature {lag_col}")
        
        return result_df
    
    def create_rolling_features(self, df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
        """Public method to create rolling window features for a specified column"""
        result_df = df.copy()
        
        if column not in result_df.columns:
            return result_df
        
        for window in windows:
            result_df[f'{column}_rolling_mean_{window}'] = result_df[column].rolling(window=window, min_periods=1).mean()
            result_df[f'{column}_rolling_std_{window}'] = result_df[column].rolling(window=window, min_periods=1).std()
            self.feature_history.append(f"Created rolling features for {column} with window {window}")
        
        return result_df
    
    def create_supply_chain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public method to create supply chain domain-specific features"""
        return self._create_supply_chain_features(df)
    
    def create_interaction_features(self, df: pd.DataFrame, interaction_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Public method to create interaction features between specified column pairs"""
        result_df = df.copy()
        
        for col1, col2 in interaction_pairs:
            if col1 in result_df.columns and col2 in result_df.columns:
                interaction_col = f'{col1}_x_{col2}'
                result_df[interaction_col] = result_df[col1] * result_df[col2]
                self.feature_history.append(f"Created interaction feature {interaction_col}")
        
        return result_df
    
    def _create_supply_chain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific supply chain features"""
        logger.info("Creating supply chain features...")
        
        # Temperature compliance features
        if 'Temperature' in df.columns:
            # Optimal temperature range compliance
            if 'OptimalTempMin' in df.columns and 'OptimalTempMax' in df.columns:
                df['Temp_InOptimalRange'] = (
                    (df['Temperature'] >= df['OptimalTempMin']) & 
                    (df['Temperature'] <= df['OptimalTempMax'])
                ).astype(int)
                
                df['Temp_DeviationFromOptimal'] = np.minimum(
                    np.abs(df['Temperature'] - df['OptimalTempMin']),
                    np.abs(df['Temperature'] - df['OptimalTempMax'])
                )
                
                df['Temp_OptimalMidpoint'] = (df['OptimalTempMin'] + df['OptimalTempMax']) / 2
                df['Temp_DeviationFromMidpoint'] = np.abs(df['Temperature'] - df['Temp_OptimalMidpoint'])
                
                self._add_feature_description('Temp_InOptimalRange', 'Whether temperature is within optimal range')
                self._add_feature_description('Temp_DeviationFromOptimal', 'Distance from optimal temperature range')
            
            # Cold chain violation indicators
            df['Temp_ColdChainViolation'] = ((df['Temperature'] < 0) | (df['Temperature'] > 8)).astype(int)
            df['Temp_SevereViolation'] = ((df['Temperature'] < -2) | (df['Temperature'] > 12)).astype(int)
            df['Temp_RiskLevel'] = pd.cut(df['Temperature'], 
                                        bins=[-np.inf, 0, 2, 6, 8, np.inf],
                                        labels=[4, 1, 0, 1, 3])  # Risk levels: 0=optimal, 4=critical
            
            # Temperature stability
            if 'Temperature_rolling_std_6' in df.columns:
                df['Temp_Stability'] = 1 / (1 + df['Temperature_rolling_std_6'])  # Higher = more stable
                self._add_feature_description('Temp_Stability', 'Temperature stability score (higher = more stable)')
        
        # Shelf life and freshness features
        if 'ShelfLifeDays' in df.columns:
            df['ShelfLife_Category'] = pd.cut(df['ShelfLifeDays'],
                                            bins=[0, 3, 7, 14, 30, np.inf],
                                            labels=[0, 1, 2, 3, 4])  # 0=very short, 4=very long
            
            df['ShelfLife_Risk'] = np.where(df['ShelfLifeDays'] <= 3, 1, 0)
            df['ShelfLife_LogDays'] = np.log1p(df['ShelfLifeDays'])  # Log transformation for skewed data
            
            # Time-dependent freshness (if we have production/expiry dates)
            if 'ProductionDate' in df.columns:
                current_date = datetime.now().date()
                df['DaysFromProduction'] = (current_date - pd.to_datetime(df['ProductionDate']).dt.date).dt.days
                df['FreshnessRatio'] = 1 - (df['DaysFromProduction'] / df['ShelfLifeDays'])
                df['FreshnessRatio'] = np.clip(df['FreshnessRatio'], 0, 1)
                
                self._add_feature_description('FreshnessRatio', 'Remaining freshness as ratio (1=fresh, 0=expired)')
        
        # Humidity optimization features
        if 'Humidity' in df.columns:
            df['Humidity_Optimal'] = ((df['Humidity'] >= 85) & (df['Humidity'] <= 95)).astype(int)
            df['Humidity_TooLow'] = (df['Humidity'] < 80).astype(int)
            df['Humidity_TooHigh'] = (df['Humidity'] > 98).astype(int)
            df['Humidity_DeviationFromOptimal'] = np.minimum(
                np.abs(df['Humidity'] - 85),
                np.abs(df['Humidity'] - 95)
            )
            
            # Mold risk calculation
            if 'Temperature' in df.columns:
                # Higher temperature + higher humidity = higher mold risk
                df['MoldRisk'] = (df['Temperature'] * df['Humidity'] / 100) / 10
                df['MoldRisk'] = np.clip(df['MoldRisk'], 0, 1)
                self._add_feature_description('MoldRisk', 'Mold growth risk based on temperature and humidity')
        
        # Gas level features
        if 'CO2Level' in df.columns:
            df['CO2_Normal'] = ((df['CO2Level'] >= 300) & (df['CO2Level'] <= 600)).astype(int)
            df['CO2_High'] = (df['CO2Level'] > 1000).astype(int)
        
        if 'EthyleneLevel' in df.columns:
            df['Ethylene_Low'] = (df['EthyleneLevel'] <= 0.05).astype(int)
            df['Ethylene_High'] = (df['EthyleneLevel'] > 0.1).astype(int)
            
            # Ripening acceleration risk
            df['RipeningRisk'] = np.clip(df['EthyleneLevel'] * 10, 0, 1)
            self._add_feature_description('RipeningRisk', 'Risk of accelerated ripening due to ethylene')
        
        # Warehouse efficiency features
        if 'WarehouseID' in df.columns:
            # Warehouse performance metrics (if we have historical data)
            warehouse_stats = df.groupby('WarehouseID').agg({
                'Temperature': ['mean', 'std'],
                'Humidity': ['mean', 'std'],
                'QualityScore': 'mean' if 'QualityScore' in df.columns else lambda x: 0.8
            }).round(3)
            
            # Flatten column names
            warehouse_stats.columns = ['_'.join(col).strip() for col in warehouse_stats.columns]
            
            # Merge back to main dataframe
            for col in warehouse_stats.columns:
                df[f'Warehouse_{col}'] = df['WarehouseID'].map(warehouse_stats[col])
        
        return df
    
    def _create_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create quality-related features"""
        logger.info("Creating quality features...")
        
        if 'QualityScore' in df.columns:
            # Quality categories
            df['Quality_Category'] = pd.cut(df['QualityScore'],
                                          bins=[0, 0.3, 0.6, 0.8, 1.0],
                                          labels=[0, 1, 2, 3])  # 0=poor, 3=excellent
            
            df['Quality_Poor'] = (df['QualityScore'] < 0.5).astype(int)
            df['Quality_Excellent'] = (df['QualityScore'] >= 0.9).astype(int)
            
            # Quality trend (if we have time series data)
            if 'QualityScore_lag_1' in df.columns:
                df['Quality_Trend'] = df['QualityScore'] - df['QualityScore_lag_1']
                df['Quality_Improving'] = (df['Quality_Trend'] > 0.05).astype(int)
                df['Quality_Deteriorating'] = (df['Quality_Trend'] < -0.05).astype(int)
                
                self._add_feature_description('Quality_Trend', 'Change in quality score from previous period')
            
            # Quality volatility
            if 'QualityScore_rolling_std_6' in df.columns:
                df['Quality_Volatility'] = df['QualityScore_rolling_std_6']
                df['Quality_Stable'] = (df['Quality_Volatility'] < 0.1).astype(int)
        
        # Environmental impact on quality
        if all(col in df.columns for col in ['Temperature', 'Humidity', 'QualityScore']):
            # Quality prediction based on environmental conditions
            df['EnvQuality_TempImpact'] = np.where(
                (df['Temperature'] >= 2) & (df['Temperature'] <= 6), 1.0,
                1.0 - np.abs(df['Temperature'] - 4) * 0.1
            )
            df['EnvQuality_HumidityImpact'] = np.where(
                (df['Humidity'] >= 85) & (df['Humidity'] <= 95), 1.0,
                1.0 - np.abs(df['Humidity'] - 90) * 0.01
            )
            
            df['EnvQuality_Combined'] = (df['EnvQuality_TempImpact'] + df['EnvQuality_HumidityImpact']) / 2
            df['EnvQuality_Deviation'] = np.abs(df['QualityScore'] - df['EnvQuality_Combined'])
            
            self._add_feature_description('EnvQuality_Combined', 'Expected quality based on environmental conditions')
        
        return df
    
    def _create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create economic and business features"""
        logger.info("Creating economic features...")
        
        # Profitability features
        if 'UnitCost' in df.columns and 'UnitPrice' in df.columns:
            df['Profit_Margin'] = (df['UnitPrice'] - df['UnitCost']) / df['UnitPrice']
            df['Profit_Absolute'] = df['UnitPrice'] - df['UnitCost']
            df['Price_CostRatio'] = df['UnitPrice'] / df['UnitCost']
            df['Is_Profitable'] = (df['UnitPrice'] > df['UnitCost']).astype(int)
            df['HighMargin_Product'] = (df['Profit_Margin'] > 0.3).astype(int)
            
            self._add_feature_description('Profit_Margin', 'Profit margin as percentage of selling price')
            self._add_feature_description('Price_CostRatio', 'Ratio of selling price to cost')
        
        # Waste cost calculation
        if all(col in df.columns for col in ['UnitCost', 'QualityScore']):
            # Estimated waste cost based on quality deterioration
            df['WasteCost_Risk'] = df['UnitCost'] * (1 - df['QualityScore'])
            df['WasteCost_Category'] = pd.cut(df['WasteCost_Risk'],
                                            bins=[0, 0.5, 1.0, 2.0, np.inf],
                                            labels=[0, 1, 2, 3])  # 0=low risk, 3=high risk
            
            self._add_feature_description('WasteCost_Risk', 'Potential waste cost based on quality deterioration')
        
        # Inventory turnover features
        if 'Quantity' in df.columns and 'ShelfLifeDays' in df.columns:
            df['Inventory_Velocity'] = df['Quantity'] / df['ShelfLifeDays']  # Units per day
            df['Inventory_Risk'] = df['Quantity'] * (1 / df['ShelfLifeDays'])  # Higher = more risk
            
            self._add_feature_description('Inventory_Velocity', 'Inventory turnover rate (units per day)')
        
        # Category-based economic features
        if 'Category' in df.columns:
            # Category profitability (if we have cost/price data)
            if 'Profit_Margin' in df.columns:
                category_margins = df.groupby('Category')['Profit_Margin'].mean()
                df['Category_AvgMargin'] = df['Category'].map(category_margins)
                df['Margin_vs_Category'] = df['Profit_Margin'] - df['Category_AvgMargin']
                
                self._add_feature_description('Margin_vs_Category', 'Product margin vs category average')
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        logger.info("Creating interaction features...")
        
        # Define important feature pairs for supply chain
        interaction_pairs = [
            ('Temperature', 'Humidity'),
            ('Temperature', 'QualityScore'),
            ('Humidity', 'QualityScore'),
            ('ShelfLifeDays', 'Temperature'),
            ('UnitCost', 'QualityScore'),
            ('CO2Level', 'EthyleneLevel'),
            ('Temperature', 'ShelfLifeDays'),
            ('Profit_Margin', 'QualityScore')
        ]
        
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplicative interaction
                interaction_col = f'{col1}_x_{col2}'
                df[interaction_col] = df[col1] * df[col2]
                self._add_feature_description(interaction_col, f'Interaction between {col1} and {col2}')
                
                # Ratio interaction (avoid division by zero)
                if (df[col2] != 0).all():
                    ratio_col = f'{col1}_div_{col2}'
                    df[ratio_col] = df[col1] / df[col2]
                    self._add_feature_description(ratio_col, f'Ratio of {col1} to {col2}')
                
                # Difference interaction
                diff_col = f'{col1}_minus_{col2}'
                df[diff_col] = df[col1] - df[col2]
                self._add_feature_description(diff_col, f'Difference between {col1} and {col2}')
        
        # Three-way interactions for critical combinations
        if all(col in df.columns for col in ['Temperature', 'Humidity', 'QualityScore']):
            df['Temp_Humidity_Quality'] = df['Temperature'] * df['Humidity'] * df['QualityScore']
            self._add_feature_description('Temp_Humidity_Quality', 'Three-way interaction: Temperature × Humidity × Quality')
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for key variables"""
        logger.info("Creating polynomial features...")
        
        # Select key numeric columns for polynomial features
        key_columns = []
        for col in ['Temperature', 'Humidity', 'QualityScore', 'ShelfLifeDays', 'Profit_Margin']:
            if col in df.columns:
                key_columns.append(col)
        
        if not key_columns:
            return df
        
        # Limit to avoid feature explosion
        selected_columns = key_columns[:3]
        
        for col in selected_columns:
            # Quadratic terms
            df[f'{col}_squared'] = df[col] ** 2
            self._add_feature_description(f'{col}_squared', f'Quadratic term for {col}')
            
            if self.config.polynomial_degree >= 3:
                # Cubic terms
                df[f'{col}_cubed'] = df[col] ** 3
                self._add_feature_description(f'{col}_cubed', f'Cubic term for {col}')
            
            # Square root (for positive values)
            if (df[col] >= 0).all():
                df[f'{col}_sqrt'] = np.sqrt(df[col])
                self._add_feature_description(f'{col}_sqrt', f'Square root of {col}')
        
        return df
    
    def _auto_generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically generate features using statistical methods"""
        logger.info("Auto-generating features...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Remove ID columns and target-like columns
        feature_columns = [col for col in numeric_columns 
                          if not any(id_term in col.lower() for id_term in ['id', 'target', 'label'])]
        
        if len(feature_columns) < 2:
            return df
        
        # Limit to prevent feature explosion
        selected_columns = feature_columns[:min(8, len(feature_columns))]
        
        # Statistical transformations
        for col in selected_columns[:5]:  # Limit transformations
            if col not in df.columns:
                continue
            
            # Log transformation (for positive values)
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log1p(df[col])
                self._add_feature_description(f'{col}_log', f'Log transformation of {col}')
            
            # Standardized values (z-score)
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
                self._add_feature_description(f'{col}_zscore', f'Z-score of {col}')
            
            # Percentile rank
            df[f'{col}_percentile'] = df[col].rank(pct=True)
            self._add_feature_description(f'{col}_percentile', f'Percentile rank of {col}')
        
        # Binning features for continuous variables
        for col in selected_columns[:3]:
            if col not in df.columns:
                continue
            
            # Equal-width binning
            try:
                df[f'{col}_bin'] = pd.cut(df[col], bins=5, labels=False)
                self._add_feature_description(f'{col}_bin', f'Binned version of {col} (5 bins)')
            except:
                pass  # Skip if binning fails
        
        return df
    
    def _select_best_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Select best features using various methods"""
        logger.info("Selecting best features...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Convert target to numeric if needed
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
        
        # Remove features with too many missing values
        missing_threshold = 0.5
        valid_features = X.columns[X.isnull().mean() < missing_threshold]
        X = X[valid_features]
        
        if X.empty:
            logger.warning("No valid features for selection")
            return df
        
        # Fill remaining missing values
        X = X.fillna(X.median())
        
        # Feature selection based on method
        if self.config.feature_selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, 
                                 k=min(self.config.max_features, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store feature scores
            self.feature_importance_scores = dict(zip(X.columns, selector.scores_))
            
        elif self.config.feature_selection_method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, 
                                 k=min(self.config.max_features, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            self.feature_importance_scores = dict(zip(X.columns, selector.scores_))
            
        elif self.config.feature_selection_method == 'rfe':
            # Recursive Feature Elimination with Random Forest
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(self.config.max_features, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Get feature importance from the estimator
            if hasattr(selector.estimator_, 'feature_importances_'):
                self.feature_importance_scores = dict(zip(selected_features, 
                                                        selector.estimator_.feature_importances_))
        
        elif self.config.feature_selection_method == 'pca':
            # Principal Component Analysis
            n_components = min(self.config.max_features, len(X.columns))
            pca = PCA(n_components=n_components)
            X_selected = pca.fit_transform(X)
            
            # Create new feature names for PCA components
            selected_features = [f'PC_{i+1}' for i in range(n_components)]
            
            # Store explained variance as importance scores
            self.feature_importance_scores = dict(zip(selected_features, pca.explained_variance_ratio_))
        
        else:
            # Default: keep all features
            selected_features = X.columns.tolist()
            X_selected = X.values
        
        # Create result dataframe
        if self.config.feature_selection_method == 'pca':
            result_df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        else:
            result_df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        
        result_df[target_column] = y.values
        
        logger.info(f"Selected {len(selected_features)} features from {len(X.columns)}")
        return result_df
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of engineered features"""
        logger.info("Cleaning up features...")
        
        # Remove features with zero variance
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        zero_var_cols = [col for col in numeric_columns if df[col].var() == 0]
        if zero_var_cols:
            df = df.drop(columns=zero_var_cols)
            logger.info(f"Removed {len(zero_var_cols)} zero-variance features")
        
        # Remove highly correlated features
        correlation_threshold = 0.95
        corr_matrix = df[numeric_columns].corr().abs()
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for col1, col2 in high_corr_pairs:
            # Keep the feature with higher importance score (if available)
            if col1 in self.feature_importance_scores and col2 in self.feature_importance_scores:
                if self.feature_importance_scores[col1] < self.feature_importance_scores[col2]:
                    features_to_remove.add(col1)
                else:
                    features_to_remove.add(col2)
            else:
                features_to_remove.add(col2)  # Default: remove second feature
        
        if features_to_remove:
            df = df.drop(columns=list(features_to_remove))
            logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
    
    def _add_feature_description(self, feature_name: str, description: str):
        """Add description for a generated feature"""
        self.feature_descriptions[feature_name] = description
        if feature_name not in self.generated_features:
            self.generated_features.append(feature_name)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features"""
        if not self.feature_importance_scores:
            return {}
        
        sorted_features = sorted(self.feature_importance_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])
    
    def get_feature_report(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering report"""
        return {
            'total_features_generated': len(self.generated_features),
            'feature_categories': {
                'time_series': len([f for f in self.generated_features if any(term in f for term in ['lag', 'rolling', 'trend'])]),
                'supply_chain': len([f for f in self.generated_features if any(term in f for term in ['Temp_', 'Quality_', 'Shelf'])]),
                'economic': len([f for f in self.generated_features if any(term in f for term in ['Profit_', 'Cost_', 'Margin'])]),
                'interaction': len([f for f in self.generated_features if '_x_' in f or '_div_' in f]),
                'polynomial': len([f for f in self.generated_features if any(term in f for term in ['squared', 'cubed', 'sqrt'])]),
                'automated': len([f for f in self.generated_features if any(term in f for term in ['log', 'zscore', 'percentile', 'bin'])])
            },
            'top_features': self.get_feature_importance(20),
            'feature_descriptions': self.feature_descriptions,
            'config_used': self.config.__dict__
        }

# Usage example
def engineer_supply_chain_features(df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, AdvancedFeatureEngineer]:
    """Example usage of advanced feature engineering for supply chain data"""
    
    # Configure feature engineering
    config = FeatureConfig(
        create_lag_features=True,
        lag_periods=[1, 3, 6, 12],
        create_rolling_features=True,
        rolling_windows=[3, 6, 12, 24],
        create_supply_chain_features=True,
        create_quality_features=True,
        create_economic_features=True,
        create_interaction_features=True,
        auto_feature_generation=True,
        feature_selection_method='mutual_info',
        max_features=40
    )
    
    # Create feature engineer
    engineer = AdvancedFeatureEngineer(config)
    
    # Engineer features
    engineered_df = engineer.engineer_features(df, target_column)
    
    # Get report
    report = engineer.get_feature_report()
    logger.info(f"Feature engineering report: {report}")
    
    return engineered_df, engineer

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'LogTime': pd.date_range('2024-01-01', periods=100, freq='H'),
        'Temperature': np.random.normal(4, 1, 100),
        'Humidity': np.random.normal(90, 5, 100),
        'QualityScore': np.random.uniform(0.6, 1.0, 100),
        'ShelfLifeDays': np.random.randint(3, 21, 100),
        'UnitCost': np.random.uniform(1, 5, 100),
        'UnitPrice': np.random.uniform(2, 8, 100),
        'Category': np.random.choice(['Fruits', 'Vegetables', 'Dairy'], 100),
        'WarehouseID': np.random.randint(1, 6, 100)
    })
    
    # Add some realistic correlations
    sample_data['QualityScore'] = np.clip(
        0.9 - np.abs(sample_data['Temperature'] - 4) * 0.1 + np.random.normal(0, 0.05, 100),
        0, 1
    )
    
    engineered_data, engineer = engineer_supply_chain_features(sample_data, 'QualityScore')
    print(f"Original features: {len(sample_data.columns)}")
    print(f"Engineered features: {len(engineered_data.columns)}")
    print(f"Top 10 features: {list(engineer.get_feature_importance(10).keys())}")
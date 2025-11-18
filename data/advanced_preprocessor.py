"""
Advanced Data Preprocessing Pipeline for Fresh Supply Chain Intelligence System
Provides sophisticated data cleaning, transformation, and feature engineering capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import re
import warnings
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Imputation strategies
    numeric_imputation: str = 'knn'  # 'mean', 'median', 'knn', 'forward_fill'
    categorical_imputation: str = 'mode'  # 'mode', 'constant', 'drop'
    
    # Scaling methods
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust', 'none'
    
    # Outlier handling
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest', 'none'
    outlier_threshold: float = 1.5
    
    # Feature engineering
    create_time_features: bool = True
    create_interaction_features: bool = True
    polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Feature selection
    feature_selection: bool = True
    max_features: int = 50
    selection_method: str = 'mutual_info'  # 'mutual_info', 'f_regression', 'variance'

class AdvancedPreprocessor:
    """Advanced data preprocessing pipeline with multiple strategies"""
    
    def __init__(self, config: PreprocessingConfig = None, 
                 imputation_strategy: str = None,
                 outlier_method: str = None,
                 scaling_method: str = None):
        self.config = config or PreprocessingConfig()
        
        # Set direct attributes for backward compatibility with tests
        if imputation_strategy is not None:
            self.config.numeric_imputation = imputation_strategy
        if outlier_method is not None:
            self.config.outlier_method = outlier_method
        if scaling_method is not None:
            self.config.scaling_method = scaling_method
        
        # Direct attributes for easy access (mapped to config)
        self.imputation_strategy = self.config.numeric_imputation
        self.outlier_method = self.config.outlier_method
        self.scaling_method = self.config.scaling_method
        
        self.fitted_transformers = {}
        self.feature_names = []
        self.preprocessing_log = []
        
    def fit_transform(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """Fit preprocessing pipeline and transform data"""
        logger.info(f"Starting advanced preprocessing for dataset with {len(df)} rows, {len(df.columns)} columns")
        
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # 1. Data type optimization
        processed_df = self._optimize_data_types(processed_df)
        
        # 2. Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # 3. Text preprocessing for string columns
        processed_df = self._preprocess_text_columns(processed_df)
        
        # 4. Handle outliers
        processed_df = self._handle_outliers(processed_df)
        
        # 5. Feature engineering
        processed_df = self._engineer_features(processed_df)
        
        # 6. Encode categorical variables
        processed_df = self._encode_categorical_variables(processed_df)
        
        # 7. Scale numerical features
        processed_df = self._scale_numerical_features(processed_df)
        
        # 8. Feature selection (if target is provided)
        if target_column and target_column in processed_df.columns:
            processed_df = self._select_features(processed_df, target_column)
        
        # 9. Final validation
        processed_df = self._final_validation(processed_df)
        
        self.feature_names = processed_df.columns.tolist()
        logger.info(f"Preprocessing completed. Final dataset: {len(processed_df)} rows, {len(processed_df.columns)} columns")
        
        return processed_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessing pipeline"""
        if not self.fitted_transformers:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        processed_df = df.copy()
        
        # Apply the same transformations in order
        processed_df = self._optimize_data_types(processed_df)
        processed_df = self._handle_missing_values(processed_df, fit=False)
        processed_df = self._preprocess_text_columns(processed_df, fit=False)
        processed_df = self._handle_outliers(processed_df, fit=False)
        processed_df = self._engineer_features(processed_df, fit=False)
        processed_df = self._encode_categorical_variables(processed_df, fit=False)
        processed_df = self._scale_numerical_features(processed_df, fit=False)
        
        # Ensure same columns as training data
        for col in self.feature_names:
            if col not in processed_df.columns:
                processed_df[col] = 0
        
        processed_df = processed_df[self.feature_names]
        
        return processed_df
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public method to optimize data types"""
        return self._optimize_data_types(df)
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        logger.info("Optimizing data types...")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isnull().all():
                    df[col] = numeric_series
                else:
                    # Check if it's a datetime
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        # Keep as string but optimize
                        df[col] = df[col].astype('category')
            
            elif df[col].dtype in ['int64', 'int32']:
                # Downcast integers
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            elif df[col].dtype in ['float64', 'float32']:
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        self._log_step("Data type optimization", f"Optimized {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public method to handle missing values"""
        return self._handle_missing_values(df, fit=True)
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Advanced missing value imputation"""
        logger.info("Handling missing values...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        # Handle numeric missing values
        if len(numeric_columns) > 0:
            if fit:
                if self.config.numeric_imputation == 'knn':
                    self.fitted_transformers['numeric_imputer'] = KNNImputer(n_neighbors=5)
                else:
                    strategy = self.config.numeric_imputation
                    if strategy not in ['mean', 'median', 'most_frequent']:
                        strategy = 'median'
                    self.fitted_transformers['numeric_imputer'] = SimpleImputer(strategy=strategy)
                
                df[numeric_columns] = self.fitted_transformers['numeric_imputer'].fit_transform(df[numeric_columns])
            else:
                df[numeric_columns] = self.fitted_transformers['numeric_imputer'].transform(df[numeric_columns])
        
        # Handle categorical missing values
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                if df[col].isnull().any():
                    if fit:
                        if self.config.categorical_imputation == 'mode':
                            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                            self.fitted_transformers[f'{col}_mode'] = mode_value
                        else:
                            self.fitted_transformers[f'{col}_mode'] = 'Unknown'
                    
                    fill_value = self.fitted_transformers.get(f'{col}_mode', 'Unknown')
                    df[col] = df[col].fillna(fill_value)
        
        missing_count = df.isnull().sum().sum()
        self._log_step("Missing value imputation", f"Remaining missing values: {missing_count}")
        return df
    
    def _preprocess_text_columns(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Advanced text preprocessing"""
        logger.info("Preprocessing text columns...")
        
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            if df[col].dtype == 'object':
                # Basic text cleaning
                df[col] = df[col].astype(str)
                df[col] = df[col].str.strip()
                df[col] = df[col].str.lower()
                
                # Remove special characters for product names/codes
                if 'name' in col.lower() or 'code' in col.lower():
                    df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
                
                # Standardize common terms
                if 'category' in col.lower():
                    category_mapping = {
                        'fruit': 'Fruits',
                        'fruits': 'Fruits',
                        'vegetable': 'Vegetables',
                        'vegetables': 'Vegetables',
                        'dairy': 'Dairy',
                        'milk': 'Dairy'
                    }
                    
                    if fit:
                        self.fitted_transformers[f'{col}_mapping'] = category_mapping
                    
                    mapping = self.fitted_transformers.get(f'{col}_mapping', {})
                    df[col] = df[col].replace(mapping)
        
        self._log_step("Text preprocessing", f"Processed {len(text_columns)} text columns")
        return df
    
    def normalize_text_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Public method to normalize text data in specified columns"""
        result_df = df.copy()
        for col in columns:
            if col in result_df.columns:
                # Only process non-null values, preserve NaN/None
                mask = result_df[col].notna()
                if mask.any():
                    # Convert non-null values to string, then normalize
                    result_df.loc[mask, col] = result_df.loc[mask, col].astype(str)
                    result_df.loc[mask, col] = result_df.loc[mask, col].str.strip()
                    result_df.loc[mask, col] = result_df.loc[mask, col].str.lower()
        return result_df
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Public method to normalize (scale) numerical data in specified columns"""
        result_df = df.copy()
        
        numeric_columns = [col for col in columns if col in result_df.columns and 
                          result_df[col].dtype in [np.number]]
        
        if len(numeric_columns) == 0:
            return result_df
        
        if self.config.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.config.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.config.scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        result_df[numeric_columns] = scaler.fit_transform(result_df[numeric_columns])
        return result_df
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> List[int]:
        """Detect outliers in specified columns and return their indices"""
        outlier_indices = []
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if self.config.outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division issues
                    lower_bound = Q1 - self.config.outlier_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_threshold * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                    # Convert to integer indices (0-based)
                    outlier_indices.extend([int(idx) for idx in outliers])
            
            elif self.config.outlier_method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                if std_val > 0:
                    z_scores = np.abs((df[col] - mean_val) / std_val)
                    outliers = df[z_scores > 3].index.tolist()
                    # Convert to integer indices (0-based)
                    outlier_indices.extend([int(idx) for idx in outliers])
        
        # Remove duplicates and return sorted list
        return sorted(list(set(outlier_indices)))
    
    def handle_outliers(self, df: pd.DataFrame, outlier_indices: List[int]) -> pd.DataFrame:
        """Handle outliers by removing rows at specified indices"""
        return df.drop(index=outlier_indices).reset_index(drop=True)
    
    def _handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Advanced outlier detection and handling"""
        logger.info("Handling outliers...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outliers_handled = 0
        
        for col in numeric_columns:
            if self.config.outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                if fit:
                    self.fitted_transformers[f'{col}_bounds'] = (lower_bound, upper_bound)
                
                bounds = self.fitted_transformers.get(f'{col}_bounds', (lower_bound, upper_bound))
                
                # Cap outliers instead of removing them
                outlier_mask = (df[col] < bounds[0]) | (df[col] > bounds[1])
                outliers_handled += outlier_mask.sum()
                
                df[col] = np.clip(df[col], bounds[0], bounds[1])
            
            elif self.config.outlier_method == 'zscore':
                if fit:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    self.fitted_transformers[f'{col}_zscore'] = (mean_val, std_val)
                
                mean_val, std_val = self.fitted_transformers.get(f'{col}_zscore', (df[col].mean(), df[col].std()))
                
                z_scores = np.abs((df[col] - mean_val) / std_val)
                outlier_mask = z_scores > 3
                outliers_handled += outlier_mask.sum()
                
                # Replace outliers with median
                median_val = df[col].median()
                df.loc[outlier_mask, col] = median_val
        
        self._log_step("Outlier handling", f"Handled {outliers_handled} outliers")
        return df
    
    def _engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Advanced feature engineering"""
        logger.info("Engineering features...")
        
        original_columns = len(df.columns)
        
        # Time-based features
        if self.config.create_time_features:
            df = self._create_time_features(df)
        
        # Domain-specific features for supply chain
        df = self._create_supply_chain_features(df)
        
        # Interaction features
        if self.config.create_interaction_features:
            df = self._create_interaction_features(df, fit)
        
        # Polynomial features (use sparingly)
        if self.config.polynomial_features:
            df = self._create_polynomial_features(df, fit)
        
        new_columns = len(df.columns)
        self._log_step("Feature engineering", f"Created {new_columns - original_columns} new features")
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_columns:
            base_name = col.replace('Time', '').replace('Date', '')
            
            # Basic time components
            df[f'{base_name}_Hour'] = df[col].dt.hour
            df[f'{base_name}_DayOfWeek'] = df[col].dt.dayofweek
            df[f'{base_name}_Month'] = df[col].dt.month
            df[f'{base_name}_Quarter'] = df[col].dt.quarter
            df[f'{base_name}_IsWeekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            
            # Business-relevant time features
            df[f'{base_name}_IsBusinessHour'] = ((df[col].dt.hour >= 8) & (df[col].dt.hour <= 17)).astype(int)
            df[f'{base_name}_Season'] = ((df[col].dt.month % 12 + 3) // 3).astype(int)
            
            # Time since reference point
            reference_date = df[col].min()
            df[f'{base_name}_DaysSinceStart'] = (df[col] - reference_date).dt.days
        
        return df
    
    def _create_supply_chain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for supply chain"""
        
        # Temperature-related features
        if 'Temperature' in df.columns:
            df['Temperature_Deviation'] = np.abs(df['Temperature'] - 4.0)  # Deviation from optimal 4°C
            df['Temperature_Risk'] = ((df['Temperature'] < 0) | (df['Temperature'] > 8)).astype(int)
        
        if 'OptimalTempMin' in df.columns and 'OptimalTempMax' in df.columns:
            df['Temperature_Range'] = df['OptimalTempMax'] - df['OptimalTempMin']
            df['Temperature_Midpoint'] = (df['OptimalTempMax'] + df['OptimalTempMin']) / 2
        
        # Shelf life features
        if 'ShelfLifeDays' in df.columns:
            df['ShelfLife_Category'] = pd.cut(df['ShelfLifeDays'], 
                                            bins=[0, 3, 7, 14, 30, float('inf')],
                                            labels=['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong'])
            df['ShelfLife_Risk'] = (df['ShelfLifeDays'] <= 3).astype(int)
        
        # Quality features
        if 'QualityScore' in df.columns:
            df['Quality_Category'] = pd.cut(df['QualityScore'],
                                          bins=[0, 0.3, 0.6, 0.8, 1.0],
                                          labels=['Poor', 'Fair', 'Good', 'Excellent'])
            df['Quality_Risk'] = (df['QualityScore'] < 0.5).astype(int)
        
        # Economic features
        if 'UnitCost' in df.columns and 'UnitPrice' in df.columns:
            df['Profit_Margin'] = (df['UnitPrice'] - df['UnitCost']) / df['UnitPrice']
            df['Price_Ratio'] = df['UnitPrice'] / df['UnitCost']
            df['Is_Profitable'] = (df['UnitPrice'] > df['UnitCost']).astype(int)
        
        # Humidity features
        if 'Humidity' in df.columns:
            df['Humidity_Optimal'] = ((df['Humidity'] >= 85) & (df['Humidity'] <= 95)).astype(int)
            df['Humidity_Risk'] = ((df['Humidity'] < 80) | (df['Humidity'] > 98)).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create interaction features between important variables"""
        
        # Define important feature pairs for supply chain
        interaction_pairs = [
            ('Temperature', 'Humidity'),
            ('Temperature', 'QualityScore'),
            ('ShelfLifeDays', 'Temperature'),
            ('UnitCost', 'ShelfLifeDays'),
            ('Temperature_Deviation', 'QualityScore')
        ]
        
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplicative interaction
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Ratio interaction (avoid division by zero)
                if (df[col2] != 0).all():
                    df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create polynomial features (use sparingly)"""
        
        # Only create polynomial features for key numeric columns
        key_numeric_cols = ['Temperature', 'Humidity', 'QualityScore', 'ShelfLifeDays']
        available_cols = [col for col in key_numeric_cols if col in df.columns]
        
        for col in available_cols[:3]:  # Limit to avoid feature explosion
            df[f'{col}_squared'] = df[col] ** 2
            if self.config.polynomial_degree >= 3:
                df[f'{col}_cubed'] = df[col] ** 3
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Advanced categorical encoding"""
        logger.info("Encoding categorical variables...")
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            unique_values = df[col].nunique()
            
            if unique_values <= 10:  # One-hot encode low cardinality
                if fit:
                    # Get unique values for consistent encoding
                    unique_vals = df[col].unique()
                    self.fitted_transformers[f'{col}_categories'] = unique_vals
                
                categories = self.fitted_transformers.get(f'{col}_categories', df[col].unique())
                
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col)
                
                # Ensure all expected columns exist
                for cat in categories:
                    col_name = f'{col}_{cat}'
                    if col_name not in dummies.columns:
                        dummies[col_name] = 0
                
                # Drop original column and add dummies
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
                
            else:  # Target encoding or frequency encoding for high cardinality
                if fit:
                    # Frequency encoding
                    freq_encoding = df[col].value_counts().to_dict()
                    self.fitted_transformers[f'{col}_frequency'] = freq_encoding
                
                freq_map = self.fitted_transformers.get(f'{col}_frequency', {})
                df[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
                df = df.drop(columns=[col])
        
        self._log_step("Categorical encoding", f"Encoded {len(categorical_columns)} categorical columns")
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        logger.info("Scaling numerical features...")
        
        if self.config.scaling_method == 'none':
            return df
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return df
        
        if fit:
            if self.config.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.config.scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif self.config.scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            self.fitted_transformers['scaler'] = scaler
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        else:
            scaler = self.fitted_transformers['scaler']
            df[numeric_columns] = scaler.transform(df[numeric_columns])
        
        self._log_step("Feature scaling", f"Scaled {len(numeric_columns)} numerical columns")
        return df
    
    def _select_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Intelligent feature selection"""
        logger.info("Performing feature selection...")
        
        if not self.config.feature_selection:
            return df
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Convert target to numeric if needed
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
        
        # Select features based on method
        if self.config.selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(self.config.max_features, len(X.columns)))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(self.config.max_features, len(X.columns)))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create new dataframe with selected features
        result_df = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        result_df[target_column] = y
        
        self._log_step("Feature selection", f"Selected {len(selected_features)} features from {len(X.columns)}")
        return result_df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup"""
        logger.info("Performing final validation...")
        
        # Remove any remaining infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Remove constant columns
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_columns:
            df = df.drop(columns=constant_columns)
            self._log_step("Constant column removal", f"Removed {len(constant_columns)} constant columns")
        
        # Ensure no duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def _log_step(self, step_name: str, details: str):
        """Log preprocessing step"""
        log_entry = {
            'timestamp': datetime.now(),
            'step': step_name,
            'details': details
        }
        self.preprocessing_log.append(log_entry)
        logger.info(f"{step_name}: {details}")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps performed"""
        return {
            'config': self.config.__dict__,
            'steps_performed': self.preprocessing_log,
            'final_features': self.feature_names,
            'transformers_fitted': list(self.fitted_transformers.keys())
        }
    
    def save_preprocessor(self, filepath: str):
        """Save fitted preprocessor for later use"""
        import pickle
        
        preprocessor_data = {
            'config': self.config,
            'fitted_transformers': self.fitted_transformers,
            'feature_names': self.feature_names,
            'preprocessing_log': self.preprocessing_log
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str):
        """Load previously fitted preprocessor"""
        import pickle
        
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        preprocessor = cls(preprocessor_data['config'])
        preprocessor.fitted_transformers = preprocessor_data['fitted_transformers']
        preprocessor.feature_names = preprocessor_data['feature_names']
        preprocessor.preprocessing_log = preprocessor_data['preprocessing_log']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor

# Usage example
def preprocess_supply_chain_data(df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """Example usage of advanced preprocessing for supply chain data"""
    
    # Configure preprocessing
    config = PreprocessingConfig(
        numeric_imputation='knn',
        categorical_imputation='mode',
        scaling_method='robust',
        outlier_method='iqr',
        outlier_threshold=1.5,
        create_time_features=True,
        create_interaction_features=True,
        feature_selection=True,
        max_features=30
    )
    
    # Create and fit preprocessor
    preprocessor = AdvancedPreprocessor(config)
    processed_df = preprocessor.fit_transform(df, target_column)
    
    # Get summary
    summary = preprocessor.get_preprocessing_summary()
    logger.info(f"Preprocessing completed: {summary}")
    
    return processed_df, preprocessor

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'Temperature': [2.5, 4.0, 6.5, 15.0, 3.2],  # One outlier
        'Humidity': [90, 85, 95, 88, None],  # One missing value
        'Category': ['Fruits', 'Dairy', 'Vegetables', 'Fruits', 'Dairy'],
        'ShelfLifeDays': [7, 14, 5, 21, 10],
        'QualityScore': [0.9, 0.8, 0.7, 0.2, 0.85],  # One poor quality
        'LogTime': pd.date_range('2024-01-01', periods=5, freq='H')
    })
    
    processed_data, preprocessor = preprocess_supply_chain_data(sample_data, 'QualityScore')
    print("Processed data shape:", processed_data.shape)
    print("Features created:", processed_data.columns.tolist())
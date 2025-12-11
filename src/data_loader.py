"""
Data Loading and Preprocessing Module
Handles CSV ingestion, normalization, train/test splitting, and data quality.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import os

try:
    from . import config
except ImportError:
    import config


class DataLoader:
    """
    Handles all data loading and preprocessing operations for the BESS optimization system.
    """
    
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.data_params = config.DATA_PARAMS
        
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV data with timestamp parsing.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with parsed timestamps
        """
        try:
            df = pd.read_csv(filepath, parse_dates=[self.data_params['datetime_column']])
            df = df.sort_values(by=self.data_params['datetime_column'])
            df = df.reset_index(drop=True)
            print(f"✓ Loaded {len(df)} records from {os.path.basename(filepath)}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using the specified method.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        method = self.data_params['missing_method']
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing == 0:
            print("✓ No missing values found")
            return df
        
        print(f"⚠ Found {initial_missing} missing values")
        
        if method == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown missing value method: {method}")
        
        final_missing = df.isnull().sum().sum()
        print(f"✓ Missing values after handling: {final_missing}")
        return df
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Detect and handle outliers using the IQR method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            
        Returns:
            DataFrame with outliers handled
        """
        method = self.data_params['outlier_method']
        
        if method == 'none':
            return df
        
        df_clean = df.copy()
        outlier_count = 0
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                threshold = self.data_params['outlier_threshold']
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outlier_count += outlier_mask.sum()
                
                # Replace outliers with boundary values
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                threshold = self.data_params['outlier_threshold']
                outlier_mask = z_scores > threshold
                outlier_count += outlier_mask.sum()
                
                # Replace outliers with median
                df_clean.loc[outlier_mask, col] = df_clean[col].median()
        
        if outlier_count > 0:
            print(f"⚠ Detected and handled {outlier_count} outliers")
        else:
            print("✓ No outliers detected")
            
        return df_clean
    
    def create_sequences(self, data: np.ndarray, seq_length: int, 
                        forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series forecasting using sliding window.
        
        Args:
            data: Input data array (time_steps, features)
            seq_length: Length of input sequence
            forecast_horizon: Length of output sequence
            
        Returns:
            X: Input sequences (samples, seq_length, features)
            y: Target sequences (samples, forecast_horizon, output_features)
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length - forecast_horizon + 1):
            # Input: past seq_length hours of [irradiance, temperature, load]
            X.append(data[i:i+seq_length])
            
            # Output: next forecast_horizon hours of [solar_power, load]
            # Assuming columns are [irradiance, temperature, load]
            # We'll predict solar_power (from irradiance) and load
            target_slice = data[i+seq_length:i+seq_length+forecast_horizon]
            
            # Extract irradiance and load for targets (indices 0 and 2)
            y.append(target_slice[:, [0, 2]])
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, train_data: np.ndarray, 
                      test_data: Optional[np.ndarray] = None) -> Tuple:
        """
        Normalize data using MinMaxScaler fitted on training data.
        
        Args:
            train_data: Training data
            test_data: Optional test data
            
        Returns:
            Normalized train_data and test_data (if provided)
        """
        train_normalized = self.scaler_X.fit_transform(train_data)
        
        if test_data is not None:
            test_normalized = self.scaler_X.transform(test_data)
            return train_normalized, test_normalized
        
        return train_normalized
    
    def prepare_data(self, filepath: str, 
                    test_size: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Args:
            filepath: Path to CSV file
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary containing training and testing data
        """
        print("\n" + "="*60)
        print("DATA PREPARATION PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_csv(filepath)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Extract feature columns
        feature_cols = [
            self.data_params['irradiance_column'],
            self.data_params['temperature_column'],
            self.data_params['load_column']
        ]
        
        # Handle outliers
        df = self.detect_and_handle_outliers(df, feature_cols)
        
        # Extract features
        data = df[feature_cols].values
        
        # Train/test split (temporal split)
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        print(f"✓ Train set: {len(train_data)} samples")
        print(f"✓ Test set: {len(test_data)} samples")
        
        # Normalize
        train_normalized, test_normalized = self.normalize_data(train_data, test_data)
        
        # Create sequences
        seq_length = config.MODEL_PARAMS['sequence_length']
        forecast_horizon = config.MODEL_PARAMS['forecast_horizon']
        
        X_train, y_train = self.create_sequences(train_normalized, seq_length, forecast_horizon)
        X_test, y_test = self.create_sequences(test_normalized, seq_length, forecast_horizon)
        
        print(f"✓ Training sequences: {X_train.shape}")
        print(f"✓ Testing sequences: {X_test.shape}")
        print("="*60 + "\n")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_columns': feature_cols,
            'raw_data': data
        }
    
    def inverse_transform(self, data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized data
            scaler: Fitted scaler object
            
        Returns:
            Original scale data
        """
        return scaler.inverse_transform(data)


def generate_sample_data(filepath: str, num_days: int = 365):
    """
    Generate sample synthetic data for testing.
    
    Args:
        filepath: Output CSV file path
        num_days: Number of days to generate
    """
    print(f"\nGenerating {num_days} days of sample data...")
    
    hours = num_days * 24
    timestamps = pd.date_range(start='2023-01-01', periods=hours, freq='H')
    
    # Generate synthetic data with realistic patterns
    hour_of_day = timestamps.hour
    day_of_year = timestamps.dayofyear
    
    # Solar irradiance (W/m²) - peak at noon, zero at night
    irradiance = np.maximum(0, 
        800 * np.sin(np.pi * (hour_of_day - 6) / 12) * 
        (1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365))
    )
    irradiance += np.random.normal(0, 50, hours)
    irradiance = np.maximum(0, irradiance)
    
    # Temperature (°C) - varies by time and season
    temperature = (
        20 + 
        10 * np.sin(2 * np.pi * day_of_year / 365) +  # Seasonal variation
        5 * np.sin(np.pi * (hour_of_day - 6) / 12) +   # Daily variation
        np.random.normal(0, 2, hours)                  # Noise
    )
    
    # Load demand (kW) - higher during day, peaks morning/evening
    base_load = 10
    daily_pattern = 5 * (np.sin(np.pi * (hour_of_day - 8) / 8) + 1)
    load = base_load + daily_pattern + np.random.normal(0, 1, hours)
    load = np.maximum(5, load)  # Minimum 5 kW
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'irradiance': irradiance,
        'temperature': temperature,
        'load': load
    })
    
    # Add some missing values (2%)
    missing_indices = np.random.choice(hours, size=int(hours * 0.02), replace=False)
    for col in ['irradiance', 'temperature', 'load']:
        df.loc[missing_indices, col] = np.nan
    
    df.to_csv(filepath, index=False)
    print(f"✓ Sample data saved to {filepath}")


if __name__ == "__main__":
    # Check if data file exists, generate only if missing
    sample_file = os.path.join(config.DATA_DIR, 'sample_data.csv')
    
    if not os.path.exists(sample_file):
        print(f"⚠ Data file not found, generating sample data...")
        generate_sample_data(sample_file, num_days=365)
    else:
        print(f"✓ Using existing data file: {sample_file}")
    
    # Test data loader
    loader = DataLoader()
    data_dict = loader.prepare_data(sample_file)
    
    print("\nData shapes:")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")

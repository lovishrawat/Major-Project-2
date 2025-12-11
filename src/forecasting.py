"""
Deep Learning Forecasting Module
Implements LSTM-based forecasting for solar generation and load demand.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict
import os

try:
    from . import config
except ImportError:
    import config


class LSTMForecaster(nn.Module):
    """
    LSTM-based sequence-to-sequence model for forecasting.
    Input: Past 24 hours of [irradiance, temperature, load]
    Output: Next 24 hours of [solar_power, load_demand]
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, forecast_horizon: int, dropout: float = 0.2):
        """
        Initialize LSTM forecaster.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            output_size: Number of output features
            forecast_horizon: Number of time steps to forecast
            dropout: Dropout rate for regularization
        """
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for each forecast step
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor (batch_size, forecast_horizon, output_size)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        out = self.fc(last_output)
        
        # Reshape to (batch_size, forecast_horizon, output_size)
        out = out.view(-1, self.forecast_horizon, self.output_size)
        
        return out


class ForecastingEngine:
    """
    Handles training, evaluation, and prediction using LSTM model.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize forecasting engine.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
        
    def build_model(self):
        """Build LSTM model based on config parameters."""
        params = config.MODEL_PARAMS
        
        self.model = LSTMForecaster(
            input_size=params['input_features'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            output_size=params['output_features'],
            forecast_horizon=params['forecast_horizon'],
            dropout=params['dropout']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=params['learning_rate']
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model built with {total_params:,} parameters")
        
    def train_epoch(self, dataloader: TorchDataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: TorchDataLoader) -> float:
        """
        Evaluate model on validation/test set.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = None, batch_size: int = None, 
              patience: int = None) -> Dict:
        """
        Train the forecasting model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target sequences
            X_val: Validation input sequences (optional)
            y_val: Validation target sequences (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = config.MODEL_PARAMS['epochs']
        if batch_size is None:
            batch_size = config.MODEL_PARAMS['batch_size']
        if patience is None:
            patience = config.MODEL_PARAMS['patience']
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), 
                torch.FloatTensor(y_val)
            )
            val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        print("\n" + "="*60)
        print("TRAINING LSTM FORECASTING MODEL")
        print("="*60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model(os.path.join(config.MODELS_DIR, 'best_model.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"✓ Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        print("="*60 + "\n")
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted sequences
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Flatten arrays for overall metrics
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        
        metrics = {}
        
        # Overall metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        metrics['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # Per-output metrics (solar and load)
        output_names = ['solar', 'load']
        for i, name in enumerate(output_names):
            y_true_i = y_true_flat[:, i]
            y_pred_i = y_pred_flat[:, i]
            
            metrics[f'rmse_{name}'] = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
            metrics[f'mae_{name}'] = mean_absolute_error(y_true_i, y_pred_i)
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights."""
        if self.model is None:
            self.build_model()
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"✓ Model loaded from {filepath}")


def forecast_future(engine: ForecastingEngine, initial_sequence: np.ndarray, 
                    num_steps: int) -> np.ndarray:
    """
    Generate multi-step ahead forecasts.
    
    Args:
        engine: Trained forecasting engine
        initial_sequence: Initial input sequence
        num_steps: Number of forecast steps
        
    Returns:
        Forecasted values
    """
    forecasts = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(num_steps):
        # Predict next step
        prediction = engine.predict(current_sequence[np.newaxis, :, :])
        forecasts.append(prediction[0])
        
        # Update sequence (rolling window)
        # This is simplified - in practice you'd update with actual features
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, [0, 2]] = prediction[0, 0]  # Update with prediction
    
    return np.array(forecasts)


if __name__ == "__main__":
    # Example usage
    try:
        from .data_loader import DataLoader
    except ImportError:
        from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    sample_file = os.path.join(config.DATA_DIR, 'sample_data.csv')
    
    if os.path.exists(sample_file):
        data_dict = loader.prepare_data(sample_file)
        
        # Initialize and train model
        engine = ForecastingEngine()
        history = engine.train(
            data_dict['X_train'], 
            data_dict['y_train'],
            data_dict['X_test'], 
            data_dict['y_test']
        )
        
        # Evaluate
        y_pred = engine.predict(data_dict['X_test'])
        metrics = engine.evaluate_metrics(data_dict['y_test'], y_pred)
        
        print("\nEvaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print(f"Sample data not found. Run data_loader.py first to generate sample data.")

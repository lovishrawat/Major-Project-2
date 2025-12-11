"""
Configuration file for BESS optimization system.
Contains all system parameters, economic assumptions, and model hyperparameters.
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# BATTERY PARAMETERS
# ============================================================================
BATTERY_PARAMS = {
    'efficiency': 0.95,              # Round-trip efficiency (95%)
    'dod': 0.80,                     # Depth of Discharge (80%)
    'soc_min': 0.20,                 # Minimum SOC (20%)
    'soc_max': 1.00,                 # Maximum SOC (100%)
    'soc_initial': 0.50,             # Initial SOC (50%)
    'cost_per_kwh': 15000,           # Battery cost (₹/kWh) - Grid-scale estimate (₹11k-19k range)
    'cost_per_kw': 12000,            # Power electronics cost (₹/kW) - Adjusted for scale
    'lifetime_cycles': 5000,         # Rated cycle life
    'calendar_life': 15,             # Calendar life (years)
    'degradation_rate': 0.02,        # Annual capacity fade (2% per year)
}

# ============================================================================
# ECONOMIC PARAMETERS
# ============================================================================
ECONOMIC_PARAMS = {
    'interest_rate': 0.10,           # Discount rate (10%) - Commercial rate
    'project_lifetime': 20,          # Project lifetime (years)
    'om_cost_percent': 0.02,         # O&M cost as % of capital (2% annually)
    'replacement_threshold': 0.70,   # Replace when capacity < 70%
}

# ============================================================================
# GRID PARAMETERS (Indian Electricity Tariff)
# ============================================================================
GRID_PARAMS = {
    'import_enabled': True,          # Allow grid import
    'export_enabled': True,          # Allow grid export (net metering)
    'tou_enabled': True,             # Time-of-Use pricing enabled
    
    # Indian TOU tariff structure (₹/kWh) - Based on typical industrial/commercial rates
    'peak_import_price': 12.0,       # Peak hours: 6 PM - 10 PM (₹12.0/kWh)
    'mid_import_price': 8.0,         # Mid hours: 10 AM - 6 PM (₹8.0/kWh)
    'offpeak_import_price': 6.0,     # Off-peak: 10 PM - 10 AM (₹6.0/kWh)
    
    # Export price (feed-in tariff / net metering rate)
    'export_price': 3.5,             # ₹3.5/kWh (APPC rate)
    
    # TOU time windows (24-hour format)
    'peak_hours': [18, 19, 20, 21],                        # 6 PM - 10 PM
    'mid_hours': [10, 11, 12, 13, 14, 15, 16, 17],        # 10 AM - 6 PM
    # Remaining hours (22, 23, 0-9) are off-peak
}

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================
OPTIMIZATION_PARAMS = {
    'lpsp_max': 0.05,                # Maximum LPSP (5%)
    'bess_kwh_min': 10,              # Minimum battery capacity (kWh)
    'bess_kwh_max': 500,             # Maximum battery capacity (kWh)
    'bess_kw_min': 5,                # Minimum power rating (kW)
    'bess_kw_max': 250,              # Maximum power rating (kW)
    'pso_swarm_size': 50,            # PSO particle count (Increased for better search)
    'pso_max_iter': 100,             # PSO maximum iterations (Increased)
    'pso_omega': 0.7,                # PSO inertia weight (Adjusted)
    'pso_phip': 0.5,                 # PSO cognitive parameter
    'pso_phig': 0.5,                 # PSO social parameter
}

# ============================================================================
# FORECASTING MODEL PARAMETERS
# ============================================================================
MODEL_PARAMS = {
    'sequence_length': 24,           # Input sequence length (24 hours)
    'forecast_horizon': 24,          # Output forecast horizon (24 hours)
    'input_features': 3,             # [irradiance, temperature, load]
    'output_features': 2,            # [solar_power, load_demand]
    'hidden_size': 64,               # LSTM hidden units
    'num_layers': 2,                 # Number of LSTM layers
    'dropout': 0.2,                  # Dropout rate
    'learning_rate': 0.001,          # Learning rate
    'batch_size': 32,                # Training batch size
    'epochs': 100,                   # Maximum training epochs
    'patience': 10,                  # Early stopping patience
    'train_ratio': 0.8,              # Train/test split ratio
}

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
SIMULATION_PARAMS = {
    'timestep': 1,                   # Simulation timestep (hours)
    'pv_efficiency': 0.18,           # PV panel efficiency
    'pv_area': 1.6,                  # PV panel area (m²)
    'stc_irradiance': 1000,          # Standard Test Condition irradiance (W/m²)
    'temp_coefficient': -0.004,      # Temperature coefficient (%/°C)
    'reference_temp': 25,            # Reference temperature (°C)
}

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================
DATA_PARAMS = {
    'datetime_column': 'timestamp',
    'irradiance_column': 'irradiance',
    'temperature_column': 'temperature',
    'load_column': 'load',
    'missing_method': 'interpolate',  # interpolate, forward_fill, drop
    'outlier_method': 'iqr',          # iqr, zscore, none
    'outlier_threshold': 1.5,         # IQR multiplier
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_config_summary():
    """Return a formatted summary of all configuration parameters."""
    summary = []
    summary.append("="*60)
    summary.append("BESS OPTIMIZATION SYSTEM CONFIGURATION")
    summary.append("="*60)
    
    summary.append("\nBattery Parameters:")
    for key, value in BATTERY_PARAMS.items():
        summary.append(f"  {key}: {value}")
    
    summary.append("\nEconomic Parameters:")
    for key, value in ECONOMIC_PARAMS.items():
        summary.append(f"  {key}: {value}")
    
    summary.append("\nOptimization Parameters:")
    for key, value in OPTIMIZATION_PARAMS.items():
        summary.append(f"  {key}: {value}")
    
    summary.append("\nModel Parameters:")
    for key, value in MODEL_PARAMS.items():
        summary.append(f"  {key}: {value}")
    
    summary.append("="*60)
    return "\n".join(summary)


if __name__ == "__main__":
    print(get_config_summary())

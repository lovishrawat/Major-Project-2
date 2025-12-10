# BESS Techno-Economic Optimization System

A comprehensive Python framework for optimizing Battery Energy Storage System (BESS) sizing using Deep Learning-based forecasting and Metaheuristic optimization algorithms.

## 🎯 Project Overview

This system determines the optimal size (Energy Capacity in kWh and Power Rating in kW) of a BESS for microgrids to minimize Total Cost of Ownership (TCO) while satisfying reliability constraints.

### Methodology

**Two-Stage Approach:**

1. **Stage 1: Forecasting** - LSTM-based deep learning to predict Solar Generation and Load Demand
2. **Stage 2: Optimization** - Particle Swarm Optimization (PSO) to find optimal BESS sizing

## 📁 Project Structure

```
Major Project 2/
├── data/               # Data storage
│   └── sample_data.csv
├── src/                # Source code modules
│   ├── __init__.py
│   ├── config.py       # System configuration
│   ├── data_loader.py  # Data processing
│   ├── forecasting.py  # LSTM forecasting
│   ├── battery_model.py # Battery physics
│   ├── simulation.py   # Energy management
│   ├── optimizer.py    # PSO optimization
│   └── utils.py        # Utilities
├── models/             # Saved model weights
├── results/            # Optimization results
├── main.py             # Main execution script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to project directory
cd "c:/Users/Victus/OneDrive/Desktop/Major Project 2"

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run complete optimization workflow
python main.py

# Train new model even if saved model exists
python main.py --train-model

# Run without plotting
python main.py --no-plot

# Custom PSO parameters
python main.py --swarm-size 50 --max-iterations 100
```

### Module-Level Usage

```python
# Example: Data Processing
from src.data_loader import DataLoader

loader = DataLoader()
data_dict = loader.prepare_data('data/sample_data.csv')

# Example: LSTM Forecasting
from src.forecasting import ForecastingEngine

engine = ForecastingEngine()
engine.train(data_dict['X_train'], data_dict['y_train'])
predictions = engine.predict(data_dict['X_test'])

# Example: BESS Optimization
from src.optimizer import BESSOptimizer

optimizer = BESSOptimizer(pv_forecast, load_forecast)
optimal_solution, optimal_cost = optimizer.optimize()
```

## 📊 Key Features

### 1. Data Processing (`data_loader.py`)
- CSV data ingestion with timestamp parsing
- MinMax normalization
- Missing value handling (interpolation)
- Outlier detection (IQR method)
- Time-series sequence generation
- Sample data generator

### 2. LSTM Forecasting (`forecasting.py`)
- Multi-layer LSTM architecture
- **Input:** Last 24 hours of [irradiance, temperature, load]
- **Output:** Next 24 hours of [solar_power, load_demand]
- Training with early stopping
- GPU support
- RMSE & MAE evaluation

### 3. Battery Modeling (`battery_model.py`)
- Physics-based SOC tracking
- Round-trip efficiency: 95%
- Depth of Discharge (DoD): 80%
- **Advanced degradation model:**
  - Cycle-based capacity fade
  - Calendar aging
  - Throughput tracking
- Power and energy constraints

### 4. Energy Management (`simulation.py`)
- **Strategy:**
  - Surplus (PV > Load): Charge battery → Curtail if full
  - Deficit (Load > PV): Discharge battery → Unmet load if empty
- LPSP calculation
- Renewable penetration metrics
- Battery utilization tracking

### 5. PSO Optimization (`optimizer.py`)
- **Objective:** Minimize Net Present Cost (NPC)
  ```
  NPC = Capital + O&M + Replacement - Revenue
  ```
- **Constraint:** LPSP < 5%
- **Variables:** [Energy_kWh, Power_kW]
- Swarm-based exploration
- Convergence tracking

### 6. Utilities (`utils.py`)
- Visualization (matplotlib/seaborn)
- Results export (JSON)
- Metrics summarization
- Plotting utilities

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
BATTERY_PARAMS = {
    'efficiency': 0.95,
    'dod': 0.80,
    'cost_per_kwh': 150,  # $/kWh
    ...
}

ECONOMIC_PARAMS = {
    'interest_rate': 0.06,
    'project_lifetime': 20,  # years
    ...
}

OPTIMIZATION_PARAMS = {
    'lpsp_max': 0.05,  # 5%
    'pso_swarm_size': 30,
    ...
}
```

## 📈 Workflow Explanation

### Data Flow: Forecast → Optimization

```
1. Historical Data (CSV)
   ↓
2. Data Preprocessing
   - Normalization, cleaning, sequences
   ↓
3. LSTM Training
   - Learn patterns from historical data
   ↓
4. Generate Forecasts
   - Predict future 24h solar + load
   ↓
5. PSO Optimization
   - Test different BESS sizes
   - Run simulation for each
   - Calculate NPC and LPSP
   ↓
6. Optimal BESS Sizing
   - Energy capacity (kWh)
   - Power rating (kW)
```

### How Forecast Feeds into Optimization

The forecasted solar and load profiles are used as **input to the simulation** within each PSO iteration:

```python
# In optimizer.py - objective function
for each particle (BESS size):
    battery = BatteryStorage(energy_kwh, power_kw)
    simulator = MicrogridSimulator(battery)
    
    # Use forecasted data
    results = simulator.simulate(pv_forecast, load_forecast)
    
    # Calculate metrics
    lpsp = results['lpsp']
    npc = calculate_npc(...)
    
    # Fitness evaluation
    if lpsp > threshold:
        fitness = npc + penalty
    else:
        fitness = npc
```

## 📊 Output Files

After running `main.py`:

```
results/
├── optimization_results.json     # Complete results
├── metrics_summary.csv          # Key metrics table
├── forecast_results.png         # Forecast accuracy
├── optimization_history.png     # PSO convergence
└── simulation_results.png       # Power flows & SOC
```

## 🧪 Example Results

```
Optimal Energy Capacity: 85.42 kWh
Optimal Power Rating:    47.23 kW
Optimal NPC:             $42,650
LPSP:                    4.2 %
Renewable Penetration:   87.3 %
Battery Cycles:          245
```

## 📝 Command-Line Arguments

```bash
Data Arguments:
  --data-file PATH          Input CSV path
  --data-days N             Days of sample data
  --test-split RATIO        Train/test split (0.2)

Forecasting Arguments:
  --train-model             Force new training
  --epochs N                Training epochs (50)
  --batch-size N            Batch size (32)

Optimization Arguments:
  --optimization-horizon N  Hours to optimize (720)
  --swarm-size N            PSO swarm size (30)
  --max-iterations N        PSO iterations (50)

Output Arguments:
  --plot / --no-plot        Generate plots (default: yes)
```

## 🔬 Technical Details

### LSTM Architecture
- **Input shape:** (batch, 24, 3)
- **Hidden layers:** 2 LSTM layers with 64 units
- **Dropout:** 0.2
- **Output shape:** (batch, 24, 2)
- **Loss:** MSE
- **Optimizer:** Adam

### PSO Configuration
- **Swarm size:** 30 particles
- **Iterations:** 50
- **Inertia (ω):** 0.5
- **Cognitive (φp):** 0.5
- **Social (φg):** 0.5

### Battery Parameters (Li-ion)
- **Efficiency:** 95% (round-trip)
- **DoD:** 80%
- **Cost:** $150/kWh + $100/kW
- **Lifetime:** 5000 cycles / 15 years
- **Degradation:** 2%/year + cycle-based

## 🎓 Academic Context

**Capstone Project Title:**  
*"Techno-Economic Optimization of Battery Energy Storage System (BESS) Sizing Using Deep Learning-Based Generation & Load Forecasting"*

**Key Contributions:**
1. Integration of deep learning forecasting with optimization
2. Advanced battery degradation modeling
3. Multi-objective techno-economic analysis
4. Modular, extensible framework

## 📚 Dependencies

- **Core:** numpy, pandas, scipy
- **ML:** torch, scikit-learn
- **Optimization:** pyswarm
- **Visualization:** matplotlib, seaborn, plotly
- **Utilities:** tqdm

## 🛠️ Extending the System

### Add a New Forecasting Model

```python
# In forecasting.py
class GRUForecaster(nn.Module):
    # Implement GRU architecture
    pass
```

### Add a New Optimization Algorithm

```python
# In optimizer.py
from deap import algorithms, base, tools

class GAOptimizer(BESSOptimizer):
    def optimize_with_ga(self):
        # Implement genetic algorithm
        pass
```

### Custom Energy Management Strategy

```python
# In simulation.py
class CustomSimulator(MicrogridSimulator):
    def energy_management_strategy(self, pv, load, dt):
        # Implement custom EMS logic
        pass
```

## ⚠️ Troubleshooting

**Issue:** CUDA out of memory  
**Solution:** Use CPU or reduce batch size
```bash
python main.py --batch-size 16
```

**Issue:** Optimization not converging  
**Solution:** Increase iterations or adjust PSO parameters
```bash
python main.py --max-iterations 100 --swarm-size 50
```

**Issue:** LPSP constraint always violated  
**Solution:** Check if load >> PV, may need larger BESS bounds

## 📞 Support

For questions or issues:
1. Check configuration in `src/config.py`
2. Review error messages carefully
3. Run individual modules for debugging

## 📄 License

Academic/Research Use

---

**Author:** Research Team  
**Version:** 1.0.0  
**Last Updated:** December 2025

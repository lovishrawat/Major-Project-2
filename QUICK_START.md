# BESS Optimization System - Quick Reference

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete system (generates sample data, trains model, optimizes)
python main.py

# Train new model
python main.py --train-model

# Quick test (reduced iterations for fast testing)
python main.py --epochs 10 --max-iterations 10 --data-days 30
```

## 📂 Project Layout

```
Major Project 2/
├── src/
│   ├── config.py          # All system parameters
│   ├── data_loader.py     # Data preprocessing & sequences
│   ├── forecasting.py     # LSTM model
│   ├── battery_model.py   # Battery physics & degradation
│   ├── simulation.py      # Energy management simulation
│   ├── optimizer.py       # PSO optimization
│   └── utils.py           # Visualization & I/O
├── main.py                # Main workflow
├── requirements.txt       # Dependencies
└── README.md             # Full documentation
```

## 🔧 Key Parameters (src/config.py)

### Battery Parameters
```python
'efficiency': 0.95         # 95% round-trip
'dod': 0.80               # 80% depth of discharge
'cost_per_kwh': 150       # $150/kWh
'cost_per_kw': 100        # $100/kW
```

### Economic Parameters
```python
'interest_rate': 0.06      # 6% discount rate
'project_lifetime': 20     # 20 years
'om_cost_percent': 0.02    # 2% annual O&M
```

### Optimization Parameters
```python
'lpsp_max': 0.05          # Max 5% LPSP
'pso_swarm_size': 30      # 30 particles
'pso_max_iter': 50        # 50 iterations
```

## 🎯 Module Usage Examples

### Data Processing
```python
from src.data_loader import DataLoader
loader = DataLoader()
data = loader.prepare_data('data/sample_data.csv')
```

### LSTM Forecasting
```python
from src.forecasting import ForecastingEngine
engine = ForecastingEngine()
engine.train(X_train, y_train, X_val, y_val)
predictions = engine.predict(X_test)
```

### Battery Simulation
```python
from src.battery_model import BatteryStorage
battery = BatteryStorage(energy_kwh=100, power_kw=50)
battery.charge(power_kw=40, dt=1.0)
battery.discharge(power_kw=30, dt=1.0)
```

### Microgrid Simulation
```python
from src.simulation import MicrogridSimulator
simulator = MicrogridSimulator(battery)
results = simulator.simulate(pv_power, load_demand)
metrics = simulator.get_metrics()
```

### BESS Optimization
```python
from src.optimizer import BESSOptimizer
optimizer = BESSOptimizer(pv_forecast, load_forecast)
solution, cost = optimizer.optimize()
```

## 📊 Output Files

After running `main.py`, check `results/` folder:
- `optimization_results.json` - Complete results
- `metrics_summary.csv` - Key metrics table
- `forecast_results.png` - Forecast accuracy plot
- `optimization_history.png` - PSO convergence
- `simulation_results.png` - Power flows & SOC

## 🔍 Testing Individual Modules

Each module can be tested standalone:

```bash
python -m src.data_loader      # Test data processing
python -m src.forecasting      # Test LSTM model
python -m src.battery_model    # Test battery physics
python -m src.simulation       # Test simulation
python -m src.optimizer        # Test optimization
python -m src.utils            # Test utilities
```

## ⚙️ Command-Line Options

```bash
# Data options
--data-file PATH          # CSV input file
--data-days N             # Days of sample data

# Training options
--train-model             # Force new training
--epochs N                # Training epochs (default: 50)
--batch-size N            # Batch size (default: 32)

# Optimization options
--swarm-size N            # PSO swarm size (default: 30)
--max-iterations N        # PSO iterations (default: 50)
--optimization-horizon N  # Forecast horizon (hours)

# Output options
--plot / --no-plot        # Enable/disable plots
```

## 🎓 Key Concepts

### Data Flow
```
Historical Data → Preprocessing → LSTM Training → Forecasting
                                                       ↓
Optimal BESS Size ← PSO Optimization ← Simulation ← Forecasts
```

### Energy Management Strategy
1. **PV > Load:** Charge battery → Curtail if full
2. **Load > PV:** Discharge battery → Unmet load if empty

### Objective Function
```
Minimize: NPC = Capital + O&M + Replacement - Revenue
Subject to: LPSP ≤ 5%
```

## 🐛 Troubleshooting

**Problem:** CUDA out of memory  
**Solution:** `--batch-size 16` or use CPU

**Problem:** Optimization not converging  
**Solution:** `--max-iterations 100 --swarm-size 50`

**Problem:** Missing dependencies  
**Solution:** `pip install -r requirements.txt`

**Problem:** No data file  
**Solution:** System auto-generates sample data

## 📈 Expected Results

Typical optimal solution:
- Energy Capacity: 50-100 kWh
- Power Rating: 25-50 kW
- NPC: $30,000-$60,000
- LPSP: 2-5%
- Renewable Penetration: 75-90%

## 🔬 Advanced Usage

### Custom Configuration
Edit `src/config.py` to modify parameters

### Custom Forecasting Model
Extend `LSTMForecaster` class in `src/forecasting.py`

### Custom Optimization Algorithm
Create new optimizer class in `src/optimizer.py`

### Grid Export Revenue
Add revenue calculation in `optimizer.calculate_npc()`

---

**For full documentation, see README.md**

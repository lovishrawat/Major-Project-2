"""
Utility Functions
Helper functions for visualization, I/O, and common operations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from typing import Dict, List

from . import config


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_forecast_results(y_true: np.ndarray, y_pred: np.ndarray, 
                          output_names: List[str] = None,
                          save_path: str = None):
    """
    Plot forecasting results comparing true vs predicted values.
    
    Args:
        y_true: True values (samples, horizon, features)
        y_pred: Predicted values (samples, horizon, features)
        output_names: Names of output features
        save_path: Path to save figure (None = display only)
    """
    if output_names is None:
        output_names = ['Solar Power', 'Load Demand']
    
    num_features = y_true.shape[-1]
    
    fig, axes = plt.subplots(num_features, 1, figsize=(14, 4*num_features))
    if num_features == 1:
        axes = [axes]
    
    # Plot first sample for visualization
    sample_idx = 0
    horizon = y_true.shape[1]
    time = np.arange(horizon)
    
    for i, (ax, name) in enumerate(zip(axes, output_names)):
        ax.plot(time, y_true[sample_idx, :, i], 
               label='True', marker='o', linewidth=2)
        ax.plot(time, y_pred[sample_idx, :, i], 
               label='Predicted', marker='s', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time Step (hours)')
        ax.set_ylabel(f'{name} (kW)')
        ax.set_title(f'{name} Forecast (Sample {sample_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()


def plot_simulation_results(results: Dict, save_path: str = None):
    """
    Plot simulation results showing power flows and battery SOC.
    
    Args:
        results: Simulation results dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    time = results['timestamps']
    
    # Plot 1: Power generation and demand
    ax1 = axes[0]
    ax1.plot(time, results['pv_generation'], label='PV Generation', linewidth=2)
    ax1.plot(time, results['load_demand'], label='Load Demand', linewidth=2)
    ax1.fill_between(time, results['pv_generation'], alpha=0.3)
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('PV Generation and Load Demand')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Battery power and SOC
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(time, results['battery_power'], 
                     label='Battery Power', color='blue', linewidth=2)
    line2 = ax2_twin.plot(time, np.array(results['battery_soc']) * 100, 
                          label='SOC', color='red', linewidth=2, alpha=0.7)
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Battery Power (kW)', color='blue')
    ax2_twin.set_ylabel('SOC (%)', color='red')
    ax2.set_title('Battery Operation')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Unmet load and curtailed PV
    ax3 = axes[2]
    ax3.plot(time, results['unmet_load'], label='Unmet Load', 
            color='red', linewidth=2)
    ax3.plot(time, results['curtailed_pv'], label='Curtailed PV', 
            color='orange', linewidth=2, alpha=0.7)
    ax3.fill_between(time, results['unmet_load'], alpha=0.3, color='red')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Power (kW)')
    ax3.set_title('Unmet Load and Curtailed PV')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()


def plot_optimization_history(history: List[Dict], save_path: str = None):
    """
    Plot optimization convergence history.
    
    Args:
        history: List of optimization iterations
        save_path: Path to save figure
    """
    if not history:
        print("No optimization history to plot")
        return
    
    df = pd.DataFrame(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Objective value over iterations
    axes[0, 0].plot(df['objective'], linewidth=2)
    axes[0, 0].set_xlabel('Evaluation')
    axes[0, 0].set_ylabel('Objective Value (₹)')
    axes[0, 0].set_title('Convergence History')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: NPC vs iterations
    axes[0, 1].plot(df['npc'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Evaluation')
    axes[0, 1].set_ylabel('NPC (₹)')
    axes[0, 1].set_title('Net Present Cost Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: LPSP distribution
    axes[1, 0].hist(df['lpsp'] * 100, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=config.OPTIMIZATION_PARAMS['lpsp_max']*100, 
                      color='red', linestyle='--', linewidth=2, 
                      label=f"LPSP Limit ({config.OPTIMIZATION_PARAMS['lpsp_max']*100:.1f}%)")
    axes[1, 0].set_xlabel('LPSP (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('LPSP Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Solution space
    scatter = axes[1, 1].scatter(df['energy_kwh'], df['power_kw'], 
                                c=df['npc'], cmap='viridis', alpha=0.6)
    axes[1, 1].set_xlabel('Energy Capacity (kWh)')
    axes[1, 1].set_ylabel('Power Rating (kW)')
    axes[1, 1].set_title('Solution Space Exploration')
    plt.colorbar(scatter, ax=axes[1, 1], label='NPC (₹)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()


def save_results(results: Dict, filepath: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Output file path
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"✓ Results loaded from {filepath}")
    return results


def calculate_metrics_summary(evaluation: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame of key metrics.
    
    Args:
        evaluation: Evaluation results dictionary
        
    Returns:
        DataFrame with formatted metrics
    """
    data = {
        'Metric': [],
        'Value': [],
        'Unit': []
    }
    
    # Sizing metrics
    data['Metric'].extend(['Energy Capacity', 'Power Rating', 'E/P Ratio'])
    data['Value'].extend([
        f"{evaluation['sizing']['energy_kwh']:.2f}",
        f"{evaluation['sizing']['power_kw']:.2f}",
        f"{evaluation['sizing']['energy_to_power_ratio']:.2f}"
    ])
    data['Unit'].extend(['kWh', 'kW', 'hours'])
    
    # Economic metrics
    data['Metric'].extend(['Capital Cost', 'Net Present Cost', 'Levelized Annual Cost'])
    data['Value'].extend([
        f"₹{evaluation['costs']['total_capital_cost']:,.2f}",
        f"₹{evaluation['economics']['npc']:,.2f}",
        f"₹{evaluation['economics']['levelized_cost']:,.2f}"
    ])
    data['Unit'].extend(['₹', '₹', '₹/year'])
    
    # Performance metrics
    perf = evaluation['performance']
    data['Metric'].extend(['LPSP', 'Renewable Penetration', 'Self-Consumption', 'Battery Cycles'])
    data['Value'].extend([
        f"{perf['lpsp']*100:.2f}",
        f"{perf['renewable_penetration']*100:.2f}",
        f"{perf['self_consumption']*100:.2f}",
        f"{perf['battery_cycles']:.2f}"
    ])
    data['Unit'].extend(['%', '%', '%', 'cycles'])
    
    df = pd.DataFrame(data)
    return df


def print_welcome_message():
    """Print welcome message."""
    print("\n" + "="*70)
    print(" "*10 + "BESS TECHNO-ECONOMIC OPTIMIZATION SYSTEM")
    print(" "*5 + "Deep Learning Forecasting + Metaheuristic Optimization")
    print("="*70)
    print("\nModules:")
    print("  ✓ Data Processing & Preprocessing")
    print("  ✓ LSTM-based Forecasting (Solar + Load)")
    print("  ✓ Battery Storage Modeling with Degradation")
    print("  ✓ Energy Management Simulation")
    print("  ✓ PSO-based Sizing Optimization")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_welcome_message()
    
    # Example: Generate and save sample results
    sample_results = {
        'energy_kwh': 100.0,
        'power_kw': 50.0,
        'npc': 50000.0,
        'lpsp': 0.03
    }
    
    output_path = os.path.join(config.RESULTS_DIR, 'sample_results.json')
    save_results(sample_results, output_path)
    
    # Load and verify
    loaded = load_results(output_path)
    print(f"\nLoaded results: {loaded}")

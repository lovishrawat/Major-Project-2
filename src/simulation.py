"""
Energy Management Simulation Module
Implements power flow simulation and energy management strategy (EMS).
"""

import numpy as np
from typing import Dict, List, Tuple
from . import config
from .battery_model import BatteryStorage


class MicrogridSimulator:
    """
    Simulates microgrid operation with solar PV, load, and battery storage.
    Implements Energy Management Strategy (EMS) and calculates performance metrics.
    """
    
    def __init__(self, battery: BatteryStorage):
        """
        Initialize microgrid simulator.
        
        Args:
            battery: BatteryStorage instance
        """
        self.battery = battery
        self.sim_params = config.SIMULATION_PARAMS
        
        # Simulation results storage
        self.results = {
            'timestamps': [],
            'pv_generation': [],
            'load_demand': [],
            'battery_soc': [],
            'battery_power': [],  # Positive = charging, Negative = discharging
            'grid_power': [],
            'unmet_load': [],
            'curtailed_pv': [],
        }
        
        # Performance metrics
        self.metrics = {
            'total_load': 0.0,
            'total_pv': 0.0,
            'total_unmet_load': 0.0,
            'total_curtailed': 0.0,
            'lpsp': 0.0,
            'renewable_penetration': 0.0,
            'self_consumption': 0.0,
            'battery_cycles': 0.0,
        }
    
    def convert_irradiance_to_power(self, irradiance: np.ndarray, 
                                   temperature: np.ndarray = None) -> np.ndarray:
        """
        Convert solar irradiance to PV power output.
        
        Args:
            irradiance: Solar irradiance (W/m²)
            temperature: Ambient temperature (°C), optional
            
        Returns:
            PV power output (kW)
        """
        # Get PV parameters
        efficiency = self.sim_params['pv_efficiency']
        area = self.sim_params['pv_area']
        stc_irradiance = self.sim_params['stc_irradiance']
        
        # Base power calculation
        pv_power = (irradiance / stc_irradiance) * efficiency * area
        
        # Temperature derating (if temperature provided)
        if temperature is not None:
            temp_coeff = self.sim_params['temp_coefficient']
            ref_temp = self.sim_params['reference_temp']
            temp_factor = 1 + temp_coeff * (temperature - ref_temp)
            pv_power = pv_power * temp_factor
        
        # Convert W to kW
        pv_power_kw = pv_power / 1000
        
        return np.maximum(0, pv_power_kw)
    
    def energy_management_strategy(self, pv_power: float, load_demand: float, 
                                   dt: float = 1.0) -> Dict[str, float]:
        """
        Implement Energy Management Strategy (EMS).
        
        Strategy:
        1. If PV > Load: Charge battery with surplus. If battery full, curtail.
        2. If Load > PV: Discharge battery to meet deficit. If battery empty, unmet load.
        
        Args:
            pv_power: PV generation (kW)
            load_demand: Load demand (kW)
            dt: Time step (hours)
            
        Returns:
            Dictionary with power flows
        """
        net_power = pv_power - load_demand
        
        battery_power = 0.0
        unmet_load = 0.0
        curtailed_pv = 0.0
        
        if net_power > 0:
            # Surplus: Try to charge battery
            actual_power, energy_stored = self.battery.charge(net_power, dt)
            battery_power = actual_power  # Positive = charging
            
            # Curtail if battery cannot accept all surplus
            curtailed_pv = net_power - actual_power
            
        elif net_power < 0:
            # Deficit: Try to discharge battery
            required_power = abs(net_power)
            actual_power, energy_delivered = self.battery.discharge(required_power, dt)
            battery_power = -actual_power  # Negative = discharging
            
            # Unmet load if battery cannot supply all deficit
            unmet_load = required_power - actual_power
        
        return {
            'battery_power': battery_power,
            'unmet_load': unmet_load,
            'curtailed_pv': curtailed_pv,
            'net_power': net_power
        }
    
    def simulate(self, pv_power: np.ndarray, load_demand: np.ndarray, 
                dt: float = 1.0, reset_battery: bool = True) -> Dict:
        """
        Run complete microgrid simulation.
        
        Args:
            pv_power: Array of PV power values (kW)
            load_demand: Array of load demand values (kW)
            dt: Time step (hours)
            reset_battery: Whether to reset battery to initial state
            
        Returns:
            Simulation results dictionary
        """
        if len(pv_power) != len(load_demand):
            raise ValueError("PV and load arrays must have same length")
        
        # Reset battery if requested
        if reset_battery:
            self.battery.reset()
        
        # Reset results
        self.results = {key: [] for key in self.results.keys()}
        
        num_steps = len(pv_power)
        
        for t in range(num_steps):
            pv = pv_power[t]
            load = load_demand[t]
            
            # Apply energy management strategy
            flows = self.energy_management_strategy(pv, load, dt)
            
            # Store results
            self.results['timestamps'].append(t)
            self.results['pv_generation'].append(pv)
            self.results['load_demand'].append(load)
            self.results['battery_soc'].append(self.battery.soc)
            self.results['battery_power'].append(flows['battery_power'])
            self.results['unmet_load'].append(flows['unmet_load'])
            self.results['curtailed_pv'].append(flows['curtailed_pv'])
            
            # Grid power (if connected)
            grid_power = flows['unmet_load']  # Import from grid
            self.results['grid_power'].append(grid_power)
        
        # Convert lists to arrays
        for key in self.results.keys():
            if key != 'timestamps':
                self.results[key] = np.array(self.results[key])
        
        # Calculate metrics
        self._calculate_metrics(dt)
        
        # Update battery degradation (assume annual update)
        years_simulated = (num_steps * dt) / 8760
        if years_simulated > 0:
            self.battery.update_degradation(dt_years=years_simulated)
        
        return self.results
    
    def _calculate_metrics(self, dt: float):
        """Calculate performance metrics from simulation results."""
        # Total energy (kWh)
        self.metrics['total_load'] = np.sum(self.results['load_demand']) * dt
        self.metrics['total_pv'] = np.sum(self.results['pv_generation']) * dt
        self.metrics['total_unmet_load'] = np.sum(self.results['unmet_load']) * dt
        self.metrics['total_curtailed'] = np.sum(self.results['curtailed_pv']) * dt
        
        # Loss of Power Supply Probability (LPSP)
        if self.metrics['total_load'] > 0:
            self.metrics['lpsp'] = self.metrics['total_unmet_load'] / self.metrics['total_load']
        else:
            self.metrics['lpsp'] = 0.0
        
        # Renewable penetration
        if self.metrics['total_load'] > 0:
            pv_consumed = self.metrics['total_pv'] - self.metrics['total_curtailed']
            self.metrics['renewable_penetration'] = pv_consumed / self.metrics['total_load']
        
        # Self-consumption ratio
        if self.metrics['total_pv'] > 0:
            pv_consumed = self.metrics['total_pv'] - self.metrics['total_curtailed']
            self.metrics['self_consumption'] = pv_consumed / self.metrics['total_pv']
        
        # Battery cycles
        self.metrics['battery_cycles'] = self.battery.total_cycles
    
    def get_metrics(self) -> Dict[str, float]:
        """Return performance metrics."""
        return self.metrics.copy()
    
    def print_summary(self):
        """Print simulation summary."""
        print("\n" + "="*60)
        print("SIMULATION RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nEnergy Balance:")
        print(f"  Total Load:           {self.metrics['total_load']:>10.2f} kWh")
        print(f"  Total PV Generation:  {self.metrics['total_pv']:>10.2f} kWh")
        print(f"  Unmet Load:           {self.metrics['total_unmet_load']:>10.2f} kWh")
        print(f"  Curtailed PV:         {self.metrics['total_curtailed']:>10.2f} kWh")
        
        print(f"\nPerformance Metrics:")
        print(f"  LPSP:                 {self.metrics['lpsp']*100:>10.2f} %")
        print(f"  Renewable Penetration:{self.metrics['renewable_penetration']*100:>10.2f} %")
        print(f"  Self-Consumption:     {self.metrics['self_consumption']*100:>10.2f} %")
        print(f"  Battery Cycles:       {self.metrics['battery_cycles']:>10.2f}")
        
        print(f"\nBattery Status:")
        status = self.battery.get_status()
        print(f"  Final SOC:            {status['soc_percent']:>10.2f} %")
        print(f"  Capacity Fade:        {status['capacity_fade_percent']:>10.2f} %")
        print(f"  Total Cycles:         {status['total_cycles']:>10.2f}")
        
        print("="*60 + "\n")


def calculate_lpsp(unmet_load: np.ndarray, load_demand: np.ndarray) -> float:
    """
    Calculate Loss of Power Supply Probability.
    
    Args:
        unmet_load: Array of unmet load values
        load_demand: Array of load demand values
        
    Returns:
        LPSP as fraction (0-1)
    """
    total_load = np.sum(load_demand)
    total_unmet = np.sum(unmet_load)
    
    if total_load > 0:
        return total_unmet / total_load
    return 0.0


if __name__ == "__main__":
    # Example simulation
    from .battery_model import BatteryStorage
    
    print("\n" + "="*60)
    print("MICROGRID SIMULATION EXAMPLE")
    print("="*60)
    
    # Create battery
    battery = BatteryStorage(energy_capacity_kwh=50, power_rating_kw=25)
    
    # Create simulator
    simulator = MicrogridSimulator(battery)
    
    # Generate sample data for 7 days (168 hours)
    hours = 168
    time = np.arange(hours)
    hour_of_day = time % 24
    
    # PV generation pattern (peak at noon)
    pv_power = 15 * np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    
    # Load demand pattern (higher during day)
    load_demand = 8 + 4 * np.sin(np.pi * (hour_of_day - 8) / 10)
    
    # Run simulation
    results = simulator.simulate(pv_power, load_demand, dt=1.0)
    
    # Print summary
    simulator.print_summary()
    
    # Check LPSP constraint
    lpsp = simulator.metrics['lpsp']
    lpsp_threshold = config.OPTIMIZATION_PARAMS['lpsp_max']
    
    if lpsp <= lpsp_threshold:
        print(f"✓ LPSP constraint satisfied ({lpsp*100:.2f}% <= {lpsp_threshold*100:.2f}%)")
    else:
        print(f"✗ LPSP constraint violated ({lpsp*100:.2f}% > {lpsp_threshold*100:.2f}%)")

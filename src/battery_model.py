"""
Battery Energy Storage System (BESS) Modeling Module
Implements battery physics, SOC tracking, and degradation modeling.
"""

import numpy as np
from typing import Dict, Tuple

try:
    # Try relative import (when used as a module)
    from . import config
except ImportError:
    # Fall back to direct import (when run as a script)
    import config


class BatteryStorage:
    """
    Battery Energy Storage System with physics-based modeling.
    Includes SOC tracking, power constraints, efficiency, and degradation.
    """
    
    def __init__(self, energy_capacity_kwh: float, power_rating_kw: float):
        """
        Initialize battery storage system.
        
        Args:
            energy_capacity_kwh: Battery energy capacity (kWh)
            power_rating_kw: Battery power rating (kW)
        """
        # Battery sizing
        self.nominal_capacity_kwh = energy_capacity_kwh
        self.current_capacity_kwh = energy_capacity_kwh  # Degrades over time
        self.power_rating_kw = power_rating_kw
        
        # Get parameters from config
        params = config.BATTERY_PARAMS
        self.efficiency = params['efficiency']
        self.soc_min = params['soc_min']
        self.soc_max = params['soc_max']
        self.soc = params['soc_initial']  # Current state of charge
        
        # Degradation parameters
        self.lifetime_cycles = params['lifetime_cycles']
        self.calendar_life = params['calendar_life']
        self.degradation_rate = params['degradation_rate']
        
        # Tracking metrics
        self.total_cycles = 0.0  # Cumulative equivalent full cycles
        self.total_throughput_kwh = 0.0  # Total energy throughput
        self.age_years = 0.0
        self.degradation_history = []
        
        # Energy tracking for current operation
        self.energy_charged = 0.0
        self.energy_discharged = 0.0
        
    def get_available_capacity_kwh(self) -> float:
        """Returns the usable capacity considering degradation and DoD."""
        usable_range = self.soc_max - self.soc_min
        return self.current_capacity_kwh * usable_range
    
    def get_energy_stored_kwh(self) -> float:
        """Returns current energy stored in battery."""
        return self.current_capacity_kwh * self.soc
    
    def get_max_charge_power(self, dt: float = 1.0) -> float:
        """
        Calculate maximum charging power at current SOC.
        
        Args:
            dt: Time step in hours
            
        Returns:
            Maximum charge power (kW)
        """
        # Power limited by inverter rating
        power_limit = self.power_rating_kw
        
        # Energy limited by remaining capacity
        remaining_capacity = self.current_capacity_kwh * (self.soc_max - self.soc)
        energy_limit = remaining_capacity / dt / self.efficiency  # Account for losses
        
        return min(power_limit, energy_limit)
    
    def get_max_discharge_power(self, dt: float = 1.0) -> float:
        """
        Calculate maximum discharging power at current SOC.
        
        Args:
            dt: Time step in hours
            
        Returns:
            Maximum discharge power (kW)
        """
        # Power limited by inverter rating
        power_limit = self.power_rating_kw
        
        # Energy limited by available energy
        available_energy = self.current_capacity_kwh * (self.soc - self.soc_min)
        energy_limit = available_energy * self.efficiency / dt
        
        return min(power_limit, energy_limit)
    
    def charge(self, power_kw: float, dt: float = 1.0) -> Tuple[float, float]:
        """
        Charge the battery.
        
        Args:
            power_kw: Charging power (kW)
            dt: Time step in hours
            
        Returns:
            (actual_power_charged, energy_accepted)
        """
        # Limit to maximum charge power
        max_power = self.get_max_charge_power(dt)
        actual_power = min(power_kw, max_power)
        
        # Calculate energy with efficiency losses
        energy_in = actual_power * dt
        energy_stored = energy_in * self.efficiency
        
        # Update SOC
        delta_soc = energy_stored / self.current_capacity_kwh
        self.soc = min(self.soc + delta_soc, self.soc_max)
        
        # Track throughput
        self.total_throughput_kwh += energy_stored
        self.energy_charged += energy_stored
        
        return actual_power, energy_stored
    
    def discharge(self, power_kw: float, dt: float = 1.0) -> Tuple[float, float]:
        """
        Discharge the battery.
        
        Args:
            power_kw: Discharging power (kW)
            dt: Time step in hours
            
        Returns:
            (actual_power_discharged, energy_delivered)
        """
        # Limit to maximum discharge power
        max_power = self.get_max_discharge_power(dt)
        actual_power = min(power_kw, max_power)
        
        # Calculate energy with efficiency losses
        energy_out = actual_power * dt
        energy_drawn = energy_out / self.efficiency
        
        # Update SOC
        delta_soc = energy_drawn / self.current_capacity_kwh
        self.soc = max(self.soc - delta_soc, self.soc_min)
        
        # Track throughput
        self.total_throughput_kwh += energy_drawn
        self.energy_discharged += energy_out
        
        return actual_power, energy_out
    
    def update_degradation(self, dt_years: float = None):
        """
        Update battery degradation based on cycling and calendar aging.
        
        Args:
            dt_years: Time step in years (None for automatic calculation)
        """
        if dt_years is None:
            # Assume degradation update called annually
            dt_years = 1.0
        
        # Calendar aging (linear fade)
        calendar_fade = self.degradation_rate * dt_years
        
        # Cycle aging (based on throughput)
        # One full cycle = 2 * nominal capacity (charge + discharge)
        cycle_increment = self.total_throughput_kwh / (2 * self.nominal_capacity_kwh)
        self.total_cycles += cycle_increment
        
        # Cycle fade (nonlinear)
        cycle_fade = (self.total_cycles / self.lifetime_cycles) * 0.20  # 20% fade at EOL
        
        # Total degradation
        total_fade = min(calendar_fade + cycle_fade, 0.30)  # Max 30% degradation
        
        # Update capacity
        self.current_capacity_kwh = self.nominal_capacity_kwh * (1 - total_fade)
        
        # Update age
        self.age_years += dt_years
        
        # Record degradation
        self.degradation_history.append({
            'age_years': self.age_years,
            'total_cycles': self.total_cycles,
            'capacity_kwh': self.current_capacity_kwh,
            'capacity_percent': (self.current_capacity_kwh / self.nominal_capacity_kwh) * 100
        })
        
        # Reset throughput counter
        self.total_throughput_kwh = 0.0
    
    def needs_replacement(self, threshold: float = None) -> bool:
        """
        Check if battery needs replacement.
        
        Args:
            threshold: Capacity threshold (fraction). Uses config default if None.
            
        Returns:
            True if capacity below threshold
        """
        if threshold is None:
            threshold = config.ECONOMIC_PARAMS['replacement_threshold']
        
        capacity_ratio = self.current_capacity_kwh / self.nominal_capacity_kwh
        return capacity_ratio < threshold
    
    def reset(self):
        """Reset battery to initial state."""
        self.current_capacity_kwh = self.nominal_capacity_kwh
        self.soc = config.BATTERY_PARAMS['soc_initial']
        self.total_cycles = 0.0
        self.total_throughput_kwh = 0.0
        self.age_years = 0.0
        self.degradation_history = []
        self.energy_charged = 0.0
        self.energy_discharged = 0.0
    
    def get_status(self) -> Dict:
        """
        Get current battery status.
        
        Returns:
            Dictionary with battery status information
        """
        return {
            'soc': self.soc,
            'soc_percent': self.soc * 100,
            'energy_stored_kwh': self.get_energy_stored_kwh(),
            'current_capacity_kwh': self.current_capacity_kwh,
            'nominal_capacity_kwh': self.nominal_capacity_kwh,
            'capacity_fade_percent': (1 - self.current_capacity_kwh / self.nominal_capacity_kwh) * 100,
            'total_cycles': self.total_cycles,
            'age_years': self.age_years,
            'max_charge_power_kw': self.get_max_charge_power(),
            'max_discharge_power_kw': self.get_max_discharge_power(),
        }
    
    def __repr__(self):
        """String representation of battery."""
        status = self.get_status()
        return (f"BatteryStorage("
                f"Capacity: {status['current_capacity_kwh']:.1f}/{status['nominal_capacity_kwh']:.1f} kWh, "
                f"SOC: {status['soc_percent']:.1f}%, "
                f"Cycles: {status['total_cycles']:.1f}, "
                f"Age: {status['age_years']:.1f} years)")


def calculate_battery_cost(energy_kwh: float, power_kw: float) -> Dict[str, float]:
    """
    Calculate battery system costs.
    
    Args:
        energy_kwh: Battery energy capacity
        power_kw: Battery power rating
        
    Returns:
        Dictionary with cost breakdown
    """
    params = config.BATTERY_PARAMS
    
    energy_cost = energy_kwh * params['cost_per_kwh']
    power_cost = power_kw * params['cost_per_kw']
    total_capital = energy_cost + power_cost
    
    return {
        'energy_cost': energy_cost,
        'power_cost': power_cost,
        'total_capital_cost': total_capital,
        'cost_per_kwh': params['cost_per_kwh'],
        'cost_per_kw': params['cost_per_kw']
    }


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*60)
    print("BATTERY STORAGE SYSTEM SIMULATION")
    print("="*60)
    
    # Create battery
    battery = BatteryStorage(energy_capacity_kwh=100, power_rating_kw=50)
    print(f"\nInitial Status:")
    print(battery)
    
    # Simulate daily cycling for 1 year
    print(f"\n\nSimulating 365 days of daily cycling...")
    for day in range(365):
        # Daytime: Charge
        for hour in range(6):
            battery.charge(power_kw=40, dt=1.0)
        
        # Evening: Discharge
        for hour in range(6):
            battery.discharge(power_kw=35, dt=1.0)
        
        # Update degradation monthly
        if (day + 1) % 30 == 0:
            battery.update_degradation(dt_years=30/365)
    
    print(f"\nStatus after 1 year:")
    print(battery)
    
    status = battery.get_status()
    print(f"\nDetailed Status:")
    for key, value in status.items():
        print(f"  {key}: {value:.2f}")
    
    # Cost calculation
    costs = calculate_battery_cost(100, 50)
    print(f"\n\nCost Breakdown:")
    for key, value in costs.items():
        print(f"  {key}: ${value:,.2f}")

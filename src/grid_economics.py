"""
Grid Economics Module
Provides utility functions for calculating grid costs with Time-of-Use pricing.
"""

import numpy as np
from typing import Dict, Tuple
try:
    from . import config
except ImportError:
    import config


def get_grid_price(hour: int) -> Tuple[float, float]:
    """
    Get grid import and export prices for a given hour.
    
    Args:
        hour: Hour of day (0-23)
        
    Returns:
        (import_price, export_price) in ₹/kWh
    """
    grid_params = config.GRID_PARAMS
    
    if not grid_params['tou_enabled']:
        # Flat rate
        return (grid_params['offpeak_import_price'], 
                grid_params['export_price'])
    
    # Time-of-Use pricing
    if hour in grid_params['peak_hours']:
        import_price = grid_params['peak_import_price']
    elif hour in grid_params['mid_hours']:
        import_price = grid_params['mid_import_price']
    else:  # Off-peak
        import_price = grid_params['offpeak_import_price']
    
    export_price = grid_params['export_price']
    
    return (import_price, export_price)


def calculate_grid_costs(unmet_load: np.ndarray, curtailed_pv: np.ndarray, 
                         start_hour: int = 0) -> Dict[str, float]:
    """
    Calculate grid import costs and export revenue with TOU pricing.
    
    Args:
        unmet_load: Array of unmet load values (kWh)
        curtailed_pv: Array of curtailed PV values (kWh)
        start_hour: Starting hour of day for simulation (default 0)
        
    Returns:
        Dictionary with grid economics (in ₹):
        - grid_import_cost: Total cost of grid imports
        - grid_export_revenue: Total revenue from grid exports
        - net_grid_cost: Net cost (import - revenue)
    """
    total_import_cost = 0.0
    total_export_revenue = 0.0
    
    num_steps = len(unmet_load)
    
    for t in range(num_steps):
        hour_of_day = (start_hour + t) % 24
        import_price, export_price = get_grid_price(hour_of_day)
        
        # Grid import cost
        import_energy = unmet_load[t]
        total_import_cost += import_energy * import_price
        
        # Grid export revenue
        export_energy = curtailed_pv[t]
        total_export_revenue += export_energy * export_price
    
    net_cost = total_import_cost - total_export_revenue
    
    return {
        'grid_import_cost': total_import_cost,
        'grid_export_revenue': total_export_revenue,
        'net_grid_cost': net_cost
    }


def calculate_annual_grid_cost(unmet_load: np.ndarray, curtailed_pv: np.ndarray,
                               start_hour: int = 0) -> float:
    """
    Calculate annualized grid cost (simplified).
    
    Args:
        unmet_load: Array of unmet load values for simulation period
        curtailed_pv: Array of curtailed PV values
        start_hour: Starting hour
        
    Returns:
        Annual grid cost in ₹
    """
    costs = calculate_grid_costs(unmet_load, curtailed_pv, start_hour)
   
    # Scale to annual if simulation is shorter than a year
    sim_hours = len(unmet_load)
    hours_per_year = 8760
    
    if sim_hours < hours_per_year:
        # Scale up to annual
        scale_factor = hours_per_year / sim_hours
        annual_cost = costs['net_grid_cost'] * scale_factor
    else:
        annual_cost = costs['net_grid_cost']
    
    return annual_cost


if __name__ == "__main__":
    # Test grid pricing
    print("\n" + "="*60)
    print("GRID PRICING TEST")
    print("="*60)
    
    # Test TOU prices at different hours
    test_hours = [0, 6, 12, 18, 22]
    for hour in test_hours:
        import_price, export_price = get_grid_price(hour)
        print(f"Hour {hour:2d}: Import ₹{import_price:.2f}/kWh, Export ₹{export_price:.2f}/kWh")
    
    # Test cost calculation
    print("\n" + "="*60)
    print("COST CALCULATION TEST")
    print("="*60)
    
    # Simulate 24 hours
    unmet = np.array([5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 8, 10, 12, 8, 6, 4])  # kWh
    curtailed = np.array([0, 0, 0, 0, 0, 0, 2, 5, 8, 10, 12, 10,
                          8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0])  # kWh
    
    costs = calculate_grid_costs(unmet, curtailed, start_hour=0)
    
    print(f"Grid Import Cost:    ₹{costs['grid_import_cost']:,.2f}")
    print(f"Grid Export Revenue: ₹{costs['grid_export_revenue']:,.2f}")
    print(f"Net Grid Cost:       ₹{costs['net_grid_cost']:,.2f}")
    print("="*60 + "\n")

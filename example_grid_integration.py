"""
Example: How to use grid_economics.py in optimization

This shows how to integrate grid cost calculations into the BESS optimizer.
"""

import numpy as np

# Import the grid economics module
from src.grid_economics import calculate_grid_costs, calculate_annual_grid_cost, get_grid_price

# Example simulation results
unmet_load = np.array([5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 8, 10, 12, 8, 6, 4] * 365)  # 365 days
curtailed_pv = np.array([0, 0, 0, 0, 0, 0, 2, 5, 8, 10, 12, 10,
                         8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0] * 365)  # 365 days

print("="*60)
print("GRID ECONOMICS INTEGRATION EXAMPLE")
print("="*60)

# Calculate grid costs
grid_costs = calculate_grid_costs(unmet_load, curtailed_pv, start_hour=0)

print(f"\n1. Grid Transaction Cost (Annual):")
print(f"   Grid Import Cost:    ₹{grid_costs['grid_import_cost']:,.2f}")
print(f"   Grid Export Revenue: ₹{grid_costs['grid_export_revenue']:,.2f}")
print(f"   Net Grid Cost:       ₹{grid_costs['net_grid_cost']:,.2f}")

# Calculate annualized cost
annual_cost = calculate_annual_grid_cost(unmet_load, curtailed_pv, start_hour=0)
print(f"\n2. Annual Grid Cost: ₹{annual_cost:,.2f}")

# Show how to add to NPC calculation
print("\n" + "="*60)
print("HOW TO ADD TO NPC CALCULATION IN OPTIMIZER.PY")
print("="*60)

print("""
In optimizer.py, in the calculate_npc() method, ADD:

```python
# Grid costs (import costs - export revenue) in ₹
# Calculate annual grid cost from simulation results
if 'unmet_load' in simulation_results and 'curtailed_pv' in simulation_results:
    annual_grid_cost = calculate_annual_grid_cost(
        simulation_results['unmet_load'],
        simulation_results['curtailed_pv'],
        start_hour=0
    )
    
    # Present value of grid costs over project lifetime
    pv_grid_cost = sum([annual_grid_cost / ((1 + interest_rate) ** year)
                       for year in range(1, project_life + 1)])
else:
    pv_grid_cost = 0.0

# Total NPC (in ₹)
npc = capital_cost + pv_om + replacement_cost + pv_grid_cost
```

REPLACE the old line:
```python
npc = capital_cost + pv_om + replacement_cost - revenue
```

WITH the new line above.
""")

print("\n" + "="*60)
print("IMPACT ON OPTIMIZATION")
print("="*60)

# Example comparison
battery_capital = 1000000  # ₹10 lakh for battery
om_costs = 200000  # ₹2 lakh O&M
replacement = 500000  # ₹5 lakh
grid_costs_annual = annual_cost

# Without grid costs
npc_without_grid = battery_capital + om_costs + replacement
print(f"\nNPC WITHOUT grid costs: ₹{npc_without_grid:,.2f}")

# With grid costs (present value over 20 years at 8% interest)
interest_rate = 0.08
project_life = 20
pv_grid = sum([grid_costs_annual / ((1 + interest_rate) ** year)
               for year in range(1, project_life + 1)])

npc_with_grid = battery_capital + om_costs + replacement + pv_grid
print(f"NPC WITH grid costs:    ₹{npc_with_grid:,.2f}")
print(f"\nDifference: ₹{npc_with_grid - npc_without_grid:,.2f}")
print(f"Grid costs are {(pv_grid/npc_with_grid)*100:.1f}% of total NPC!")

print("\n" + "="*60)
print("TOU PRICING SCHEDULE")
print("="*60)
print("\nHourly grid prices:")
for hour in range(24):
    import_price, export_price = get_grid_price(hour)
    print(f"Hour {hour:2d}: Import ₹{import_price:.2f}/kWh")

"""
Optimization Module
Implements PSO-based optimization for BESS sizing with techno-economic objectives.
"""

import numpy as np
from pyswarm import pso
from typing import Tuple, Dict, Callable
import warnings

from . import config
from .battery_model import BatteryStorage, calculate_battery_cost
from .simulation import MicrogridSimulator


class BESSOptimizer:
    """
    Optimizes BESS sizing using Particle Swarm Optimization (PSO).
    Minimizes Net Present Cost (NPC) subject to LPSP constraint.
    """
    
    def __init__(self, pv_power: np.ndarray, load_demand: np.ndarray):
        """
        Initialize BESS optimizer.
        
        Args:
            pv_power: Forecasted PV power array (kW)
            load_demand: Forecasted load demand array (kW)
        """
        self.pv_power = pv_power
        self.load_demand = load_demand
        
        # Get parameters
        self.opt_params = config.OPTIMIZATION_PARAMS
        self.econ_params = config.ECONOMIC_PARAMS
        self.battery_params = config.BATTERY_PARAMS
        
        # Optimization results
        self.best_solution = None
        self.best_cost = None
        self.optimization_history = []
        
        # Evaluation counter
        self.eval_count = 0
    
    def calculate_npc(self, energy_kwh: float, power_kw: float, 
                     simulation_results: Dict) -> float:
        """
        Calculate Net Present Cost (NPC) for a BESS configuration.
        
        NPC = Capital Cost + O&M Costs + Replacement Costs - Revenue
        
        Args:
            energy_kwh: Battery energy capacity
            power_kw: Battery power rating
            simulation_results: Results from simulation
            
        Returns:
            Net Present Cost (₹)
        """
        # Capital cost
        costs = calculate_battery_cost(energy_kwh, power_kw)
        capital_cost = costs['total_capital_cost']
        
        # Economic parameters
        interest_rate = self.econ_params['interest_rate']
        project_life = self.econ_params['project_lifetime']
        om_percent = self.econ_params['om_cost_percent']
        
        # Annual O&M costs (present value)
        annual_om = capital_cost * om_percent
        pv_om = sum([annual_om / ((1 + interest_rate) ** year) 
                     for year in range(1, project_life + 1)])
        
        # Replacement cost (if needed during project life)
        replacement_cost = 0.0
        battery_life = self.battery_params['calendar_life']
        
        if battery_life < project_life:
            # Calculate number of replacements needed
            num_replacements = int(np.floor(project_life / battery_life))
            
            for replacement in range(1, num_replacements + 1):
                year = replacement * battery_life
                if year < project_life:
                    # Present value of replacement
                    replacement_cost += capital_cost / ((1 + interest_rate) ** year)
        
        # Revenue from grid export (if any) - simplified
        # In this case, we assume no grid export revenue as we focus on self-consumption
        revenue = 0.0
        
        # Total NPC
        npc = capital_cost + pv_om + replacement_cost - revenue
        
        return npc
    
    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for optimization.
        Minimizes NPC while penalizing LPSP violations.
        
        Args:
            x: Decision variables [energy_kwh, power_kw]
            
        Returns:
            Objective value (cost + penalties)
        """
        self.eval_count += 1
        
        energy_kwh, power_kw = x
        
        # Create battery and simulator
        try:
            battery = BatteryStorage(energy_kwh, power_kw)
            simulator = MicrogridSimulator(battery)
            
            # Run simulation
            results = simulator.simulate(self.pv_power, self.load_demand, dt=1.0)
            
            # Get metrics
            metrics = simulator.get_metrics()
            lpsp = metrics['lpsp']
            
            # Calculate NPC
            npc = self.calculate_npc(energy_kwh, power_kw, results)
            
            # LPSP penalty (soft constraint)
            lpsp_threshold = self.opt_params['lpsp_max']
            if lpsp > lpsp_threshold:
                # Heavy penalty for violating LPSP constraint
                penalty = 1e6 * (lpsp - lpsp_threshold)
                objective = npc + penalty
            else:
                objective = npc
            
            # Store in history
            self.optimization_history.append({
                'energy_kwh': energy_kwh,
                'power_kw': power_kw,
                'npc': npc,
                'lpsp': lpsp,
                'objective': objective
            })
            
            return objective
            
        except Exception as e:
            # If simulation fails, return large penalty
            warnings.warn(f"Simulation failed for x={x}: {e}")
            return 1e9
    
    def constraint_lpsp(self, x: np.ndarray) -> float:
        """
        LPSP constraint function.
        Constraint is satisfied when result >= 0.
        
        Args:
            x: Decision variables [energy_kwh, power_kw]
            
        Returns:
            lpsp_max - lpsp (>= 0 means constraint satisfied)
        """
        energy_kwh, power_kw = x
        
        try:
            battery = BatteryStorage(energy_kwh, power_kw)
            simulator = MicrogridSimulator(battery)
            results = simulator.simulate(self.pv_power, self.load_demand, dt=1.0)
            
            lpsp = simulator.get_metrics()['lpsp']
            lpsp_threshold = self.opt_params['lpsp_max']
            
            return lpsp_threshold - lpsp  # >= 0 means satisfied
            
        except:
            return -1  # Constraint violated
    
    def optimize(self, swarm_size: int = None, max_iter: int = None,
                verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization to find optimal BESS sizing.
        
        Args:
            swarm_size: Number of particles (None uses config default)
            max_iter: Maximum iterations (None uses config default)
            verbose: Print progress
            
        Returns:
            (optimal_solution, optimal_cost)
        """
        if swarm_size is None:
            swarm_size = self.opt_params['pso_swarm_size']
        if max_iter is None:
            max_iter = self.opt_params['pso_max_iter']
        
        # Decision variable bounds
        lb = [self.opt_params['bess_kwh_min'], self.opt_params['bess_kw_min']]
        ub = [self.opt_params['bess_kwh_max'], self.opt_params['bess_kw_max']]
        
        if verbose:
            print("\n" + "="*60)
            print("PARTICLE SWARM OPTIMIZATION")
            print("="*60)
            print(f"Swarm size: {swarm_size}")
            print(f"Max iterations: {max_iter}")
            print(f"Decision variables:")
            print(f"  Energy capacity: [{lb[0]:.1f}, {ub[0]:.1f}] kWh")
            print(f"  Power rating:    [{lb[1]:.1f}, {ub[1]:.1f}] kW")
            print(f"Constraint: LPSP <= {self.opt_params['lpsp_max']*100:.1f}%")
            print("="*60 + "\n")
        
        # Reset counter
        self.eval_count = 0
        self.optimization_history = []
        
        # Run PSO
        try:
            xopt, fopt = pso(
                func=self.objective_function,
                lb=lb,
                ub=ub,
                swarmsize=swarm_size,
                maxiter=max_iter,
                omega=self.opt_params['pso_omega'],
                phip=self.opt_params['pso_phip'],
                phig=self.opt_params['pso_phig'],
                debug=verbose
            )
            
            self.best_solution = xopt
            self.best_cost = fopt
            
            if verbose:
                print("\n" + "="*60)
                print("OPTIMIZATION RESULTS")
                print("="*60)
                print(f"Optimal Energy Capacity: {xopt[0]:.2f} kWh")
                print(f"Optimal Power Rating:   {xopt[1]:.2f} kW")
                print(f"Optimal NPC:            ₹{fopt:,.2f}")
                print(f"Total evaluations:      {self.eval_count}")
                print("="*60 + "\n")
            
            return xopt, fopt
            
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")
    
    def evaluate_solution(self, energy_kwh: float, power_kw: float, 
                         verbose: bool = True) -> Dict:
        """
        Evaluate a specific BESS configuration in detail.
        
        Args:
            energy_kwh: Battery energy capacity
            power_kw: Battery power rating
            verbose: Print detailed results
            
        Returns:
            Dictionary with complete evaluation results
        """
        # Create battery and simulator
        battery = BatteryStorage(energy_kwh, power_kw)
        simulator = MicrogridSimulator(battery)
        
        # Run simulation
        results = simulator.simulate(self.pv_power, self.load_demand, dt=1.0)
        metrics = simulator.get_metrics()
        
        # Calculate economics
        npc = self.calculate_npc(energy_kwh, power_kw, results)
        costs = calculate_battery_cost(energy_kwh, power_kw)
        
        # Compile evaluation
        evaluation = {
            'sizing': {
                'energy_kwh': energy_kwh,
                'power_kw': power_kw,
                'energy_to_power_ratio': energy_kwh / power_kw if power_kw > 0 else 0
            },
            'costs': costs,
            'economics': {
                'npc': npc,
                'levelized_cost': npc / self.econ_params['project_lifetime']
            },
            'performance': metrics,
            'battery_status': battery.get_status()
        }
        
        if verbose:
            self._print_evaluation(evaluation)
        
        return evaluation
    
    def _print_evaluation(self, evaluation: Dict):
        """Print formatted evaluation results."""
        print("\n" + "="*60)
        print("BESS CONFIGURATION EVALUATION")
        print("="*60)
        
        print("\nSizing:")
        for key, value in evaluation['sizing'].items():
            print(f"  {key}: {value:.2f}")
        
        print("\nCosts:")
        for key, value in evaluation['costs'].items():
            if 'cost' not in key.lower() or value > 100:
                print(f"  {key}: ₹{value:,.2f}")
            else:
                print(f"  {key}: ₹{value:.2f}")
        
        print("\nEconomics:")
        for key, value in evaluation['economics'].items():
            print(f"  {key}: ₹{value:,.2f}")
        
        print("\nPerformance:")
        perf = evaluation['performance']
        print(f"  LPSP:                 {perf['lpsp']*100:.2f} %")
        print(f"  Renewable Penetration:{perf['renewable_penetration']*100:.2f} %")
        print(f"  Self-Consumption:     {perf['self_consumption']*100:.2f} %")
        print(f"  Battery Cycles:       {perf['battery_cycles']:.2f}")
        
        print("="*60 + "\n")


def multi_objective_optimization(pv_power: np.ndarray, load_demand: np.ndarray,
                                 weights: Dict[str, float] = None) -> Dict:
    """
    Perform multi-objective optimization with weighted objectives.
    
    Args:
        pv_power: PV generation forecast
        load_demand: Load demand forecast
        weights: Weights for objectives {'cost': 0.7, 'lpsp': 0.2, 'cycles': 0.1}
        
    Returns:
        Optimization results
    """
    if weights is None:
        weights = {'cost': 0.7, 'lpsp': 0.2, 'reliability': 0.1}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    optimizer = BESSOptimizer(pv_power, load_demand)
    
    # Modified objective function
    original_obj = optimizer.objective_function
    
    def multi_obj(x):
        cost = original_obj(x)
        # Add other objectives as needed
        return cost  # Simplified for now
    
    optimizer.objective_function = multi_obj
    return optimizer.optimize()


if __name__ == "__main__":
    # Example optimization
    print("\n" + "="*60)
    print("BESS OPTIMIZATION EXAMPLE")
    print("="*60)
    
    # Generate sample data for 30 days
    hours = 24 * 30
    time = np.arange(hours)
    hour_of_day = time % 24
    
    # PV generation
    pv_power = 20 * np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
    
    # Load demand
    load_demand = 12 + 5 * np.sin(np.pi * (hour_of_day - 8) / 10)
    
    # Initialize optimizer
    optimizer = BESSOptimizer(pv_power, load_demand)
    
    # Run optimization
    optimal_solution, optimal_cost = optimizer.optimize(
        swarm_size=20,  # Reduced for quick demo
        max_iter=10,    # Reduced for quick demo
        verbose=True
    )
    
    # Detailed evaluation of optimal solution
    evaluation = optimizer.evaluate_solution(
        optimal_solution[0], 
        optimal_solution[1],
        verbose=True
    )

"""
Main Execution Script
Orchestrates the complete BESS optimization workflow:
1. Data loading and preprocessing
2. Forecasting model training or loading
3. Future forecasting
4. BESS sizing optimization
5. Results visualization and export
"""

import numpy as np
import os
import argparse

from src import config
from src.data_loader import DataLoader, generate_sample_data
from src.forecasting import ForecastingEngine
from src.battery_model import BatteryStorage
from src.simulation import MicrogridSimulator
from src.optimizer import BESSOptimizer
from src import utils


def main(args):
    """Main execution function."""
    
    # Print welcome message
    utils.print_welcome_message()
    
    # ========================================================================
    # STAGE 1: DATA PREPARATION
    # ========================================================================
    print("\n" + "▶"*30)
    print("STAGE 1: DATA PREPARATION")
    print("▶"*30 + "\n")
    
    # Check if data file exists, generate if not
    data_file = args.data_file
    if not os.path.exists(data_file):
        print(f"Data file not found. Generating sample data...")
        generate_sample_data(data_file, num_days=args.data_days)
    
    # Load and preprocess data
    loader = DataLoader()
    data_dict = loader.prepare_data(data_file, test_size=args.test_split)
    
    # ========================================================================
    # STAGE 2: FORECASTING
    # ========================================================================
    print("\n" + "▶"*30)
    print("STAGE 2: DEEP LEARNING FORECASTING")
    print("▶"*30 + "\n")
    
    # Initialize forecasting engine
    engine = ForecastingEngine()
    
    # Check if pre-trained model exists
    model_path = os.path.join(config.MODELS_DIR, 'best_model.pth')
    
    if args.train_model or not os.path.exists(model_path):
        print("Training new forecasting model...")
        
        # Train model
        history = engine.train(
            data_dict['X_train'],
            data_dict['y_train'],
            data_dict['X_test'],
            data_dict['y_test'],
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save model
        engine.save_model(model_path)
        
        # Evaluate
        y_pred = engine.predict(data_dict['X_test'])
        metrics = engine.evaluate_metrics(data_dict['y_test'], y_pred)
        
        print("\n" + "="*60)
        print("FORECASTING MODEL EVALUATION")
        print("="*60)
        for key, value in metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
        print("="*60 + "\n")
        
        # Plot forecast results
        if args.plot:
            plot_path = os.path.join(config.RESULTS_DIR, 'forecast_results.png')
            utils.plot_forecast_results(data_dict['y_test'], y_pred, 
                                       save_path=plot_path)
    else:
        print(f"Loading pre-trained model from {model_path}...")
        engine.load_model(model_path)
    
    # Generate forecasts for optimization
    print("\nGenerating forecasts for optimization period...")
    
    # Use test set or generate new forecasts
    if args.use_test_data:
        # Use actual test data
        y_forecast = engine.predict(data_dict['X_test'])
        
        # Take first forecast (24 hours)
        pv_forecast = y_forecast[0, :, 0] * config.SIMULATION_PARAMS['stc_irradiance']
        load_forecast = y_forecast[0, :, 1] * 50  # Scale back to kW
        
        # Denormalize if needed
        # In practice, you'd use the scaler from data_loader
        
    else:
        # Generate synthetic data for demonstration
        hours = args.optimization_horizon
        time = np.arange(hours)
        hour_of_day = time % 24
        
        # PV generation (Increased to 50kW peak to match ~12kW avg load)
        pv_forecast = 50 * np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
        load_forecast = 12 + 5 * np.sin(np.pi * (hour_of_day - 8) / 10)
    
    print(f"✓ Generated {len(pv_forecast)} hour forecast")
    
    # ========================================================================
    # STAGE 3: BESS OPTIMIZATION
    # ========================================================================
    print("\n" + "▶"*30)
    print("STAGE 3: BESS SIZING OPTIMIZATION")
    print("▶"*30 + "\n")
    
    # Initialize optimizer
    optimizer = BESSOptimizer(pv_forecast, load_forecast)
    
    # Run optimization
    optimal_solution, optimal_cost = optimizer.optimize(
        swarm_size=args.swarm_size,
        max_iter=args.max_iterations,
        verbose=True
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Optimal Energy Capacity: {optimal_solution[0]:.2f} kWh")
    print(f"  Optimal Power Rating:    {optimal_solution[1]:.2f} kW")
    print(f"  Optimal NPC:             ₹{optimal_cost:,.2f}")
    
    # ========================================================================
    # STAGE 4: DETAILED EVALUATION
    # ========================================================================
    print("\n" + "▶"*30)
    print("STAGE 4: DETAILED EVALUATION")
    print("▶"*30 + "\n")
    
    # Evaluate optimal solution
    evaluation = optimizer.evaluate_solution(
        optimal_solution[0],
        optimal_solution[1],
        verbose=True
    )
    
    # ========================================================================
    # STAGE 5: RESULTS EXPORT AND VISUALIZATION
    # ========================================================================
    print("\n" + "▶"*30)
    print("STAGE 5: RESULTS EXPORT")
    print("▶"*30 + "\n")
    
    # Save results
    results_file = os.path.join(config.RESULTS_DIR, 'optimization_results.json')
    utils.save_results(evaluation, results_file)
    
    # Create metrics summary
    summary_df = utils.calculate_metrics_summary(evaluation)
    print("\nMetrics Summary:")
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_file = os.path.join(config.RESULTS_DIR, 'metrics_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to {summary_file}")
    
    # Plot optimization history
    if args.plot and optimizer.optimization_history:
        opt_plot_path = os.path.join(config.RESULTS_DIR, 'optimization_history.png')
        utils.plot_optimization_history(optimizer.optimization_history, 
                                       save_path=opt_plot_path)
    
    # Run final simulation and plot
    if args.plot:
        battery = BatteryStorage(optimal_solution[0], optimal_solution[1])
        simulator = MicrogridSimulator(battery)
        sim_results = simulator.simulate(pv_forecast, load_forecast)
        
        sim_plot_path = os.path.join(config.RESULTS_DIR, 'simulation_results.png')
        utils.plot_simulation_results(sim_results, save_path=sim_plot_path)
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "="*70)
    print(" "*20 + "✓ OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"  - optimization_results.json")
    print(f"  - metrics_summary.csv")
    if args.plot:
        print(f"  - forecast_results.png")
        print(f"  - optimization_history.png")
        print(f"  - simulation_results.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BESS Techno-Economic Optimization System"
    )
    
    # Data arguments
    parser.add_argument('--data-file', type=str, 
                       default=os.path.join(config.DATA_DIR, 'sample_data.csv'),
                       help='Path to input CSV data file')
    parser.add_argument('--data-days', type=int, default=365,
                       help='Number of days to generate if creating sample data')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Train/test split ratio')
    
    # Forecasting arguments
    parser.add_argument('--train-model', action='store_true',
                       help='Force training new model even if saved model exists')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--use-test-data', action='store_true',
                       help='Use test data for optimization instead of synthetic')
    
    # Optimization arguments
    parser.add_argument('--optimization-horizon', type=int, default=24*30,
                       help='Optimization horizon in hours')
    parser.add_argument('--swarm-size', type=int, default=30,
                       help='PSO swarm size')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='PSO maximum iterations')
    
    # Output arguments
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate plots')
    parser.add_argument('--no-plot', action='store_false', dest='plot',
                       help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Run main workflow
    main(args)

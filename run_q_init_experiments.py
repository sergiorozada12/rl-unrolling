#!/usr/bin/env python3
"""
Script to run q-bar initialization experiments with different initialization methods.
"""

import json
import time
from typing import Dict, Any
import wandb
from main import unrl

def run_experiment(init_q: str, experiment_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment with the given q initialization method."""
    print(f"Running experiment with init_q='{init_q}'")
    
    # Set up experiment parameters
    params = experiment_params.copy()
    params['init_q'] = init_q
    params['group'] = f"q_init_comparison_{init_q}"
    
    # Run the experiment
    start_time = time.time()
    unrl(**params)
    end_time = time.time()
    
    # Extract final metrics from the wandb run
    # Note: Since wandb.finish() is called in unrl, we need to get the last run
    api = wandb.Api()
    runs = api.runs("rl-unrolling", filters={"group": params['group']})
    if runs:
        last_run = runs[0]  # Get the most recent run
        
        metrics = {
            'init_q': init_q,
            'final_loss': last_run.summary.get('loss', None),
            'final_bellman_error': getattr(last_run.summary, 'bellman_error', None),
            'final_reward_smoothness': getattr(last_run.summary, 'reward_smoothness', None),
            'training_time': end_time - start_time,
            'run_id': last_run.id,
            'run_name': last_run.name
        }
        
        # Get last logged values if summary doesn't have them
        if metrics['final_loss'] is None and last_run.history():
            history = last_run.history()
            if not history.empty and 'loss' in history.columns:
                metrics['final_loss'] = history['loss'].iloc[-1]
        
        return metrics
    else:
        print(f"Warning: Could not find wandb run for init_q='{init_q}'")
        return {'init_q': init_q, 'error': 'Run not found in wandb'}

def main():
    """Run experiments with different q initialization methods."""
    
    # Common experiment parameters
    experiment_params = {
        'K': 10,
        'num_unrolls': 10,
        'tau': 100,
        'beta': 1.0,
        'lr': 5e-3,
        'N': 500,
        'weight_sharing': False,
    }
    
    # Initialize methods to test
    init_methods = ['zeros', 'ones', 'random']
    
    # Store results
    results = []
    
    print("Starting q-bar initialization comparison experiments...")
    print(f"Testing initialization methods: {init_methods}")
    print(f"Common parameters: {experiment_params}")
    print("="*60)
    
    # Run experiments for each initialization method
    for init_q in init_methods:
        try:
            result = run_experiment(init_q, experiment_params)
            results.append(result)
            print(f"Completed experiment for init_q='{init_q}'")
            print(f"Results: {result}")
            print("-"*40)
            
            # Small delay between experiments
            time.sleep(2)
            
        except Exception as e:
            print(f"Error running experiment for init_q='{init_q}': {e}")
            results.append({
                'init_q': init_q,
                'error': str(e)
            })
    
    # Save results to file
    results_file = 'q_init_experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*60)
    print("All experiments completed!")
    print(f"Results saved to: {results_file}")
    
    # Print comparison table
    print("\nCOMPARISON TABLE:")
    print("="*80)
    print(f"{'Init Method':<12} {'Final Loss':<12} {'Bellman Error':<15} {'Reward Smooth':<15} {'Time (s)':<10}")
    print("-"*80)
    
    for result in results:
        if 'error' not in result:
            init_q = result['init_q']
            loss = f"{result['final_loss']:.6f}" if result['final_loss'] is not None else "N/A"
            bell_err = f"{result['final_bellman_error']:.6f}" if result['final_bellman_error'] is not None else "N/A"
            rew_smooth = f"{result['final_reward_smoothness']:.6f}" if result['final_reward_smoothness'] is not None else "N/A"
            time_taken = f"{result['training_time']:.1f}"
            
            print(f"{init_q:<12} {loss:<12} {bell_err:<15} {rew_smooth:<15} {time_taken:<10}")
        else:
            print(f"{result['init_q']:<12} ERROR: {result['error']}")
    
    return results

if __name__ == "__main__":
    results = main()
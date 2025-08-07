#!/usr/bin/env python3
"""
Analyze results from q-bar initialization experiments.
"""

import wandb
import pandas as pd
from typing import Dict, List, Any

def get_experiment_results() -> List[Dict[str, Any]]:
    """Fetch experiment results from wandb."""
    api = wandb.Api()
    
    # Get runs from the q_init_comparison group
    runs = api.runs("rl-unrolling", filters={"group": "q_init_comparison"})
    
    results = []
    for run in runs:
        # Extract init_q from run name
        if "initzeros" in run.name:
            init_q = "zeros"
        elif "initones" in run.name:
            init_q = "ones"
        elif "initrandom" in run.name:
            init_q = "random"
        else:
            continue
            
        # Get metrics
        history = run.history()
        
        result = {
            'init_q': init_q,
            'run_name': run.name,
            'run_id': run.id,
            'final_loss': None,
            'min_loss': None,
            'final_bellman_error': None,
            'min_bellman_error': None,
            'final_reward_smoothness': None,
            'training_time': run.summary.get('fit_time_sec', None),
            'total_epochs': len(history) if not history.empty else 0
        }
        
        if not history.empty:
            # Get final values (last epoch)
            if 'loss' in history.columns:
                result['final_loss'] = history['loss'].iloc[-1]
                result['min_loss'] = history['loss'].min()
            
            if 'bellman_error' in history.columns:
                result['final_bellman_error'] = history['bellman_error'].iloc[-1]
                result['min_bellman_error'] = history['bellman_error'].min()
                
            if 'reward_smoothness' in history.columns:
                result['final_reward_smoothness'] = history['reward_smoothness'].iloc[-1]
        
        results.append(result)
    
    return results

def create_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison table from results."""
    df = pd.DataFrame(results)
    
    # Sort by init_q for consistent ordering
    df = df.sort_values('init_q')
    
    # Select and reorder columns for display
    display_cols = [
        'init_q', 'final_loss', 'min_loss', 
        'final_bellman_error', 'min_bellman_error',
        'final_reward_smoothness', 'total_epochs'
    ]
    
    df_display = df[display_cols].copy()
    
    # Round numerical values for better display
    numerical_cols = ['final_loss', 'min_loss', 'final_bellman_error', 
                     'min_bellman_error', 'final_reward_smoothness']
    
    for col in numerical_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)
    
    return df_display

def print_detailed_analysis(results: List[Dict[str, Any]]):
    """Print detailed analysis of the results."""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF Q-BAR INITIALIZATION EXPERIMENTS")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"\nNumber of experiments: {len(df)}")
    print(f"Initialization methods tested: {sorted(df['init_q'].unique())}")
    
    # Group by init_q and calculate statistics
    grouped = df.groupby('init_q').agg({
        'final_loss': ['mean', 'std', 'min'],
        'min_loss': ['mean', 'std', 'min'],
        'final_bellman_error': ['mean', 'std', 'min'],
        'min_bellman_error': ['mean', 'std', 'min'],
        'final_reward_smoothness': ['mean', 'std'],
        'total_epochs': ['mean']
    }).round(4)
    
    print("\nGROUPED STATISTICS:")
    print(grouped)
    
    # Find best performing initialization for each metric
    print("\nBEST PERFORMING INITIALIZATION BY METRIC:")
    
    if 'final_loss' in df.columns and not df['final_loss'].isna().all():
        best_final_loss = df.loc[df['final_loss'].idxmin()]
        print(f"Lowest Final Loss: {best_final_loss['init_q']} ({best_final_loss['final_loss']:.4f})")
    
    if 'min_loss' in df.columns and not df['min_loss'].isna().all():
        best_min_loss = df.loc[df['min_loss'].idxmin()]
        print(f"Lowest Min Loss: {best_min_loss['init_q']} ({best_min_loss['min_loss']:.4f})")
        
    if 'final_bellman_error' in df.columns and not df['final_bellman_error'].isna().all():
        best_final_bellman = df.loc[df['final_bellman_error'].idxmin()]
        print(f"Lowest Final Bellman Error: {best_final_bellman['init_q']} ({best_final_bellman['final_bellman_error']:.4f})")
    
    if 'min_bellman_error' in df.columns and not df['min_bellman_error'].isna().all():
        best_min_bellman = df.loc[df['min_bellman_error'].idxmin()]
        print(f"Lowest Min Bellman Error: {best_min_bellman['init_q']} ({best_min_bellman['min_bellman_error']:.4f})")

def main():
    """Main analysis function."""
    print("Fetching experiment results from Weights & Biases...")
    
    try:
        results = get_experiment_results()
        
        if not results:
            print("No experiment results found!")
            print("Make sure the experiments have been run and are available in wandb.")
            return
        
        # Create comparison table
        comparison_df = create_comparison_table(results)
        
        print("\nCOMPARISON TABLE:")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Detailed analysis
        print_detailed_analysis(results)
        
        # Save results to CSV
        comparison_df.to_csv('q_init_comparison_results.csv', index=False)
        print(f"\nResults saved to: q_init_comparison_results.csv")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        print("Make sure you have access to the wandb project and the experiments have been run.")

if __name__ == "__main__":
    main()
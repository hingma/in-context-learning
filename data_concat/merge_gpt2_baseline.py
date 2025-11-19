#!/usr/bin/env python3
"""
Script to merge GPT-2 metrics from metrics_gpt2.json into metrics.json as a baseline model.
"""

import json
from pathlib import Path


def merge_gpt2_metrics(metrics_file, gpt2_file, output_file=None):
    """
    Merge GPT-2 metrics into the main metrics file.
    
    Args:
        metrics_file: Path to the main metrics.json file
        gpt2_file: Path to the metrics_gpt2.json file
        output_file: Path to save the merged output (defaults to metrics_file)
    """
    # Read both JSON files
    print(f"Reading {metrics_file}...")
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print(f"Reading {gpt2_file}...")
    with open(gpt2_file, 'r') as f:
        gpt2_metrics = json.load(f)
    
    # Merge the metrics
    print("Merging GPT-2 metrics into main metrics...")
    models_added = 0
    models_updated = 0
    
    for task_type, models in gpt2_metrics.items():
        # Ensure the task type exists in the main metrics
        if task_type not in metrics:
            metrics[task_type] = {}
        
        # Add or update each GPT-2 model configuration
        for model_name, model_data in models.items():
            if model_name in metrics[task_type]:
                print(f"  skipping existing model: {task_type}/{model_name}")
                models_updated += 1
                continue
            else:
                print(f"  Adding new model: {task_type}/{model_name}")
                models_added += 1
            
            metrics[task_type][model_name] = model_data
    
    # Save the merged metrics
    if output_file is None:
        output_file = metrics_file
    
    print(f"\nSaving merged metrics to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDone! Added {models_added} new models, updated {models_updated} existing models.")


if __name__ == "__main__":
    # Define file paths
    script_dir = Path(__file__).parent
    metrics_file = script_dir / "metrics.json"
    gpt2_file = script_dir / "metrics_gpt2.json"
    
    # Create a backup of the original metrics.json
    backup_file = script_dir / "metrics_backup.json"
    print(f"Creating backup at {backup_file}...")
    with open(metrics_file, 'r') as f:
        backup_data = json.load(f)
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    # Merge the metrics
    merge_gpt2_metrics(metrics_file, gpt2_file)
    
    print(f"\nBackup of original metrics saved to: {backup_file}")


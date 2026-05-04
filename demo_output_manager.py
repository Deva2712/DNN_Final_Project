"""
Demo script for output management functionality.

Demonstrates how to use the OutputManager class for organizing and saving
all outputs from the CIFAR-10 disagreement predictor project.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.output_manager import OutputManager


def demo_output_manager():
    """Demonstrate output manager functionality."""
    
    print("="*70)
    print("CIFAR-10 Disagreement Predictor - Output Manager Demo")
    print("="*70)
    print()
    
    # Initialize output manager
    print("1. Initializing OutputManager...")
    manager = OutputManager(base_dir="outputs")
    print(f"   ✓ Created OutputManager with timestamp: {manager.timestamp}")
    print()
    
    # Create directory structure
    print("2. Creating directory structure...")
    manager.create_directory_structure()
    print("   ✓ Created all output directories:")
    print("     - outputs/data_visualizations/")
    print("     - outputs/training_logs/")
    print("     - outputs/evaluation_results/")
    print("     - outputs/ablation_studies/")
    print("     - outputs/explainability/")
    print("     - checkpoints/")
    print()
    
    # Demo 1: Save visualization
    print("3. Saving visualization with timestamp...")
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin(x)')
    ax.plot(x, np.cos(x), label='cos(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Demo Plot')
    ax.legend()
    ax.grid(alpha=0.3)
    
    manager.save_visualization(
        fig=fig,
        experiment_type='evaluation',
        filename='demo_plot.png',
        include_timestamp=True
    )
    
    viz_path = manager.get_visualization_path(
        'evaluation', 'demo_plot.png', True
    )
    print(f"   ✓ Saved visualization to: {viz_path}")
    plt.close(fig)
    print()
    
    # Demo 2: Export metrics to JSON
    print("4. Exporting metrics to JSON with metadata...")
    metrics = {
        'mean_kl': 0.1234,
        'std_kl': 0.0456,
        'mean_js': 0.0789,
        'std_js': 0.0234,
        'pearson_r': 0.8123,
        'pearson_p': 1.23e-10,
        'precision_at_100': 0.65,
        'precision_at_200': 0.72,
        'precision_at_500': 0.81
    }
    
    config = {
        'model': 'ResNet18',
        'loss_function': 'kl_divergence',
        'learning_rate': 1e-4,
        'batch_size': 64,
        'num_epochs': 50
    }
    
    manager.export_metrics_json(
        metrics=metrics,
        experiment_type='evaluation',
        filename='demo_metrics.json',
        config=config,
        seed=42,
        include_timestamp=True
    )
    print(f"   ✓ Exported metrics to JSON with metadata")
    print(f"     - Timestamp: {manager.timestamp}")
    print(f"     - Random seed: 42")
    print(f"     - Config included: Yes")
    print()
    
    # Demo 3: Export comparison table to CSV
    print("5. Exporting comparison table to CSV...")
    comparison_data = {
        'loss_function': ['kl', 'js', 'custom'],
        'mean_kl': [0.1234, 0.1456, 0.1098],
        'mean_js': [0.0789, 0.0823, 0.0756],
        'pearson_r': [0.8123, 0.8234, 0.8456],
        'precision_at_100': [0.65, 0.67, 0.70]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    manager.export_comparison_csv(
        comparison_df=comparison_df,
        experiment_type='ablation',
        filename='loss_function_comparison.csv',
        config={'experiment': 'loss_ablation'},
        seed=42,
        include_timestamp=True
    )
    print(f"   ✓ Exported comparison table to CSV")
    print(f"     - Rows: {len(comparison_df)}")
    print(f"     - Columns: {list(comparison_df.columns)}")
    print(f"     - Metadata file created: Yes")
    print()
    
    # Demo 4: Generate checkpoint paths
    print("6. Generating checkpoint paths...")
    
    checkpoint_examples = [
        ('disagreement_predictor', 'kl', None, True),
        ('disagreement_predictor', 'js', None, True),
        ('disagreement_predictor', 'custom', None, True),
        ('disagreement_predictor', 'kl', 25, False),
    ]
    
    for model_name, loss_fn, epoch, is_best in checkpoint_examples:
        path = manager.get_checkpoint_path(
            model_name=model_name,
            loss_function=loss_fn,
            epoch=epoch,
            is_best=is_best
        )
        status = "best" if is_best else f"epoch {epoch}"
        print(f"   ✓ {loss_fn} loss ({status}): {path}")
    print()
    
    # Demo 5: Export training history
    print("7. Exporting training history...")
    history = {
        'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28],
        'val_loss': [0.6, 0.5, 0.45, 0.42, 0.40],
        'val_kl': [0.25, 0.20, 0.18, 0.16, 0.15],
        'val_js': [0.12, 0.10, 0.09, 0.08, 0.075]
    }
    
    manager.export_training_history(
        history=history,
        loss_function='kl',
        config=config,
        seed=42
    )
    print(f"   ✓ Exported training history for KL loss")
    print(f"     - Epochs: {len(history['train_loss'])}")
    print(f"     - Metrics tracked: {list(history.keys())}")
    print()
    
    # Demo 6: Create experiment summary
    print("8. Creating experiment summary...")
    results = {
        'best_loss_function': 'custom',
        'final_metrics': metrics,
        'training_time_minutes': 45.3,
        'num_parameters': 11_133_632,
        'convergence_epoch': 38
    }
    
    manager.create_experiment_summary(
        experiment_name='final_evaluation',
        results=results
    )
    print(f"   ✓ Created comprehensive experiment summary")
    print(f"     - Experiment: final_evaluation")
    print(f"     - Timestamp: {manager.timestamp}")
    print()
    
    # Summary
    print("="*70)
    print("Demo Complete!")
    print("="*70)
    print()
    print("All outputs have been saved with:")
    print("  ✓ Descriptive filenames")
    print("  ✓ Timestamps for versioning")
    print("  ✓ Organized directory structure")
    print("  ✓ Metadata (config, seed, timestamp)")
    print()
    print("Check the following directories:")
    print("  - outputs/evaluation_results/")
    print("  - outputs/ablation_studies/")
    print("  - outputs/training_logs/")
    print("  - checkpoints/")
    print()


if __name__ == '__main__':
    demo_output_manager()

"""
Integrated demo showing how to use OutputManager with existing modules.

Demonstrates integration of OutputManager with:
- Data pipeline visualizations
- Training monitoring
- Evaluation metrics
- Ablation studies
- Explainability visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.output_manager import OutputManager


def demo_data_pipeline_integration():
    """Demo: Using OutputManager with data pipeline visualizations."""
    print("\n" + "="*70)
    print("1. DATA PIPELINE INTEGRATION")
    print("="*70)
    
    manager = OutputManager()
    manager.create_directory_structure()
    
    # Simulate entropy data
    entropies = np.random.beta(2, 5, 10000) * 3.32  # Entropy values in [0, 3.32]
    
    # Create entropy histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(entropies, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(entropies.mean(), color='red', linestyle='--', 
               label=f'Mean: {entropies.mean():.2f}', linewidth=2)
    ax.set_xlabel('Shannon Entropy (bits)', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Distribution of Human Disagreement (CIFAR-10H)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Save using OutputManager
    manager.save_visualization(
        fig=fig,
        experiment_type='data',
        filename='entropy_histogram.png',
        include_timestamp=True
    )
    plt.close(fig)
    
    print("✓ Saved entropy histogram with timestamp")
    print(f"  Location: {manager.get_visualization_path('data', 'entropy_histogram.png', True)}")


def demo_training_integration():
    """Demo: Using OutputManager for training logs and checkpoints."""
    print("\n" + "="*70)
    print("2. TRAINING INTEGRATION")
    print("="*70)
    
    manager = OutputManager()
    
    # Simulate training history
    history = {
        'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25],
        'val_loss': [0.6, 0.5, 0.45, 0.42, 0.40, 0.39, 0.38],
        'val_kl': [0.25, 0.20, 0.18, 0.16, 0.15, 0.14, 0.135],
        'val_js': [0.12, 0.10, 0.09, 0.08, 0.075, 0.07, 0.068]
    }
    
    config = {
        'model': 'ResNet18',
        'loss_function': 'kl_divergence',
        'learning_rate': 1e-4,
        'batch_size': 64,
        'optimizer': 'AdamW',
        'weight_decay': 1e-4
    }
    
    # Export training history
    manager.export_training_history(
        history=history,
        loss_function='kl',
        config=config,
        seed=42
    )
    
    print("✓ Exported training history with metadata")
    
    # Generate checkpoint paths
    best_checkpoint = manager.get_checkpoint_path(
        model_name='disagreement_predictor',
        loss_function='kl',
        is_best=True
    )
    
    print(f"✓ Best checkpoint path: {best_checkpoint}")
    print("  (Use this path when saving model with torch.save())")


def demo_evaluation_integration():
    """Demo: Using OutputManager for evaluation metrics."""
    print("\n" + "="*70)
    print("3. EVALUATION INTEGRATION")
    print("="*70)
    
    manager = OutputManager()
    
    # Simulate evaluation metrics
    metrics = {
        'mean_kl': 0.1234,
        'std_kl': 0.0456,
        'mean_js': 0.0789,
        'std_js': 0.0234,
        'mean_cosine': 0.9123,
        'std_cosine': 0.0345,
        'pearson_r': 0.8123,
        'pearson_p': 1.23e-10,
        'spearman_r': 0.7956,
        'spearman_p': 2.45e-9,
        'precision_at_100': 0.65,
        'precision_at_200': 0.72,
        'precision_at_500': 0.81
    }
    
    config = {
        'model': 'ResNet18',
        'loss_function': 'kl_divergence',
        'test_set_size': 2000
    }
    
    # Export metrics
    manager.export_metrics_json(
        metrics=metrics,
        experiment_type='evaluation',
        filename='evaluation_metrics.json',
        config=config,
        seed=42,
        include_timestamp=True
    )
    
    print("✓ Exported evaluation metrics with metadata")
    
    # Create entropy correlation plot
    true_entropy = np.random.beta(2, 5, 2000) * 3.32
    pred_entropy = true_entropy + np.random.normal(0, 0.2, 2000)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_entropy, pred_entropy, alpha=0.5, s=20)
    ax.plot([0, 3.32], [0, 3.32], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Entropy (bits)', fontsize=12)
    ax.set_ylabel('Predicted Entropy (bits)', fontsize=12)
    ax.set_title(f'Entropy Prediction Quality\nPearson r = {metrics["pearson_r"]:.3f}', 
                 fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    manager.save_visualization(
        fig=fig,
        experiment_type='evaluation',
        filename='entropy_correlation.png',
        include_timestamp=True
    )
    plt.close(fig)
    
    print("✓ Saved entropy correlation plot")


def demo_ablation_integration():
    """Demo: Using OutputManager for ablation studies."""
    print("\n" + "="*70)
    print("4. ABLATION STUDY INTEGRATION")
    print("="*70)
    
    manager = OutputManager()
    
    # Simulate loss function comparison
    comparison_data = {
        'loss_function': ['kl', 'js', 'custom'],
        'mean_kl': [0.1234, 0.1456, 0.1098],
        'mean_js': [0.0789, 0.0823, 0.0756],
        'pearson_r': [0.8123, 0.8234, 0.8456],
        'spearman_r': [0.7956, 0.8012, 0.8234],
        'precision_at_100': [0.65, 0.67, 0.70],
        'precision_at_200': [0.72, 0.74, 0.77],
        'precision_at_500': [0.81, 0.82, 0.85]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Export comparison table
    manager.export_comparison_csv(
        comparison_df=comparison_df,
        experiment_type='ablation',
        filename='loss_function_comparison.csv',
        config={'experiment': 'loss_ablation', 'num_models': 3},
        seed=42,
        include_timestamp=True
    )
    
    print("✓ Exported loss function comparison table")
    print(f"  Rows: {len(comparison_df)}")
    print(f"  Metrics compared: {len(comparison_df.columns) - 1}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['mean_kl', 'pearson_r', 'precision_at_100']
    titles = ['KL Divergence (lower is better)', 
              'Pearson Correlation (higher is better)',
              'Precision@100 (higher is better)']
    
    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        ax.bar(comparison_df['loss_function'], comparison_df[metric], 
               color=['steelblue', 'coral', 'lightgreen'], alpha=0.7)
        ax.set_xlabel('Loss Function', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    manager.save_visualization(
        fig=fig,
        experiment_type='ablation',
        filename='loss_function_comparison.png',
        include_timestamp=True
    )
    plt.close(fig)
    
    print("✓ Saved loss function comparison visualization")


def demo_explainability_integration():
    """Demo: Using OutputManager for explainability visualizations."""
    print("\n" + "="*70)
    print("5. EXPLAINABILITY INTEGRATION")
    print("="*70)
    
    manager = OutputManager()
    
    # Simulate failure case analysis
    num_failures = 5
    
    fig, axes = plt.subplots(num_failures, 3, figsize=(12, 4 * num_failures))
    
    for i in range(num_failures):
        # Simulate image
        img = np.random.rand(32, 32, 3)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Failure Case {i+1}\nKL Div: {0.5 + i*0.1:.3f}')
        axes[i, 0].axis('off')
        
        # Simulate true distribution
        true_dist = np.random.dirichlet(np.ones(10) * 2)
        axes[i, 1].bar(range(10), true_dist, color='steelblue', alpha=0.7)
        axes[i, 1].set_title('True Distribution')
        axes[i, 1].set_ylim(0, 1)
        
        # Simulate predicted distribution
        pred_dist = np.random.dirichlet(np.ones(10) * 2)
        axes[i, 2].bar(range(10), pred_dist, color='coral', alpha=0.7)
        axes[i, 2].set_title('Predicted Distribution')
        axes[i, 2].set_ylim(0, 1)
    
    plt.suptitle('Top 5 Failure Cases (Highest KL Divergence)', fontsize=16)
    plt.tight_layout()
    
    manager.save_visualization(
        fig=fig,
        experiment_type='explainability',
        filename='failure_cases.png',
        include_timestamp=True
    )
    plt.close(fig)
    
    print("✓ Saved failure case analysis")
    
    # Export failure case metadata
    failure_metadata = {
        'num_cases': num_failures,
        'kl_range': [0.5, 0.9],
        'analysis_date': manager.timestamp
    }
    
    manager.export_metrics_json(
        metrics=failure_metadata,
        experiment_type='explainability',
        filename='failure_cases_metadata.json',
        include_timestamp=True
    )
    
    print("✓ Exported failure case metadata")


def demo_complete_experiment():
    """Demo: Complete experiment workflow with OutputManager."""
    print("\n" + "="*70)
    print("6. COMPLETE EXPERIMENT WORKFLOW")
    print("="*70)
    
    manager = OutputManager()
    manager.create_directory_structure()
    
    # Collect all results
    all_results = {
        'data_statistics': {
            'num_train': 6000,
            'num_val': 2000,
            'num_test': 2000,
            'mean_entropy': 1.23,
            'std_entropy': 0.45
        },
        'training': {
            'num_epochs': 38,
            'final_train_loss': 0.25,
            'final_val_loss': 0.38,
            'training_time_minutes': 45.3
        },
        'evaluation': {
            'mean_kl': 0.1098,
            'pearson_r': 0.8456,
            'precision_at_100': 0.70
        },
        'ablation': {
            'best_loss_function': 'custom',
            'best_initialization': 'cifar10_pretrained',
            'best_architecture': 'two_layer_mlp'
        }
    }
    
    # Create comprehensive summary
    manager.create_experiment_summary(
        experiment_name='complete_evaluation',
        results=all_results
    )
    
    print("✓ Created comprehensive experiment summary")
    print(f"  Timestamp: {manager.timestamp}")
    print(f"  Sections: {list(all_results.keys())}")
    print()
    print("Summary includes:")
    print("  - Data statistics")
    print("  - Training metrics")
    print("  - Evaluation results")
    print("  - Ablation study findings")


def main():
    """Run all integration demos."""
    print("\n" + "="*70)
    print("CIFAR-10 DISAGREEMENT PREDICTOR")
    print("Integrated Output Management Demo")
    print("="*70)
    
    demo_data_pipeline_integration()
    demo_training_integration()
    demo_evaluation_integration()
    demo_ablation_integration()
    demo_explainability_integration()
    demo_complete_experiment()
    
    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)
    print()
    print("The OutputManager provides:")
    print("  ✓ Organized directory structure")
    print("  ✓ Timestamped filenames for versioning")
    print("  ✓ Metadata tracking (config, seed, timestamp)")
    print("  ✓ Consistent naming conventions")
    print("  ✓ Easy integration with existing modules")
    print()
    print("Check outputs in:")
    print("  - outputs/data_visualizations/")
    print("  - outputs/training_logs/")
    print("  - outputs/evaluation_results/")
    print("  - outputs/ablation_studies/")
    print("  - outputs/explainability/")
    print("  - checkpoints/")
    print()


if __name__ == '__main__':
    main()

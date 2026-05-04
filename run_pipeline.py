#!/usr/bin/env python3
"""
End-to-End Pipeline Script

Runs the complete CIFAR-10 disagreement prediction pipeline from data preparation
through training to evaluation. Supports configuration via JSON files and generates
a comprehensive report.

Usage:
    python run_pipeline.py [options]

Examples:
    # Run complete pipeline with default settings
    python run_pipeline.py

    # Use custom configuration file
    python run_pipeline.py --config my_config.json

    # Run specific phases only
    python run_pipeline.py --phases data train evaluate

    # Quick test run with reduced epochs
    python run_pipeline.py --pretrain-epochs 10 --finetune-epochs 5
"""

import argparse
import logging
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logging_config import setup_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run complete CIFAR-10 disagreement prediction pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON configuration file (overrides command-line arguments)'
    )
    
    # Pipeline phases
    parser.add_argument(
        '--phases',
        type=str,
        nargs='+',
        default=['data', 'train', 'evaluate'],
        choices=['data', 'train', 'evaluate', 'ablations', 'all'],
        help='Which pipeline phases to run'
    )
    
    # Data directories
    parser.add_argument(
        '--cifar10-dir',
        type=str,
        default='./data',
        help='Directory for CIFAR-10 dataset'
    )
    parser.add_argument(
        '--cifar10h-dir',
        type=str,
        default='./cifar-10h-1.0.0/data',
        help='Directory for CIFAR-10H dataset'
    )
    
    # Output directories
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Base output directory for all results'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory for model checkpoints'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=100,
        help='Number of pretraining epochs'
    )
    parser.add_argument(
        '--finetune-epochs',
        type=int,
        default=50,
        help='Number of fine-tuning epochs'
    )
    parser.add_argument(
        '--loss-functions',
        type=str,
        nargs='+',
        default=['kl', 'js', 'custom'],
        choices=['kl', 'js', 'custom'],
        help='Loss functions to train with'
    )
    
    # Evaluation options
    parser.add_argument(
        '--generate-visualizations',
        action='store_true',
        default=True,
        help='Generate visualizations during evaluation'
    )
    parser.add_argument(
        '--evaluate-robustness',
        action='store_true',
        help='Evaluate robustness to corruptions'
    )
    
    # Device and seed
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Report generation
    parser.add_argument(
        '--generate-report',
        action='store_true',
        default=True,
        help='Generate comprehensive markdown report'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(args, output_path):
    """Save pipeline configuration to JSON file."""
    config = vars(args)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def run_data_preparation(args, logger):
    """Run data preparation phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: DATA PREPARATION")
    logger.info("="*70)
    
    import subprocess
    
    cmd = [
        'python', 'prepare_data.py',
        '--cifar10-dir', args.cifar10_dir,
        '--cifar10h-dir', args.cifar10h_dir,
        '--output-dir', os.path.join(args.output_dir, 'data_visualizations'),
        '--seed', str(args.seed),
        '--log-level', args.log_level
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("✓ Data preparation complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data preparation failed: {e}")
        return False


def run_training(args, logger):
    """Run training phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: TRAINING")
    logger.info("="*70)
    
    import subprocess
    
    cmd = [
        'python', 'train.py',
        '--cifar10-dir', args.cifar10_dir,
        '--cifar10h-dir', args.cifar10h_dir,
        '--checkpoint-dir', args.checkpoint_dir,
        '--log-dir', os.path.join(args.output_dir, 'training_logs'),
        '--pretrain-epochs', str(args.pretrain_epochs),
        '--finetune-epochs', str(args.finetune_epochs),
        '--loss-functions'] + args.loss_functions + [
        '--device', args.device,
        '--seed', str(args.seed),
        '--log-level', args.log_level
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("✓ Training complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return False


def run_evaluation(args, logger):
    """Run evaluation phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: EVALUATION")
    logger.info("="*70)
    
    import subprocess
    
    cmd = [
        'python', 'evaluate.py',
        '--checkpoint-dir', args.checkpoint_dir,
        '--cifar10-dir', args.cifar10_dir,
        '--cifar10h-dir', args.cifar10h_dir,
        '--output-dir', os.path.join(args.output_dir, 'evaluation_results'),
        '--device', args.device,
        '--seed', str(args.seed),
        '--log-level', args.log_level
    ]
    
    if args.generate_visualizations:
        cmd.append('--generate-visualizations')
    
    if args.evaluate_robustness:
        cmd.append('--evaluate-robustness')
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("✓ Evaluation complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e}")
        return False


def run_ablations(args, logger):
    """Run ablation studies phase."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: ABLATION STUDIES")
    logger.info("="*70)
    
    import subprocess
    
    cmd = [
        'python', 'run_ablations.py',
        '--cifar10-dir', args.cifar10_dir,
        '--cifar10h-dir', args.cifar10h_dir,
        '--output-dir', os.path.join(args.output_dir, 'ablation_studies'),
        '--checkpoint-dir', os.path.join(args.checkpoint_dir, 'ablations'),
        '--pretrain-epochs', str(args.pretrain_epochs),
        '--finetune-epochs', str(args.finetune_epochs),
        '--device', args.device,
        '--seed', str(args.seed),
        '--log-level', args.log_level
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("✓ Ablation studies complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ablation studies failed: {e}")
        return False


def generate_report(args, logger, phase_results, start_time, end_time):
    """Generate comprehensive markdown report."""
    logger.info("\n" + "="*70)
    logger.info("GENERATING COMPREHENSIVE REPORT")
    logger.info("="*70)
    
    report_path = os.path.join(args.output_dir, 'pipeline_report.md')
    
    try:
        with open(report_path, 'w') as f:
            # Header
            f.write("# CIFAR-10 Human Disagreement Predictor - Pipeline Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Runtime:** {end_time - start_time:.2f} seconds\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- **Device:** {args.device}\n")
            f.write(f"- **Random Seed:** {args.seed}\n")
            f.write(f"- **Pretraining Epochs:** {args.pretrain_epochs}\n")
            f.write(f"- **Fine-tuning Epochs:** {args.finetune_epochs}\n")
            f.write(f"- **Loss Functions:** {', '.join(args.loss_functions)}\n")
            f.write(f"- **Output Directory:** {args.output_dir}\n\n")
            
            # Phase Results
            f.write("## Pipeline Phases\n\n")
            for phase, success in phase_results.items():
                status = "✓ Success" if success else "✗ Failed"
                f.write(f"- **{phase.title()}:** {status}\n")
            f.write("\n")
            
            # Data Preparation Results
            if phase_results.get('data', False):
                f.write("## Data Preparation\n\n")
                f.write("- CIFAR-10 test set: 10,000 images\n")
                f.write("- CIFAR-10H soft labels: 10,000 distributions\n")
                f.write("- Dataset split: 6,000 train / 2,000 val / 2,000 test\n")
                f.write("- Visualizations generated in `data_visualizations/`\n\n")
            
            # Training Results
            if phase_results.get('train', False):
                f.write("## Training\n\n")
                f.write(f"Trained {len(args.loss_functions)} model(s) with different loss functions:\n\n")
                for loss_name in args.loss_functions:
                    f.write(f"- **{loss_name.upper()} Loss Model**\n")
                    checkpoint_path = os.path.join(args.checkpoint_dir, f'finetuned_{loss_name}_best.pth')
                    if os.path.exists(checkpoint_path):
                        f.write(f"  - Checkpoint: `{checkpoint_path}`\n")
                    
                    # Try to load training history
                    history_path = os.path.join(args.output_dir, 'training_logs', f'finetune_{loss_name}_history.json')
                    if os.path.exists(history_path):
                        with open(history_path, 'r') as hf:
                            history = json.load(hf)
                            best_val_kl = min(history['val_kl'])
                            best_val_js = min(history['val_js'])
                            f.write(f"  - Best Validation KL: {best_val_kl:.4f}\n")
                            f.write(f"  - Best Validation JS: {best_val_js:.4f}\n")
                f.write("\n")
            
            # Evaluation Results
            if phase_results.get('evaluate', False):
                f.write("## Evaluation Results\n\n")
                
                # Try to load evaluation metrics
                eval_dir = os.path.join(args.output_dir, 'evaluation_results')
                comparison_path = os.path.join(eval_dir, 'model_comparison.csv')
                
                if os.path.exists(comparison_path):
                    import pandas as pd
                    comparison_df = pd.read_csv(comparison_path)
                    
                    f.write("### Model Comparison\n\n")
                    f.write(comparison_df.to_markdown(index=False))
                    f.write("\n\n")
                    
                    # Identify best models
                    best_kl_idx = comparison_df['mean_kl'].idxmin()
                    best_pearson_idx = comparison_df['pearson_r'].idxmax()
                    best_precision_idx = comparison_df['precision@100'].idxmax()
                    
                    f.write("### Best Performing Models\n\n")
                    f.write(f"- **Lowest KL Divergence:** {comparison_df.loc[best_kl_idx, 'model']} ")
                    f.write(f"({comparison_df.loc[best_kl_idx, 'mean_kl']:.4f})\n")
                    f.write(f"- **Highest Pearson Correlation:** {comparison_df.loc[best_pearson_idx, 'model']} ")
                    f.write(f"({comparison_df.loc[best_pearson_idx, 'pearson_r']:.4f})\n")
                    f.write(f"- **Highest Precision@100:** {comparison_df.loc[best_precision_idx, 'model']} ")
                    f.write(f"({comparison_df.loc[best_precision_idx, 'precision@100']:.4f})\n\n")
                
                f.write("### Generated Outputs\n\n")
                f.write("- Evaluation metrics: `evaluation_results/*/evaluation_metrics.json`\n")
                if args.generate_visualizations:
                    f.write("- Grad-CAM visualizations: `evaluation_results/*/gradcam_comparison.png`\n")
                    f.write("- Failure case analysis: `evaluation_results/*/failure_cases.png`\n")
                    f.write("- Entropy correlation plots: `evaluation_results/*/entropy_correlation.png`\n")
                if args.evaluate_robustness:
                    f.write("- Corruption robustness: `evaluation_results/*/corruption_robustness.png`\n")
                f.write("\n")
            
            # Ablation Studies Results
            if phase_results.get('ablations', False):
                f.write("## Ablation Studies\n\n")
                f.write("Comprehensive ablation studies completed. Results saved in `ablation_studies/`:\n\n")
                f.write("- Loss function comparison\n")
                f.write("- Initialization strategy comparison\n")
                f.write("- Training strategy comparison\n")
                f.write("- Prediction head architecture comparison\n\n")
            
            # Directory Structure
            f.write("## Output Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{args.output_dir}/\n")
            f.write("├── data_visualizations/\n")
            f.write("│   ├── entropy_histogram.png\n")
            f.write("│   ├── per_class_entropy.png\n")
            f.write("│   └── example_grid.png\n")
            f.write("├── training_logs/\n")
            f.write("│   ├── pretrain_history.json\n")
            f.write("│   └── finetune_*_history.json\n")
            f.write("├── evaluation_results/\n")
            f.write("│   ├── model_comparison.csv\n")
            f.write("│   └── finetuned_*/\n")
            f.write("│       ├── evaluation_metrics.json\n")
            f.write("│       ├── per_class_performance.csv\n")
            f.write("│       └── visualizations...\n")
            if phase_results.get('ablations', False):
                f.write("├── ablation_studies/\n")
                f.write("│   ├── loss_functions/\n")
                f.write("│   ├── initialization/\n")
                f.write("│   ├── training_strategy/\n")
                f.write("│   └── architecture/\n")
            f.write("└── pipeline_report.md (this file)\n")
            f.write("```\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            successful_phases = sum(1 for success in phase_results.values() if success)
            total_phases = len(phase_results)
            f.write(f"Pipeline completed {successful_phases}/{total_phases} phases successfully.\n\n")
            
            if all(phase_results.values()):
                f.write("All phases completed successfully! The disagreement prediction models have been ")
                f.write("trained and evaluated. Review the evaluation results to identify the best-performing ")
                f.write("model for your use case.\n")
            else:
                f.write("Some phases failed. Please review the logs for error details.\n")
        
        logger.info(f"✓ Report generated: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return False


def main():
    """Main pipeline orchestration."""
    args = parse_args()
    
    # Load configuration from file if provided
    if args.config is not None:
        logger = logging.getLogger(__name__)
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("CIFAR-10 HUMAN DISAGREEMENT PREDICTOR")
    logger.info("END-TO-END PIPELINE")
    logger.info("="*70)
    logger.info(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Random Seed: {args.seed}")
    
    # Handle 'all' option
    if 'all' in args.phases:
        args.phases = ['data', 'train', 'evaluate', 'ablations']
    
    logger.info(f"Pipeline phases: {', '.join(args.phases)}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save pipeline configuration
    config_path = os.path.join(args.output_dir, 'pipeline_config.json')
    save_config(args, config_path)
    logger.info(f"\n✓ Pipeline configuration saved to {config_path}")
    
    # Track phase results
    phase_results = {}
    start_time = time.time()
    
    # Run pipeline phases
    if 'data' in args.phases:
        phase_results['data'] = run_data_preparation(args, logger)
        if not phase_results['data']:
            logger.error("Data preparation failed. Stopping pipeline.")
            sys.exit(1)
    
    if 'train' in args.phases:
        phase_results['train'] = run_training(args, logger)
        if not phase_results['train']:
            logger.error("Training failed. Stopping pipeline.")
            sys.exit(1)
    
    if 'evaluate' in args.phases:
        phase_results['evaluate'] = run_evaluation(args, logger)
        # Continue even if evaluation fails
    
    if 'ablations' in args.phases:
        phase_results['ablations'] = run_ablations(args, logger)
        # Continue even if ablations fail
    
    end_time = time.time()
    
    # Generate comprehensive report
    if args.generate_report:
        generate_report(args, logger, phase_results, start_time, end_time)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Runtime: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.1f} minutes)")
    logger.info(f"\nPhase Results:")
    for phase, success in phase_results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {phase.title()}: {status}")
    logger.info(f"\nAll outputs saved to: {args.output_dir}")
    logger.info(f"Model checkpoints saved to: {args.checkpoint_dir}")
    if args.generate_report:
        logger.info(f"Comprehensive report: {os.path.join(args.output_dir, 'pipeline_report.md')}")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()

"""
Output Management Module

Implements comprehensive output management for the CIFAR-10 disagreement predictor project.
Handles directory structure creation, visualization saving with timestamps, and metrics export.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Manages all output operations including directory structure, file naming, and metadata.
    
    Provides utilities for:
    - Creating organized directory structure
    - Saving visualizations with descriptive, timestamped filenames
    - Exporting metrics to JSON and CSV with metadata
    """
    
    def __init__(self, base_dir: str = "outputs"):
        """
        Initialize output manager with base directory.
        
        Args:
            base_dir: Base directory for all outputs (default: "outputs")
        """
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Initialized OutputManager with base_dir={base_dir}, timestamp={self.timestamp}")
    
    def create_directory_structure(self):
        """
        Create comprehensive output directory structure.
        
        Creates:
        - outputs/data_visualizations/
        - outputs/training_logs/
        - outputs/evaluation_results/
        - outputs/ablation_studies/
        - outputs/explainability/
        - checkpoints/
        
        Requirements: 29.1, 29.7
        """
        logger.info("Creating output directory structure")
        
        directories = [
            os.path.join(self.base_dir, "data_visualizations"),
            os.path.join(self.base_dir, "training_logs"),
            os.path.join(self.base_dir, "evaluation_results"),
            os.path.join(self.base_dir, "ablation_studies"),
            os.path.join(self.base_dir, "explainability"),
            "checkpoints"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        
        logger.info("Output directory structure created successfully")
    
    def get_visualization_path(
        self,
        experiment_type: str,
        filename: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Generate path for saving visualization with descriptive filename.
        
        Args:
            experiment_type: Type of experiment (e.g., 'data_exploration', 'training', 
                           'evaluation', 'ablation', 'explainability')
            filename: Base filename (e.g., 'entropy_histogram.png')
            include_timestamp: Whether to include timestamp in filename
        
        Returns:
            Full path for saving visualization
        
        Requirements: 29.2, 29.3, 29.7
        """
        # Map experiment types to subdirectories
        type_to_dir = {
            'data_exploration': 'data_visualizations',
            'data': 'data_visualizations',
            'training': 'training_logs',
            'evaluation': 'evaluation_results',
            'ablation': 'ablation_studies',
            'explainability': 'explainability',
            'robustness': 'evaluation_results'
        }
        
        subdir = type_to_dir.get(experiment_type, 'data_visualizations')
        
        # Add timestamp to filename if requested
        if include_timestamp:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{self.timestamp}{ext}"
        
        path = os.path.join(self.base_dir, subdir, filename)
        logger.debug(f"Generated visualization path: {path}")
        
        return path
    
    def save_visualization(
        self,
        fig,
        experiment_type: str,
        filename: str,
        include_timestamp: bool = True,
        dpi: int = 300
    ):
        """
        Save matplotlib figure with descriptive filename and timestamp.
        
        Args:
            fig: Matplotlib figure object
            experiment_type: Type of experiment
            filename: Base filename
            include_timestamp: Whether to include timestamp
            dpi: Resolution for saving (default: 300)
        
        Requirements: 29.2, 29.3, 29.4, 29.5, 29.6, 29.7
        """
        path = self.get_visualization_path(experiment_type, filename, include_timestamp)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save figure
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved visualization to {path}")
    
    def export_metrics_json(
        self,
        metrics: Dict[str, Any],
        experiment_type: str,
        filename: str,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        include_timestamp: bool = True
    ):
        """
        Export metrics to JSON file with metadata.
        
        Args:
            metrics: Dictionary of metrics to export
            experiment_type: Type of experiment
            filename: Base filename (e.g., 'evaluation_metrics.json')
            config: Optional configuration dictionary
            seed: Optional random seed used
            include_timestamp: Whether to include timestamp in filename
        
        Requirements: 31.1, 31.3, 31.4
        """
        # Add timestamp to filename if requested
        if include_timestamp:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{self.timestamp}{ext}"
        
        # Determine subdirectory based on experiment type
        type_to_dir = {
            'evaluation': 'evaluation_results',
            'ablation': 'ablation_studies',
            'training': 'training_logs',
            'data': 'data_visualizations'
        }
        
        subdir = type_to_dir.get(experiment_type, 'evaluation_results')
        path = os.path.join(self.base_dir, subdir, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create output dictionary with metadata
        output = {
            'timestamp': self.timestamp,
            'experiment_type': experiment_type,
            'metrics': metrics
        }
        
        # Add optional metadata
        if config is not None:
            output['config'] = config
        
        if seed is not None:
            output['random_seed'] = seed
        
        # Save to JSON
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Exported metrics to JSON: {path}")
    
    def export_comparison_csv(
        self,
        comparison_df: pd.DataFrame,
        experiment_type: str,
        filename: str,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        include_timestamp: bool = True
    ):
        """
        Export comparison table to CSV file with metadata.
        
        Args:
            comparison_df: Pandas DataFrame with comparison results
            experiment_type: Type of experiment
            filename: Base filename (e.g., 'loss_function_comparison.csv')
            config: Optional configuration dictionary
            seed: Optional random seed used
            include_timestamp: Whether to include timestamp in filename
        
        Requirements: 31.2, 31.3, 31.4
        """
        # Add timestamp to filename if requested
        if include_timestamp:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{self.timestamp}{ext}"
        
        # Determine subdirectory based on experiment type
        type_to_dir = {
            'ablation': 'ablation_studies',
            'evaluation': 'evaluation_results',
            'training': 'training_logs'
        }
        
        subdir = type_to_dir.get(experiment_type, 'ablation_studies')
        path = os.path.join(self.base_dir, subdir, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save DataFrame to CSV
        comparison_df.to_csv(path, index=False)
        logger.info(f"Exported comparison table to CSV: {path}")
        
        # Save metadata to companion JSON file
        metadata_path = path.replace('.csv', '_metadata.json')
        metadata = {
            'timestamp': self.timestamp,
            'experiment_type': experiment_type,
            'csv_file': os.path.basename(path),
            'num_rows': len(comparison_df),
            'columns': list(comparison_df.columns)
        }
        
        if config is not None:
            metadata['config'] = config
        
        if seed is not None:
            metadata['random_seed'] = seed
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exported metadata to JSON: {metadata_path}")
    
    def get_checkpoint_path(
        self,
        model_name: str,
        loss_function: Optional[str] = None,
        epoch: Optional[int] = None,
        is_best: bool = False
    ) -> str:
        """
        Generate path for saving model checkpoint.
        
        Args:
            model_name: Name of the model (e.g., 'disagreement_predictor')
            loss_function: Loss function used (e.g., 'kl', 'js', 'custom')
            epoch: Epoch number (optional)
            is_best: Whether this is the best checkpoint
        
        Returns:
            Full path for saving checkpoint
        """
        # Build filename
        parts = [model_name]
        
        if loss_function is not None:
            parts.append(loss_function)
        
        if is_best:
            parts.append('best')
        elif epoch is not None:
            parts.append(f'epoch{epoch}')
        
        filename = '_'.join(parts) + '.pth'
        path = os.path.join('checkpoints', filename)
        
        logger.debug(f"Generated checkpoint path: {path}")
        return path
    
    def export_training_history(
        self,
        history: Dict[str, list],
        loss_function: str,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        """
        Export training history to JSON file.
        
        Args:
            history: Dictionary with training history (e.g., {'train_loss': [...], 'val_loss': [...]})
            loss_function: Loss function used
            config: Optional training configuration
            seed: Optional random seed used
        """
        filename = f"training_history_{loss_function}.json"
        self.export_metrics_json(
            metrics=history,
            experiment_type='training',
            filename=filename,
            config=config,
            seed=seed,
            include_timestamp=True
        )
    
    def create_experiment_summary(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive experiment summary with all results and metadata.
        
        Args:
            experiment_name: Name of the experiment
            results: Dictionary with all experiment results
            save_path: Optional custom save path (if None, uses default location)
        """
        if save_path is None:
            save_path = os.path.join(
                self.base_dir,
                'evaluation_results',
                f'{experiment_name}_summary_{self.timestamp}.json'
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create summary
        summary = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'results': results
        }
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created experiment summary: {save_path}")


# Convenience functions for backward compatibility

def create_output_directories():
    """
    Create all required output directories.
    
    Convenience function that creates the standard directory structure.
    """
    manager = OutputManager()
    manager.create_directory_structure()


def get_timestamped_filename(base_filename: str) -> str:
    """
    Add timestamp to filename.
    
    Args:
        base_filename: Base filename (e.g., 'metrics.json')
    
    Returns:
        Filename with timestamp (e.g., 'metrics_20240115_143022.json')
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(base_filename)
    return f"{name}_{timestamp}{ext}"


def save_metrics_with_metadata(
    metrics: Dict[str, Any],
    filepath: str,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
):
    """
    Save metrics to JSON with metadata.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save JSON file
        config: Optional configuration dictionary
        seed: Optional random seed
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create output with metadata
    output = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'metrics': metrics
    }
    
    if config is not None:
        output['config'] = config
    
    if seed is not None:
        output['random_seed'] = seed
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved metrics with metadata to {filepath}")

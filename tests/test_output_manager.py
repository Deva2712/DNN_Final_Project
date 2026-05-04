"""
Unit tests for output management module.

Tests directory creation, visualization saving, and metrics export functionality.
"""

import os
import json
import tempfile
import shutil
from datetime import datetime
import pytest
import pandas as pd
import matplotlib.pyplot as plt

from src.output_manager import (
    OutputManager,
    create_output_directories,
    get_timestamped_filename,
    save_metrics_with_metadata
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def output_manager(temp_output_dir):
    """Create OutputManager instance with temporary directory."""
    return OutputManager(base_dir=temp_output_dir)


class TestOutputManager:
    """Test suite for OutputManager class."""
    
    def test_initialization(self, temp_output_dir):
        """Test OutputManager initialization."""
        manager = OutputManager(base_dir=temp_output_dir)
        
        assert manager.base_dir == temp_output_dir
        assert isinstance(manager.timestamp, str)
        assert len(manager.timestamp) == 15  # Format: YYYYMMDD_HHMMSS
    
    def test_create_directory_structure(self, output_manager):
        """Test creation of output directory structure."""
        output_manager.create_directory_structure()
        
        # Check all required directories exist
        expected_dirs = [
            os.path.join(output_manager.base_dir, "data_visualizations"),
            os.path.join(output_manager.base_dir, "training_logs"),
            os.path.join(output_manager.base_dir, "evaluation_results"),
            os.path.join(output_manager.base_dir, "ablation_studies"),
            os.path.join(output_manager.base_dir, "explainability"),
            "checkpoints"
        ]
        
        for directory in expected_dirs:
            assert os.path.exists(directory), f"Directory {directory} was not created"
            assert os.path.isdir(directory), f"{directory} is not a directory"
    
    def test_get_visualization_path_with_timestamp(self, output_manager):
        """Test visualization path generation with timestamp."""
        path = output_manager.get_visualization_path(
            experiment_type='evaluation',
            filename='test_plot.png',
            include_timestamp=True
        )
        
        # Check path structure
        assert output_manager.base_dir in path
        assert 'evaluation_results' in path
        assert 'test_plot' in path
        assert output_manager.timestamp in path
        assert path.endswith('.png')
    
    def test_get_visualization_path_without_timestamp(self, output_manager):
        """Test visualization path generation without timestamp."""
        path = output_manager.get_visualization_path(
            experiment_type='data',
            filename='histogram.png',
            include_timestamp=False
        )
        
        # Check path structure
        assert output_manager.base_dir in path
        assert 'data_visualizations' in path
        assert path.endswith('histogram.png')
        assert output_manager.timestamp not in path
    
    def test_get_visualization_path_experiment_types(self, output_manager):
        """Test visualization path generation for different experiment types."""
        type_to_expected_dir = {
            'data_exploration': 'data_visualizations',
            'data': 'data_visualizations',
            'training': 'training_logs',
            'evaluation': 'evaluation_results',
            'ablation': 'ablation_studies',
            'explainability': 'explainability',
            'robustness': 'evaluation_results'
        }
        
        for exp_type, expected_dir in type_to_expected_dir.items():
            path = output_manager.get_visualization_path(
                experiment_type=exp_type,
                filename='test.png',
                include_timestamp=False
            )
            assert expected_dir in path, f"Expected {expected_dir} in path for {exp_type}"
    
    def test_save_visualization(self, output_manager):
        """Test saving matplotlib figure."""
        output_manager.create_directory_structure()
        
        # Create simple matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title('Test Plot')
        
        # Save visualization
        output_manager.save_visualization(
            fig=fig,
            experiment_type='evaluation',
            filename='test_plot.png',
            include_timestamp=True
        )
        
        # Check file was created
        path = output_manager.get_visualization_path(
            experiment_type='evaluation',
            filename='test_plot.png',
            include_timestamp=True
        )
        assert os.path.exists(path), f"Visualization file {path} was not created"
        
        # Cleanup
        plt.close(fig)
    
    def test_export_metrics_json(self, output_manager):
        """Test exporting metrics to JSON."""
        output_manager.create_directory_structure()
        
        metrics = {
            'mean_kl': 0.123,
            'mean_js': 0.045,
            'pearson_r': 0.789
        }
        
        config = {
            'learning_rate': 1e-4,
            'batch_size': 64
        }
        
        output_manager.export_metrics_json(
            metrics=metrics,
            experiment_type='evaluation',
            filename='test_metrics.json',
            config=config,
            seed=42,
            include_timestamp=True
        )
        
        # Check file was created
        expected_path = os.path.join(
            output_manager.base_dir,
            'evaluation_results',
            f'test_metrics_{output_manager.timestamp}.json'
        )
        assert os.path.exists(expected_path), f"Metrics file {expected_path} was not created"
        
        # Check file contents
        with open(expected_path, 'r') as f:
            data = json.load(f)
        
        assert 'timestamp' in data
        assert 'experiment_type' in data
        assert 'metrics' in data
        assert 'config' in data
        assert 'random_seed' in data
        
        assert data['metrics'] == metrics
        assert data['config'] == config
        assert data['random_seed'] == 42
        assert data['experiment_type'] == 'evaluation'
    
    def test_export_metrics_json_without_optional_fields(self, output_manager):
        """Test exporting metrics without config and seed."""
        output_manager.create_directory_structure()
        
        metrics = {'accuracy': 0.95}
        
        output_manager.export_metrics_json(
            metrics=metrics,
            experiment_type='evaluation',
            filename='simple_metrics.json',
            include_timestamp=False
        )
        
        # Check file was created
        expected_path = os.path.join(
            output_manager.base_dir,
            'evaluation_results',
            'simple_metrics.json'
        )
        assert os.path.exists(expected_path)
        
        # Check file contents
        with open(expected_path, 'r') as f:
            data = json.load(f)
        
        assert 'metrics' in data
        assert 'config' not in data
        assert 'random_seed' not in data
    
    def test_export_comparison_csv(self, output_manager):
        """Test exporting comparison table to CSV."""
        output_manager.create_directory_structure()
        
        # Create sample DataFrame
        comparison_df = pd.DataFrame({
            'loss_function': ['kl', 'js', 'custom'],
            'mean_kl': [0.123, 0.145, 0.110],
            'pearson_r': [0.789, 0.801, 0.823]
        })
        
        config = {'experiment': 'loss_comparison'}
        
        output_manager.export_comparison_csv(
            comparison_df=comparison_df,
            experiment_type='ablation',
            filename='loss_comparison.csv',
            config=config,
            seed=42,
            include_timestamp=True
        )
        
        # Check CSV file was created
        csv_path = os.path.join(
            output_manager.base_dir,
            'ablation_studies',
            f'loss_comparison_{output_manager.timestamp}.csv'
        )
        assert os.path.exists(csv_path), f"CSV file {csv_path} was not created"
        
        # Check CSV contents
        loaded_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded_df, comparison_df)
        
        # Check metadata file was created
        metadata_path = csv_path.replace('.csv', '_metadata.json')
        assert os.path.exists(metadata_path), f"Metadata file {metadata_path} was not created"
        
        # Check metadata contents
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert 'timestamp' in metadata
        assert 'experiment_type' in metadata
        assert 'csv_file' in metadata
        assert 'num_rows' in metadata
        assert 'columns' in metadata
        assert 'config' in metadata
        assert 'random_seed' in metadata
        
        assert metadata['num_rows'] == 3
        assert metadata['columns'] == ['loss_function', 'mean_kl', 'pearson_r']
        assert metadata['config'] == config
        assert metadata['random_seed'] == 42
    
    def test_get_checkpoint_path(self, output_manager):
        """Test checkpoint path generation."""
        # Test basic checkpoint
        path = output_manager.get_checkpoint_path(
            model_name='disagreement_predictor',
            loss_function='kl'
        )
        assert path == 'checkpoints/disagreement_predictor_kl.pth'
        
        # Test best checkpoint
        path = output_manager.get_checkpoint_path(
            model_name='disagreement_predictor',
            loss_function='js',
            is_best=True
        )
        assert path == 'checkpoints/disagreement_predictor_js_best.pth'
        
        # Test epoch checkpoint
        path = output_manager.get_checkpoint_path(
            model_name='disagreement_predictor',
            loss_function='custom',
            epoch=25
        )
        assert path == 'checkpoints/disagreement_predictor_custom_epoch25.pth'
        
        # Test without loss function
        path = output_manager.get_checkpoint_path(
            model_name='pretrained_model',
            is_best=True
        )
        assert path == 'checkpoints/pretrained_model_best.pth'
    
    def test_export_training_history(self, output_manager):
        """Test exporting training history."""
        output_manager.create_directory_structure()
        
        history = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'val_kl': [0.2, 0.18, 0.15]
        }
        
        config = {'learning_rate': 1e-4}
        
        output_manager.export_training_history(
            history=history,
            loss_function='kl',
            config=config,
            seed=42
        )
        
        # Check file was created
        expected_path = os.path.join(
            output_manager.base_dir,
            'training_logs',
            f'training_history_kl_{output_manager.timestamp}.json'
        )
        assert os.path.exists(expected_path)
        
        # Check contents
        with open(expected_path, 'r') as f:
            data = json.load(f)
        
        assert data['metrics'] == history
        assert data['config'] == config
        assert data['random_seed'] == 42
    
    def test_create_experiment_summary(self, output_manager):
        """Test creating experiment summary."""
        output_manager.create_directory_structure()
        
        results = {
            'mean_kl': 0.123,
            'pearson_r': 0.789,
            'precision_at_100': 0.65
        }
        
        output_manager.create_experiment_summary(
            experiment_name='final_evaluation',
            results=results
        )
        
        # Check file was created
        expected_path = os.path.join(
            output_manager.base_dir,
            'evaluation_results',
            f'final_evaluation_summary_{output_manager.timestamp}.json'
        )
        assert os.path.exists(expected_path)
        
        # Check contents
        with open(expected_path, 'r') as f:
            data = json.load(f)
        
        assert data['experiment_name'] == 'final_evaluation'
        assert 'timestamp' in data
        assert data['results'] == results


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_create_output_directories(self, temp_output_dir, monkeypatch):
        """Test create_output_directories convenience function."""
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_output_dir)
        
        try:
            create_output_directories()
            
            # Check directories were created
            assert os.path.exists('outputs/data_visualizations')
            assert os.path.exists('outputs/training_logs')
            assert os.path.exists('outputs/evaluation_results')
            assert os.path.exists('outputs/ablation_studies')
            assert os.path.exists('outputs/explainability')
            assert os.path.exists('checkpoints')
        finally:
            os.chdir(original_cwd)
    
    def test_get_timestamped_filename(self):
        """Test get_timestamped_filename function."""
        filename = get_timestamped_filename('metrics.json')
        
        # Check format
        assert 'metrics_' in filename
        assert filename.endswith('.json')
        assert len(filename) == len('metrics_20240115_143022.json')
    
    def test_save_metrics_with_metadata(self, temp_output_dir):
        """Test save_metrics_with_metadata convenience function."""
        metrics = {'accuracy': 0.95}
        config = {'batch_size': 64}
        filepath = os.path.join(temp_output_dir, 'test', 'metrics.json')
        
        save_metrics_with_metadata(
            metrics=metrics,
            filepath=filepath,
            config=config,
            seed=42
        )
        
        # Check file was created
        assert os.path.exists(filepath)
        
        # Check contents
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert 'timestamp' in data
        assert data['metrics'] == metrics
        assert data['config'] == config
        assert data['random_seed'] == 42


class TestIntegration:
    """Integration tests for output manager."""
    
    def test_complete_workflow(self, output_manager):
        """Test complete workflow: create dirs, save viz, export metrics."""
        # Step 1: Create directory structure
        output_manager.create_directory_structure()
        
        # Step 2: Save visualization
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        output_manager.save_visualization(
            fig=fig,
            experiment_type='evaluation',
            filename='test_plot.png'
        )
        plt.close(fig)
        
        # Step 3: Export metrics
        metrics = {'mean_kl': 0.123}
        output_manager.export_metrics_json(
            metrics=metrics,
            experiment_type='evaluation',
            filename='metrics.json'
        )
        
        # Step 4: Export comparison table
        df = pd.DataFrame({'loss': ['kl', 'js'], 'value': [0.1, 0.2]})
        output_manager.export_comparison_csv(
            comparison_df=df,
            experiment_type='ablation',
            filename='comparison.csv'
        )
        
        # Step 5: Create summary
        output_manager.create_experiment_summary(
            experiment_name='test_experiment',
            results={'final_score': 0.95}
        )
        
        # Verify all files exist
        viz_path = output_manager.get_visualization_path(
            'evaluation', 'test_plot.png', True
        )
        assert os.path.exists(viz_path)
        
        metrics_path = os.path.join(
            output_manager.base_dir,
            'evaluation_results',
            f'metrics_{output_manager.timestamp}.json'
        )
        assert os.path.exists(metrics_path)
        
        csv_path = os.path.join(
            output_manager.base_dir,
            'ablation_studies',
            f'comparison_{output_manager.timestamp}.csv'
        )
        assert os.path.exists(csv_path)
        
        summary_path = os.path.join(
            output_manager.base_dir,
            'evaluation_results',
            f'test_experiment_summary_{output_manager.timestamp}.json'
        )
        assert os.path.exists(summary_path)

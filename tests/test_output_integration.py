"""
Integration tests for OutputManager with actual project modules.

Tests that OutputManager integrates correctly with:
- Data pipeline
- Evaluation module
- Visualization module
"""

import os
import json
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import torch

from src.output_manager import OutputManager
from src.evaluation import EvaluationMetrics


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


class TestEvaluationIntegration:
    """Test integration with evaluation module."""
    
    def test_export_evaluation_metrics_dataclass(self, output_manager):
        """Test exporting EvaluationMetrics dataclass."""
        output_manager.create_directory_structure()
        
        # Create sample metrics
        metrics = EvaluationMetrics(
            mean_kl=0.1234,
            std_kl=0.0456,
            mean_js=0.0789,
            std_js=0.0234,
            mean_cosine=0.9123,
            std_cosine=0.0345,
            pearson_r=0.8123,
            pearson_p=1.23e-10,
            spearman_r=0.7956,
            spearman_p=2.45e-9,
            precision_at_100=0.65,
            precision_at_200=0.72,
            precision_at_500=0.81
        )
        
        # Export using OutputManager
        output_manager.export_metrics_json(
            metrics=metrics.to_dict(),
            experiment_type='evaluation',
            filename='test_metrics.json',
            seed=42
        )
        
        # Verify file was created
        expected_path = os.path.join(
            output_manager.base_dir,
            'evaluation_results',
            f'test_metrics_{output_manager.timestamp}.json'
        )
        assert os.path.exists(expected_path)
        
        # Verify contents
        with open(expected_path, 'r') as f:
            data = json.load(f)
        
        assert 'metrics' in data
        assert data['metrics']['mean_kl'] == 0.1234
        assert data['metrics']['pearson_r'] == 0.8123
        assert data['random_seed'] == 42
    
    def test_export_comparison_dataframe(self, output_manager):
        """Test exporting comparison DataFrame from evaluation module."""
        output_manager.create_directory_structure()
        
        # Create sample comparison DataFrame (like from compare_loss_functions)
        comparison_df = pd.DataFrame({
            'loss_function': ['kl', 'js', 'custom'],
            'mean_kl': [0.1234, 0.1456, 0.1098],
            'mean_js': [0.0789, 0.0823, 0.0756],
            'pearson_r': [0.8123, 0.8234, 0.8456],
            'precision_at_100': [0.65, 0.67, 0.70]
        })
        
        # Export using OutputManager
        output_manager.export_comparison_csv(
            comparison_df=comparison_df,
            experiment_type='ablation',
            filename='loss_comparison.csv',
            seed=42
        )
        
        # Verify CSV was created
        csv_path = os.path.join(
            output_manager.base_dir,
            'ablation_studies',
            f'loss_comparison_{output_manager.timestamp}.csv'
        )
        assert os.path.exists(csv_path)
        
        # Verify metadata was created
        metadata_path = csv_path.replace('.csv', '_metadata.json')
        assert os.path.exists(metadata_path)
        
        # Verify CSV contents
        loaded_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded_df, comparison_df)


class TestDataPipelineIntegration:
    """Test integration with data pipeline."""
    
    def test_entropy_computation_export(self, output_manager):
        """Test exporting entropy computation results."""
        output_manager.create_directory_structure()
        
        # Simulate entropy computation results
        entropies = np.random.beta(2, 5, 10000) * 3.32
        
        stats = {
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'min_entropy': float(np.min(entropies)),
            'max_entropy': float(np.max(entropies)),
            'median_entropy': float(np.median(entropies))
        }
        
        # Export using OutputManager
        output_manager.export_metrics_json(
            metrics=stats,
            experiment_type='data',
            filename='entropy_statistics.json',
            seed=42
        )
        
        # Verify file was created
        expected_path = os.path.join(
            output_manager.base_dir,
            'data_visualizations',
            f'entropy_statistics_{output_manager.timestamp}.json'
        )
        assert os.path.exists(expected_path)
        
        # Verify contents
        with open(expected_path, 'r') as f:
            data = json.load(f)
        
        assert 'metrics' in data
        assert 'mean_entropy' in data['metrics']
        assert 0 <= data['metrics']['mean_entropy'] <= 3.32


class TestTrainingIntegration:
    """Test integration with training module."""
    
    def test_training_history_export(self, output_manager):
        """Test exporting training history."""
        output_manager.create_directory_structure()
        
        # Simulate training history
        history = {
            'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28],
            'val_loss': [0.6, 0.5, 0.45, 0.42, 0.40],
            'val_kl': [0.25, 0.20, 0.18, 0.16, 0.15],
            'val_js': [0.12, 0.10, 0.09, 0.08, 0.075]
        }
        
        config = {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'optimizer': 'AdamW',
            'loss_function': 'kl_divergence'
        }
        
        # Export using OutputManager
        output_manager.export_training_history(
            history=history,
            loss_function='kl',
            config=config,
            seed=42
        )
        
        # Verify file was created
        expected_path = os.path.join(
            output_manager.base_dir,
            'training_logs',
            f'training_history_kl_{output_manager.timestamp}.json'
        )
        assert os.path.exists(expected_path)
        
        # Verify contents
        with open(expected_path, 'r') as f:
            data = json.load(f)
        
        assert 'metrics' in data
        assert data['metrics'] == history
        assert data['config'] == config
        assert data['random_seed'] == 42
    
    def test_checkpoint_path_generation(self, output_manager):
        """Test checkpoint path generation for different scenarios."""
        # Test best checkpoint for each loss function
        loss_functions = ['kl', 'js', 'custom']
        
        for loss_fn in loss_functions:
            path = output_manager.get_checkpoint_path(
                model_name='disagreement_predictor',
                loss_function=loss_fn,
                is_best=True
            )
            
            assert 'checkpoints' in path
            assert loss_fn in path
            assert 'best' in path
            assert path.endswith('.pth')
        
        # Test epoch checkpoint
        path = output_manager.get_checkpoint_path(
            model_name='disagreement_predictor',
            loss_function='kl',
            epoch=25
        )
        
        assert 'epoch25' in path


class TestCompleteWorkflow:
    """Test complete workflow integration."""
    
    def test_end_to_end_experiment(self, output_manager):
        """Test complete experiment workflow."""
        output_manager.create_directory_structure()
        
        # Step 1: Data pipeline outputs
        data_stats = {
            'num_train': 6000,
            'num_val': 2000,
            'num_test': 2000,
            'mean_entropy': 1.23
        }
        
        output_manager.export_metrics_json(
            metrics=data_stats,
            experiment_type='data',
            filename='data_statistics.json',
            seed=42
        )
        
        # Step 2: Training outputs
        training_history = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4]
        }
        
        output_manager.export_training_history(
            history=training_history,
            loss_function='kl',
            seed=42
        )
        
        # Step 3: Evaluation outputs
        eval_metrics = {
            'mean_kl': 0.123,
            'pearson_r': 0.789
        }
        
        output_manager.export_metrics_json(
            metrics=eval_metrics,
            experiment_type='evaluation',
            filename='evaluation_metrics.json',
            seed=42
        )
        
        # Step 4: Ablation study outputs
        ablation_df = pd.DataFrame({
            'loss_function': ['kl', 'js'],
            'mean_kl': [0.123, 0.145]
        })
        
        output_manager.export_comparison_csv(
            comparison_df=ablation_df,
            experiment_type='ablation',
            filename='ablation_results.csv',
            seed=42
        )
        
        # Step 5: Create summary
        all_results = {
            'data': data_stats,
            'training': {'final_loss': training_history['val_loss'][-1]},
            'evaluation': eval_metrics
        }
        
        output_manager.create_experiment_summary(
            experiment_name='complete_experiment',
            results=all_results
        )
        
        # Verify all files were created
        data_path = os.path.join(
            output_manager.base_dir,
            'data_visualizations',
            f'data_statistics_{output_manager.timestamp}.json'
        )
        assert os.path.exists(data_path)
        
        training_path = os.path.join(
            output_manager.base_dir,
            'training_logs',
            f'training_history_kl_{output_manager.timestamp}.json'
        )
        assert os.path.exists(training_path)
        
        eval_path = os.path.join(
            output_manager.base_dir,
            'evaluation_results',
            f'evaluation_metrics_{output_manager.timestamp}.json'
        )
        assert os.path.exists(eval_path)
        
        ablation_path = os.path.join(
            output_manager.base_dir,
            'ablation_studies',
            f'ablation_results_{output_manager.timestamp}.csv'
        )
        assert os.path.exists(ablation_path)
        
        summary_path = os.path.join(
            output_manager.base_dir,
            'evaluation_results',
            f'complete_experiment_summary_{output_manager.timestamp}.json'
        )
        assert os.path.exists(summary_path)

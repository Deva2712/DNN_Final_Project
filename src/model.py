"""
Model Architecture Module

Implements the modified ResNet-18 backbone and MLP prediction head
for predicting human disagreement distributions.
"""

import logging
import json
from typing import Optional
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torchvision.models import resnet18

from .data_pipeline import ConfigParseError

logger = logging.getLogger(__name__)


class DisagreementPredictionHead(nn.Module):
    """
    MLP head that predicts probability distributions over 10 classes.
    
    Architecture: 512 → 256 → 10 with ReLU activation and Softmax output.
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 10):
        """
        Initialize prediction head.
        
        Args:
            input_dim: Dimension of input features (default: 512)
            hidden_dim: Dimension of hidden layer (default: 256)
            num_classes: Number of output classes (default: 10)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        logger.info(f"Initialized DisagreementPredictionHead: {input_dim}→{hidden_dim}→{num_classes}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through prediction head.
        
        Args:
            features: Tensor of shape (batch_size, input_dim)
        
        Returns:
            probs: Tensor of shape (batch_size, num_classes) with probability distributions
        """
        x = self.fc1(features)
        x = self.relu(x)
        logits = self.fc2(x)
        probs = self.softmax(logits)
        return probs


def create_modified_resnet18() -> nn.Module:
    """
    Create ResNet-18 modified for 32×32 CIFAR-10 images.
    
    Modifications:
    - Replace initial 7×7 conv (stride 2) with 3×3 conv (stride 1)
    - Remove max pooling layer
    - Remove final fully connected layer
    
    Returns:
        Backbone model outputting 512-dimensional features
    """
    logger.info("Creating modified ResNet-18 backbone for 32×32 images")
    
    model = resnet18(pretrained=False)
    
    # Replace initial 7×7 conv with 3×3 conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove max pooling layer
    model.maxpool = nn.Identity()
    
    # Remove final fully connected layer (we'll add custom head)
    model.fc = nn.Identity()
    
    logger.info("Modified ResNet-18 backbone created successfully")
    return model


class DisagreementPredictor(nn.Module):
    """
    Complete model: Modified ResNet-18 backbone + MLP prediction head.
    
    Predicts probability distributions over 10 CIFAR-10 classes that reflect
    human annotator disagreement.
    """
    
    def __init__(self):
        """Initialize the complete disagreement predictor model."""
        super().__init__()
        self.backbone = create_modified_resnet18()
        self.head = DisagreementPredictionHead(input_dim=512, hidden_dim=256, num_classes=10)
        
        logger.info("Initialized DisagreementPredictor model")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            x: Input images of shape (batch_size, 3, 32, 32)
        
        Returns:
            probs: Predicted probability distributions of shape (batch_size, 10)
        """
        features = self.backbone(x)
        probs = self.head(features)
        return probs
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 512-dim features for analysis.
        
        Args:
            x: Input images of shape (batch_size, 3, 32, 32)
        
        Returns:
            features: Tensor of shape (batch_size, 512)
        """
        return self.backbone(x)


# Configuration serialization
@dataclass
class ModelConfig:
    """
    Configuration for model architecture parameters.
    
    Attributes:
        backbone_type: Type of backbone architecture (default: 'resnet18')
        input_dim: Dimension of backbone output features (default: 512)
        hidden_dim: Dimension of MLP hidden layer (default: 256)
        num_classes: Number of output classes (default: 10)
        pretrained: Whether to use pretrained weights (default: False)
    """
    backbone_type: str = 'resnet18'
    input_dim: int = 512
    hidden_dim: int = 256
    num_classes: int = 10
    pretrained: bool = False
    
    def to_json(self, filepath: str):
        """
        Serialize configuration to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Saved model configuration to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ModelConfig':
        """
        Deserialize configuration from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            ModelConfig instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            config = cls(**data)
            config.validate()
            
            logger.info(f"Loaded model configuration from {filepath}")
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigParseError(f"Invalid JSON in configuration file: {e}")
        except TypeError as e:
            raise ConfigParseError(f"Invalid configuration fields: {e}")
    
    def validate(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        valid_backbones = ['resnet18', 'resnet34', 'resnet50']
        if self.backbone_type not in valid_backbones:
            raise ValueError(
                f"Invalid backbone_type '{self.backbone_type}'. Must be one of {valid_backbones}"
            )
        
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        
        if not isinstance(self.pretrained, bool):
            raise ValueError(f"pretrained must be boolean, got {type(self.pretrained)}")
    
    @staticmethod
    def get_json_schema() -> dict:
        """
        Get JSON schema for validation.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "backbone_type": {
                    "type": "string",
                    "enum": ["resnet18", "resnet34", "resnet50"]
                },
                "input_dim": {"type": "integer", "minimum": 1},
                "hidden_dim": {"type": "integer", "minimum": 1},
                "num_classes": {"type": "integer", "minimum": 1},
                "pretrained": {"type": "boolean"}
            },
            "required": ["backbone_type", "input_dim", "hidden_dim", "num_classes", "pretrained"]
        }

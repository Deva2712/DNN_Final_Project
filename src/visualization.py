"""
Visualization Module

Implements visualization functions for data exploration, training monitoring,
evaluation results, Grad-CAM attention maps, and failure analysis.
"""

import logging
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def plot_entropy_histogram(entropies: np.ndarray, save_path: str):
    """
    Plot histogram of Shannon entropy values across all images.
    
    Shows distribution of disagreement levels in the dataset.
    
    Args:
        entropies: Array of entropy values
        save_path: Path to save the plot
    """
    import os
    
    logger.info(f"Plotting entropy histogram to {save_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(entropies, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Shannon Entropy (bits)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Distribution of Human Disagreement (CIFAR-10H)', fontsize=14)
    plt.axvline(entropies.mean(), color='red', linestyle='--', 
                label=f'Mean: {entropies.mean():.2f}', linewidth=2)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved entropy histogram to {save_path}")


def plot_per_class_entropy(
    entropies: np.ndarray,
    hard_labels: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """
    Plot box plots showing entropy distribution for each class.
    
    Reveals which classes have more annotator disagreement.
    
    Args:
        entropies: Array of entropy values
        hard_labels: Array of hard class labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    import os
    
    logger.info(f"Plotting per-class entropy to {save_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Group entropies by class
    data_by_class = [entropies[hard_labels == i] for i in range(len(class_names))]
    
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(data_by_class, labels=class_names, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Shannon Entropy (bits)', fontsize=12)
    plt.title('Human Disagreement by Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved per-class entropy plot to {save_path}")


def plot_example_grid(
    images: np.ndarray,
    entropies: np.ndarray,
    soft_labels: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None
):
    """
    Display grid of low/medium/high entropy images with their distributions.
    
    Shows 3 rows (low/medium/high entropy) × 5 columns = 15 images.
    
    Args:
        images: Array of images (N, 3, 32, 32) or (N, 32, 32, 3)
        entropies: Array of entropy values
        soft_labels: Array of soft label distributions (N, 10)
        save_path: Path to save the plot
        class_names: Optional list of class names for x-axis labels
    """
    import os
    
    logger.info(f"Plotting example grid to {save_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f'C{i}' for i in range(10)]
    
    # Select images at different entropy levels
    num_per_row = 5
    
    # Low entropy: bottom 5
    low_idx = np.argsort(entropies)[:num_per_row]
    # Medium entropy: 5 closest to median
    median_entropy = np.median(entropies)
    med_idx = np.argsort(np.abs(entropies - median_entropy))[:num_per_row]
    # High entropy: top 5
    high_idx = np.argsort(entropies)[-num_per_row:]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, num_per_row, hspace=0.4, wspace=0.3)
    
    categories = [
        ('Low Entropy (High Agreement)', low_idx),
        ('Medium Entropy (Moderate Disagreement)', med_idx),
        ('High Entropy (High Disagreement)', high_idx)
    ]
    
    for row_idx, (category_name, indices) in enumerate(categories):
        for col_idx, img_idx in enumerate(indices):
            # Create subplot for image and distribution
            ax_img = fig.add_subplot(gs[row_idx, col_idx])
            
            # Get image and convert to displayable format
            img = images[img_idx]
            if img.shape[0] == 3:  # (3, 32, 32) -> (32, 32, 3)
                img = img.transpose(1, 2, 0)
            
            # Normalize to [0, 1] for display
            img_display = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Display image
            ax_img.imshow(img_display)
            ax_img.set_title(f'H={entropies[img_idx]:.2f} bits', fontsize=10)
            ax_img.axis('off')
            
            # Add distribution bar chart below image
            # Create inset axes for the bar chart
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_bar = inset_axes(ax_img, width="100%", height="40%", 
                               loc='lower center', bbox_to_anchor=(0, -0.6, 1, 1),
                               bbox_transform=ax_img.transAxes, borderpad=0)
            
            # Plot distribution
            ax_bar.bar(range(10), soft_labels[img_idx], color='steelblue', alpha=0.7)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xticks(range(10))
            ax_bar.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
            ax_bar.set_ylabel('Prob', fontsize=8)
            ax_bar.tick_params(axis='y', labelsize=7)
            ax_bar.grid(alpha=0.3, axis='y')
        
        # Add row label
        fig.text(0.02, 0.83 - row_idx * 0.33, category_name, 
                rotation=90, va='center', fontsize=12, fontweight='bold')
    
    plt.suptitle('Example Images by Disagreement Level', fontsize=16, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved example grid to {save_path}")


def plot_training_curves(history: dict, save_path: str):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    logger.info(f"Plotting training curves to {save_path}")
    raise NotImplementedError("To be implemented in task 6.4")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visualizing model attention.
    
    Uses forward and backward hooks to capture activations and gradients from
    the target convolutional layer (layer4 in ResNet-18).
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM with forward and backward hooks.
        
        Args:
            model: The model to visualize
            target_layer: The convolutional layer to target (e.g., model.backbone.layer4)
        """
        logger.info("Initializing Grad-CAM")
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
        
        logger.info("Grad-CAM initialized with hooks on target layer")
    
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image.
        
        Args:
            input_image: Input image tensor of shape (1, 3, 32, 32) or (3, 32, 32)
            target_class: Class index to visualize (if None, use predicted class with highest probability)
        
        Returns:
            heatmap: Grad-CAM heatmap as numpy array of shape (32, 32) with values in [0, 1]
        """
        logger.debug("Generating Grad-CAM heatmap")
        
        # Ensure input has batch dimension
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        
        # Ensure model is in eval mode and gradients are enabled
        self.model.eval()
        input_image.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)
        
        # Select target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute weights (global average pooling of gradients)
        # Shape: (batch, channels, height, width) -> (batch, channels, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        # Shape: (batch, channels, height, width)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU to focus on positive contributions
        cam = torch.nn.functional.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        
        # Handle edge case where cam is all zeros
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        # Resize to input image size (32x32 for CIFAR-10) using torch interpolation
        # Convert back to tensor for resizing
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_resized = torch.nn.functional.interpolate(
            cam_tensor, 
            size=(32, 32), 
            mode='bilinear', 
            align_corners=False
        )
        cam = cam_resized.squeeze().numpy()
        
        logger.debug(f"Generated Grad-CAM heatmap for class {target_class}")
        return cam
    
    def remove_hooks(self):
        """Remove registered hooks to free memory."""
        self.forward_hook.remove()
        self.backward_hook.remove()
        logger.debug("Removed Grad-CAM hooks")


def visualize_gradcam_comparison(
    model: nn.Module,
    low_entropy_images: torch.Tensor,
    high_entropy_images: torch.Tensor,
    save_path: str,
    device: str = 'cuda'
):
    """
    Create visualization grid comparing Grad-CAM for low vs high entropy images.
    
    Shows attention patterns for images with high agreement (low entropy) vs
    high disagreement (high entropy) to understand what visual features the model
    focuses on in each case.
    
    Args:
        model: Trained model
        low_entropy_images: Images with low entropy (high agreement), shape (N, 3, 32, 32)
        high_entropy_images: Images with high entropy (high disagreement), shape (N, 3, 32, 32)
        save_path: Path to save the visualization
        device: 'cuda' or 'cpu'
    """
    import os
    
    logger.info(f"Creating Grad-CAM comparison visualization to {save_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Initialize Grad-CAM targeting layer4 (final convolutional layer)
    gradcam = GradCAM(model, model.backbone.layer4[-1])
    
    # Determine number of images to display
    num_low = min(5, len(low_entropy_images))
    num_high = min(5, len(high_entropy_images))
    
    # Create visualization grid: 2 rows (low/high entropy) x num_images columns
    fig, axes = plt.subplots(2, max(num_low, num_high), figsize=(15, 6))
    
    # Ensure axes is 2D even if only one column
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)
    
    # Visualize low entropy images (high agreement)
    for i in range(num_low):
        image = low_entropy_images[i:i+1].to(device)
        
        # Generate Grad-CAM heatmap
        cam = gradcam.generate_cam(image)
        
        # Prepare original image for display
        img_np = low_entropy_images[i].cpu().permute(1, 2, 0).numpy()
        # Normalize to [0, 1] for display
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Create heatmap overlay using matplotlib colormap
        # Apply jet colormap to heatmap
        cmap = plt.cm.jet
        heatmap_colored = cmap(cam)[:, :, :3]  # Get RGB, discard alpha
        
        # Blend original image with heatmap
        overlay = 0.6 * img_np + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        axes[0, i].imshow(overlay)
        axes[0, i].set_title(f'Low Entropy\n(High Agreement)', fontsize=10)
        axes[0, i].axis('off')
    
    # Hide unused subplots in first row
    for i in range(num_low, axes.shape[1]):
        axes[0, i].axis('off')
    
    # Visualize high entropy images (high disagreement)
    for i in range(num_high):
        image = high_entropy_images[i:i+1].to(device)
        
        # Generate Grad-CAM heatmap
        cam = gradcam.generate_cam(image)
        
        # Prepare original image for display
        img_np = high_entropy_images[i].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Create heatmap overlay using matplotlib colormap
        cmap = plt.cm.jet
        heatmap_colored = cmap(cam)[:, :, :3]  # Get RGB, discard alpha
        
        # Blend original image with heatmap
        overlay = 0.6 * img_np + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'High Entropy\n(High Disagreement)', fontsize=10)
        axes[1, i].axis('off')
    
    # Hide unused subplots in second row
    for i in range(num_high, axes.shape[1]):
        axes[1, i].axis('off')
    
    plt.suptitle('Grad-CAM Attention Patterns: Low vs High Entropy Images', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Clean up hooks
    gradcam.remove_hooks()
    
    logger.info(f"Saved Grad-CAM comparison visualization to {save_path}")


def visualize_failure_cases(
    model: nn.Module,
    test_loader,
    num_cases: int = 10,
    save_path: str = None,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None
):
    """
    Visualize failure cases with highest KL divergence.
    
    Displays original image, true distribution, predicted distribution,
    true and predicted entropy values for the worst predictions.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        num_cases: Number of failure cases to display
        save_path: Path to save the visualization
        device: 'cuda' or 'cpu'
        class_names: List of CIFAR-10 class names (default: ['airplane', 'automobile', ...])
    """
    import os
    from src.evaluation import identify_failure_cases
    
    logger.info(f"Visualizing top {num_cases} failure cases")
    
    # Default class names if not provided
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Identify failure cases
    failure_cases = identify_failure_cases(model, test_loader, num_cases, device)
    
    # Create visualization grid: num_cases rows × 3 columns
    # Column 1: Image, Column 2: True distribution, Column 3: Predicted distribution
    fig, axes = plt.subplots(num_cases, 3, figsize=(12, 4 * num_cases))
    
    # Ensure axes is 2D even if only one case
    if num_cases == 1:
        axes = axes.reshape(1, 3)
    
    for i, case in enumerate(failure_cases):
        # Column 1: Display image
        img = case['image'].permute(1, 2, 0).numpy()
        # Normalize to [0, 1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(
            f"Image (Class: {class_names[case['hard_label']]})\n"
            f"KL Divergence: {case['kl_divergence']:.4f}",
            fontsize=10
        )
        axes[i, 0].axis('off')
        
        # Column 2: True distribution
        axes[i, 1].bar(range(10), case['true_dist'].numpy(), color='steelblue', alpha=0.7)
        axes[i, 1].set_title(
            f"True Distribution\nEntropy: {case['true_entropy']:.3f} bits",
            fontsize=10
        )
        axes[i, 1].set_xticks(range(10))
        axes[i, 1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_ylabel('Probability', fontsize=9)
        axes[i, 1].grid(alpha=0.3, axis='y')
        
        # Column 3: Predicted distribution
        axes[i, 2].bar(range(10), case['pred_dist'].numpy(), color='coral', alpha=0.7)
        axes[i, 2].set_title(
            f"Predicted Distribution\nEntropy: {case['pred_entropy']:.3f} bits",
            fontsize=10
        )
        axes[i, 2].set_xticks(range(10))
        axes[i, 2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        axes[i, 2].set_ylim(0, 1)
        axes[i, 2].set_ylabel('Probability', fontsize=9)
        axes[i, 2].grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'Top {num_cases} Failure Cases (Highest KL Divergence)', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved failure case visualization to {save_path}")
    
    plt.close()
    
    logger.info(f"Completed failure case visualization")


def plot_corruption_robustness(results: dict, save_path: str):
    """
    Plot entropy change vs corruption severity for each corruption type.
    
    Args:
        results: Dict mapping corruption_type to dict of {severity: entropy_change}
        save_path: Path to save the plot
    """
    import os
    
    logger.info(f"Plotting corruption robustness to {save_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    corruption_types = ['gaussian_noise', 'gaussian_blur', 'contrast_reduction']
    severities = [1, 3, 5]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, corruption_type in enumerate(corruption_types):
        if corruption_type in results:
            entropy_changes = [results[corruption_type][s] for s in severities]
            
            axes[idx].plot(severities, entropy_changes, marker='o', linewidth=2, 
                          markersize=8, color='steelblue')
            axes[idx].set_xlabel('Severity', fontsize=12)
            axes[idx].set_ylabel('Mean Absolute Entropy Change (bits)', fontsize=12)
            axes[idx].set_title(corruption_type.replace('_', ' ').title(), fontsize=13)
            axes[idx].grid(alpha=0.3)
            axes[idx].set_xticks(severities)
            axes[idx].set_ylim(bottom=0)
    
    plt.suptitle('Model Robustness to Image Corruptions', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved corruption robustness plot to {save_path}")


def manual_categorization_interface(
    model: nn.Module,
    test_loader,
    num_images: int = 25,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None
) -> dict:
    """
    Interactive interface for manually categorizing disagreement sources.
    
    Selects 20-30 highest entropy images and allows manual categorization into:
    - ambiguous_identity: Object genuinely looks like multiple classes
    - poor_image_quality: Low resolution, blur, occlusion
    - multi_object_scene: Multiple objects present, unclear focus
    - boundary_case: Object at edge of class definition
    - other: Uncategorized reasons
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        num_images: Number of high-entropy images to categorize (20-30 recommended)
        device: 'cuda' or 'cpu'
        class_names: List of CIFAR-10 class names
    
    Returns:
        categorization: Dict mapping image_idx to category
    """
    import os
    
    logger.info(f"Starting manual categorization interface for {num_images} images")
    
    # Default class names if not provided
    if class_names is None:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Collect all images and their entropies
    all_images = []
    all_soft_labels = []
    all_hard_labels = []
    all_entropies = []
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for batch in test_loader:
            images, soft_labels, hard_labels, true_entropy = batch
            all_images.append(images)
            all_soft_labels.append(soft_labels)
            all_hard_labels.append(hard_labels)
            all_entropies.append(true_entropy)
    
    # Concatenate all batches
    all_images = torch.cat(all_images)
    all_soft_labels = torch.cat(all_soft_labels)
    all_hard_labels = torch.cat(all_hard_labels)
    all_entropies = torch.cat(all_entropies)
    
    # Select highest entropy images
    high_entropy_indices = torch.argsort(all_entropies, descending=True)[:num_images]
    
    # Define categories
    categories = [
        'ambiguous_identity',
        'poor_image_quality',
        'multi_object_scene',
        'boundary_case',
        'other'
    ]
    
    categorization = {}
    
    print("\n" + "="*70)
    print("MANUAL DISAGREEMENT CATEGORIZATION INTERFACE")
    print("="*70)
    print("\nCategories:")
    print("  1. Ambiguous Identity - Object genuinely looks like multiple classes")
    print("  2. Poor Image Quality - Low resolution, blur, occlusion")
    print("  3. Multi-Object Scene - Multiple objects present, unclear focus")
    print("  4. Boundary Case - Object at edge of class definition")
    print("  5. Other - Uncategorized reasons")
    print("\n" + "="*70 + "\n")
    
    for i, idx in enumerate(high_entropy_indices):
        idx = idx.item()
        
        # Display image and distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Image
        img = all_images[idx].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[0].imshow(img)
        axes[0].set_title(
            f"Image {i+1}/{num_images}\n"
            f"Class: {class_names[all_hard_labels[idx].item()]}\n"
            f"Entropy: {all_entropies[idx]:.3f} bits",
            fontsize=12
        )
        axes[0].axis('off')
        
        # Distribution
        axes[1].bar(range(10), all_soft_labels[idx].numpy(), color='steelblue', alpha=0.7)
        axes[1].set_xticks(range(10))
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].set_title('Annotator Distribution', fontsize=12)
        axes[1].set_ylabel('Probability', fontsize=11)
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Prompt for category
        while True:
            try:
                choice = input(f"\nSelect category (1-5) for image {i+1}: ").strip()
                choice_int = int(choice)
                if 1 <= choice_int <= 5:
                    categorization[i] = categories[choice_int - 1]
                    print(f"✓ Categorized as: {categories[choice_int - 1].replace('_', ' ').title()}\n")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 5.")
            except KeyboardInterrupt:
                print("\n\nCategorization interrupted by user.")
                plt.close('all')
                return categorization
        
        plt.close()
    
    print("\n" + "="*70)
    print("CATEGORIZATION COMPLETE")
    print("="*70 + "\n")
    
    logger.info(f"Completed manual categorization of {len(categorization)} images")
    
    return categorization


def generate_categorization_summary(
    categorization: dict,
    save_path: Optional[str] = None
) -> dict:
    """
    Generate summary report with category frequencies.
    
    Args:
        categorization: Dict mapping image_idx to category
        save_path: Optional path to save summary as JSON
    
    Returns:
        summary: Dict with category counts and percentages
    """
    import os
    from collections import Counter
    import json
    
    logger.info("Generating categorization summary")
    
    # Count categories
    counts = Counter(categorization.values())
    total = len(categorization)
    
    # Create summary
    summary = {
        'total_images': total,
        'categories': {}
    }
    
    for category, count in counts.items():
        summary['categories'][category] = {
            'count': count,
            'percentage': 100.0 * count / total
        }
    
    # Sort by count (descending)
    summary['categories'] = dict(
        sorted(summary['categories'].items(), 
               key=lambda x: x[1]['count'], 
               reverse=True)
    )
    
    # Print summary
    print("\n" + "="*70)
    print("DISAGREEMENT SOURCE CATEGORIZATION SUMMARY")
    print("="*70)
    print(f"\nTotal Images Categorized: {total}\n")
    print(f"{'Category':<30} {'Count':<10} {'Percentage':<10}")
    print("-" * 70)
    
    for category, data in summary['categories'].items():
        category_name = category.replace('_', ' ').title()
        print(f"{category_name:<30} {data['count']:<10} {data['percentage']:.1f}%")
    
    print("="*70 + "\n")
    
    # Save to JSON if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved categorization summary to {save_path}")
    
    logger.info("Categorization summary complete")
    
    return summary

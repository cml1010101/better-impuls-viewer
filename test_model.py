#!/usr/bin/env python3
"""
Enhanced model evaluation and testing script.
Tests model loading, inference, and provides performance metrics.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import project modules
from periodizer import MultiBranchStarModelHybrid, StarModelConfig, ModelInput, ModelOutput
from train import SyntheticLightCurveDataset, collate_fn
from torch.utils.data import DataLoader

def load_trained_model(model_path: str) -> tuple[MultiBranchStarModelHybrid, Dict[str, Any]]:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the .pth model file
        
    Returns:
        Tuple of (model, metadata)
    """
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract components
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    class_names = checkpoint.get('class_names', [])
    training_info = checkpoint.get('training_info', {})
    
    # Create and load model
    model = MultiBranchStarModelHybrid(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    metadata = {
        'config': config,
        'class_names': class_names,
        'training_info': training_info,
        'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
    }
    
    print(f"✓ Model loaded successfully")
    print(f"  - Model size: {metadata['model_size_mb']:.2f} MB")
    print(f"  - Training epochs: {training_info.get('completed_epochs', 'Unknown')}")
    print(f"  - Final loss: {training_info.get('final_loss', 'Unknown')}")
    print(f"  - Classes: {len(class_names)}")
    
    return model, metadata

def evaluate_model_on_test_data(model: MultiBranchStarModelHybrid, num_samples: int = 50) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        num_samples: Number of test samples to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating model on {num_samples} test samples...")
    
    # Create test dataset
    test_dataset = SyntheticLightCurveDataset(n_per_class=num_samples//14, seed=999)  # Different seed for test
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_period_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    from periodizer import MultiTaskLoss
    loss_fn = MultiTaskLoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Prepare inputs
            model_input = ModelInput(
                lc=batch['raw_lc'],
                pgram=batch['periodogram'],
                folded_data=batch['folded_data'],
                folded_periods=batch['candidate_periods']
            )
            
            # Forward pass
            outputs = model(model_input)
            
            # Calculate loss
            loss_output = loss_fn(outputs, batch['class_labels'], batch['target_periods'])
            
            total_loss += loss_output.total.item()
            total_cls_loss += loss_output.cls.item()
            total_period_loss += loss_output.period.item()
            
            # Classification accuracy
            predictions = torch.argmax(outputs.type_logits, dim=1)
            correct_predictions += (predictions == batch['class_labels']).sum().item()
            total_predictions += batch['class_labels'].size(0)
    
    # Calculate metrics
    num_batches = len(test_loader)
    metrics = {
        'test_loss': total_loss / num_batches,
        'test_cls_loss': total_cls_loss / num_batches,
        'test_period_loss': total_period_loss / num_batches,
        'classification_accuracy': correct_predictions / total_predictions,
        'num_test_samples': total_predictions
    }
    
    print("📊 Evaluation Results:")
    print(f"  - Test Loss: {metrics['test_loss']:.4f}")
    print(f"  - Classification Loss: {metrics['test_cls_loss']:.4f}")
    print(f"  - Period Loss: {metrics['test_period_loss']:.4f}")
    print(f"  - Classification Accuracy: {metrics['classification_accuracy']:.4f} ({metrics['classification_accuracy']*100:.1f}%)")
    print(f"  - Test Samples: {metrics['num_test_samples']}")
    
    return metrics

def test_model_inference_speed(model: MultiBranchStarModelHybrid, num_iterations: int = 100) -> float:
    """
    Test model inference speed.
    
    Args:
        model: Trained model
        num_iterations: Number of inference iterations
        
    Returns:
        Average inference time in milliseconds
    """
    print(f"Testing inference speed over {num_iterations} iterations...")
    
    # Create dummy input
    batch_size = 1
    model_input = ModelInput(
        lc=torch.randn(batch_size, 1, 1024),
        pgram=torch.randn(batch_size, 1, 512),
        folded_data=torch.randn(batch_size, 4, 200),
        folded_periods=torch.randn(batch_size, 4)
    )
    
    model.eval()
    import time
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(model_input)
    
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(model_input)
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / num_iterations) * 1000
    
    print(f"⏱️  Average inference time: {avg_time_ms:.2f} ms")
    
    return avg_time_ms

def analyze_model_architecture(model: MultiBranchStarModelHybrid) -> Dict[str, Any]:
    """
    Analyze model architecture and parameter counts.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with architecture analysis
    """
    print("🔍 Analyzing model architecture...")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by component
    component_params = {}
    for name, module in model.named_children():
        component_params[name] = sum(p.numel() for p in module.parameters())
    
    analysis = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'component_parameters': component_params,
        'model_size_estimate_mb': (total_params * 4) / (1024 * 1024)  # Assuming float32
    }
    
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Estimated model size: {analysis['model_size_estimate_mb']:.2f} MB")
    print("  - Parameters by component:")
    for name, count in component_params.items():
        percentage = (count / total_params) * 100
        print(f"    • {name}: {count:,} ({percentage:.1f}%)")
    
    return analysis

def main():
    """Main testing and evaluation function."""
    print("🚀 Enhanced Model Evaluation")
    print("=" * 50)
    
    # Check for available models
    model_files = list(Path('.').glob('*.pth'))
    if not model_files:
        print("❌ No model files found. Please train a model first.")
        return 1
    
    print(f"📁 Available models: {[str(f) for f in model_files]}")
    
    # Use the first available model (or specify one)
    model_path = str(model_files[0])
    print(f"🔬 Testing model: {model_path}")
    print()
    
    try:
        # Load model
        model, metadata = load_trained_model(model_path)
        print()
        
        # Analyze architecture
        architecture_analysis = analyze_model_architecture(model)
        print()
        
        # Test inference speed
        inference_time = test_model_inference_speed(model)
        print()
        
        # Evaluate on test data
        evaluation_metrics = evaluate_model_on_test_data(model)
        print()
        
        # Summary report
        print("📋 SUMMARY REPORT")
        print("=" * 50)
        print(f"Model: {model_path}")
        print(f"Size: {metadata['model_size_mb']:.2f} MB")
        print(f"Parameters: {architecture_analysis['total_parameters']:,}")
        print(f"Test Accuracy: {evaluation_metrics['classification_accuracy']*100:.1f}%")
        print(f"Test Loss: {evaluation_metrics['test_loss']:.4f}")
        print(f"Inference Time: {inference_time:.2f} ms")
        
        # Check if performance is reasonable
        if evaluation_metrics['classification_accuracy'] > 0.5:
            print("✅ Model appears to be learning successfully!")
        else:
            print("⚠️ Model accuracy is low - may need more training or architectural improvements")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
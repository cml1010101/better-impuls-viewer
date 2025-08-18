#!/usr/bin/env python3
"""
Model optimization and validation script.
Implements architectural improvements and performance benchmarking.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from periodizer import MultiBranchStarModelHybrid, StarModelConfig, ModelInput, MultiTaskLoss
from train import SyntheticLightCurveDataset, collate_fn
from torch.utils.data import DataLoader

class ModelOptimizer:
    """Class for optimizing and validating model performance."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        self.config = checkpoint['config']
        self.model = MultiBranchStarModelHybrid(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def benchmark_inference_speed(self, batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[int, float]:
        """Benchmark inference speed for different batch sizes."""
        print("🚀 Benchmarking inference speed...")
        
        results = {}
        for batch_size in batch_sizes:
            # Create test input
            model_input = ModelInput(
                lc=torch.randn(batch_size, 1, 1024),
                pgram=torch.randn(batch_size, 1, 512),
                folded_data=torch.randn(batch_size, 4, 200),
                folded_periods=torch.randn(batch_size, 4)
            )
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(model_input)
            
            # Benchmark
            num_iterations = 100
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.model(model_input)
            
            end_time = time.time()
            avg_time_ms = ((end_time - start_time) / num_iterations) * 1000
            throughput = batch_size / (avg_time_ms / 1000)  # samples per second
            
            results[batch_size] = {
                'time_ms': avg_time_ms,
                'throughput': throughput
            }
            
            print(f"  Batch size {batch_size:2d}: {avg_time_ms:6.2f} ms/batch, {throughput:6.1f} samples/sec")
        
        return results
    
    def validate_model_outputs(self, num_samples: int = 100) -> Dict[str, float]:
        """Validate that model outputs are reasonable."""
        print("🔍 Validating model outputs...")
        
        # Create test dataset
        test_dataset = SyntheticLightCurveDataset(n_per_class=num_samples//14, seed=123)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        
        all_type_logits = []
        all_period_preds = []
        all_true_labels = []
        all_true_periods = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                model_input = ModelInput(
                    lc=batch['raw_lc'],
                    pgram=batch['periodogram'],
                    folded_data=batch['folded_data'],
                    folded_periods=batch['candidate_periods']
                )
                
                outputs = self.model(model_input)
                
                all_type_logits.append(outputs.type_logits)
                all_period_preds.append(torch.stack([outputs.logP1_pred, outputs.logP2_pred], dim=1))
                all_true_labels.append(batch['class_labels'])
                all_true_periods.append(batch['target_periods'])
        
        # Concatenate all results
        type_logits = torch.cat(all_type_logits, dim=0)
        period_preds = torch.cat(all_period_preds, dim=0)
        true_labels = torch.cat(all_true_labels, dim=0)
        true_periods = torch.cat(all_true_periods, dim=0)
        
        # Calculate metrics
        type_predictions = torch.argmax(type_logits, dim=1)
        classification_accuracy = (type_predictions == true_labels).float().mean().item()
        
        # Period prediction analysis
        period_mse = nn.MSELoss()(period_preds, true_periods).item()
        period_mae = nn.L1Loss()(period_preds, true_periods).item()
        
        # Check for numerical stability
        has_nan_outputs = torch.isnan(type_logits).any() or torch.isnan(period_preds).any()
        has_inf_outputs = torch.isinf(type_logits).any() or torch.isinf(period_preds).any()
        
        # Type prediction distribution
        type_pred_entropy = -torch.sum(torch.softmax(type_logits, dim=1) * torch.log_softmax(type_logits, dim=1), dim=1).mean().item()
        
        validation_results = {
            'classification_accuracy': classification_accuracy,
            'period_mse': period_mse,
            'period_mae': period_mae,
            'type_prediction_entropy': type_pred_entropy,
            'has_nan_outputs': has_nan_outputs,
            'has_inf_outputs': has_inf_outputs,
            'num_samples_tested': len(true_labels)
        }
        
        print(f"  📊 Classification accuracy: {classification_accuracy:.4f} ({classification_accuracy*100:.1f}%)")
        print(f"  📊 Period MSE: {period_mse:.4f}")
        print(f"  📊 Period MAE: {period_mae:.4f}")
        print(f"  📊 Type prediction entropy: {type_pred_entropy:.4f}")
        print(f"  📊 Has NaN outputs: {has_nan_outputs}")
        print(f"  📊 Has Inf outputs: {has_inf_outputs}")
        
        return validation_results
    
    def analyze_model_components(self) -> Dict[str, Dict]:
        """Analyze individual model components."""
        print("🔬 Analyzing model components...")
        
        # Parameter analysis by component
        component_analysis = {}
        total_params = sum(p.numel() for p in self.model.parameters())
        
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            component_analysis[name] = {
                'parameters': params,
                'percentage': (params / total_params) * 100,
                'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
            print(f"  • {name}: {params:,} params ({component_analysis[name]['percentage']:.1f}%)")
        
        return component_analysis
    
    def memory_profiling(self) -> Dict[str, float]:
        """Profile memory usage during inference."""
        print("💾 Profiling memory usage...")
        
        # Test with different input sizes
        memory_results = {}
        
        for batch_size in [1, 4, 8]:
            model_input = ModelInput(
                lc=torch.randn(batch_size, 1, 1024),
                pgram=torch.randn(batch_size, 1, 512),
                folded_data=torch.randn(batch_size, 4, 200),
                folded_periods=torch.randn(batch_size, 4)
            )
            
            # Estimate memory usage (rough approximation)
            input_memory = sum(t.numel() * 4 for t in [model_input.lc, model_input.pgram, 
                                                      model_input.folded_data, model_input.folded_periods]) / (1024**2)
            
            # Model parameters memory
            model_memory = sum(p.numel() * 4 for p in self.model.parameters()) / (1024**2)
            
            total_memory_mb = input_memory + model_memory
            memory_results[batch_size] = total_memory_mb
            
            print(f"  Batch size {batch_size}: ~{total_memory_mb:.1f} MB")
        
        return memory_results
    
    def suggest_optimizations(self, validation_results: Dict) -> List[str]:
        """Suggest optimizations based on analysis results."""
        suggestions = []
        
        if validation_results['classification_accuracy'] < 0.5:
            suggestions.append("Low classification accuracy - consider more training epochs or data")
        
        if validation_results['period_mae'] > 2.0:
            suggestions.append("High period prediction error - review period normalization")
        
        if validation_results['type_prediction_entropy'] < 1.0:
            suggestions.append("Low prediction entropy - model may be overconfident")
        
        if validation_results['has_nan_outputs']:
            suggestions.append("NaN outputs detected - check for numerical instability")
        
        if validation_results['has_inf_outputs']:
            suggestions.append("Infinite outputs detected - review loss function and gradients")
        
        # Performance suggestions
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params > 5_000_000:
            suggestions.append("Large model size - consider model compression or pruning")
        
        return suggestions

def create_performance_report(model_path: str) -> Dict:
    """Create a comprehensive performance report."""
    print("📋 Creating Performance Report")
    print("=" * 60)
    
    optimizer = ModelOptimizer(model_path)
    
    # Run all analyses
    speed_results = optimizer.benchmark_inference_speed()
    validation_results = optimizer.validate_model_outputs()
    component_analysis = optimizer.analyze_model_components()
    memory_results = optimizer.memory_profiling()
    optimization_suggestions = optimizer.suggest_optimizations(validation_results)
    
    report = {
        'model_path': model_path,
        'model_config': optimizer.config,
        'speed_benchmark': speed_results,
        'validation_results': validation_results,
        'component_analysis': component_analysis,
        'memory_profiling': memory_results,
        'optimization_suggestions': optimization_suggestions,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Print summary
    print("\n📋 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Parameters: {sum(p.numel() for p in optimizer.model.parameters()):,}")
    print(f"Classification Accuracy: {validation_results['classification_accuracy']*100:.1f}%")
    print(f"Period MAE: {validation_results['period_mae']:.4f}")
    print(f"Inference Speed (batch=1): {speed_results[1]['time_ms']:.2f} ms")
    print(f"Throughput (batch=8): {speed_results[8]['throughput']:.1f} samples/sec")
    
    if optimization_suggestions:
        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        for i, suggestion in enumerate(optimization_suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print("\n✅ No major issues detected!")
    
    return report

def main():
    """Main function."""
    # Find available models
    model_files = list(Path('.').glob('*.pth'))
    if not model_files:
        print("❌ No model files found. Please train a model first.")
        return 1
    
    print(f"📁 Available models: {[str(f) for f in model_files]}")
    
    # Test all available models
    for model_file in model_files:
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model_file}")
        print(f"{'='*80}")
        
        try:
            report = create_performance_report(str(model_file))
            
            # Save report
            report_file = str(model_file).replace('.pth', '_performance_report.txt')
            with open(report_file, 'w') as f:
                f.write(f"Performance Report for {model_file}\n")
                f.write(f"Generated: {report['timestamp']}\n\n")
                f.write(f"Classification Accuracy: {report['validation_results']['classification_accuracy']*100:.1f}%\n")
                f.write(f"Period MAE: {report['validation_results']['period_mae']:.4f}\n")
                f.write(f"Inference Time: {report['speed_benchmark'][1]['time_ms']:.2f} ms\n")
                f.write(f"Parameters: {sum(p.numel() for p in MultiBranchStarModelHybrid(report['model_config']).parameters()):,}\n\n")
                if report['optimization_suggestions']:
                    f.write("Optimization Suggestions:\n")
                    for i, suggestion in enumerate(report['optimization_suggestions'], 1):
                        f.write(f"  {i}. {suggestion}\n")
            
            print(f"\n📄 Report saved to: {report_file}")
            
        except Exception as e:
            print(f"❌ Error testing {model_file}: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main())
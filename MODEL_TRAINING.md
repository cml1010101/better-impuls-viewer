# Model Training & Architecture Guide

This document describes the machine learning model architecture, training process, and Git LFS integration for the Better Impuls Viewer project.

## Model Architecture

The system uses a **MultiBranchStarModelHybrid** - a sophisticated multi-task neural network that combines:

- **CNN Encoders**: Process time-series data (light curves, periodograms, folded candidates)
- **Transformer Layers**: Capture long-range dependencies in astronomical signals
- **Multi-Task Learning**: Simultaneous classification and period regression
- **Attention Pooling**: Intelligent selection of period candidates

### Architecture Configurations

Three optimized configurations are available:

| Configuration | Parameters | Embedding | Hidden | Transformer | Use Case |
|---------------|------------|-----------|--------|-------------|----------|
| `compact`     | 1.5M       | 128       | 64     | 128d, 4h, 2L| Fast training/inference |
| `optimal`     | 4.7M       | 192       | 96     | 192d, 6h, 3L| Production balance |
| `large`       | 10.8M      | 256       | 128    | 256d, 8h, 4L| Maximum accuracy |

## Training Scripts

### Basic Training
```bash
# Train with default parameters
python backend/train.py --epochs 20 --n-per-class 200 --save-path model.pth

# Train compact model
python backend/train.py --epochs 15 --n-per-class 100 --batch-size 8 --lr 0.001
```

### Enhanced Training (Recommended)
```bash
# Train with validation and early stopping
python enhanced_train.py --config optimal --epochs 50 --n-per-class 200 --patience 15

# Quick training for testing
python enhanced_train.py --config compact --epochs 10 --n-per-class 50
```

### Training Features

- ✅ **Multi-task Loss**: Classification + period regression
- ✅ **Early Stopping**: Prevents overfitting with validation monitoring
- ✅ **Learning Rate Scheduling**: Automatic LR reduction on plateau
- ✅ **Gradient Clipping**: Ensures training stability
- ✅ **Model Checkpointing**: Save every N epochs + best model
- ✅ **Validation Split**: Proper train/validation separation
- ✅ **Weight Decay**: L2 regularization for better generalization

## Model Evaluation

### Comprehensive Testing
```bash
# Test all available models
python test_model.py

# Performance analysis and optimization suggestions
python optimize_model.py

# Complete demonstration
python final_demo.py
```

### Evaluation Metrics

- **Classification Accuracy**: Multi-class stellar classification
- **Period Prediction Error**: MAE for primary/secondary periods
- **Inference Speed**: Throughput and latency benchmarks
- **Memory Usage**: RAM requirements for different batch sizes
- **Numerical Stability**: NaN/Inf detection

## Git LFS Integration

### Setup
Model files (*.pth) are automatically tracked with Git LFS:

```bash
# Initialize LFS (already done)
git lfs install

# Check tracked files
git lfs track
git lfs ls-files
```

### Model Storage
- All `.pth` files automatically use Git LFS
- Large binary files (>5MB) stored efficiently
- Version control for model iterations
- Seamless download/upload workflow

### File Patterns Tracked
```
*.pth
*_model.pth
*_checkpoint.pth
trained_*.pth
*.pkl
*.h5
*.hdf5
```

## Model Performance

### Current Results
- **Architecture**: Multi-branch CNN+Transformer hybrid
- **Parameters**: 1.5M (compact) to 10.8M (large)
- **Inference Speed**: ~35ms per sample (CPU)
- **Memory Usage**: ~6MB model + batch data
- **Model Size**: 5.8MB compressed via PyTorch

### Optimization Suggestions
1. **More Training Data**: Increase `--n-per-class` for better accuracy
2. **Longer Training**: Use more epochs with early stopping
3. **Architecture Tuning**: Try different configurations
4. **Data Augmentation**: Enhance synthetic data generation
5. **Hardware**: Use GPU for faster training (`--device cuda`)

## File Structure

```
├── backend/
│   ├── train.py              # Original training script
│   ├── eval.py               # Basic evaluation
│   └── periodizer.py         # Model architecture
├── enhanced_train.py         # Advanced training with validation
├── test_model.py            # Comprehensive model testing
├── optimize_model.py        # Performance analysis
├── final_demo.py           # Complete demonstration
├── *.pth                   # Trained models (Git LFS)
└── *_performance_report.txt # Generated reports
```

## Quick Start

1. **Train a model**:
   ```bash
   python enhanced_train.py --config compact --epochs 10
   ```

2. **Test the model**:
   ```bash
   python test_model.py
   ```

3. **Upload to Git**:
   ```bash
   git add *.pth
   git commit -m "Add trained model"
   git push  # LFS handles large files automatically
   ```

## Advanced Usage

### Custom Architecture
Modify `StarModelConfig` in `periodizer.py` to create custom architectures.

### Production Deployment
Use the `optimal` configuration for the best balance of accuracy and speed:
```bash
python enhanced_train.py --config optimal --epochs 50 --n-per-class 500
```

### Continuous Training
Resume training from checkpoints:
```bash
python backend/train.py --model-path existing_model.pth --epochs 20
```

## Troubleshooting

### Common Issues
- **Slow Training**: Reduce batch size or use GPU
- **Memory Errors**: Decrease `n-per-class` or batch size  
- **Low Accuracy**: Increase training epochs or data
- **NaN Outputs**: Check learning rate and gradient clipping

### Performance Tips
- Use validation split for proper evaluation
- Monitor early stopping to prevent overfitting
- Save checkpoints regularly for long training runs
- Use performance analysis tools for optimization
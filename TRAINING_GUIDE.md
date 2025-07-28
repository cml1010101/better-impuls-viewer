# CSV Model Training Guide

## Overview

The Better Impuls Viewer supports automatic model training using data from CSV files. This enables the CNN period validation model to be trained on real astronomical data with known periods and classifications.

## Getting Started

### 1. Configure CSV Data Path

Add your CSV data path to the `.env` file:

```bash
# CSV Training Data Configuration
CSV_TRAINING_DATA_PATH=sample_training_data.csv
```

### 2. CSV Format

The CSV file should contain the following columns:

- **star_number**: Integer (e.g., 1, 2, 3...)
- **period_1**: Float (primary period in days, -9 for no period)
- **period_2**: Float (secondary period in days, -9 for no period, optional)
- **lc_category**: String (light curve category)
- **sensor**: String (sensor/instrument name, optional)

Example:
```csv
star_number,period_1,period_2,lc_category,sensor
1,2.5,-9,dipper,csv
2,7.8,-9,resolved distant peaks,csv
3,15.3,-9,sinusoidal,csv
```

### 3. Start Training

Use the API or command line:

```python
from backend.model_training import ModelTrainer

trainer = ModelTrainer()
result = trainer.train_from_csv()
```

### Command Line Usage

```bash
python backend/model_training.py --csv-file sample_training_data.csv --stars "1:5"
```

## File Structure

The training system consists of:

1. **config.py**: Environment configuration and CSV paths
2. **models.py**: Data models and training result structures
3. **csv_data_loader.py**: CSV data loader with category normalization
4. **model_training.py**: CNN model training pipeline
5. **period_detection.py**: Period analysis and CNN architecture

## Training Pipeline

CSV → Data Extraction → Category Normalization → Time Series Loading → Phase Folding → CNN Training → Model Saving

### Key Features

- **Category Normalization**: Automatic mapping of CSV categories to model classes
- **Star Range Selection**: Train on specific stars or ranges
- **Phase Folding**: Converts time series to phase-folded curves for CNN input
- **Early Stopping**: Prevents overfitting during training

## API Endpoints

- `POST /train_model`: Train CNN model from CSV data
- `GET /model_status`: Check current model status

## Example Usage

```python
from backend.csv_data_loader import CSVDataLoader

loader = CSVDataLoader('sample_training_data.csv')
training_data = loader.extract_training_data(['1', '2', '3'])
print(f"Loaded {len(training_data)} training examples")
```

### Training Configuration

Training parameters can be customized:

- **Epochs**: Number of training epochs (default: 100)
- **Batch Size**: Training batch size (default: 32)
- **Learning Rate**: Model learning rate (default: 0.001)
- **Validation Split**: Fraction for validation (default: 0.2)

## Model Output

The trained model provides:

- **Period Validation**: Confidence scores for given periods
- **Category Classification**: Light curve type classification
- **Performance Metrics**: Training loss and validation accuracy

## Advantages

- **Simple Setup**: No external API dependencies
- **Flexible Input**: Standard CSV format
- **Portable**: Works offline without internet connection
- **Scalable**: Easy to add new training data via CSV files

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure CSV path is correct in `.env`
2. **No Training Data**: Check CSV format and star data files
3. **Memory Issues**: Reduce batch size or use fewer stars
4. **Low Accuracy**: Increase training data or adjust parameters

### Debugging

Enable debug logging:

```bash
python backend/model_training.py --csv-file your_data.csv --verbose
```
# Google Sheets Integration and Model Training

## Overview

The Better Impuls Viewer now supports automatic model training using data from Google Sheets. This enables the CNN period validation model to be trained on real astronomical data with known periods and classifications.

## Setup

### 1. Configure Google Sheets URL

Add your Google Sheets URL to the `.env` file:

```bash
# Google Sheets Configuration
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id-here/edit#gid=0
```

### 2. Google Sheets Format

The system expects the following column layout:

- **Column A**: Star number (integer)
- **Column AK**: Period 1 (float, -9 for no valid period)
- **Column AL**: Period 2 (float, -9 for no valid period)  
- **Column AN**: LC category (string: "dipper", "distant peaks", "close peak", "sinusoidal", etc.)

### 3. Sample Data Files

Ensure corresponding data files exist in the `sample_data` directory with naming format:
```
{star_number}-{telescope}.tbl
```

## Training the Model

### Option 1: API Endpoint

```bash
curl -X POST http://localhost:8000/train_model
```

### Option 2: Python Script

```python
from backend.model_training import ModelTrainer

trainer = ModelTrainer()
result = trainer.train_from_google_sheets()

if result.success:
    print(f"Training completed! Model saved to: {result.model_path}")
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Training samples: {result.training_samples}")
```

### Option 3: Direct Execution

```bash
cd backend
python model_training.py
```

## How It Works

### 1. Data Loading
- Reads Google Sheets via CSV export URL
- Extracts star numbers and periods from specified columns
- Filters out invalid periods (-9 values)
- Loads corresponding time series data from sample_data

### 2. Data Preprocessing
- Phase-folds light curves at known periods
- Generates confidence labels based on period quality metrics
- Encodes LC categories for classification
- Creates train/validation splits

### 3. Model Training
- Uses CNN architecture with convolutional layers for pattern recognition
- Multi-task learning: period confidence + variability classification
- Early stopping to prevent overfitting
- Saves best model based on validation loss

### 4. Model Integration
- Trained model automatically used by `/auto_periods/` endpoint
- Provides improved period validation and classification
- Replaces synthetic training data with real astronomical examples

## Architecture

### Backend Modules

1. **config.py**: Environment configuration
2. **models.py**: Pydantic data models
3. **data_processing.py**: Data preprocessing utilities
4. **period_detection.py**: Period analysis algorithms
5. **google_sheets.py**: Google Sheets data loader
6. **model_training.py**: CNN training pipeline
7. **server.py**: FastAPI endpoints

### Training Pipeline

```
Google Sheets → Data Extraction → Time Series Loading → Phase Folding → CNN Training → Model Saving
```

## Model Output

The trained model is saved as `trained_cnn_model.pth` and includes:

- Model state dictionary
- Training metadata (loss, epochs, samples)
- Class label encoder
- Model configuration

## API Changes

### New Endpoint

- `POST /train_model`: Train CNN model from Google Sheets data

### Enhanced Endpoints

- `/auto_periods/`: Now uses trained CNN model for better validation
- Improved period confidence and classification accuracy

## Example Usage

```python
# Load training data
from backend.google_sheets import GoogleSheetsLoader
loader = GoogleSheetsLoader()
training_data = loader.extract_training_data()

# Train model
from backend.model_training import ModelTrainer
trainer = ModelTrainer()
result = trainer.train_from_google_sheets()

# Use trained model
from backend.model_training import load_trained_model
model, class_names = load_trained_model()
```

## Benefits

- **Real Data**: Train on actual astronomical data instead of synthetic examples
- **Modular Design**: Clean separation of concerns across modules
- **Scalable**: Easy to add new training data via Google Sheets
- **Automated**: Complete pipeline from data loading to model deployment
- **Flexible**: Support for various LC categories and period configurations
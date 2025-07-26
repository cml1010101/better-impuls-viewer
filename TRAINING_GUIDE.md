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
- **Column AN**: LC category (string)

### 3. Supported LC Categories

The system now supports enhanced classification types:

- **dipper**: Transit-like dips (exoplanet/eclipsing binary patterns)
- **distant peaks**: Double-peaked variables with significant separation
- **close peak**: Multiple peaks close together (close binary/pulsating)
- **sinusoidal**: Clean sinusoidal patterns (regular variables)
- **other**: Irregular or unclassified variability

Categories can include question marks (e.g., "dipper?") which are automatically normalized.

### 4. Sample Data Files

Ensure corresponding data files exist in the `sample_data` directory with naming format:
```
{star_number}-{telescope}.tbl
```

The system now includes sample data for 5 different stars showcasing all classification types:
- Stars 1: Dipper type (eclipsing/transiting objects)
- Stars 2: Distant peaks type (double-peaked variables) 
- Stars 3: Sinusoidal type (regular variables)
- Stars 4: Close peak type (close binary/pulsating)
- Stars 5: Other type (irregular/complex)

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
- Normalizes LC categories to standard classifications
- Loads corresponding time series data from sample_data

### 2. Data Preprocessing
- Phase-folds light curves at known periods
- Generates confidence labels based on period quality metrics and category
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

## Enhanced Classification System

### New Classification Types

The updated system provides more sophisticated classification:

1. **Eclipsing Objects** (`eclipsing_object`): Single dip patterns
2. **Eclipsing Binaries** (`eclipsing_binary`): Multiple period systems
3. **Regular Variables** (`regular_variable`): Clean sinusoidal patterns
4. **Double-Peaked** (`double_peaked`): Distant peak patterns
5. **Close Binaries** (`close_binary`): Close peak patterns
6. **Complex Multi-period** (`complex_multiperiod`): Multiple period systems
7. **Irregular** (`irregular`): Unclassified variability
8. **Uncertain** (`uncertain`): Low confidence detections

### Frontend Display

The frontend now displays all new classification types with color-coded badges:
- Green: Regular variables
- Red: Eclipsing objects
- Orange: Binary systems
- Blue: Double-peaked variables
- Purple: Complex systems
- Gray: Irregular/uncertain

## Architecture

### Backend Modules

1. **config.py**: Environment configuration and Google Sheets URL
2. **models.py**: Pydantic data models for training data
3. **data_processing.py**: Data preprocessing utilities
4. **period_detection.py**: Enhanced CNN with 5 classification types
5. **google_sheets.py**: Google Sheets data loader with category normalization
6. **model_training.py**: CNN training pipeline with real data support
7. **server.py**: FastAPI endpoints including `/train_model`

### Training Pipeline

```
Google Sheets → Data Extraction → Category Normalization → Time Series Loading → Phase Folding → CNN Training → Model Saving
```

## Model Output

The trained model is saved as `trained_cnn_model.pth` and includes:

- Model state dictionary
- Training metadata (loss, epochs, samples)
- Class label encoder
- Model configuration

## API Changes

### New Features

- **Enhanced Classifications**: 5 new classification types beyond basic regular/binary/other
- **Category Normalization**: Automatic mapping of Google Sheets categories to model classes
- **Improved Confidence**: Category-specific confidence scoring
- **Better Sample Data**: Realistic examples of each classification type

### Enhanced Endpoints

- `POST /train_model`: Train CNN model from Google Sheets data
- `/auto_periods/`: Now uses enhanced CNN model with new classification types
- Improved period confidence and classification accuracy

## Testing

### Sample Data Testing

```python
# Test with sample data
from backend.period_detection import determine_automatic_periods
from backend.google_sheets import GoogleSheetsLoader
import numpy as np

loader = GoogleSheetsLoader('https://dummy-url')
for star_num in [1, 2, 3, 4, 5]:
    time_series, flux_series = loader._load_star_data(star_num)
    data = np.column_stack([time_series, flux_series])
    result = determine_automatic_periods(data)
    print(f"Star {star_num}: {result['classification']['type']}")
```

### Training Pipeline Testing

```python
# Test complete training workflow
from backend.model_training import ModelTrainer
from backend.models import TrainingDataPoint

trainer = ModelTrainer()
# ... training code
```

## Benefits

- **Real Data**: Train on actual astronomical data instead of synthetic examples
- **Enhanced Classifications**: More detailed and accurate variability typing
- **Modular Design**: Clean separation of concerns across modules
- **Scalable**: Easy to add new training data via Google Sheets
- **Automated**: Complete pipeline from data loading to model deployment
- **Flexible**: Support for various LC categories and period configurations
- **Professional UI**: Color-coded classification display in frontend
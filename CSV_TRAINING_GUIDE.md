# CSV Training Data Input

## Overview

The Better Impuls Viewer now supports loading training data from CSV files in addition to Google Sheets. This provides a simpler alternative for users who want to provide their own training data without setting up Google Sheets API authentication.

## Installation

### Full Installation (Google Sheets + CSV)

```bash
pip install -r requirements.txt
```

### CSV-Only Installation (Minimal Dependencies)

If you only plan to use CSV input and don't need Google Sheets integration:

```bash
pip install -r requirements-csv-only.txt
```

This excludes the Google API dependencies (`gspread`, `google-auth`, `google-api-python-client`).

### Environment Variables

Add the CSV file path to your `.env` file:

```bash
# CSV Training Data Configuration
CSV_TRAINING_DATA_PATH=training_data.csv
```

If both `GOOGLE_SHEET_URL` and `CSV_TRAINING_DATA_PATH` are configured, the system will prefer Google Sheets by default. You can explicitly specify the data source when training models.

## CSV Format

### Required Columns

- **star_number**: Integer star identifier (must match data files in sample_data directory)
- **period_1**: Primary period in days (use -9 or leave empty for no valid period)
- **lc_category**: Light curve category (see supported categories below)

### Optional Columns

- **period_2**: Secondary period in days (use -9 or leave empty for no valid period)
- **sensor**: Sensor/instrument name (defaults to 'csv' if not provided)

### Example CSV

```csv
star_number,period_1,period_2,lc_category,sensor
1,2.5,-9,dipper,csv
2,7.8,-9,resolved distant peaks,csv
3,15.3,-9,sinusoidal,csv
4,5.2,-9,resolved close peaks,csv
5,12.7,-9,stochastic,csv
```

## Supported Light Curve Categories

The following categories are supported (case-insensitive):

- `sinusoidal` - Clean sinusoidal patterns (regular variables)
- `double dip` - Double-peaked patterns
- `shape changer` - Variables with evolving morphology
- `beater` - Beat frequency patterns
- `beater/complex peak` - Multi-frequency patterns
- `resolved close peaks` - Close binary systems
- `resolved distant peaks` - Separated double systems
- `eclipsing binaries` - Transit/eclipse systems
- `pulsator` - Multi-harmonic pulsations
- `burster` - Episodic outbursts
- `dipper` - YSO-like dipping patterns
- `co-rotating optically thin material` - Spotted stars
- `long term trend` - Secular evolution
- `stochastic` - Irregular/noise-dominated

## Usage

### Basic CSV Loading

```python
from google_sheets import CSVDataLoader

# Load training data from CSV
loader = CSVDataLoader("training_data.csv")
training_data = loader.extract_training_data()
```

### Model Training with CSV

```python
from model_training import ModelTrainer

# Train model using CSV data
trainer = ModelTrainer()
training_data = trainer.load_training_data(
    data_source="csv",
    csv_file_path="training_data.csv"
)
```

### Automatic Data Source Selection

```python
# Will use CSV if Google Sheets URL is not configured
trainer = ModelTrainer()
training_data = trainer.load_training_data(data_source="auto")
```

### Star Range Filtering

```python
# Extract specific stars
training_data = loader.extract_training_data(stars_to_extract="1:5")

# Extract specific star list
training_data = loader.extract_training_data(stars_to_extract=[1, 3, 5])
```

## Data File Requirements

For each star in your CSV, you must have corresponding data files in the `sample_data` directory with the format:

```
{star_number}-{telescope}.tbl
```

For example:
- `1-hubble.tbl`
- `2-kepler.tbl`
- `3-tess.tbl`

The system will automatically find and use the first available data file for each star.

## Command Line Usage

You can test CSV functionality using the provided test script:

```bash
python test_csv_functionality.py
```

## Migration from Google Sheets

If you're currently using Google Sheets and want to switch to CSV:

1. Export your Google Sheets data to CSV format
2. Ensure the CSV has the required columns: `star_number`, `period_1`, `lc_category`
3. Add optional columns: `period_2`, `sensor` as needed
4. Set `CSV_TRAINING_DATA_PATH` in your `.env` file
5. Use `data_source="csv"` when training models

## Error Handling

The CSV loader provides clear error messages for common issues:

- Missing required columns
- Invalid star numbers
- Missing data files
- Invalid period values
- Unknown light curve categories (defaults to 'stochastic')

## Performance

CSV loading is significantly faster than Google Sheets API calls and doesn't require internet connectivity or API authentication. It's recommended for:

- Local development
- Automated training pipelines
- Environments without Google API access
- Large datasets that exceed Google Sheets limits
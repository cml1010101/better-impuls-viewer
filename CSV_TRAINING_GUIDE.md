# CSV Training Data Input

## Overview

The Better Impuls Viewer supports loading training data from CSV files using a format that matches the original Google Sheets structure with multiple telescopes/sensors per star. This provides a comprehensive way to train models with data from multiple instruments.

## Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Add the CSV file path to your `.env` file:

```bash
# CSV Training Data Configuration
CSV_TRAINING_DATA_PATH=sample_training_data.csv
```

## CSV Format

### Column Structure (No Headers)

The CSV file should contain **no header row** and use the following hardcoded column positions that match the original Google Sheets format:

- **Column 0 (A)**: Star number
- **Columns 5-6 (F-G)**: CDIPS period 1 & 2
- **Columns 7-8 (H-I)**: ELEANOR period 1 & 2  
- **Columns 9-10 (J-K)**: QLP period 1 & 2
- **Columns 11-12 (L-M)**: SPOC period 1 & 2
- **Columns 13-14 (N-O)**: TESS 16 period 1 & 2
- **Columns 15-16 (P-Q)**: TASOC period 1 & 2
- **Columns 17-18 (R-S)**: TGLC period 1 & 2
- **Columns 19-20 (T-U)**: EVEREST period 1 & 2
- **Columns 21-22 (V-W)**: K2SC period 1 & 2
- **Columns 23-24 (X-Y)**: K2SFF period 1 & 2
- **Columns 25-26 (Z-AA)**: K2VARCAT period 1 & 2
- **Columns 27-28 (AB-AC)**: ZTF_R period 1 & 2
- **Columns 29-30 (AD-AE)**: ZTF_G period 1 & 2
- **Columns 31-32 (AF-AG)**: W1 period 1 & 2
- **Columns 33-34 (AH-AI)**: W2 period 1 & 2
- **Column 40**: LC category

### Period Values

- Use valid float numbers for detected periods (e.g., `2.5`, `15.3`)
- Use `-9` or `"no"` for no valid period detected
- Leave blank/empty for no data

### Example CSV (First Few Columns)

```csv
1,,,,,,2.5,-9,7.8,-9,15.3,-9,5.2,-9,12.7,-9,8.4,-9,6.1,-9,4.3,-9,9.6,-9,3.7,-9,11.2,-9,2.8,-9,14.5,-9,7.9,-9,5.8,-9,,,,,dipper
2,,,,,,3.1,-9,8.2,-9,16.7,-9,6.4,-9,13.1,-9,9.8,-9,7.2,-9,5.5,-9,10.3,-9,4.1,-9,12.6,-9,3.2,-9,15.8,-9,8.7,-9,6.9,-9,,,,,resolved distant peaks
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
from csv_data_loader import CSVDataLoader

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
# Will use CSV if  URL is not configured
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

## Migration from 

If you're currently using  and want to switch to CSV:

1. Export your  data to CSV format
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

CSV loading is significantly faster than  API calls and doesn't require internet connectivity or API authentication. It's recommended for:

- Local development
- Automated training pipelines
- Environments on local systems
- Large datasets that exceed  limits
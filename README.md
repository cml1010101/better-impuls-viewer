# Better Impuls Viewer

A modern web application for astronomical data analysis, providing interactive visualization of light curves, periodogram analysis, and phase folding capabilities.

![Dashboard Preview](https://github.com/user-attachments/assets/a95f9e76-4b76-44f8-a31e-b7854e7fad4c)

## 🚀 Features

- **Multi-Star Selection**: Choose from available astronomical targets (5 sample stars with different variability types)
- **Multi-Telescope Support**: Analyze data from Hubble, Kepler, and TESS
- **Campaign Management**: Automatically identifies and displays the 3 most massive campaigns per dataset
- **Interactive Light Curves**: Scatter plot visualization of time-series photometry
- **Lomb-Scargle Periodogram**: Frequency analysis with logarithmic period display
- **Phase Folding**: Manual period input with real-time phase-folded light curve generation
- **🆕 Enhanced Automatic Period Detection**: Dual-method approach combining traditional periodogram analysis with CNN validation
- **🆕 Advanced Variability Classification**: Intelligent classification into 14 astronomical categories:
  - Sinusoidal (regular variables)
  - Double dip (eclipsing systems)
  - Shape changer (morphology evolution)
  - Beater (beat frequency patterns)
  - Beater/complex peak (multi-frequency)
  - Resolved close peaks (close binary systems)
  - Resolved distant peaks (separated double systems)
  - Eclipsing binaries (transit systems)
  - Pulsator (multi-harmonic pulsations)
  - Burster (episodic outbursts)
  - Dipper (YSO-like dipping)
  - Co-rotating optically thin material (spotted stars)
  - Long term trend (secular evolution)
  - Stochastic (irregular/noise-dominated)
- **🆕 Enhanced CSV Integration**: Complete data pipeline supporting 15+ sensor types (CDIPS, ELEANOR, QLP, SPOC, TESS, TASOC, TGLC, EVEREST, K2SC, K2SFF, K2VARCAT, ZTF, WISE)
- **🆕 
- **🆕 5-Period Training Strategy**: Advanced ML training approach that generates:
  - 1-2 correct periods from catalog data (high confidence)
  - 2 incorrect periodogram peaks (medium confidence) 
  - 2 random periods (low confidence)
- **🆕 Robust CNN Architecture**: Multi-layer convolutional network for period validation and classification
- **🆕 Model Persistence**: Automatic model storage and loading - no need to retrain unnecessarily
- **🆕 CSV Export**: Export training data to CSV format for external analysis and development
- **Modern UI**: Responsive design with color-coded classification badges and intuitive controls

## 🛠 Technology Stack

### Web Application
- **React 19**: Modern UI framework
- **TypeScript**: Type-safe development  
- **Vite**: Fast build tool and dev server
- **FastAPI**: High-performance backend API
- **Node.js**: Development tooling and scripts

### Backend
- **FastAPI**: High-performance API framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **astropy**: Astronomical calculations (Lomb-Scargle periodogram)
- **PyTorch**: Deep learning framework for CNN period validation
- **scipy**: Scientific computing for peak detection and signal analysis
- **scikit-learn**: Machine learning utilities for training pipeline
- **python-dotenv**: Environment variable management
- **requests**: HTTP client for CSV integration
- **CORS Support**: Cross-origin requests for frontend communication

### Frontend
- **React 19**: Modern UI framework
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **Recharts**: Interactive chart library for data visualization
- **Custom CSS**: Beautiful gradient design with responsive layout

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Quick Start (Web App)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Build and run the web app
npm start
```

### Development Mode
```bash
# Install dependencies (if not done above)
pip install -r requirements.txt
npm install

# Start development mode with both frontend and backend
npm run dev
```

### Manual Setup (Backend and Frontend separately)

#### Backend Setup
```bash
cd backend
pip install -r ../requirements.txt
python server.py
```

#### Frontend Setup  
```bash
cd frontend
npm install
npm run dev
```

## 🚦 Usage

### Web App Usage
1. **Quick Start**: Run `npm start` to build and serve the web application
2. **Development**: Run `npm run dev` for development mode with frontend hot reloading and backend auto-restart
3. **Manual Setup**: Start backend with `npm run dev:backend` and frontend with `npm run dev:frontend`
4. **Testing**: Run `npm run test:webapp` to verify all components work

### Accessing the Application
- **Frontend**: Navigate to `http://localhost:5173` (dev) or `http://localhost:4173` (production preview)
- **Backend API**: `http://localhost:8000/docs` for API documentation

### Using the Dashboard

1. **Select Star**: Choose from the available star numbers (1, 2, 3)
2. **Choose Telescope**: Pick between Hubble, Kepler, or TESS instruments
3. **Pick Campaign**: Select from the 3 most massive observing campaigns
4. **View Data**: Observe the light curve in the top chart
5. **Analyze Periods**: Examine the periodogram for significant frequencies
6. **Phase Fold**: Enter a period value and click "Fold Data" to see the phase-folded light curve

## 📊 Data Processing

The application implements several astronomical data processing techniques:

- **Campaign Extraction**: Identifies continuous observing segments using time gap analysis
- **Outlier Removal**: Uses Interquartile Range (IQR) method to filter anomalous data points
- **Lomb-Scargle Periodogram**: Calculates frequency spectrum for unevenly sampled time series
- **Phase Folding**: Wraps time series data to a specified period for periodic signal analysis

### 🤖 Automatic Period Determination (NEW)

The system now includes intelligent period detection using two complementary methods:

#### 1. Enhanced Periodogram Analysis
- **Robust Peak Detection**: Uses median absolute deviation for noise-resistant thresholds
- **Period Weighting**: Prioritizes astronomically reasonable periods (0.5-50 days)
- **Harmonic Filtering**: Avoids spurious detections from high-frequency noise

#### 2. PyTorch Sinusoidal Regression
- **Deep Learning Approach**: Fits multiple sinusoidal components using gradient descent
- **Flexible Modeling**: Automatically determines amplitudes, periods, and phases
- **Early Stopping**: Prevents overfitting with patience-based convergence

#### 3. Intelligent Classification
- **Regular Variables**: Single dominant period systems
- **Binary Systems**: Multiple period detection with ratio analysis
- **Complex Objects**: Irregular or multi-component variability

**Example API Response:**
```json
{
  "primary_period": 2.361,
  "secondary_period": 10.303,
  "classification": {
    "type": "regular",
    "confidence": 0.88,
    "description": "Regular variable star with period 2.361 days"
  },
  "methods": {
    "periodogram": {"success": true, "periods": [...]},
    "torch_fitting": {"success": true, "periods": [...]}
  }
}
```

## 🎯 Sample Data

The repository includes simulated astronomical data with:
- 3 different stars with varying periodic signals
- 3 telescope datasets (Hubble, Kepler, TESS) per star
- 3 campaigns per telescope with different durations (30-90 days)
- Realistic noise and outlier characteristics

## 🔧 API Endpoints

- `GET /stars` - List available star numbers
- `GET /telescopes/{star_number}` - Get telescopes for a star
- `GET /campaigns/{star_number}/{telescope}` - Get top 3 campaigns
- `GET /data/{star_number}/{telescope}/{campaign_id}` - Get processed light curve data
- `GET /periodogram/{star_number}/{telescope}/{campaign_id}` - Get Lomb-Scargle periodogram
- `GET /phase_fold/{star_number}/{telescope}/{campaign_id}?period={period}` - Get phase-folded data
- `GET /auto_periods/{star_number}/{telescope}/{campaign_id}` - **Enhanced**: CNN-powered period detection and classification
- `GET /model_status` - **NEW**: Check trained model status and information
- `POST /train_model` - **NEW**: Train CNN model using CSV data with model persistence
- `POST /export_training_csv` - **NEW**: Export 

## 🔬 Machine Learning Features

### Automatic Period Detection
The system uses a sophisticated dual-method approach:

1. **Enhanced Lomb-Scargle Periodogram**: Improved peak detection using robust statistics and period weighting
2. **CNN Period Validation**: Convolutional neural network analyzes phase-folded light curves for pattern recognition

### Advanced Classification System
Objects are automatically classified into detailed categories:

- **🟢 Regular Variable**: Clean sinusoidal patterns with single dominant period
- **🔴 Eclipsing Object**: Transit-like dips indicating exoplanets or eclipsing systems  
- **🟠 Eclipsing Binary**: Multiple period systems with ratio analysis
- **🔵 Double-Peaked**: Variables with distant peak separation
- **🟤 Close Binary**: Close peak patterns indicating tight binary systems
- **🟣 Complex Multi-period**: Systems with multiple significant periods
- **⚫ Irregular**: Unclassified or chaotic variability
- **⚪ Uncertain**: Low confidence detections requiring manual review

### CSV Training Data Input
Train models using CSV files instead of CSV:

```bash
# Create CSV training data file
echo "star_number,period_1,period_2,lc_category,sensor" > training_data.csv
echo "1,2.5,-9,dipper,csv" >> training_data.csv
echo "2,7.8,-9,resolved distant peaks,csv" >> training_data.csv

# Set CSV path in .env
CSV_TRAINING_DATA_PATH=training_data.csv

# Train model with CSV data
curl -X POST http://localhost:8000/train_model \
  -H "Content-Type: application/json" \
  -d '{"data_source": "csv"}'

# Test CSV loading from command line
python backend/csv_data_loader.py --csv-input training_data.csv --stars "1:3"
```

**CSV Format Requirements:**
- `star_number`: Integer star identifier
- `period_1`: Primary period in days (-9 for no period)
- `lc_category`: Light curve category (see classification types above)
- `period_2`: Secondary period (optional)
- `sensor`: Sensor name (optional, defaults to 'csv')

See [CSV_TRAINING_GUIDE.md](CSV_TRAINING_GUIDE.md) for detailed documentation.

### Synthetic Training Data Generation
Generate realistic synthetic astronomical datasets for training and testing:

```bash
# Generate training dataset with synthetic light curves
python backend/generate_training_data.py --n-stars 100 --output-dir synthetic_training

# Customize parameters
python backend/generate_training_data.py \
  --n-stars 50 \
  --surveys hubble kepler tess \
  --max-days 100 \
  --noise-level 0.02 \
  --output-dir custom_dataset

# Use the generator directly in Python
python backend/generator.py --action generate-tbl --n-stars 20

# Demo the synthetic generator
python backend/generator.py --action demo
```

**Synthetic Data Features:**
- **14 Variability Classes**: Sinusoidal, eclipsing binaries, pulsators, bursters, dippers, and more
- **Realistic Noise**: Photometric errors based on flux levels and observational characteristics  
- **Multiple Surveys**: Generate data for different telescope/survey combinations
- **Period Diversity**: Logarithmic period distribution from 0.1 to 20 days
- **Metadata Export**: Includes CSV file with star catalog and classification information
- **.tbl Format**: Compatible with existing data pipeline and training code

**Supported Variability Classes:**
- Sinusoidal (stellar rotation/spots)
- Double dip (binary with two eclipses)
- Shape changer (spot evolution)
- Beater patterns (beat frequencies)
- Resolved close/distant peaks
- Eclipsing binaries
- Pulsating variables
- Bursters and dippers
- Co-rotating material
- Long-term trends
- Stochastic variability

### CSV Training Pipeline
Train the CNN model using real astronomical data:

```bash
# Set up your CSV URL in .env
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id/edit

# Check model status before training
curl http://localhost:8000/model_status

# Train the model with all available stars (skips training if model exists)
curl -X POST http://localhost:8000/train_model

# Force retrain even if model exists
curl -X POST http://localhost:8000/train_model \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": true}'

# Train with specific stars only
curl -X POST http://localhost:8000/train_model \
  -H "Content-Type: application/json" \
  -d '{"stars_to_extract": [1, 2, 3, 4, 5]}'
```

### Model Persistence Features
- **Automatic Model Storage**: Trained models are saved locally as `trained_cnn_model.pth`
- **Smart Loading**: System automatically uses existing trained models instead of retraining
- **Model Information**: Check model status, training metadata, and class information
- **Force Retrain**: Option to retrain models when needed for updates

### CSV Export for External Analysis
Export training data to CSV format for external analysis and model development:

```bash
# Export all training data to CSV
curl -X POST http://localhost:8000/export_training_csv

# Export specific stars to custom directory
curl -X POST http://localhost:8000/export_training_csv \
  -H "Content-Type: application/json" \
  -d '{"stars_to_extract": [1, 2, 3], "output_dir": "custom_dataset"}'
```

**CSV Output Features:**
- **Phase-Folded Data**: Each row contains phase and flux values for machine learning
- **Rich Metadata**: Includes star number, period, category, sensor, period type, and confidence
- **Complete Training Set**: Exports all 5 periods per light curve with realistic confidence scores
- **Analysis Ready**: CSV format suitable for pandas, R, or other analysis tools

### 🎯 Enhanced 5-Period Training Strategy

The system implements an advanced training methodology that generates **5 periods per light curve** to create a robust and balanced dataset:

#### Period Types Generated:
1. **Correct Periods (1-2 samples)**: 
   - Source: CSV catalog data (columns AK, AL for legacy; F-AI for multi-sensor)
   - Confidence: 0.85-0.95 (high)
   - Purpose: Teaches the CNN what genuine periods look like

2. **Incorrect Periodogram Peaks (2 samples)**:
   - Source: Lomb-Scargle periodogram peaks that are NOT close to correct periods  
   - Confidence: 0.3-0.6 (medium)
   - Purpose: Teaches the CNN to distinguish real periods from spurious peaks

3. **Random Periods (1-2 samples)**:
   - Source: Randomly generated within astronomical ranges (0.1-50 days)
   - Confidence: 0.05-0.25 (low)
   - Purpose: Provides negative examples for robust validation

#### Benefits:
- **Balanced Dataset**: Equal representation of good, questionable, and bad periods
- **Realistic Training**: CNN learns from the same types of false positives it will encounter in real analysis
- **Confidence Calibration**: Network learns to assign appropriate confidence scores
- **Robust Validation**: Significantly reduces false positive period detections

#### Multi-Sensor Support:
The system supports 15+ astronomical survey sensors:
- **TESS**: CDIPS, ELEANOR, QLP, SPOC, TESS-16, TASOC, TGLC
- **Kepler/K2**: EVEREST, K2SC, K2SFF, K2VARCAT  
- **Ground-based**: ZTF (R/G bands), WISE (W1/W2)

Each sensor contributes independent training samples, maximizing dataset diversity and model generalization.

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete documentation.

## 🎨 Architecture

```
better-impuls-viewer/
├── main.js                  # Electron main process (app orchestration)
├── package.json            # Electron app configuration and scripts
├── dev.js                  # Development environment launcher
├── test-backend.js         # Backend integration test
├── backend/
│   ├── server.py            # FastAPI application
│   ├── config.py           # Configuration and environment variables
│   ├── models.py           # Pydantic data models
│   ├── data_processing.py  # Data preprocessing utilities
│   ├── period_detection.py # CNN period validation and periodogram analysis
│   ├── csv_data_loader.py    # CSV integration
│   ├── model_training.py   # CNN training pipeline
│   ├── process.py          # Legacy data processing functions
│   ├── display.py          # Original matplotlib visualization
│   └── app.py             # Original command-line interface
├── frontend/
│   ├── src/
│   │   ├── Dashboard.tsx   # Main dashboard component
│   │   ├── App.tsx        # Root application component
│   │   └── *.css          # Styling files with classification colors
│   ├── dist/              # Built frontend (for Electron production)
│   └── package.json       # Frontend dependencies and scripts
├── sample_data/           # Enhanced sample datasets (5 star types)
├── ml-dataset/           # Machine learning dataset generation
├── TRAINING_GUIDE.md     # Comprehensive training documentation
└── README.md            # This file
```

### Electron App Workflow
1. **main.js** starts and manages the application lifecycle
2. Python **FastAPI backend** is spawned as a subprocess on port 8000
3. **React frontend** is served (dev server in development, built files in production)
4. Electron **BrowserWindow** displays the frontend with backend API integration
5. Both processes are managed and terminated together

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built upon existing astronomical data processing code
- Uses modern web technologies for enhanced user experience
- Implements industry-standard astronomical analysis techniques
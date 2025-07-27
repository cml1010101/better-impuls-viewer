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
- **🆕 Enhanced Google Sheets Integration**: Complete data pipeline supporting 15+ sensor types (CDIPS, ELEANOR, QLP, SPOC, TESS, TASOC, TGLC, EVEREST, K2SC, K2SFF, K2VARCAT, ZTF, WISE)
- **🆕 5-Period Training Strategy**: Advanced ML training approach that generates:
  - 1-2 correct periods from catalog data (high confidence)
  - 2 incorrect periodogram peaks (medium confidence) 
  - 2 random periods (low confidence)
- **🆕 Robust CNN Architecture**: Multi-layer convolutional network for period validation and classification
- **Modern UI**: Responsive design with color-coded classification badges and intuitive controls

## 🛠 Technology Stack

### Backend
- **FastAPI**: High-performance API framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **astropy**: Astronomical calculations (Lomb-Scargle periodogram)
- **PyTorch**: Deep learning framework for CNN period validation
- **scipy**: Scientific computing for peak detection and signal analysis
- **scikit-learn**: Machine learning utilities for training pipeline
- **python-dotenv**: Environment variable management
- **requests**: HTTP client for Google Sheets integration
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

### Backend Setup
```bash
cd backend
pip install fastapi uvicorn pandas numpy astropy torch scipy scikit-learn python-dotenv requests pydantic
python server.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🚦 Usage

1. **Start the Backend**: Run `python server.py` in the `backend` directory (serves on port 8000)
2. **Start the Frontend**: Run `npm run dev` in the `frontend` directory (serves on port 5173)
3. **Open Browser**: Navigate to `http://localhost:5173`

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
- `POST /train_model` - **NEW**: Train CNN model using Google Sheets data

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

### Google Sheets Training Pipeline
Train the CNN model using real astronomical data:

```bash
# Set up your Google Sheets URL in .env
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/your-sheet-id/edit

# Train the model with all available stars
curl -X POST http://localhost:8000/train_model

# Train with specific stars only
curl -X POST http://localhost:8000/train_model \
  -H "Content-Type: application/json" \
  -d '{"stars_to_extract": [1, 2, 3, 4, 5]}'
```

### 🎯 Enhanced 5-Period Training Strategy

The system implements an advanced training methodology that generates **5 periods per light curve** to create a robust and balanced dataset:

#### Period Types Generated:
1. **Correct Periods (1-2 samples)**: 
   - Source: Google Sheets catalog data (columns AK, AL for legacy; F-AI for multi-sensor)
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
├── backend/
│   ├── server.py            # FastAPI application
│   ├── config.py           # Configuration and environment variables
│   ├── models.py           # Pydantic data models
│   ├── data_processing.py  # Data preprocessing utilities
│   ├── period_detection.py # CNN period validation and periodogram analysis
│   ├── google_sheets.py    # Google Sheets integration
│   ├── model_training.py   # CNN training pipeline
│   ├── process.py          # Legacy data processing functions
│   ├── display.py          # Original matplotlib visualization
│   └── app.py             # Original command-line interface
├── frontend/
│   ├── src/
│   │   ├── Dashboard.tsx   # Main dashboard component
│   │   ├── App.tsx        # Root application component
│   │   └── *.css          # Styling files with classification colors
│   └── package.json       # Dependencies and scripts
├── sample_data/           # Enhanced sample datasets (5 star types)
├── ml-dataset/           # Machine learning dataset generation
├── TRAINING_GUIDE.md     # Comprehensive training documentation
└── README.md            # This file
```

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
# Better Impuls Viewer

A modern web application for astronomical data analysis, providing interactive visualization of light curves, periodogram analysis, and phase folding capabilities.

![Dashboard Preview](https://github.com/user-attachments/assets/a95f9e76-4b76-44f8-a31e-b7854e7fad4c)

## 🚀 Features

- **Multi-Star Selection**: Choose from available astronomical targets
- **Multi-Telescope Support**: Analyze data from Hubble, Kepler, and TESS
- **Campaign Management**: Automatically identifies and displays the 3 most massive campaigns per dataset
- **Interactive Light Curves**: Scatter plot visualization of time-series photometry
- **Lomb-Scargle Periodogram**: Frequency analysis with logarithmic period display
- **Phase Folding**: Manual period input with real-time phase-folded light curve generation
- **Modern UI**: Responsive design with gradient backgrounds and intuitive controls

## 🛠 Technology Stack

### Backend
- **FastAPI**: High-performance API framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **astropy**: Astronomical calculations (Lomb-Scargle periodogram)
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
pip install fastapi uvicorn pandas matplotlib astropy
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

## 🎨 Architecture

```
better-impuls-viewer/
├── backend/
│   ├── server.py          # FastAPI application
│   ├── process.py         # Data processing functions
│   ├── display.py         # Original matplotlib visualization
│   └── app.py            # Original command-line interface
├── frontend/
│   ├── src/
│   │   ├── Dashboard.tsx  # Main dashboard component
│   │   ├── App.tsx       # Root application component
│   │   └── *.css         # Styling files
│   └── package.json      # Dependencies and scripts
├── sample_data/          # Generated astronomical datasets
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
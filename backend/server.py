from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import pandas as pd
import numpy as np
from process import find_longest_x_campaign, sort_data, calculate_lomb_scargle, remove_y_outliers

app = FastAPI(title="Better Impuls Viewer API", version="1.0.0")

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and other common dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class CampaignInfo(BaseModel):
    campaign_id: str
    telescope: str
    star_number: int
    data_points: int
    duration: float

class ProcessedData(BaseModel):
    time: List[float]
    flux: List[float]
    error: List[float]

class PeriodogramData(BaseModel):
    periods: List[float]
    powers: List[float]

class PhaseFoldedData(BaseModel):
    phase: List[float]
    flux: List[float]

# Configuration
DEFAULT_DATA_FOLDER = '/home/runner/work/better-impuls-viewer/better-impuls-viewer/sample_data'

def get_data_folder():
    """Get the data folder path"""
    return DEFAULT_DATA_FOLDER

def load_data_file(filepath: str) -> np.ndarray:
    """Load data from a .tbl file"""
    try:
        data = pd.read_table(filepath, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        data_array = data.to_numpy()
        
        # If data has more than 2 columns, use only the first 2 (time, flux)
        if data_array.shape[1] > 2:
            return data_array[:, :2]
        
        return data_array
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data file: {str(e)}")

def get_campaigns_for_star_telescope(star_number: int, telescope: str) -> List[CampaignInfo]:
    """Get all campaigns for a specific star and telescope"""
    folder = get_data_folder()
    all_files = os.listdir(folder)
    
    campaigns = []
    pattern = f"{star_number}-{telescope}"
    
    for filename in all_files:
        if filename.startswith(pattern) and filename.endswith('.tbl'):
            filepath = os.path.join(folder, filename)
            
            # Extract campaign info from filename
            parts = filename.replace('.tbl', '').split('-')
            if len(parts) >= 3:
                campaign_id = parts[2]  # e.g., 'c1', 'c2', etc.
                
                # Load data to get info
                try:
                    data = load_data_file(filepath)
                    duration = data[-1, 0] - data[0, 0] if len(data) > 0 else 0
                    
                    campaigns.append(CampaignInfo(
                        campaign_id=campaign_id,
                        telescope=telescope,
                        star_number=star_number,
                        data_points=len(data),
                        duration=duration
                    ))
                except Exception:
                    continue  # Skip files that can't be loaded
    
    # Sort by number of data points (largest first) to get "most massive" campaigns
    campaigns.sort(key=lambda x: x.data_points, reverse=True)
    return campaigns[:3]  # Return top 3 most massive

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Better Impuls Viewer API"}

@app.get("/stars")
async def get_available_stars() -> List[int]:
    """Get list of available star numbers"""
    folder = get_data_folder()
    if not os.path.exists(folder):
        return []
    
    all_files = os.listdir(folder)
    stars = set()
    
    for filename in all_files:
        if filename.endswith('.tbl'):
            try:
                star_num = int(filename.split('-')[0])
                stars.add(star_num)
            except (ValueError, IndexError):
                continue
    
    return sorted(list(stars))

@app.get("/telescopes/{star_number}")
async def get_telescopes_for_star(star_number: int) -> List[str]:
    """Get available telescopes/sensors for a specific star"""
    folder = get_data_folder()
    if not os.path.exists(folder):
        return []
    
    all_files = os.listdir(folder)
    telescopes = set()
    
    for filename in all_files:
        if filename.startswith(f"{star_number}-") and filename.endswith('.tbl'):
            try:
                parts = filename.split('-')
                if len(parts) >= 2:
                    telescope = parts[1].split('.')[0].replace('c1', '').replace('c2', '').replace('c3', '')
                    if telescope:
                        telescopes.add(telescope)
            except (ValueError, IndexError):
                continue
    
    return sorted(list(telescopes))

@app.get("/campaigns/{star_number}/{telescope}")
async def get_campaigns(star_number: int, telescope: str) -> List[CampaignInfo]:
    """Get the 3 most massive campaigns for a star and telescope"""
    return get_campaigns_for_star_telescope(star_number, telescope)

@app.get("/data/{star_number}/{telescope}/{campaign_id}")
async def get_campaign_data(star_number: int, telescope: str, campaign_id: str) -> ProcessedData:
    """Get processed data for a specific campaign"""
    folder = get_data_folder()
    filename = f"{star_number}-{telescope}-{campaign_id}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load raw data with all columns
        raw_data = pd.read_table(filepath, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        raw_array = raw_data.to_numpy()
        
        # For processing, use only first 2 columns (time, flux)
        data_for_processing = raw_array[:, :2]
        
        # Find longest campaign
        data_for_processing = find_longest_x_campaign(data_for_processing, 0.1)
        
        # Remove outliers
        data_for_processing = remove_y_outliers(data_for_processing)
        
        # Sort data
        data_for_processing = sort_data(data_for_processing)
        
        # Extract error column if available
        if raw_array.shape[1] > 2:
            # Create a mapping from original indices to processed indices
            # For simplicity, use a default error for processed data
            error_values = [0.005] * len(data_for_processing)
        else:
            error_values = [0.005] * len(data_for_processing)
        
        return ProcessedData(
            time=data_for_processing[:, 0].tolist(),
            flux=data_for_processing[:, 1].tolist(),
            error=error_values
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/periodogram/{star_number}/{telescope}/{campaign_id}")
async def get_periodogram(star_number: int, telescope: str, campaign_id: str) -> PeriodogramData:
    """Get Lomb-Scargle periodogram for a specific campaign"""
    folder = get_data_folder()
    filename = f"{star_number}-{telescope}-{campaign_id}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load and process data
        data = load_data_file(filepath)
        data = find_longest_x_campaign(data, 0.1)
        data = remove_y_outliers(data)
        data = sort_data(data)
        
        # Calculate periodogram
        frequencies, powers = calculate_lomb_scargle(data)
        
        # Convert frequencies to periods
        periods = np.zeros_like(frequencies)
        non_zero_freq_indices = frequencies != 0
        periods[non_zero_freq_indices] = 1.0 / frequencies[non_zero_freq_indices]
        
        # Filter valid periods
        valid_period_indices = (periods > 0.1) & (periods < 20) & np.isfinite(periods)
        periods = periods[valid_period_indices]
        powers = powers[valid_period_indices]
        
        return PeriodogramData(
            periods=periods.tolist(),
            powers=powers.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating periodogram: {str(e)}")

@app.get("/phase_fold/{star_number}/{telescope}/{campaign_id}")
async def get_phase_folded_data(
    star_number: int, 
    telescope: str, 
    campaign_id: str, 
    period: float
) -> PhaseFoldedData:
    """Get phase-folded data for a specific campaign and period"""
    if period <= 0 or not np.isfinite(period):
        raise HTTPException(status_code=400, detail="Period must be positive and finite")
    
    folder = get_data_folder()
    filename = f"{star_number}-{telescope}-{campaign_id}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load and process data
        data = load_data_file(filepath)
        data = find_longest_x_campaign(data, 0.1)
        data = remove_y_outliers(data)
        data = sort_data(data)
        
        # Calculate phase
        phase = (data[:, 0] % period) / period
        
        return PhaseFoldedData(
            phase=phase.tolist(),
            flux=data[:, 1].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating phase-folded data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
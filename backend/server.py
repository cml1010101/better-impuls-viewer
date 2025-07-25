from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import pandas as pd
import numpy as np
from process import find_all_campaigns, sort_data, calculate_lomb_scargle, remove_y_outliers

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
DEFAULT_DATA_FOLDER = '~/Documents/impuls-data'
DEFAULT_DATA_FOLDER = os.path.expanduser(DEFAULT_DATA_FOLDER)

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
    """Get the top 3 campaigns for a specific star and telescope from a single data file"""
    folder = get_data_folder()
    filename = f"{star_number}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        return []
    
    try:
        # Load data from the single file
        data = load_data_file(filepath)
        data = sort_data(data)
        
        # Find all campaigns in the data using time threshold
        # Use 10 day threshold to identify gaps between campaigns (since campaigns are separated by 120 days)
        campaigns_data = find_all_campaigns(data, 1.0)
        
        campaigns = []
        for i, campaign_data in enumerate(campaigns_data):
            # Remove outliers from this campaign
            clean_campaign_data = remove_y_outliers(campaign_data)
            
            if len(clean_campaign_data) > 10:  # Only include campaigns with reasonable amount of data
                duration = clean_campaign_data[-1, 0] - clean_campaign_data[0, 0] if len(clean_campaign_data) > 0 else 0
                
                campaigns.append(CampaignInfo(
                    campaign_id=f"c{i+1}",
                    telescope=telescope,
                    star_number=star_number,
                    data_points=len(clean_campaign_data),
                    duration=duration
                ))
        
        # Sort by number of data points (largest first) to get "most massive" campaigns
        campaigns.sort(key=lambda x: x.data_points, reverse=True)
        return campaigns[:3]  # Return top 3 most massive
        
    except Exception as e:
        print(f"Error processing campaigns for {star_number}-{telescope}: {str(e)}")
        return []

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
                    telescope = parts[1].replace('.tbl', '')
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
    filename = f"{star_number}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load raw data with all columns
        raw_data = pd.read_table(filepath, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        raw_array = raw_data.to_numpy()
        
        # For processing, use only first 2 columns (time, flux)
        data_for_processing = raw_array[:, :2]
        data_for_processing = sort_data(data_for_processing)
        
        # Find all campaigns
        campaigns_data = find_all_campaigns(data_for_processing, 10.0)
        
        # Extract campaign index from campaign_id (e.g., "c1" -> 0, "c2" -> 1)
        try:
            campaign_index = int(campaign_id.replace('c', '')) - 1
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid campaign ID: {campaign_id}")
        
        if campaign_index < 0 or campaign_index >= len(campaigns_data):
            raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
        
        # Get the specific campaign data
        campaign_data = campaigns_data[campaign_index]
        
        # Remove outliers
        campaign_data = remove_y_outliers(campaign_data)
        
        # Sort data
        campaign_data = sort_data(campaign_data)
        
        # Extract error column if available
        if raw_array.shape[1] > 2:
            # Create a mapping from original indices to processed indices
            # For simplicity, use a default error for processed data
            error_values = [0.005] * len(campaign_data)
        else:
            error_values = [0.005] * len(campaign_data)
        
        return ProcessedData(
            time=campaign_data[:, 0].tolist(),
            flux=campaign_data[:, 1].tolist(),
            error=error_values
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/periodogram/{star_number}/{telescope}/{campaign_id}")
async def get_periodogram(star_number: int, telescope: str, campaign_id: str) -> PeriodogramData:
    """Get Lomb-Scargle periodogram for a specific campaign"""
    folder = get_data_folder()
    filename = f"{star_number}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load and process data
        data = load_data_file(filepath)
        data = sort_data(data)
        
        # Find all campaigns
        campaigns_data = find_all_campaigns(data, 1.0)
        
        # Extract campaign index from campaign_id
        try:
            campaign_index = int(campaign_id.replace('c', '')) - 1
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid campaign ID: {campaign_id}")
        
        if campaign_index < 0 or campaign_index >= len(campaigns_data):
            raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
        
        # Get the specific campaign data
        campaign_data = campaigns_data[campaign_index]
        campaign_data = remove_y_outliers(campaign_data)
        campaign_data = sort_data(campaign_data)
        
        # Calculate periodogram
        frequencies, powers = calculate_lomb_scargle(campaign_data)
        
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
    filename = f"{star_number}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load and process data
        data = load_data_file(filepath)
        data = sort_data(data)
        
        # Find all campaigns
        campaigns_data = find_all_campaigns(data, 1.0)
        
        # Extract campaign index from campaign_id
        try:
            campaign_index = int(campaign_id.replace('c', '')) - 1
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid campaign ID: {campaign_id}")
        
        if campaign_index < 0 or campaign_index >= len(campaigns_data):
            raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
        
        # Get the specific campaign data
        campaign_data = campaigns_data[campaign_index]
        campaign_data = remove_y_outliers(campaign_data)
        campaign_data = sort_data(campaign_data)
        
        # Calculate phase
        phase = (campaign_data[:, 0] % period) / period
        
        return PhaseFoldedData(
            phase=phase.tolist(),
            flux=campaign_data[:, 1].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating phase-folded data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
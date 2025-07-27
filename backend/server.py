from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from typing import List, Dict, Any, Optional
import os
import pandas as pd
import numpy as np
from functools import lru_cache
import hashlib
import time
from pydantic import BaseModel

# Import from our modular structure
from config import Config, DATA_DIR
from models import (
    CampaignInfo, ProcessedData, PeriodogramData, PhaseFoldedData, 
    AutoPeriodsData, ModelTrainingResult
)
from data_processing import (
    find_all_campaigns, sort_data, remove_y_outliers, 
    load_star_data_file, calculate_data_statistics
)
from period_detection import calculate_lomb_scargle, determine_automatic_periods
from model_training import ModelTrainer
from credentials_manager import get_credentials_manager
from google_oauth import get_oauth_manager

app = FastAPI(title="Better Impuls Viewer API", version="1.0.0")

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests
class CredentialsRequest(BaseModel):
    google_sheets_url: Optional[str] = None
    sed_url: Optional[str] = None
    sed_username: Optional[str] = None
    sed_password: Optional[str] = None

class ModelTrainingRequest(BaseModel):
    stars_to_extract: Optional[List[int]] = None
    force_retrain: bool = False

# Cache for processed data to improve performance
_file_cache = {}
_campaigns_cache = {}
_periodogram_cache = {}
_processed_data_cache = {}

def get_file_hash(filepath: str) -> str:
    """Get a hash of the file for caching purposes"""
    try:
        # Use file modification time and size for a quick hash
        stat = os.stat(filepath)
        content = f"{filepath}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    except:
        return hashlib.md5(filepath.encode()).hexdigest()

def get_data_folder():
    """Get the data folder path"""
    return DATA_DIR

def load_data_file(filepath: str) -> np.ndarray:
    """Load data from a .tbl file with caching"""
    file_hash = get_file_hash(filepath)
    
    if file_hash in _file_cache:
        return _file_cache[file_hash]
    
    try:
        data = pd.read_table(filepath, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        data_array = data.to_numpy()
        
        # If data has more than 2 columns, use only the first 2 (time, flux)
        if data_array.shape[1] > 2:
            result = data_array[:, :2]
        else:
            result = data_array
        
        # Cache the result
        _file_cache[file_hash] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data file: {str(e)}")

def get_campaigns_from_data(data: np.ndarray, threshold: float = None) -> List[np.ndarray]:
    """Get campaigns from data with caching. Uses dynamic threshold if none provided."""
    # Create a cache key based on data hash and threshold
    threshold_str = str(threshold) if threshold is not None else "dynamic"
    data_hash = hashlib.md5(f"{data.tobytes()}_{threshold_str}".encode()).hexdigest()
    
    if data_hash in _campaigns_cache:
        return _campaigns_cache[data_hash]
    
    # Sort data first
    sorted_data = sort_data(data)
    
    # Find all campaigns (will use dynamic threshold if threshold is None)
    campaigns_data = find_all_campaigns(sorted_data, threshold)
    
    # Cache the result
    _campaigns_cache[data_hash] = campaigns_data
    return campaigns_data

def get_campaigns_for_star_telescope(star_number: int, telescope: str) -> List[CampaignInfo]:
    """Get the top 3 campaigns for a specific star and telescope from a single data file with caching"""
    cache_key = f"campaigns_{star_number}_{telescope}"
    
    # Check if we already have this in cache
    if cache_key in _processed_data_cache:
        cache_entry = _processed_data_cache[cache_key]
        # Check if cache is still valid (file hasn't changed)
        folder = get_data_folder()
        filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            current_hash = get_file_hash(filepath)
            if cache_entry.get('file_hash') == current_hash:
                return cache_entry['campaigns']
    
    folder = get_data_folder()
    filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        return []
    
    try:
        # Load data from the single file (cached)
        data = load_data_file(filepath)
        
        # Find all campaigns in the data using dynamic threshold
        campaigns_data = get_campaigns_from_data(data, None)
        
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
        result = campaigns[:3]  # Return top 3 most massive
        
        # Cache the result
        _processed_data_cache[cache_key] = {
            'campaigns': result,
            'file_hash': get_file_hash(filepath),
            'timestamp': time.time()
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing campaigns for {star_number}-{telescope}: {str(e)}")
        return []

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Better Impuls Viewer API"}

# === Credentials Management Endpoints ===

@app.get("/credentials/status")
async def get_credentials_status():
    """Get status of all credential types"""
    credentials_manager = get_credentials_manager()
    oauth_manager = get_oauth_manager()
    
    status = credentials_manager.get_credentials_status()
    oauth_status = oauth_manager.get_oauth_status()
    
    return {
        "google_sheets": {
            "url_configured": status['google_sheets_url_set'],
            "authenticated": oauth_status['authenticated'],
            "user_info": oauth_status.get('user_info'),
            "has_refresh_token": oauth_status['has_refresh_token']
        },
        "sed_service": {
            "configured": status['sed_service']
        }
    }

@app.post("/credentials/configure")
async def configure_credentials(request: CredentialsRequest):
    """Configure application credentials"""
    credentials_manager = get_credentials_manager()
    
    try:
        # Update Google Sheets URL if provided
        if request.google_sheets_url:
            credentials_manager.set_google_sheets_url(request.google_sheets_url)
        
        # Update SED credentials if provided
        if request.sed_url and request.sed_username and request.sed_password:
            credentials_manager.set_sed_credentials(
                request.sed_url, 
                request.sed_username, 
                request.sed_password
            )
        
        return {"success": True, "message": "Credentials updated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating credentials: {str(e)}")

@app.get("/oauth/google/authorize")
async def get_google_auth_url():
    """Get Google OAuth authorization URL"""
    oauth_manager = get_oauth_manager()
    
    try:
        auth_url = oauth_manager.get_authorization_url()
        return {"authorization_url": auth_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating authorization URL: {str(e)}")

@app.get("/oauth/google/callback")
async def google_oauth_callback(request: Request):
    """Handle Google OAuth callback"""
    oauth_manager = get_oauth_manager()
    
    # Get authorization code from query parameters
    query_params = dict(request.query_params)
    code = query_params.get('code')
    error = query_params.get('error')
    
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")
    
    try:
        # Exchange code for tokens
        token_data = oauth_manager.exchange_code_for_tokens(code)
        
        # Get user info
        user_info = oauth_manager.get_user_info()
        
        # Return a simple success page or redirect to frontend
        return RedirectResponse(
            url=f"/?oauth_success=true&user_email={user_info.get('email', '') if user_info else ''}",
            status_code=302
        )
    
    except Exception as e:
        return RedirectResponse(
            url=f"/?oauth_error={str(e)}",
            status_code=302
        )

@app.post("/oauth/google/revoke")
async def revoke_google_auth():
    """Revoke Google OAuth authentication"""
    oauth_manager = get_oauth_manager()
    
    try:
        success = oauth_manager.revoke_authentication()
        return {"success": success, "message": "Authentication revoked successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error revoking authentication: {str(e)}")

@app.get("/oauth/google/status")
async def get_google_oauth_status():
    """Get current Google OAuth status"""
    oauth_manager = get_oauth_manager()
    return oauth_manager.get_oauth_status()

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics for monitoring"""
    return {
        "file_cache_size": len(_file_cache),
        "campaigns_cache_size": len(_campaigns_cache),
        "periodogram_cache_size": len(_periodogram_cache),
        "processed_data_cache_size": len(_processed_data_cache)
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    global _file_cache, _campaigns_cache, _periodogram_cache, _processed_data_cache
    _file_cache.clear()
    _campaigns_cache.clear()
    _periodogram_cache.clear()
    _processed_data_cache.clear()
    return {"message": "All caches cleared successfully"}

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
        if filename.startswith(f"{str(star_number).zfill(3)}-") and filename.endswith('.tbl'):
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
    """Get processed data for a specific campaign with caching"""
    cache_key = f"data_{star_number}_{telescope}_{campaign_id}"
    
    # Check cache first
    if cache_key in _processed_data_cache:
        cache_entry = _processed_data_cache[cache_key]
        # Check if cache is still valid
        folder = get_data_folder()
        filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            current_hash = get_file_hash(filepath)
            if cache_entry.get('file_hash') == current_hash:
                return cache_entry['data']
    
    folder = get_data_folder()
    filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load raw data with all columns
        raw_data = pd.read_table(filepath, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        raw_array = raw_data.to_numpy()
        
        # For processing, use only first 2 columns (time, flux)
        data_for_processing = raw_array[:, :2]
        
        # Use cached campaigns function with same threshold as campaigns endpoint
        campaigns_data = get_campaigns_from_data(data_for_processing, None)
        
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
        
        result = ProcessedData(
            time=campaign_data[:, 0].tolist(),
            flux=campaign_data[:, 1].tolist(),
            error=error_values
        )
        
        # Cache the result
        _processed_data_cache[cache_key] = {
            'data': result,
            'file_hash': get_file_hash(filepath),
            'timestamp': time.time()
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/periodogram/{star_number}/{telescope}/{campaign_id}")
async def get_periodogram(star_number: int, telescope: str, campaign_id: str) -> PeriodogramData:
    """Get Lomb-Scargle periodogram for a specific campaign with caching"""
    cache_key = f"periodogram_{star_number}_{telescope}_{campaign_id}"
    
    # Check cache first
    if cache_key in _periodogram_cache:
        cache_entry = _periodogram_cache[cache_key]
        # Check if cache is still valid
        folder = get_data_folder()
        filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            current_hash = get_file_hash(filepath)
            if cache_entry.get('file_hash') == current_hash:
                return cache_entry['data']
    
    folder = get_data_folder()
    filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load and process data (cached)
        data = load_data_file(filepath)
        
        # Find all campaigns (cached) - use same threshold as campaigns endpoint
        campaigns_data = get_campaigns_from_data(data, None)
        
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
        
        result = PeriodogramData(
            periods=periods.tolist(),
            powers=powers.tolist()
        )
        
        # Cache the result
        _periodogram_cache[cache_key] = {
            'data': result,
            'file_hash': get_file_hash(filepath),
            'timestamp': time.time()
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating periodogram: {str(e)}")

@app.get("/phase_fold/{star_number}/{telescope}/{campaign_id}")
async def get_phase_folded_data(
    star_number: int, 
    telescope: str, 
    campaign_id: str, 
    period: float
) -> PhaseFoldedData:
    """Get phase-folded data for a specific campaign and period using cached data"""
    if period <= 0 or not np.isfinite(period):
        raise HTTPException(status_code=400, detail="Period must be positive and finite")
    
    folder = get_data_folder()
    filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load and process data (cached)
        data = load_data_file(filepath)
        
        # Find all campaigns (cached) - use same threshold as campaigns endpoint
        campaigns_data = get_campaigns_from_data(data, None)
        
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

@app.get("/auto_periods/{star_number}/{telescope}/{campaign_id}")
async def get_automatic_periods(star_number: int, telescope: str, campaign_id: str) -> AutoPeriodsData:
    """Get automatically determined periods for a specific campaign using multiple methods"""
    cache_key = f"auto_periods_{star_number}_{telescope}_{campaign_id}"
    
    # Check cache first
    if cache_key in _processed_data_cache:
        cache_entry = _processed_data_cache[cache_key]
        # Check if cache is still valid
        folder = get_data_folder()
        filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            current_hash = get_file_hash(filepath)
            if cache_entry.get('file_hash') == current_hash:
                return cache_entry['data']
    
    folder = get_data_folder()
    filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Data file not found: {filename}")
    
    try:
        # Load and process data (cached)
        data = load_data_file(filepath)
        
        # Find all campaigns (cached) - use same threshold as campaigns endpoint
        campaigns_data = get_campaigns_from_data(data, None)
        
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
        
        # Perform automatic period determination
        period_analysis = determine_automatic_periods(campaign_data)
        
        result = AutoPeriodsData(
            primary_period=period_analysis.get("primary_period"),
            secondary_period=period_analysis.get("secondary_period"),
            classification=period_analysis.get("classification", {}),
            methods=period_analysis.get("methods", {}),
            error=period_analysis.get("error")
        )
        
        # Cache the result
        _processed_data_cache[cache_key] = {
            'data': result,
            'file_hash': get_file_hash(filepath),
            'timestamp': time.time()
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error determining automatic periods: {str(e)}")

from fastapi.responses import Response
import requests

@app.get("/model_status")
async def get_model_status() -> Dict[str, Any]:
    """
    Get information about the current trained model status.
    
    Returns model information if a trained model exists, or status if no model is available.
    """
    try:
        from model_training import get_model_info, model_exists
        
        if model_exists():
            model_info = get_model_info()
            if model_info:
                return {
                    "model_available": True,
                    "model_info": model_info
                }
        
        return {
            "model_available": False,
            "message": "No trained model found. Use /train_model to train a new model.",
            "model_path": Config.MODEL_SAVE_PATH
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")


@app.post("/train_model")
async def train_model_from_sheets(request: ModelTrainingRequest) -> ModelTrainingResult:
    """
    Train the CNN model using data from Google Sheets with enhanced 5-period strategy.
    
    Parameters:
    - stars_to_extract: Optional list of star numbers to include in training. 
                       If None, all available stars will be used.
    - force_retrain: If True, trains a new model even if one already exists.
                    If False, returns existing model info if available.
    
    The enhanced system generates 5 training samples per light curve:
    - 1-2 correct periods (high confidence)
    - 2 periodogram peaks that are not correct (medium confidence) 
    - 2 random periods (low confidence)
    """
    try:
        credentials_manager = get_credentials_manager()
        oauth_manager = get_oauth_manager()
        
        # Check authentication
        if not oauth_manager.is_authenticated():
            raise HTTPException(
                status_code=401, 
                detail="Google Sheets authentication required. Please authenticate in app settings."
            )
        
        # Check if Google Sheets URL is configured
        if not credentials_manager.get_google_sheets_url():
            raise HTTPException(
                status_code=400, 
                detail="Google Sheets URL not configured. Please configure in app settings."
            )
        
        from model_training import model_exists, get_model_info
        
        # Check if model already exists and force_retrain is False
        if not request.force_retrain and model_exists():
            model_info = get_model_info()
            if model_info:
                return ModelTrainingResult(
                    success=True,
                    epochs_trained=model_info['training_metadata'].get('epochs_trained', 0),
                    final_loss=model_info['training_metadata'].get('final_loss', 0.0),
                    model_path=model_info['model_path'],
                    training_samples=model_info['training_metadata'].get('training_samples', 0)
                )
        
        trainer = ModelTrainer()
        result = trainer.train_from_google_sheets(request.stars_to_extract)
        
        if not result.success:
            raise HTTPException(status_code=500, detail="Model training failed")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.post("/export_training_csv")
async def export_training_data_to_csv(request: ModelTrainingRequest) -> Dict[str, Any]:
    """
    Export training data from Google Sheets to CSV format for external analysis.
    
    Creates CSV files containing phase-folded light curves with corresponding
    category and confidence information suitable for machine learning training.
    
    Parameters:
    - stars_to_extract: Optional list of star numbers to include in export. 
                       If None, all available stars will be used.
    
    Returns:
    - Dictionary with export summary information
    """
    try:
        credentials_manager = get_credentials_manager()
        oauth_manager = get_oauth_manager()
        
        # Check authentication
        if not oauth_manager.is_authenticated():
            raise HTTPException(
                status_code=401, 
                detail="Google Sheets authentication required. Please authenticate in app settings."
            )
        
        # Check if Google Sheets URL is configured
        if not credentials_manager.get_google_sheets_url():
            raise HTTPException(
                status_code=400, 
                detail="Google Sheets URL not configured. Please configure in app settings."
            )
        
        from google_sheets import GoogleSheetsLoader
        
        loader = GoogleSheetsLoader()
        csv_path = loader.export_training_data_to_csv(
            output_dir="ml-dataset",
            stars_to_extract=request.stars_to_extract
        )
        
        # Get file info
        import os
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            
            # Count rows in CSV
            with open(csv_path, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
        else:
            file_size = 0
            row_count = 0
        
        return {
            "success": True,
            "csv_path": csv_path,
            "file_size_bytes": file_size,
            "total_rows": row_count,
            "output_directory": "ml-dataset",
            "stars_requested": request.stars_to_extract,
            "message": f"Successfully exported training data to {csv_path}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting training data: {str(e)}")


@app.get("/sed/{star_number}")
async def get_sed_image(star_number: int) -> Response:
    """Get SED image URL for a specific star"""
    credentials_manager = get_credentials_manager()
    sed_creds = credentials_manager.get_sed_credentials()
    
    if not sed_creds['url'] or not sed_creds['username'] or not sed_creds['password']:
        raise HTTPException(
            status_code=400, 
            detail="SED service credentials not configured. Please configure in app settings."
        )
    
    # Construct the SED image URL
    sed_url = f"http://{sed_creds['username']}:{sed_creds['password']}@{sed_creds['url']}/{star_number}_SED.png"

    response = requests.get(sed_url)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching SED image")
    return Response(content=response.content, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
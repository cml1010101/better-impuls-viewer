from fastapi import FastAPI, HTTPException, Request, File, UploadFile
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
from config import Config, get_data_folder
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


def get_data_filepath(star_number: int, telescope: str) -> str:
    """Get the correct filepath for a star and telescope, trying both naming formats"""
    folder = get_data_folder()
    
    # Try 3-digit padded format first
    filename = f"{str(star_number).zfill(3)}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    # Try simple format
    filename = f"{star_number}-{telescope}.tbl"
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    # If neither exists, return the padded format (for error reporting)
    return os.path.join(folder, f"{str(star_number).zfill(3)}-{telescope}.tbl")

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
    sed_url: Optional[str] = None
    sed_username: Optional[str] = None
    sed_password: Optional[str] = None
    data_folder_path: Optional[str] = None

class ModelTrainingRequest(BaseModel):
    stars_to_extract: Optional[List[int]] = None
    force_retrain: bool = False
    csv_filename: Optional[str] = None  # Specify which uploaded CSV to use

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
    
    status = credentials_manager.get_credentials_status()
    
    return {
        "sed_service": {
            "configured": status['sed_service']
        },
        "data_folder": {
            "configured": status['data_folder_configured'],
            "current_path": get_data_folder()
        }
    }

@app.post("/credentials/configure")
async def configure_credentials(request: CredentialsRequest):
    """Configure application credentials"""
    credentials_manager = get_credentials_manager()
    
    try:
        # Update SED credentials if provided
        if request.sed_url and request.sed_username and request.sed_password:
            credentials_manager.set_sed_credentials(
                request.sed_url, 
                request.sed_username, 
                request.sed_password
            )
        
        # Update data folder path if provided
        if request.data_folder_path:
            try:
                credentials_manager.set_data_folder_path(request.data_folder_path)
                # Clear relevant caches since data folder changed
                global _file_cache, _campaigns_cache, _periodogram_cache, _processed_data_cache
                _file_cache.clear()
                _campaigns_cache.clear()
                _periodogram_cache.clear()
                _processed_data_cache.clear()
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        return {"success": True, "message": "Credentials updated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating credentials: {str(e)}")

# === CSV Training Data Upload Endpoints ===

@app.post("/training_data/upload")
async def upload_training_data_csv(file: UploadFile = File(...)):
    """Upload CSV file for training data"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(Config.get_data_folder(), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(uploads_dir, f"training_data_{int(time.time())}.csv")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Validate CSV format
        from csv_training_data import CSVTrainingDataLoader
        try:
            loader = CSVTrainingDataLoader(file_path)
            # Test loading raw data to validate format
            df = loader.load_raw_data()
            
            return {
                "success": True,
                "message": f"CSV uploaded successfully. Found {len(df)} rows.",
                "file_path": file_path,
                "filename": file.filename
            }
        except Exception as e:
            # Remove invalid file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/training_data/files")
async def list_training_data_files():
    """List uploaded training data CSV files"""
    try:
        uploads_dir = os.path.join(Config.get_data_folder(), "uploads")
        
        if not os.path.exists(uploads_dir):
            return {"files": []}
        
        files = []
        for filename in os.listdir(uploads_dir):
            if filename.endswith('.csv') and filename.startswith('training_data_'):
                file_path = os.path.join(uploads_dir, filename)
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "upload_time": stat.st_mtime,
                    "size_bytes": stat.st_size
                })
        
        # Sort by upload time, newest first
        files.sort(key=lambda x: x['upload_time'], reverse=True)
        
        return {"files": files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@app.delete("/training_data/files/{filename}")
async def delete_training_data_file(filename: str):
    """Delete an uploaded training data CSV file"""
    try:
        uploads_dir = os.path.join(Config.get_data_folder(), "uploads")
        file_path = os.path.join(uploads_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        if not filename.endswith('.csv') or not filename.startswith('training_data_'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        os.remove(file_path)
        
        return {"success": True, "message": f"File {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/data_folder/browse")
async def browse_data_folders(path: str = None):
    """Browse available directories for data folder selection"""
    import platform
    
    # Default starting paths based on operating system
    if path is None:
        if platform.system() == "Windows":
            base_paths = [os.path.expanduser("~"), "C:\\"]
        else:
            base_paths = [os.path.expanduser("~"), "/"]
    else:
        if not os.path.exists(path) or not os.path.isdir(path):
            raise HTTPException(status_code=400, detail="Invalid directory path")
        base_paths = [path]
    
    try:
        directories = []
        for base_path in base_paths:
            if os.path.exists(base_path) and os.path.isdir(base_path):
                try:
                    for item in sorted(os.listdir(base_path)):
                        item_path = os.path.join(base_path, item)
                        if os.path.isdir(item_path) and not item.startswith('.'):
                            # Check if this directory contains .tbl files
                            tbl_count = 0
                            try:
                                files = os.listdir(item_path)
                                tbl_count = len([f for f in files if f.endswith('.tbl')])
                            except (PermissionError, OSError):
                                continue
                            
                            directories.append({
                                "name": item,
                                "path": item_path,
                                "tbl_files_count": tbl_count,
                                "parent": base_path
                            })
                except (PermissionError, OSError):
                    continue
        
        return {
            "current_path": base_paths[0] if base_paths else None,
            "parent_path": os.path.dirname(base_paths[0]) if base_paths and base_paths[0] != "/" else None,
            "directories": directories[:50]  # Limit to 50 directories for performance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error browsing directories: {str(e)}")

@app.post("/data_folder/validate")
async def validate_data_folder(path: str):
    """Validate that a directory is suitable as a data folder"""
    if not path:
        raise HTTPException(status_code=400, detail="Path cannot be empty")
    
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Directory does not exist")
    
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    try:
        # Check if directory contains .tbl files
        files = os.listdir(path)
        tbl_files = [f for f in files if f.endswith('.tbl')]
        
        # Check for valid star file naming pattern (e.g., 001-telescopename.tbl)
        star_files = []
        for f in tbl_files:
            try:
                parts = f.split('-')
                if len(parts) >= 2:
                    star_num = int(parts[0])
                    telescope = parts[1].replace('.tbl', '')
                    if telescope:
                        star_files.append({'star': star_num, 'telescope': telescope, 'file': f})
            except (ValueError, IndexError):
                continue
        
        return {
            "valid": True,
            "path": os.path.abspath(path),
            "total_files": len(files),
            "tbl_files_count": len(tbl_files),
            "valid_star_files": len(star_files),
            "sample_files": tbl_files[:10],  # Show first 10 files as examples
            "message": f"Found {len(star_files)} valid star data files"
        }
    
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied accessing directory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating directory: {str(e)}")

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
        # Try both 3-digit padded and simple format
        star_prefix_padded = f"{str(star_number).zfill(3)}-"
        star_prefix_simple = f"{star_number}-"
        
        if (filename.startswith(star_prefix_padded) or filename.startswith(star_prefix_simple)) and filename.endswith('.tbl'):
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
async def train_model_from_csv(request: ModelTrainingRequest) -> ModelTrainingResult:
    """
    Train the CNN model using data from uploaded CSV file with enhanced 5-period strategy.
    
    Parameters:
    - csv_filename: Name of the uploaded CSV file to use for training
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
        # Check if CSV filename is provided
        if not request.csv_filename:
            raise HTTPException(
                status_code=400, 
                detail="CSV filename is required. Please upload a training data CSV first."
            )
        
        # Verify CSV file exists
        uploads_dir = os.path.join(Config.get_data_folder(), "uploads")
        csv_path = os.path.join(uploads_dir, request.csv_filename)
        
        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404, 
                detail=f"CSV file '{request.csv_filename}' not found. Please upload the file first."
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
        result = trainer.train_from_csv(csv_path, request.stars_to_extract)
        
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
    Export training data from uploaded CSV to processed CSV format for external analysis.
    
    Creates CSV files containing phase-folded light curves with corresponding
    category and confidence information suitable for machine learning training.
    
    Parameters:
    - csv_filename: Name of the uploaded CSV file to process
    - stars_to_extract: Optional list of star numbers to include in export. 
                       If None, all available stars will be used.
    
    Returns:
    - Dictionary with export summary information
    """
    try:
        # Check if CSV filename is provided
        if not request.csv_filename:
            raise HTTPException(
                status_code=400, 
                detail="CSV filename is required. Please upload a training data CSV first."
            )
        
        # Verify CSV file exists
        uploads_dir = os.path.join(Config.get_data_folder(), "uploads")
        csv_path = os.path.join(uploads_dir, request.csv_filename)
        
        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=404, 
                detail=f"CSV file '{request.csv_filename}' not found. Please upload the file first."
            )
        
        from csv_training_data import CSVTrainingDataLoader
        
        loader = CSVTrainingDataLoader(csv_path)
        output_csv_path = loader.export_training_data_to_csv(
            output_dir="ml-dataset",
            stars_to_extract=request.stars_to_extract
        )
        
        # Get file info
        if os.path.exists(output_csv_path):
            file_size = os.path.getsize(output_csv_path)
            
            # Count rows in CSV
            with open(output_csv_path, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
        else:
            file_size = 0
            row_count = 0
        
        return {
            "success": True,
            "csv_path": output_csv_path,
            "file_size_bytes": file_size,
            "total_rows": row_count,
            "output_directory": "ml-dataset",
            "stars_requested": request.stars_to_extract,
            "message": f"Successfully exported training data to {output_csv_path}"
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
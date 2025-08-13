"""
Online data source management for Better Impuls Viewer.
Handles downloading, caching, and managing data from online astronomical databanks.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from config import Config


class OnlineDataManager:
    """Manages online data sources and local caching."""
    
    def __init__(self):
        self.cache_dir = Config.ONLINE_CACHE_DIR
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        self._ensure_cache_dir()
        self._current_source = Config.DEFAULT_DATA_SOURCE
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def set_data_source(self, source: str) -> bool:
        """Set the current data source (local or online)."""
        if source not in ["local", "online"]:
            return False
        self._current_source = source
        return True
    
    def get_data_source(self) -> str:
        """Get the current data source."""
        return self._current_source
    
    def get_cache_info(self) -> Dict:
        """Get information about the cache."""
        cache_info = {
            "source": self._current_source,
            "cache_dir": self.cache_dir,
            "cached_files": 0,
            "total_size_mb": 0.0,
            "last_updated": None
        }
        
        if os.path.exists(self.cache_dir):
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('.tbl')]
            cache_info["cached_files"] = len(files)
            
            # Calculate total size
            total_size = 0
            for file in files:
                file_path = os.path.join(self.cache_dir, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            cache_info["total_size_mb"] = total_size / (1024 * 1024)
            
            # Get last updated from metadata
            if os.path.exists(self.metadata_file):
                try:
                    with open(self.metadata_file, 'r') as f:
                        metadata = json.load(f)
                        cache_info["last_updated"] = metadata.get("last_updated")
                except:
                    pass
        
        return cache_info
    
    def clear_cache(self) -> bool:
        """Clear the online data cache."""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            self._ensure_cache_dir()
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def _simulate_online_download(self, star_number: int, telescope: str) -> Optional[np.ndarray]:
        """
        Simulate downloading data from online astronomical databanks.
        In a real implementation, this would connect to actual databases like:
        - MAST (Mikulski Archive for Space Telescopes)
        - ESA Archive
        - IRSA (Infrared Science Archive)
        etc.
        """
        # Simulate different data characteristics for different sources
        np.random.seed(star_number * 1000 + hash(telescope) % 1000)
        
        # Create realistic astronomical time series data
        if telescope.lower() == "hubble":
            # Hubble: shorter observations, higher precision
            duration = np.random.uniform(20, 60)  # days
            n_points = int(duration * 48)  # ~twice daily observations
            time_noise = 0.001
        elif telescope.lower() == "kepler":
            # Kepler: long-term monitoring, very high precision
            duration = np.random.uniform(80, 120)  # days
            n_points = int(duration * 24 * 2)  # ~every 30 minutes
            time_noise = 0.0001
        elif telescope.lower() == "tess":
            # TESS: 27-day sectors, high cadence
            duration = np.random.uniform(25, 30)  # days
            n_points = int(duration * 24 * 24)  # ~every hour
            time_noise = 0.0005
        else:
            # Default parameters
            duration = np.random.uniform(30, 90)
            n_points = int(duration * 12)
            time_noise = 0.001
        
        # Generate time array
        time = np.linspace(0, duration, n_points)
        time += np.random.normal(0, time_noise, n_points)
        time = np.sort(time)
        
        # Generate flux with realistic variability
        base_flux = 1.0
        
        # Add periodic variability (simulating different types of stars)
        if star_number % 5 == 0:
            # Eclipsing binary
            period = np.random.uniform(2, 10)
            eclipse_depth = np.random.uniform(0.01, 0.1)
            flux = base_flux - eclipse_depth * np.where(
                (time % period) < (period * 0.1), 1.0, 0.0
            )
        elif star_number % 5 == 1:
            # Pulsating variable
            period = np.random.uniform(0.5, 5)
            amplitude = np.random.uniform(0.005, 0.05)
            flux = base_flux + amplitude * np.sin(2 * np.pi * time / period)
        elif star_number % 5 == 2:
            # Multi-periodic variable
            period1 = np.random.uniform(1, 8)
            period2 = np.random.uniform(0.5, 3)
            amp1 = np.random.uniform(0.01, 0.03)
            amp2 = np.random.uniform(0.005, 0.02)
            flux = base_flux + amp1 * np.sin(2 * np.pi * time / period1) + \
                   amp2 * np.sin(2 * np.pi * time / period2)
        elif star_number % 5 == 3:
            # Irregular variable
            flux = base_flux + np.random.normal(0, 0.01, len(time))
            # Add some larger variations
            for _ in range(5):
                start = np.random.randint(0, len(time))
                width = np.random.randint(10, 50)
                end = min(start + width, len(time))
                flux[start:end] += np.random.uniform(-0.05, 0.05)
        else:
            # Relatively stable star with noise
            flux = base_flux + np.random.normal(0, 0.002, len(time))
        
        # Add realistic noise based on telescope characteristics
        if telescope.lower() == "hubble":
            noise_level = np.random.uniform(0.001, 0.003)
        elif telescope.lower() == "kepler":
            noise_level = np.random.uniform(0.0001, 0.0005)
        elif telescope.lower() == "tess":
            noise_level = np.random.uniform(0.0005, 0.002)
        else:
            noise_level = np.random.uniform(0.001, 0.003)
        
        flux += np.random.normal(0, noise_level, len(flux))
        
        # Generate realistic error estimates
        error = np.full_like(flux, noise_level) + np.random.uniform(0, noise_level * 0.5, len(flux))
        
        # Create data array in the same format as local files
        data = np.column_stack([time, flux, error])
        
        return data
    
    def download_star_data(self, star_number: int, telescope: str, force_refresh: bool = False) -> bool:
        """Download data for a specific star and telescope from online sources."""
        filename = f"{star_number}-{telescope}.tbl"
        filepath = os.path.join(self.cache_dir, filename)
        
        # Check if file already exists and force_refresh is False
        if os.path.exists(filepath) and not force_refresh:
            return True
        
        try:
            # Simulate download
            data = self._simulate_online_download(star_number, telescope)
            if data is None:
                return False
            
            # Save to cache with same format as local files
            with open(filepath, 'w') as f:
                f.write("# Time (days)\tFlux\tError\n")
                for row in data:
                    f.write(f"{row[0]:.6f}\t{row[1]:.6f}\t{row[2]:.6f}\n")
            
            # Update metadata
            self._update_metadata(star_number, telescope)
            
            return True
            
        except Exception as e:
            print(f"Error downloading data for star {star_number}, telescope {telescope}: {e}")
            return False
    
    def download_multiple_stars(self, star_numbers: List[int], telescopes: List[str], 
                               force_refresh: bool = False) -> Dict[str, bool]:
        """Download data for multiple stars and telescopes."""
        results = {}
        
        for star_number in star_numbers:
            for telescope in telescopes:
                key = f"{star_number}-{telescope}"
                results[key] = self.download_star_data(star_number, telescope, force_refresh)
        
        return results
    
    def _update_metadata(self, star_number: int, telescope: str):
        """Update metadata file with download information."""
        metadata = {}
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        
        if "files" not in metadata:
            metadata["files"] = {}
        
        key = f"{star_number}-{telescope}"
        metadata["files"][key] = {
            "downloaded_at": datetime.now().isoformat(),
            "star_number": star_number,
            "telescope": telescope
        }
        metadata["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error updating metadata: {e}")
    
    def get_data_directory(self) -> str:
        """Get the appropriate data directory based on current source."""
        if self._current_source == "online":
            return self.cache_dir
        else:
            return Config.DATA_DIR
    
    def is_data_available(self, star_number: int, telescope: str) -> bool:
        """Check if data is available for the given star and telescope."""
        data_dir = self.get_data_directory()
        filename = f"{star_number}-{telescope}.tbl"
        filepath = os.path.join(data_dir, filename)
        return os.path.exists(filepath)
    
    def get_available_stars(self) -> List[int]:
        """Get list of available stars for current data source."""
        data_dir = self.get_data_directory()
        
        if not os.path.exists(data_dir):
            return []
        
        stars = set()
        for filename in os.listdir(data_dir):
            if filename.endswith('.tbl'):
                try:
                    star_part = filename.split('-')[0]
                    star_number = int(star_part)
                    stars.add(star_number)
                except ValueError:
                    continue
        
        return sorted(list(stars))
    
    def get_available_telescopes(self, star_number: int) -> List[str]:
        """Get list of available telescopes for a given star."""
        data_dir = self.get_data_directory()
        
        if not os.path.exists(data_dir):
            return []
        
        telescopes = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.tbl') and filename.startswith(f"{star_number}-"):
                try:
                    telescope = filename.split('-', 1)[1].replace('.tbl', '')
                    telescopes.append(telescope)
                except:
                    continue
        
        return sorted(telescopes)


# Global instance
online_data_manager = OnlineDataManager()
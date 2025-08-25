#!/usr/bin/env python3
"""
Dataset management for Better Impuls Viewer backend.
Replaces database.py with efficient dataset file format.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import NamedTuple, Dict, List, Optional
from astropy.coordinates import SkyCoord, Angle
import glob
from pathlib import Path

# Re-export classes that need to be compatible with existing code
class StarMetadata(NamedTuple):
    star_number: int
    name: str | None
    coordinates: SkyCoord
    def __hash__(self) -> int:
        return hash((self.star_number, self.name, self.coordinates.ra.deg, self.coordinates.dec.deg))

DEFAULT_SURVEYS = [
    'cdips', 'eleanor', 'qlp', 'spoc', 't16', 'tasoc', 'tglc',
    'everest', 'k2sc', 'k2sff', 'k2varcat', 'ztf-r', 'ztf-g',
    'neowise-1', 'neowise-2']


class StarDataset:
    """Dataset-based star management class to replace StarList."""
    
    def __init__(self, dataset_file: Optional[str] = None):
        self.stars: Dict[int, StarMetadata] = {}
        self.survey_data: Dict[int, Dict[str, np.ndarray]] = {}
        self.dataset_file = dataset_file
        
        if dataset_file and os.path.exists(dataset_file):
            self.load_from_dataset(dataset_file)

    def add_star(self, star_number: int, ra: float, dec: float, *, name: str | None = None):
        """Add a star to the dataset with its name and coordinates."""
        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        self.stars[star_number] = StarMetadata(star_number=star_number, name=name, coordinates=coord)

    def get_star(self, star_number: int) -> Optional[StarMetadata]:
        """Retrieve a star's metadata by its number."""
        return self.stars.get(star_number)

    def list_stars(self) -> List[int]:
        """List all star numbers in the dataset."""
        return list(self.stars.keys())
    
    def coords_to_str(self, coords: SkyCoord) -> str:
        """Convert coordinates to string format."""
        sign = '-' if coords.dec.deg < 0 else '+'
        return f"{coords.ra.arcsec:.6f}{sign}{abs(coords.dec.arcsec):.6f}"
    
    def str_to_coords(self, coords_str: str) -> SkyCoord:
        """Convert a string representation of coordinates to SkyCoord."""
        ra_str = coords_str[:9]
        dec_str = coords_str[9:]
        ra_hr = ra_str[:2]
        ra_min = ra_str[2:4]
        ra_sec = ra_str[4:]
        dec_deg = dec_str[:3]
        dec_min = dec_str[3:5]
        dec_sec = dec_str[5:]
        ra = Angle(f"{ra_hr}h{ra_min}m{ra_sec}s")
        dec = Angle(f"{dec_deg}d{dec_min}m{dec_sec}s")
        return SkyCoord(ra=ra, dec=dec, unit='deg')

    def load_from_csv(self, csv_path: str):
        """Load star metadata from CSV file (compatible with database.py format)."""
        self.stars.clear()
        data = pd.read_csv(csv_path, header=None, names=['star_number', 'name', 'coordinates'])
        for _, row in data.iterrows():
            star_number = int(row['star_number'])
            name = row['name'] if pd.notna(row['name']) else None
            coordinates = self.str_to_coords(row['coordinates'])
            self.add_star(star_number, coordinates.ra.deg, coordinates.dec.deg, name=name)

    def add_survey_data(self, star_number: int, survey: str, data: np.ndarray):
        """Add survey data for a star."""
        if star_number not in self.survey_data:
            self.survey_data[star_number] = {}
        self.survey_data[star_number][survey] = data

    def get_survey_data(self, star_metadata: StarMetadata) -> Dict[str, np.ndarray]:
        """Get survey data for a star."""
        return self.survey_data.get(star_metadata.star_number, {})

    def save_to_dataset(self, dataset_path: str):
        """Save the complete dataset to a file."""
        dataset = {
            'stars': self.stars,
            'survey_data': self.survey_data,
            'metadata': {
                'version': '1.0',
                'description': 'Better Impuls Viewer star dataset'
            }
        }
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {dataset_path}")

    def load_from_dataset(self, dataset_path: str):
        """Load the complete dataset from a file."""
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        self.stars = dataset.get('stars', {})
        self.survey_data = dataset.get('survey_data', {})
        print(f"Dataset loaded from {dataset_path}")


class DatasetStarDatabase:
    """Dataset-based database to replace PinputStarDatabase."""
    
    def __init__(self, dataset: StarDataset, *, surveys: List[str] = DEFAULT_SURVEYS):
        self.dataset = dataset
        self.surveys = surveys

    def get_survey_data(self, star_metadata: StarMetadata) -> Dict[str, np.ndarray]:
        """Retrieve survey data for a star from the dataset."""
        all_data = self.dataset.get_survey_data(star_metadata)
        # Filter by requested surveys
        return {survey: data for survey, data in all_data.items() if survey in self.surveys}


def load_tbl_file(filepath: str) -> np.ndarray:
    """Load a .tbl file and return its contents."""
    try:
        data = pd.read_table(filepath, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        data_array = data.to_numpy()
        
        # For .tbl files, typically columns are: time, time_err, flux, flux_err
        # We want time (column 0) and flux (column 2)
        if data_array.shape[1] >= 3:
            result = np.column_stack([data_array[:, 0], data_array[:, 2]])
        elif data_array.shape[1] >= 2:
            result = data_array[:, :2]  # fallback to first 2 columns
        else:
            result = data_array
        
        return result
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.array([])


def convert_directory_to_dataset(
    data_dir: str,
    star_csv_path: str,
    output_dataset_path: str,
    surveys: List[str] = DEFAULT_SURVEYS
) -> StarDataset:
    """
    Main function to convert directory structure to dataset files.
    
    Args:
        data_dir: Directory containing .tbl files
        star_csv_path: Path to star metadata CSV file
        output_dataset_path: Path where to save the dataset file
        surveys: List of surveys to include
    
    Returns:
        StarDataset: The created dataset
    """
    print(f"Converting directory {data_dir} to dataset...")
    
    # Create new dataset
    dataset = StarDataset()
    
    # Load star metadata from CSV
    if os.path.exists(star_csv_path):
        print(f"Loading star metadata from {star_csv_path}")
        dataset.load_from_csv(star_csv_path)
        print(f"Loaded {len(dataset.stars)} stars")
    else:
        print(f"Warning: Star CSV file not found at {star_csv_path}")
    
    # Scan for .tbl files and load survey data
    data_files_found = 0
    if os.path.exists(data_dir):
        for survey in surveys:
            pattern = os.path.join(data_dir, f"*-{survey}.tbl")
            files = glob.glob(pattern)
            
            for filepath in files:
                filename = os.path.basename(filepath)
                # Extract star number from filename like "001-survey.tbl"
                try:
                    star_number = int(filename.split('-')[0])
                    data = load_tbl_file(filepath)
                    if len(data) > 0:
                        dataset.add_survey_data(star_number, survey, data)
                        data_files_found += 1
                        print(f"Loaded {survey} data for star {star_number}: {data.shape}")
                except (ValueError, IndexError) as e:
                    print(f"Error parsing filename {filename}: {e}")
    
    print(f"Loaded {data_files_found} survey data files")
    
    # Save the dataset
    dataset.save_to_dataset(output_dataset_path)
    
    return dataset


def main():
    """Main function to convert directories into dataset files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert directory structure to dataset files')
    parser.add_argument('--data-dir', type=str, default='impuls-data', 
                       help='Directory containing .tbl files')
    parser.add_argument('--star-csv', type=str, default='impuls-data/impuls_stars.csv',
                       help='Path to star metadata CSV file')
    parser.add_argument('--output', type=str, default='stars_dataset.pkl',
                       help='Output dataset file path')
    parser.add_argument('--surveys', type=str, nargs='+', default=DEFAULT_SURVEYS,
                       help='List of surveys to include')
    
    args = parser.parse_args()
    
    # Convert directory to dataset
    dataset = convert_directory_to_dataset(
        data_dir=args.data_dir,
        star_csv_path=args.star_csv,
        output_dataset_path=args.output,
        surveys=args.surveys
    )
    
    print(f"\nDataset conversion complete!")
    print(f"Stars: {len(dataset.stars)}")
    print(f"Stars with survey data: {len(dataset.survey_data)}")
    
    # Print summary
    total_surveys = sum(len(surveys) for surveys in dataset.survey_data.values())
    print(f"Total survey datasets: {total_surveys}")


if __name__ == "__main__":
    main()
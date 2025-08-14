from astropy.coordinates import SkyCoord
from typing import NamedTuple

class StarMetadata(NamedTuple):
    star_number: int
    name: str | None
    coordinates: SkyCoord
    def __hash__(self) -> int:
        return hash((self.star_number, self.name, self.coordinates.ra.deg, self.coordinates.dec.deg)) 

import pandas as pd
from astropy.coordinates import Angle

DEFAULT_SURVEYS = [
    'cdips', 'eleanor', 'qlp', 'spoc', 't16', 'tasoc', 'tglc',
    'everest', 'k2sc', 'k2sff', 'k2varcat', 'ztf-r', 'ztf-g',
    'neowise-1', 'neowise-2']

class StarList:
    def __init__(self):
        self.stars: dict[int, StarMetadata] = {}

    def add_star(self, star_number: int, ra: float, dec: float, *, name: str | None):
        """Add a star to the database with its name and coordinates.
        Args:
            name (str): The name of the star.
            ra (float): Right Ascension in degrees.
            dec (float): Declination in degrees.
        """
        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        self.stars[star_number] = StarMetadata(star_number=star_number, name=name, coordinates=coord)

    def get_star(self, name: str) -> StarMetadata:
        """Retrieve a star's metadata by its name."""
        return self.stars.get(name)

    def list_stars(self):
        """List all stars in the database."""
        return list(self.stars.keys())
    
    def coords_to_str(self, coords: SkyCoord) -> str:
        sign = '-' if coords.dec.deg < 0 else '+'
        return f"{coords.ra.arcsec:.6f}{sign}{abs(coords.dec.arcsec):.6f}"
    
    def save_to_file(self, path: str):
        """Save the star list to a file."""
        with open(path, 'w') as f:
            for star in self.stars.values():
                f.write(f"{star.star_number},{star.name},{self.coords_to_str(star.coordinates)}\n")
    
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

    def load_from_file(self, path: str):
        """Load the star list from a file."""
        self.stars.clear()
        data = pd.read_csv(path, header=None, names=['star_number', 'name', 'coordinates'])
        for _, row in data.iterrows():
            star_number = int(row['star_number'])
            name = row['name'] if pd.notna(row['name']) else None
            coordinates = self.str_to_coords(row['coordinates'])
            self.add_star(star_number, coordinates.ra.deg, coordinates.dec.deg, name=name)

import numpy as np

class StarDatabase:
    def get_survey_data(self, star_metadata: StarMetadata) -> dict[str, np.ndarray]:
        """Retrieve survey data for a star."""
        raise NotImplementedError("This method should be implemented by subclasses.")

from functools import lru_cache

class PinputStarDatabase(StarDatabase):
    def __init__(self, path_to_pinput: str, *, surveys: list[str] = DEFAULT_SURVEYS):
        """
        Initialize the PinputStarDatabase with the path to the pinput directory.
        Args:
            path_to_pinput (str): Path to the directory containing pinput files.
            surveys (list[str]): List of surveys to include. Defaults to None.
        """
        self.surveys = surveys
        self.path_to_pinput = path_to_pinput
    
    def get_path_to_lc(self, star_number: int, survey: str):
        filename = f"{star_number:03d}-{survey}.tbl"
        return f"{self.path_to_pinput}/{filename}"
    
    def load_tbl_file(self, filepath: str) -> np.ndarray:
        """Load a .tbl file and return its contents.
        This is a placeholder implementation. Replace with actual file reading logic.
        """
        data = pd.read_table(filepath, header=None, sep=r'\s+', skiprows=[0, 1, 2])
        data_array = data.to_numpy()
        
        # If data has more than 2 columns, use only the first 2 (time, flux)
        if data_array.shape[1] > 2:
            result = data_array[:, :2]
        else:
            result = data_array
        
        return result
    
    @lru_cache(maxsize=128)
    def get_survey_data(self, star_metadata: StarMetadata) -> dict[str, np.ndarray]:
        """Retrieve survey data for a star."""
        survey_data = {}
        for survey in self.surveys:
            path = self.get_path_to_lc(star_metadata.star_number, survey)
            try:
                data = self.load_tbl_file(path)
                survey_data[survey] = data
            except FileNotFoundError:
                print(f"File not found: {path}")
        return survey_data

import lightkurve as lk

class MASTStarDatabase(StarDatabase):
    SURVEY_SUBSTITUTIONS = {
        'gsfc-eleanor-lite': 'eleanor'
    }
    def __init__(self, *, surveys: list[str] = DEFAULT_SURVEYS):
        """
        Initialize the MASTStarDatabase.
        Args:
            surveys (list[str]): List of surveys to include. Defaults to None.
        """
        self.surveys = surveys
    @lru_cache(maxsize=128)
    def get_survey_data(self, star_metadata: StarMetadata) -> dict[str, np.ndarray]:
        """Retrieve survey data for a star from MAST."""
        survey_data: dict[str, lk.SearchResult] = {}
        search_result = lk.search_lightcurve(star_metadata.coordinates, mission='TESS')
        for lc in search_result:
            survey_name = lc.author[0].lower()
            if survey_name.startswith('tess-'):
                survey_name = survey_name[5:]
            if survey_name in self.SURVEY_SUBSTITUTIONS:
                survey_name = self.SURVEY_SUBSTITUTIONS[survey_name]
            if survey_name not in self.surveys:
                continue
            if lc.year is None or lc.year < 2015:
                print(f"Skipping {star_metadata.star_number} {survey_name} due to missing year")
                continue
            if survey_name not in survey_data:
                survey_data[survey_name] = lc
            elif survey_data[survey_name].year < lc.year:
                survey_data[survey_name] = lc
        results: dict[str, np.ndarray] = {}
        for survey, lc in survey_data.items():
            lc_data: lk.lightcurve.LightCurve = lc.download()
            if lc_data is not None:
                results[survey] = np.vstack((lc_data.time.value, lc_data.flux.value)).T
                print(results[survey])
                print(lc_data.flux.value)
        return results

# import os

# if __name__ == "__main__":
#     path_to_impuls_stars = os.path.join(os.path.dirname(__file__), 'impuls_stars.csv')
#     star_list = StarList()
#     star_list.load_from_file(path_to_impuls_stars)
#     mast_star_db = MASTStarDatabase()
#     # Load star #1 from MAST and print its data
#     import asyncio
#     star_metadata = star_list.get_star(1)
#     if star_metadata:
#         survey_data = asyncio.run(mast_star_db.get_survey_data(star_metadata))
#         for survey, data in survey_data.items():
#             print(f"Survey: {survey}, Data shape: {data.shape}")
#     else:
#         print("Star #1 not found in the star list.")
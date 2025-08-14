"""
Pydantic data models for Better Impuls Viewer API.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Coordinates(BaseModel):
    ra: float  # Right Ascension in degrees
    dec: float  # Declination in degrees

class StarInfo(BaseModel):
    star_number: int
    name: str
    coordinates: Coordinates

class StarSurveys(BaseModel):
    star_number: int
    surveys: List[str]

class CampaignInfo(BaseModel):
    campaign_id: int
    survey: str
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
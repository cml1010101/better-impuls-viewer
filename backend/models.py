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


class CandidatePeriod(BaseModel):
    period: float
    score: float
    rank: int


class PeriodizationResult(BaseModel):
    predicted_period: float
    predicted_class: str
    class_confidence: float
    detected_period: float
    candidate_periods: List[CandidatePeriod]


class DatasetInfo(BaseModel):
    name: str
    path: str
    n_stars: int
    n_files: int
    surveys: List[str]
    created_at: Optional[str] = None
    total_data_points: int


class SyntheticStarInfo(BaseModel):
    star_id: int
    name: str
    variability_class: str
    primary_period: Optional[float]
    secondary_period: Optional[float]
    surveys: List[str]


class DatasetGenerationRequest(BaseModel):
    name: str
    n_stars: int = 50
    surveys: List[str] = ["hubble", "kepler", "tess"]
    max_days: float = 50.0
    min_days: float = 10.0
    noise_level: float = 0.02
    seed: Optional[int] = None


class SEDData(BaseModel):
    url: str
    available: bool
    message: Optional[str] = None
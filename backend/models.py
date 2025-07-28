"""
Pydantic data models for Better Impuls Viewer API.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


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


class AutoPeriodsData(BaseModel):
    primary_period: Optional[float]
    secondary_period: Optional[float]
    classification: Dict[str, Any]
    methods: Dict[str, Any]
    error: Optional[str] = None


class ClassificationResult(BaseModel):
    type: str
    confidence: float
    description: str


class MethodResult(BaseModel):
    success: bool
    periods: List[float]
    confidence: Optional[float] = None


class TrainingDataPoint(BaseModel):
    """Data model for a single training example from CSV data."""
    star_number: int
    period_1: Optional[float]
    period_2: Optional[float]
    lc_category: str
    time_series: List[float]
    flux_series: List[float]
    # Additional metadata for enhanced training
    sensor: Optional[str] = None
    period_type: Optional[str] = None  # 'correct', 'periodogram_peak', or 'random'
    period_confidence: Optional[float] = None


class ModelTrainingResult(BaseModel):
    """Result of model training process."""
    success: bool
    epochs_trained: int
    final_loss: float
    model_path: str
    training_samples: int
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/api/")
async def read_root():
    return {"message": "Welcome to the Better IMPULS Viewer API!"}

from database import StarList, PinputStarDatabase, MASTStarDatabase

star_list = StarList()

pinput_star_db = PinputStarDatabase(config.Config.DATA_DIR)
mast_star_db = MASTStarDatabase()

from models import StarInfo, StarSurveys, Coordinates

@app.get("/api/stars")
async def list_stars() -> list[int]:
    """List all stars in the database."""
    return star_list.list_stars()

@app.get("/api/star/{star_number}")
async def get_star(star_number: int) -> StarInfo:
    """Get metadata for a specific star."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        return {"error": "Star not found"}
    return StarInfo(
        star_number=star_metadata.star_number,
        name=star_metadata.name,
        coordinates=Coordinates(ra=star_metadata.coordinates.ra.deg, dec=star_metadata.coordinates.dec.deg)
    )

@app.get("/api/star/{star_number}/surveys")
async def get_star_surveys(star_number: int, use_mast: bool = False) -> StarSurveys:
    """Get survey data for a specific star."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        return {"error": "Star not found"}
    
    if use_mast:
        survey_data = mast_star_db.get_survey_data(star_metadata)
    else:
        survey_data = pinput_star_db.get_survey_data(star_metadata)
    
    return StarSurveys(
        star_number=star_number,
        surveys=list(survey_data.keys())
    )

from fastapi import HTTPException
from models import ProcessedData

@app.get("/api/star/{star_number}/survey/{survey_name}/raw")
async def get_star_survey_data_by_name(star_number: int, survey_name: str, use_mast: bool = False) -> ProcessedData:
    """Get survey data for a specific star and survey."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        return {"error": "Star not found"}
    
    if use_mast:
        survey_data = mast_star_db.get_survey_data(star_metadata)
    else:
        survey_data = pinput_star_db.get_survey_data(star_metadata)
    
    if survey_name not in survey_data:
        raise HTTPException(status_code=404, detail=f"Survey '{survey_name}' not found for star {star_number}")
    
    # Remove all NaN values from the survey data
    return ProcessedData(
        time=survey_data[survey_name][:, 0],
        flux=survey_data[survey_name][:, 1],
        error=[0.0] * len(survey_data[survey_name])  # Placeholder for error values
    )

from data_processing import *

from functools import lru_cache

from models import CampaignInfo

import numpy as np

@lru_cache(maxsize=128)
def get_campaigns_for_survey(star_number: int, survey_name: str, use_mast: bool = False) -> list[tuple[CampaignInfo, np.ndarray]]:
    """Get all campaigns for a specific survey."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        raise ValueError("Star not found")
    if use_mast:
        survey_data = mast_star_db.get_survey_data(star_metadata)
    else:
        survey_data = pinput_star_db.get_survey_data(star_metadata)
    campaigns = find_all_campaigns(survey_data[survey_name], config.Config.DEFAULT_THRESHOLD)
    campaign_infos = []
    for i, campaign in enumerate(campaigns):
        clean_campaign_data = remove_y_outliers(campaign)
        duration = clean_campaign_data[-1, 0] - clean_campaign_data[0, 0] if len(clean_campaign_data) > 0 else 0
        campaign_info = CampaignInfo(
            campaign_id=i,
            survey=survey_name,
            star_number=star_number,
            data_points=len(campaign.data),
            duration=duration
        )
        campaign_infos.append((campaign_info, clean_campaign_data))
    return campaign_infos

@app.get("/api/star/{star_number}/survey/{survey_name}/campaigns")
async def get_star_survey_campaigns(star_number: int, survey_name: str, use_mast: bool = False) -> list[CampaignInfo]:
    """Get campaigns for a specific star and survey."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        raise HTTPException(status_code=404, detail="Star not found")
    
    try:
        campaigns = get_campaigns_for_survey(star_number, survey_name, use_mast)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return [campaign_info[0] for campaign_info in campaigns]

@app.get("/api/star/{star_number}/survey/{survey_name}/campaigns/{campaign_id}/raw")
async def get_star_survey_campaign(star_number: int, survey_name: str, campaign_id: int, use_mast: bool = False) -> ProcessedData:
    """Get a specific campaign for a star and survey."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        raise HTTPException(status_code=404, detail="Star not found")
    
    try:
        campaigns = get_campaigns_for_survey(star_number, survey_name, use_mast)
        if campaign_id < 0 or campaign_id >= len(campaigns):
            raise HTTPException(status_code=404, detail="Campaign not found")
        campaign_data = campaigns[campaign_id][1]
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ProcessedData(
        time=campaign_data[:, 0].tolist(),
        flux=campaign_data[:, 1].tolist(),
        error=[0.0] * len(campaign_data)  # Placeholder for error values
    )

from models import PeriodogramData, PhaseFoldedData

@app.get("/api/star/{star_number}/survey/{survey_name}/campaigns/{campaign_id}/periodogram")
async def get_star_survey_campaign_periodogram(star_number: int, survey_name: str, campaign_id: int, use_mast: bool = False) -> PeriodogramData:
    """Get periodogram data for a specific campaign."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        raise HTTPException(status_code=404, detail="Star not found")
    
    try:
        campaigns = get_campaigns_for_survey(star_number, survey_name, use_mast)
        if campaign_id < 0 or campaign_id >= len(campaigns):
            raise HTTPException(status_code=404, detail="Campaign not found")
        campaign_data = campaigns[campaign_id][1]
        frequencies, powers = calculate_lomb_scargle(campaign_data)
        periods = 1 / frequencies
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return PeriodogramData(
        periods=periods.tolist(),
        powers=powers.tolist()
    )

@app.get("/api/star/{star_number}/survey/{survey_name}/campaigns/{campaign_id}/phase_folded")
async def get_star_survey_campaign_phase_folded(star_number: int, survey_name: str, campaign_id: int, period: float, use_mast: bool = False) -> PhaseFoldedData:
    """Get phase-folded data for a specific campaign."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        raise HTTPException(status_code=404, detail="Star not found")
    
    try:
        campaigns = get_campaigns_for_survey(star_number, survey_name, use_mast)
        if campaign_id < 0 or campaign_id >= len(campaigns):
            raise HTTPException(status_code=404, detail="Campaign not found")
        campaign_data = campaigns[campaign_id][1]
        
        # Phase folding logic
        time = campaign_data[:, 0]
        flux = campaign_data[:, 1]
        phase = (time % period) / period  # Normalize phase to [0, 1)
        
        return PhaseFoldedData(
            phase=phase.tolist(),
            flux=flux.tolist()
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

import argparse
import uvicorn

def main(args = None):
    parser = argparse.ArgumentParser(description="Run the Better IMPULS Viewer API")
    parser.add_argument('-h,--help', action='help', help='Show this help message and exit')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API on (default: 8000)')
    args = parser.parse_args(args)
    star_list.load_from_file(config.Config.IMPULS_STARS_PATH)
    uvicorn.run(app, host='0.0.0.0', port=args.port, log_level="info")

if __name__ == "__main__":
    main()
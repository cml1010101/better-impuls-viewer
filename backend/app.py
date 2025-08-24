from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import io

import config

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Better IMPULS Viewer API!"}

from database import StarList, PinputStarDatabase, MASTStarDatabase

star_list = StarList()

pinput_star_db = PinputStarDatabase(config.Config.DATA_DIR)
mast_star_db = MASTStarDatabase()

from models import StarInfo, StarSurveys, Coordinates, SEDData

@app.get("/stars")
async def list_stars() -> list[int]:
    """List all stars in the database."""
    return star_list.list_stars()

@app.get("/star/{star_number}")
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

@app.get("/star/{star_number}/sed")
async def get_star_sed(star_number: int) -> SEDData:
    """Get SED information for a specific star."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        raise HTTPException(status_code=404, detail="Star not found")
    
    # Check if SED configuration is available
    if not config.Config.SED_API_URL:
        return SEDData(
            url="",
            available=False,
            message="SED API URL not configured"
        )
    
    # Return local backend URL that will serve the image
    local_sed_url = f"/star/{star_number}/sed/image"
    
    return SEDData(
        url=local_sed_url,
        available=True,
        message="SED data available"
    )

@app.get("/star/{star_number}/sed/image")
async def get_star_sed_image(star_number: int):
    """Serve the SED PNG image for a specific star."""
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        raise HTTPException(status_code=404, detail="Star not found")
    
    # Check if SED configuration is available
    if not config.Config.SED_API_URL:
        raise HTTPException(status_code=503, detail="SED API URL not configured")
    
    # Build external SED URL with credentials
    external_sed_url = f"http://{config.Config.SED_USERNAME}:{config.Config.SED_PASSWORD}@{config.Config.SED_API_URL}/{star_number:03d}_SED.png"
    
    try:
        # Fetch the image from external URL
        async with httpx.AsyncClient() as client:
            response = await client.get(external_sed_url)
            response.raise_for_status()
            
            # Stream the image back to the frontend
            image_stream = io.BytesIO(response.content)
            
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"inline; filename=star_{star_number}_sed.png"
                }
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch SED image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving SED image: {str(e)}")

@app.get("/star/{star_number}/surveys")
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

@app.get("/star/{star_number}/survey/{survey_name}/raw")
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

@app.get("/star/{star_number}/survey/{survey_name}/campaigns")
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

@app.get("/star/{star_number}/survey/{survey_name}/campaigns/{campaign_id}/raw")
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

@app.get("/star/{star_number}/survey/{survey_name}/campaigns/{campaign_id}/periodogram")
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

@app.get("/star/{star_number}/survey/{survey_name}/campaigns/{campaign_id}/phase_folded")
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

# ML Model imports and utilities
import torch
from periodizer import MultiBranchStarModelHybrid, StarModelConfig, multitask_loss
from models import PeriodizationResult
import os

# Global model variable
_loaded_model = None
_model_config = None

def load_model():
    """Load the trained MultiBranchStarModelHybrid model if available."""
    global _loaded_model, _model_config
    
    if _loaded_model is not None:
        return _loaded_model, _model_config
    
    model_path = config.Config.MODEL_PATH
    if not os.path.exists(model_path):
        return None, None
    
    try:
        # Load model state
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        _model_config = checkpoint['config']
        
        # Create model and load state
        _loaded_model = MultiBranchStarModelHybrid(_model_config)
        _loaded_model.load_state_dict(checkpoint['model_state_dict'])
        _loaded_model.eval()
        
        print(f"Loaded trained model from {model_path}")
        return _loaded_model, _model_config
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

def create_multi_branch_data(campaign_data: np.ndarray) -> dict:
    """Create multi-branch input data from campaign light curve."""
    
    # Remove outliers and normalize
    cleaned_data = remove_y_outliers(campaign_data)
    time_clean = cleaned_data[:, 0]
    flux_clean = cleaned_data[:, 1]
    
    # Detect period and get periodogram
    true_period, frequency, power = detect_period_lomb_scargle(cleaned_data)
    
    # Normalize raw light curve
    flux_norm = (flux_clean - np.mean(flux_clean)) / (np.std(flux_clean) + 1e-8)
    
    # Create periodogram data
    pgram_data = power / (np.max(power) + 1e-8)
    
    # Generate period candidates
    candidate_periods = generate_candidate_periods(true_period, num_candidates=4)
    
    # Create folded candidates
    folded_candidates = []
    for period in candidate_periods:
        phase, flux_folded = phase_fold_data(time_clean, flux_norm, period)
        # Resample to consistent length
        phase_grid = np.linspace(0, 1, 200)
        flux_interp = np.interp(phase_grid, phase, flux_folded)
        folded_candidates.append(flux_interp)
    
    return {
        'raw_lc': flux_norm,
        'periodogram': pgram_data,
        'folded_candidates': folded_candidates,
        'candidate_periods': candidate_periods,
        'detected_period': true_period
    }

def prepare_model_input(multi_branch_data: dict) -> tuple:
    """Prepare data for model inference."""
    # Raw light curve
    lc = torch.tensor(multi_branch_data['raw_lc'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    
    # Periodogram 
    pgram = torch.tensor(multi_branch_data['periodogram'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    
    # Folded candidates
    folded_list = []
    logP_list = []
    for folded_data, period in zip(multi_branch_data['folded_candidates'], multi_branch_data['candidate_periods']):
        folded_tensor = torch.tensor(folded_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        folded_list.append(folded_tensor)
        logP_list.append(torch.tensor([np.log10(period)], dtype=torch.float32))
    
    return lc, pgram, folded_list, logP_list

@app.get("/star/{star_number}/survey/{survey_name}/campaigns/{campaign_id}/auto_analysis")
async def get_auto_periodization_classification(star_number: int, survey_name: str, campaign_id: int, use_mast: bool = False) -> PeriodizationResult:
    """Get automatic periodization and classification for a campaign using the trained ML model."""
    
    # Load the trained model
    model, model_config = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Trained model not available. Please train the model first using the training script.")
    
    # Get campaign data (same as other endpoints)
    star_metadata = star_list.get_star(star_number)
    if star_metadata is None:
        raise HTTPException(status_code=404, detail="Star not found")
    
    try:
        campaigns = get_campaigns_for_survey(star_number, survey_name, use_mast)
        if campaign_id < 0 or campaign_id >= len(campaigns):
            raise HTTPException(status_code=404, detail="Campaign not found")
        campaign_data = campaigns[campaign_id][1]
        
        if len(campaign_data) < 10:
            raise HTTPException(status_code=400, detail="Campaign has insufficient data points for analysis")
        
        # Create multi-branch data
        multi_branch_data = create_multi_branch_data(campaign_data)
        
        # Prepare model input
        lc, pgram, folded_list, logP_list = prepare_model_input(multi_branch_data)
        
        # Run model inference
        with torch.no_grad():
            outputs = model(lc, pgram, folded_list, logP_list)
        
        # Extract predictions
        predicted_logP = float(outputs['logP_pred'][0])
        predicted_period = 10 ** predicted_logP
        
        type_logits = outputs['type_logits'][0]
        predicted_class_idx = int(torch.argmax(type_logits))
        confidence = float(torch.softmax(type_logits, dim=0)[predicted_class_idx])
        
        # Get class name
        class_names = config.Config.CLASS_NAMES
        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else "unknown"
        
        # Get candidate scores
        cand_weights = outputs['cand_weights'][0].tolist()
        candidate_info = []
        for i, (period, weight) in enumerate(zip(multi_branch_data['candidate_periods'], cand_weights)):
            candidate_info.append({
                "period": period,
                "score": weight,
                "rank": i + 1
            })
        
        # Sort candidates by score
        candidate_info.sort(key=lambda x: x['score'], reverse=True)
        for i, cand in enumerate(candidate_info):
            cand['rank'] = i + 1
        
        return PeriodizationResult(
            predicted_period=predicted_period,
            predicted_class=predicted_class,
            class_confidence=confidence,
            detected_period=multi_branch_data['detected_period'],
            candidate_periods=candidate_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in auto analysis: {str(e)}")

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
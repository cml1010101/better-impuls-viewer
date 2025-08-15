interface Survey {
  name: string;
  campaigns: Campaign[];
}

interface Campaign {
  campaign_id: number;
  data_points: number;
  duration: number;
}

interface DataPoint {
  time: number;
  flux: number;
  error: number;
}

interface PeriodogramPoint {
  period: number;
  power: number;
}

interface PhaseFoldedPoint {
  phase: number;
  flux: number;
}

export interface CandidatePeriod {
  period: number;
  score: number;
  rank: number;
}

export interface AutoAnalysisResult {
  predicted_period: number;
  predicted_class: string;
  class_confidence: number;
  detected_period: number;
  candidate_periods: CandidatePeriod[];
}

const API_BASE = 'http://localhost:8000/api';

// Mock data for when API is not available
const getMockSurveys = () => [
  {
    name: 'hubble',
    campaigns: [
      { campaign_id: 0, data_points: 1250, duration: 45.2 },
      { campaign_id: 1, data_points: 980, duration: 32.1 }
    ]
  },
  {
    name: 'kepler',
    campaigns: [
      { campaign_id: 0, data_points: 2340, duration: 89.7 },
      { campaign_id: 1, data_points: 1890, duration: 67.3 }
    ]
  },
  {
    name: 'tess',
    campaigns: [
      { campaign_id: 0, data_points: 3200, duration: 27.4 }
    ]
  }
];

const getMockCampaignData = (): DataPoint[] => {
  // Generate synthetic light curve data
  const points: DataPoint[] = [];
  for (let i = 0; i < 100; i++) {
    const time = i * 0.5; // 0.5 day intervals
    const flux = 1.0 + 0.02 * Math.sin(2 * Math.PI * time / 7.8) + 0.01 * Math.random(); // 7.8 day period
    const error = 0.005 + 0.002 * Math.random();
    points.push({ time, flux, error });
  }
  return points;
};

const getMockPeriodogramData = (): PeriodogramPoint[] => {
  // Generate synthetic periodogram data with a peak at 7.8 days
  const points: PeriodogramPoint[] = [];
  for (let i = 0; i < 200; i++) {
    const period = 0.5 + i * 0.1; // Periods from 0.5 to 20.5 days
    let power = 0.1 + 0.3 * Math.random();
    
    // Add a peak around 7.8 days
    if (period > 7.0 && period < 8.5) {
      power += 0.8 * Math.exp(-Math.pow((period - 7.8) / 0.3, 2));
    }
    
    points.push({ period, power });
  }
  return points;
};

const getMockPhaseFoldedData = (period: number): PhaseFoldedPoint[] => {
  // Generate synthetic phase-folded data based on the period
  const points: PhaseFoldedPoint[] = [];
  for (let i = 0; i < 100; i++) {
    const phase = i / 100.0; // 0 to 1
    // Use period to create realistic amplitude and shape variations
    const amplitude = Math.max(0.01, 0.05 / Math.sqrt(period));
    const flux = 1.0 + amplitude * Math.sin(2 * Math.PI * phase) + 0.005 * Math.random();
    points.push({ phase, flux });
  }
  return points;
};

export const fetchSurveys = async (starNumber: number): Promise<Survey[]> => {
  try {
    const response = await fetch(`${API_BASE}/star/${starNumber}/surveys`);
    if (!response.ok) {
      console.error(`Error fetching surveys: ${response.status}`);
      return getMockSurveys();
    }
    const data = await response.json();
    
    // If no surveys available from API, use mock data for demonstration
    if (!data.surveys || data.surveys.length === 0) {
      console.log('No surveys from API, using mock data for demonstration');
      return getMockSurveys();
    }
    
    // Fetch campaigns for each survey
    const surveysWithCampaigns = await Promise.all(
      data.surveys.map(async (surveyName: string) => {
        try {
          const campaignsResponse = await fetch(`${API_BASE}/star/${starNumber}/survey/${surveyName}/campaigns`);
          if (campaignsResponse.ok) {
            const campaigns = await campaignsResponse.json();
            return {
              name: surveyName,
              campaigns: campaigns
            };
          }
        } catch (error) {
          console.error(`Error fetching campaigns for ${surveyName}:`, error);
        }
        return {
          name: surveyName,
          campaigns: []
        };
      })
    );
    
    return surveysWithCampaigns;
  } catch (error) {
    console.error('Error fetching surveys:', error);
    return getMockSurveys();
  }
};

export const fetchCampaignData = async (
  starNumber: number, 
  surveyName: string, 
  campaignId: number
): Promise<DataPoint[]> => {
  try {
    const response = await fetch(`${API_BASE}/star/${starNumber}/survey/${surveyName}/campaigns/${campaignId}/raw`);
    if (!response.ok) {
      console.error(`Error fetching campaign data: ${response.status}`);
      return getMockCampaignData();
    }
    const data = await response.json();
    
    if (!data.time || !data.flux) {
      console.error('Invalid campaign data format received');
      return getMockCampaignData();
    }
    
    const formattedData = data.time.map((time: number, index: number) => ({
      time,
      flux: data.flux[index],
      error: data.error?.[index] || 0,
    }));
    
    return formattedData;
  } catch (error) {
    console.error('Error fetching campaign data:', error);
    return getMockCampaignData();
  }
};

export const fetchPeriodogramData = async (
  starNumber: number, 
  surveyName: string, 
  campaignId: number
): Promise<PeriodogramPoint[]> => {
  try {
    const response = await fetch(`${API_BASE}/star/${starNumber}/survey/${surveyName}/campaigns/${campaignId}/periodogram`);
    if (!response.ok) {
      console.error(`Error fetching periodogram: ${response.status}`);
      return getMockPeriodogramData();
    }
    const data = await response.json();
    
    if (!data.periods || !data.powers) {
      console.error('Invalid periodogram data format received');
      return getMockPeriodogramData();
    }
    
    const formattedData = data.periods.map((period: number, index: number) => ({
      period,
      power: data.powers[index],
    }));
    
    return formattedData;
  } catch (error) {
    console.error('Error fetching periodogram:', error);
    return getMockPeriodogramData();
  }
};

export const fetchPhaseFoldedData = async (
  starNumber: number, 
  surveyName: string, 
  campaignId: number, 
  period: number
): Promise<PhaseFoldedPoint[]> => {
  try {
    const response = await fetch(`${API_BASE}/star/${starNumber}/survey/${surveyName}/campaigns/${campaignId}/phase_folded?period=${period}`);
    if (!response.ok) {
      console.error(`Error fetching phase-folded data: ${response.status}`);
      return getMockPhaseFoldedData(period);
    }
    const data = await response.json();
    
    if (!data.phase || !data.flux) {
      console.error('Invalid phase-folded data format received');
      return getMockPhaseFoldedData(period);
    }
    
    const formattedData = data.phase.map((phase: number, index: number) => ({
      phase,
      flux: data.flux[index],
    }));
    
    return formattedData;
  } catch (error) {
    console.error('Error fetching phase-folded data:', error);
    return getMockPhaseFoldedData(period);
  }
};

const getMockAutoAnalysis = (): AutoAnalysisResult => {
  // Generate mock auto analysis data for demonstration
  return {
    predicted_period: 7.8234,
    predicted_class: "sinusoidal",
    class_confidence: 0.876,
    detected_period: 7.8156,
    candidate_periods: [
      { period: 7.8234, score: 0.876, rank: 1 },
      { period: 3.9117, score: 0.654, rank: 2 },
      { period: 15.6468, score: 0.432, rank: 3 },
      { period: 2.6078, score: 0.321, rank: 4 },
      { period: 11.7351, score: 0.198, rank: 5 }
    ]
  };
};

export const fetchAutoAnalysis = async (starNumber: number, surveyName: string, campaignId: number): Promise<AutoAnalysisResult | null> => {
  try {
    const response = await fetch(`${API_BASE}/star/${starNumber}/survey/${surveyName}/campaigns/${campaignId}/auto_analysis`);
    if (!response.ok) {
      console.error(`Error fetching auto analysis: ${response.status}, using mock data for demonstration`);
      return getMockAutoAnalysis();
    }
    const data = await response.json();
    return data as AutoAnalysisResult;
  } catch (error) {
    console.error('Error fetching auto analysis:', error, ', using mock data for demonstration');
    return getMockAutoAnalysis();
  }
};

// Export types for use in components
export type { Survey, Campaign, DataPoint, PeriodogramPoint, PhaseFoldedPoint };
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

export const fetchSurveys = async (starNumber: number): Promise<Survey[]> => {
  try {
    const response = await fetch(`${API_BASE}/star/${starNumber}/surveys`);
    if (!response.ok) {
      console.error(`Error fetching surveys: ${response.status}`);
      return getMockSurveys();
    }
    const data = await response.json();
    
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
      return [];
    }
    const data = await response.json();
    
    if (!data.time || !data.flux) {
      console.error('Invalid campaign data format received');
      return [];
    }
    
    const formattedData = data.time.map((time: number, index: number) => ({
      time,
      flux: data.flux[index],
      error: data.error?.[index] || 0,
    }));
    
    return formattedData;
  } catch (error) {
    console.error('Error fetching campaign data:', error);
    return [];
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
      return [];
    }
    const data = await response.json();
    
    if (!data.periods || !data.powers) {
      console.error('Invalid periodogram data format received');
      return [];
    }
    
    const formattedData = data.periods.map((period: number, index: number) => ({
      period,
      power: data.powers[index],
    }));
    
    return formattedData;
  } catch (error) {
    console.error('Error fetching periodogram:', error);
    return [];
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
      return [];
    }
    const data = await response.json();
    
    if (!data.phase || !data.flux) {
      console.error('Invalid phase-folded data format received');
      return [];
    }
    
    const formattedData = data.phase.map((phase: number, index: number) => ({
      phase,
      flux: data.flux[index],
    }));
    
    return formattedData;
  } catch (error) {
    console.error('Error fetching phase-folded data:', error);
    return [];
  }
};

// Export types for use in components
export type { Survey, Campaign, DataPoint, PeriodogramPoint, PhaseFoldedPoint };
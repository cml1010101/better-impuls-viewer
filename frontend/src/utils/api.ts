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

interface SEDData {
  url: string;
  available: boolean;
  message?: string;
}

export const API_BASE = `${window.location.protocol}//${window.location.hostname}/api`;

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
  // Generate synthetic phase-folded data based on period
  const points: PhaseFoldedPoint[] = [];
  for (let i = 0; i < 100; i++) {
    const phase = i / 100.0; // 0 to 1
    // Use period to modulate the signal slightly for realism
    const periodFactor = Math.log10(period) / 2.0;
    const flux = 1.0 + 0.02 * Math.sin(2 * Math.PI * phase * periodFactor) + 0.005 * Math.random();
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

export const fetchSEDData = async (starNumber: number): Promise<SEDData> => {
  try {
    const response = await fetch(`${API_BASE}/star/${starNumber}/sed`);
    if (!response.ok) {
      console.error(`Error fetching SED data: ${response.status}`);
      return {
        url: "",
        available: false,
        message: "Failed to fetch SED data"
      };
    }
    const data = await response.json();
    
    // If the URL is relative, convert it to absolute URL using API_BASE
    if (data.url && data.url.startsWith('/api/')) {
      data.url = `${API_BASE.replace('/api', '')}${data.url}`;
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching SED data:', error);
    return {
      url: "",
      available: false,
      message: "Error fetching SED data"
    };
  }
};

// Export types for use in components
export type { Survey, Campaign, DataPoint, PeriodogramPoint, PhaseFoldedPoint, SEDData };
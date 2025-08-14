import React, { useState, useEffect } from 'react';
import LightCurveChart from '../Charts/LightCurveChart';
import PeriodogramChart from '../Charts/PeriodogramChart';
import PhaseFoldedChart from '../Charts/PhaseFoldedChart';
import styles from './StarPage.module.css';

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

interface StarPageProps {
  starNumber: number;
  onBackToStarList: () => void;
}

const API_BASE = 'http://localhost:8000/api';

const StarPage: React.FC<StarPageProps> = ({ starNumber, onBackToStarList }) => {
  const [surveys, setSurveys] = useState<Survey[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedSurvey, setExpandedSurvey] = useState<string | null>(null);
  const [selectedCampaign, setSelectedCampaign] = useState<{survey: string, campaign: number} | null>(null);
  
  // Data states
  const [campaignData, setCampaignData] = useState<DataPoint[]>([]);
  const [periodogramData, setPeriodogramData] = useState<PeriodogramPoint[]>([]);
  const [phaseFoldedData, setPhaseFoldedData] = useState<PhaseFoldedPoint[]>([]);
  const [selectedPeriod, setSelectedPeriod] = useState<number | null>(null);
  const [periodInputValue, setPeriodInputValue] = useState<string>('');
  const [dataLoading, setDataLoading] = useState(false);

  useEffect(() => {
    fetchSurveys();
  }, [starNumber]);

  // Fetch campaign data when a campaign is selected
  useEffect(() => {
    if (selectedCampaign) {
      fetchCampaignData(selectedCampaign.survey, selectedCampaign.campaign);
      fetchPeriodogramData(selectedCampaign.survey, selectedCampaign.campaign);
    }
  }, [selectedCampaign, starNumber]);

  // Fetch phase-folded data when period is selected
  useEffect(() => {
    if (selectedCampaign && selectedPeriod) {
      fetchPhaseFoldedData(selectedCampaign.survey, selectedCampaign.campaign, selectedPeriod);
    }
  }, [selectedCampaign, selectedPeriod, starNumber]);

  const fetchSurveys = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/star/${starNumber}/surveys`);
      if (!response.ok) {
        console.error(`Error fetching surveys: ${response.status}`);
        // For demo purposes, show mock data when API is not available
        const mockSurveys = [
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
        setSurveys(mockSurveys);
        return;
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
      
      setSurveys(surveysWithCampaigns);
    } catch (error) {
      console.error('Error fetching surveys:', error);
      // For demo purposes, show mock data when API is not available
      const mockSurveys = [
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
      setSurveys(mockSurveys);
    } finally {
      setLoading(false);
    }
  };

  const fetchCampaignData = async (surveyName: string, campaignId: number) => {
    try {
      setDataLoading(true);
      const response = await fetch(`${API_BASE}/star/${starNumber}/survey/${surveyName}/campaigns/${campaignId}/raw`);
      if (!response.ok) {
        console.error(`Error fetching campaign data: ${response.status}`);
        setCampaignData([]);
        return;
      }
      const data = await response.json();
      
      if (!data.time || !data.flux) {
        console.error('Invalid campaign data format received');
        setCampaignData([]);
        return;
      }
      
      const formattedData = data.time.map((time: number, index: number) => ({
        time,
        flux: data.flux[index],
        error: data.error?.[index] || 0,
      }));
      
      setCampaignData(formattedData);
    } catch (error) {
      console.error('Error fetching campaign data:', error);
      setCampaignData([]);
    } finally {
      setDataLoading(false);
    }
  };

  const fetchPeriodogramData = async (surveyName: string, campaignId: number) => {
    try {
      const response = await fetch(`${API_BASE}/star/${starNumber}/survey/${surveyName}/campaigns/${campaignId}/periodogram`);
      if (!response.ok) {
        console.error(`Error fetching periodogram: ${response.status}`);
        setPeriodogramData([]);
        return;
      }
      const data = await response.json();
      
      if (!data.periods || !data.powers) {
        console.error('Invalid periodogram data format received');
        setPeriodogramData([]);
        return;
      }
      
      const formattedData = data.periods.map((period: number, index: number) => ({
        period,
        power: data.powers[index],
      }));
      
      setPeriodogramData(formattedData);
    } catch (error) {
      console.error('Error fetching periodogram:', error);
      setPeriodogramData([]);
    }
  };

  const fetchPhaseFoldedData = async (surveyName: string, campaignId: number, period: number) => {
    try {
      const response = await fetch(`${API_BASE}/star/${starNumber}/survey/${surveyName}/campaigns/${campaignId}/phase_folded?period=${period}`);
      if (!response.ok) {
        console.error(`Error fetching phase-folded data: ${response.status}`);
        setPhaseFoldedData([]);
        return;
      }
      const data = await response.json();
      
      if (!data.phase || !data.flux) {
        console.error('Invalid phase-folded data format received');
        setPhaseFoldedData([]);
        return;
      }
      
      const formattedData = data.phase.map((phase: number, index: number) => ({
        phase,
        flux: data.flux[index],
      }));
      
      setPhaseFoldedData(formattedData);
    } catch (error) {
      console.error('Error fetching phase-folded data:', error);
      setPhaseFoldedData([]);
    }
  };

  const handlePeriodogramClick = (data: Plotly.PlotMouseEvent) => {
    if (data && data.points && data.points[0]) {
      const period = data.points[0].x;
      if (period && typeof period === 'number') {
        setSelectedPeriod(period);
        setPeriodInputValue(period.toFixed(4));
      }
    }
  };

  const handlePeriodInputChange = (value: string) => {
    setPeriodInputValue(value);
  };

  const handlePeriodSubmit = () => {
    const value = parseFloat(periodInputValue);
    if (value && value >= 0.1 && value <= 100) {
      setSelectedPeriod(value);
    }
  };

  const toggleSurvey = (surveyName: string) => {
    setExpandedSurvey(expandedSurvey === surveyName ? null : surveyName);
  };

  const selectCampaign = (surveyName: string, campaignId: number) => {
    setSelectedCampaign({ survey: surveyName, campaign: campaignId });
    // Reset chart data
    setCampaignData([]);
    setPeriodogramData([]);
    setPhaseFoldedData([]);
    setSelectedPeriod(null);
    setPeriodInputValue('');
  };

  if (loading) {
    return <div className={styles.loading}>Loading surveys...</div>;
  }

  return (
    <div className={styles.starPage}>
      <div className={styles.header}>
        <button onClick={onBackToStarList} className={styles.backButton}>
          ← Back to Star List
        </button>
        <h1>Star {starNumber}</h1>
      </div>

      <div className={styles.content}>
        <div className={styles.surveysPanel}>
          <h2>Surveys</h2>
          {surveys.length === 0 ? (
            <p>No surveys available for this star.</p>
          ) : (
            surveys.map((survey) => (
              <div key={survey.name} className={styles.surveySection}>
                <button
                  className={`${styles.surveyHeader} ${expandedSurvey === survey.name ? styles.expanded : ''}`}
                  onClick={() => toggleSurvey(survey.name)}
                >
                  <span className={styles.surveyName}>{survey.name.toUpperCase()}</span>
                  <span className={styles.campaignCount}>
                    {survey.campaigns.length} campaign{survey.campaigns.length !== 1 ? 's' : ''}
                  </span>
                  <span className={styles.expandIcon}>
                    {expandedSurvey === survey.name ? '−' : '+'}
                  </span>
                </button>
                
                {expandedSurvey === survey.name && (
                  <div className={styles.campaignsList}>
                    {survey.campaigns.length === 0 ? (
                      <p className={styles.noCampaigns}>No campaigns available</p>
                    ) : (
                      survey.campaigns.map((campaign) => (
                        <button
                          key={campaign.campaign_id}
                          className={`${styles.campaignItem} ${
                            selectedCampaign?.survey === survey.name && 
                            selectedCampaign?.campaign === campaign.campaign_id ? 
                            styles.selected : ''
                          }`}
                          onClick={() => selectCampaign(survey.name, campaign.campaign_id)}
                        >
                          <div className={styles.campaignDetails}>
                            <span className={styles.campaignId}>Campaign {campaign.campaign_id}</span>
                            <span className={styles.campaignInfo}>
                              {campaign.data_points} points, {campaign.duration.toFixed(1)} days
                            </span>
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        <div className={styles.dataPanel}>
          {selectedCampaign ? (
            <div className={styles.campaignData}>
              <h3>
                {selectedCampaign.survey.toUpperCase()} - Campaign {selectedCampaign.campaign}
              </h3>
              {dataLoading ? (
                <div className={styles.dataLoading}>Loading campaign data...</div>
              ) : (
                <div className={styles.dataViews}>
                  <div className={styles.lightCurveSection}>
                    {campaignData.length > 0 ? (
                      <LightCurveChart campaignData={campaignData} />
                    ) : (
                      <div className={styles.placeholder}>No light curve data available</div>
                    )}
                  </div>
                  <div className={styles.periodogramSection}>
                    {periodogramData.length > 0 ? (
                      <PeriodogramChart
                        periodogramData={periodogramData}
                        selectedPeriod={selectedPeriod}
                        periodInputValue={periodInputValue}
                        onPeriodogramClick={handlePeriodogramClick}
                        onPeriodInputChange={handlePeriodInputChange}
                        onPeriodSubmit={handlePeriodSubmit}
                      />
                    ) : (
                      <div className={styles.placeholder}>No periodogram data available</div>
                    )}
                  </div>
                  <div className={styles.phaseFoldedSection}>
                    {phaseFoldedData.length > 0 && selectedPeriod ? (
                      <PhaseFoldedChart
                        phaseFoldedData={phaseFoldedData}
                        selectedPeriod={selectedPeriod}
                      />
                    ) : (
                      <div className={styles.placeholder}>
                        {selectedPeriod ? 'Loading phase folded data...' : 'Select a period from the periodogram to view phase folded data'}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className={styles.selectPrompt}>
              <p>Select a campaign from the surveys panel to view data</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StarPage;
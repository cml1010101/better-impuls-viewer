import React, { useState, useEffect, useCallback } from 'react';
import * as Plotly from 'plotly.js';
import LightCurveChart from '../Charts/LightCurveChart';
import PeriodogramChart from '../Charts/PeriodogramChart';
import PhaseFoldedChart from '../Charts/PhaseFoldedChart';
import {
  fetchSurveys,
  fetchCampaignData,
  fetchPeriodogramData,
  fetchPhaseFoldedData,
  fetchAutoAnalysis,
  type Survey,
  type DataPoint,
  type PeriodogramPoint,
  type PhaseFoldedPoint,
  type AutoAnalysisResult
} from '../../utils/api';
import { PeriodCache } from '../../utils/periodCache';
import styles from './StarPage.module.css';

interface StarPageProps {
  starNumber: number;
  onBackToStarList: () => void;
}

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
  const [cachedPeriods, setCachedPeriods] = useState<Array<{
    survey: string;
    campaignId: number;
    period: number;
    timestamp: number;
    isPrimary: boolean;
  }>>([]);
  
  // Auto-normalize state
  const [autoNormalize, setAutoNormalize] = useState<boolean>(false);
  const [autoAnalysisResult, setAutoAnalysisResult] = useState<AutoAnalysisResult | null>(null);

  useEffect(() => {
    loadSurveys();
    loadCachedPeriods();
  }, [starNumber]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadSurveys = useCallback(async () => {
    try {
      setLoading(true);
      const surveysData = await fetchSurveys(starNumber);
      setSurveys(surveysData);
    } finally {
      setLoading(false);
    }
  }, [starNumber]);

  const loadCachedPeriods = useCallback(() => {
    const cached = PeriodCache.getAllCachedPeriodsForStar(starNumber);
    setCachedPeriods(cached);
  }, [starNumber]);

  const loadCampaignData = useCallback(async (surveyName: string, campaignId: number) => {
    try {
      setDataLoading(true);
      const data = await fetchCampaignData(starNumber, surveyName, campaignId);
      setCampaignData(data);
    } finally {
      setDataLoading(false);
    }
  }, [starNumber]);

  const loadPeriodogramData = useCallback(async (surveyName: string, campaignId: number) => {
    const data = await fetchPeriodogramData(starNumber, surveyName, campaignId, autoNormalize);
    setPeriodogramData(data);
  }, [starNumber, autoNormalize]);

  const loadPhaseFoldedData = useCallback(async (surveyName: string, campaignId: number, period: number) => {
    const data = await fetchPhaseFoldedData(starNumber, surveyName, campaignId, period, autoNormalize);
    setPhaseFoldedData(data);
  }, [starNumber, autoNormalize]);

  const loadAutoAnalysis = useCallback(async (surveyName: string, campaignId: number) => {
    const result = await fetchAutoAnalysis(starNumber, surveyName, campaignId, autoNormalize);
    setAutoAnalysisResult(result);
  }, [starNumber, autoNormalize]);

  // Fetch campaign data when a campaign is selected
  useEffect(() => {
    if (selectedCampaign) {
      loadCampaignData(selectedCampaign.survey, selectedCampaign.campaign);
      loadPeriodogramData(selectedCampaign.survey, selectedCampaign.campaign);
      
      // Load cached period for this campaign if available
      const cachedPeriod = PeriodCache.getCachedPeriod(
        starNumber, 
        selectedCampaign.survey, 
        selectedCampaign.campaign
      );
      if (cachedPeriod) {
        setSelectedPeriod(cachedPeriod);
        setPeriodInputValue(cachedPeriod.toFixed(4));
      } else {
        setSelectedPeriod(null);
        setPeriodInputValue('');
      }
    }
  }, [selectedCampaign, starNumber]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch phase-folded data when period is selected
  useEffect(() => {
    if (selectedCampaign && selectedPeriod) {
      loadPhaseFoldedData(selectedCampaign.survey, selectedCampaign.campaign, selectedPeriod);
    }
  }, [selectedCampaign, selectedPeriod, starNumber]); // eslint-disable-line react-hooks/exhaustive-deps

  // Reload data when auto-normalize setting changes
  useEffect(() => {
    if (selectedCampaign) {
      loadPeriodogramData(selectedCampaign.survey, selectedCampaign.campaign);
      if (selectedPeriod) {
        loadPhaseFoldedData(selectedCampaign.survey, selectedCampaign.campaign, selectedPeriod);
      }
    }
  }, [autoNormalize]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleSurvey = (surveyName: string) => {
    setExpandedSurvey(expandedSurvey === surveyName ? null : surveyName);
  };

  const handlePeriodogramClick = (data: Plotly.PlotMouseEvent) => {
    if (data && data.points && data.points[0]) {
      const period = data.points[0].x;
      if (period && typeof period === 'number') {
        setSelectedPeriod(period);
        setPeriodInputValue(period.toFixed(4));
        
        // Cache the selected period
        if (selectedCampaign) {
          PeriodCache.setCachedPeriod(
            starNumber,
            selectedCampaign.survey,
            selectedCampaign.campaign,
            period
          );
          loadCachedPeriods(); // Refresh cached periods display
          
          // Immediately trigger phase folding
          loadPhaseFoldedData(selectedCampaign.survey, selectedCampaign.campaign, period);
        }
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
      
      // Cache the manually entered period
      if (selectedCampaign) {
        PeriodCache.setCachedPeriod(
          starNumber,
          selectedCampaign.survey,
          selectedCampaign.campaign,
          value
        );
        loadCachedPeriods(); // Refresh cached periods display
      }
    }
  };

  const handleLoadCachedPeriod = (period: number) => {
    setSelectedPeriod(period);
    setPeriodInputValue(period.toFixed(4));
  };

  const handleSetPrimaryPeriod = (survey: string, campaignId: number) => {
    PeriodCache.setPrimaryPeriod(starNumber, survey, campaignId);
    loadCachedPeriods(); // Refresh cached periods display
  };

  const handleAutoNormalizeToggle = (enabled: boolean) => {
    setAutoNormalize(enabled);
  };

  const handleRunAutoAnalysis = async () => {
    if (selectedCampaign) {
      await loadAutoAnalysis(selectedCampaign.survey, selectedCampaign.campaign);
    }
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
        <div className={styles.controls}>
          <label className={styles.toggleLabel}>
            <input
              type="checkbox"
              checked={autoNormalize}
              onChange={(e) => handleAutoNormalizeToggle(e.target.checked)}
              className={styles.toggleInput}
            />
            <span className={styles.toggleText}>Auto-Linearization</span>
            <span className={styles.toggleDescription}>
              Remove linear trends from data
            </span>
          </label>
        </div>
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
          
          {/* Cached Periods Section */}
          {cachedPeriods.length > 0 && (
            <div className={styles.cachedPeriodsSection}>
              <h3>Cached Periods</h3>
              <p className={styles.cachedDescription}>
                Previously selected periods for this star
              </p>
              {cachedPeriods.map((cached, index) => (
                <div key={index} className={styles.cachedPeriodItemContainer}>
                  <button
                    className={`${styles.cachedPeriodItem} ${
                      selectedCampaign?.survey === cached.survey && 
                      selectedCampaign?.campaign === cached.campaignId &&
                      selectedPeriod === cached.period ? 
                      styles.active : ''
                    } ${cached.isPrimary ? styles.primary : ''}`}
                    onClick={() => handleLoadCachedPeriod(cached.period)}
                    title={`Cached ${new Date(cached.timestamp).toLocaleString()}`}
                  >
                    <div className={styles.cachedPeriodDetails}>
                      <span className={styles.cachedPeriodValue}>
                        {cached.period.toFixed(4)} days
                        {cached.isPrimary && <span className={styles.primaryBadge}>PRIMARY</span>}
                      </span>
                      <span className={styles.cachedPeriodSource}>
                        {cached.survey.toUpperCase()} Campaign {cached.campaignId}
                      </span>
                    </div>
                  </button>
                  <button
                    className={`${styles.primaryButton} ${cached.isPrimary ? styles.primaryActive : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSetPrimaryPeriod(cached.survey, cached.campaignId);
                    }}
                    title={cached.isPrimary ? "Remove as primary" : "Set as primary"}
                  >
                    ⭐
                  </button>
                </div>
              ))}
            </div>
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
                      <div>
                        <div className={styles.periodogramControls}>
                          <button 
                            className={styles.autoAnalysisButton}
                            onClick={handleRunAutoAnalysis}
                            disabled={!selectedCampaign}
                          >
                            Run Auto Analysis
                          </button>
                          {autoAnalysisResult && (
                            <div className={styles.autoAnalysisResult}>
                              <span>Auto-detected period: <strong>{autoAnalysisResult.predicted_period.toFixed(4)} days</strong></span>
                              <span>Confidence: {(autoAnalysisResult.confidence * 100).toFixed(1)}%</span>
                              <span>Type: {autoAnalysisResult.variability_type}</span>
                            </div>
                          )}
                        </div>
                        <PeriodogramChart
                          periodogramData={periodogramData}
                          selectedPeriod={selectedPeriod}
                          periodInputValue={periodInputValue}
                          onPeriodogramClick={handlePeriodogramClick}
                          onPeriodInputChange={handlePeriodInputChange}
                          onPeriodSubmit={handlePeriodSubmit}
                        />
                      </div>
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
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
  fetchCategories,
  type Survey,
  type DataPoint,
  type PeriodogramPoint,
  type PhaseFoldedPoint
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
    period: number | null;
    timestamp: number;
    isPrimary: boolean;
    category?: string;
  }>>([]);
  const [totalCachedCount, setTotalCachedCount] = useState(0);
  const [starsWithCache, setStarsWithCache] = useState<number[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [showCsvPanel, setShowCsvPanel] = useState(false);

  useEffect(() => {
    loadSurveys();
    loadCachedPeriods();
    loadCategories();
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
    setTotalCachedCount(PeriodCache.getTotalCachedCount());
    setStarsWithCache(PeriodCache.getStarsWithCachedData());
  }, [starNumber]);

  const loadCategories = useCallback(async () => {
    try {
      const fetchedCategories = await fetchCategories();
      setCategories(fetchedCategories);
    } catch (error) {
      console.error('Error loading categories:', error);
    }
  }, []);

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
    const data = await fetchPeriodogramData(starNumber, surveyName, campaignId);
    setPeriodogramData(data);
  }, [starNumber]);

  const loadPhaseFoldedData = useCallback(async (surveyName: string, campaignId: number, period: number) => {
    const data = await fetchPhaseFoldedData(starNumber, surveyName, campaignId, period);
    setPhaseFoldedData(data);
  }, [starNumber]);

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

  const handleLoadCachedPeriod = (period: number | null) => {
    if (period === null) {
      setSelectedPeriod(null);
      setPeriodInputValue('');
    } else {
      setSelectedPeriod(period);
      setPeriodInputValue(period.toFixed(4));
    }
  };

  const handleSetPrimaryPeriod = (survey: string, campaignId: number) => {
    PeriodCache.setPrimaryPeriod(starNumber, survey, campaignId);
    loadCachedPeriods(); // Refresh cached periods display
  };

  const handleSetPeriodCategory = (survey: string, campaignId: number, category: string) => {
    PeriodCache.setCachedPeriodCategory(starNumber, survey, campaignId, category);
    loadCachedPeriods(); // Refresh cached periods display
  };

  const handleRemoveCachedPeriod = (survey: string, campaignId: number) => {
    PeriodCache.removeCachedPeriod(starNumber, survey, campaignId);
    loadCachedPeriods(); // Refresh cached periods display
  };

  const handleExportCSV = () => {
    PeriodCache.downloadCSV(`star_periods_${starNumber}.csv`);
  };

  const handleExportAllCSV = () => {
    PeriodCache.downloadCSV(`all_star_periods.csv`);
  };

  const handleImportCSV = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const csvContent = e.target?.result as string;
        const result = PeriodCache.importFromCSV(csvContent, false);
        
        if (result.success) {
          alert(`Successfully imported ${result.imported} period(s).${result.errors.length > 0 ? ` Warnings: ${result.errors.join(', ')}` : ''}`);
          loadCachedPeriods(); // Refresh display
        } else {
          alert(`Import failed: ${result.errors.join(', ')}`);
        }
      };
      reader.readAsText(file);
    }
    // Reset the input value so the same file can be selected again
    event.target.value = '';
  };

  const handleClearAllCache = () => {
    if (confirm(`Clear ALL cached periods for ALL stars? This will remove ${totalCachedCount} cached periods across ${starsWithCache.length} stars. This action cannot be undone.`)) {
      PeriodCache.clearCache();
      loadCachedPeriods(); // Refresh display
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
                  <div className={styles.cachedPeriodMain}>
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
                          {cached.period !== null ? `${cached.period.toFixed(4)} days` : 'No period data'}
                          {cached.isPrimary && <span className={styles.primaryBadge}>PRIMARY</span>}
                          {cached.category && <span className={styles.categoryBadge}>{cached.category}</span>}
                        </span>
                        <span className={styles.cachedPeriodSource}>
                          {cached.survey.toUpperCase()} Campaign {cached.campaignId}
                        </span>
                      </div>
                    </button>
                    <div className={styles.cachedPeriodActions}>
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
                      <button
                        className={styles.removeButton}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm('Remove this cached period?')) {
                            handleRemoveCachedPeriod(cached.survey, cached.campaignId);
                          }
                        }}
                        title="Remove from cache"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                  <div className={styles.categorySelectContainer}>
                    <select
                      className={styles.categorySelect}
                      value={cached.category || ''}
                      onChange={(e) => handleSetPeriodCategory(cached.survey, cached.campaignId, e.target.value)}
                      title="Set category"
                    >
                      <option value="">No category</option>
                      {categories.map(category => (
                        <option key={category} value={category}>{category}</option>
                      ))}
                    </select>
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {/* CSV Data Management Panel */}
          <div className={styles.csvManagementSection}>
            <h3>
              Data Management
              <button 
                className={styles.toggleButton}
                onClick={() => setShowCsvPanel(!showCsvPanel)}
                title={showCsvPanel ? "Hide CSV options" : "Show CSV options"}
              >
                {showCsvPanel ? '−' : '+'}
              </button>
            </h3>
            
            {showCsvPanel && (
              <div className={styles.csvPanel}>
                <p className={styles.csvDescription}>
                  Export cached periods to CSV or import period data from CSV file
                </p>
                
                <div className={styles.csvSectionHeader}>
                  <h4>Current Star ({starNumber})</h4>
                </div>
                <div className={styles.csvActions}>
                  <button
                    className={styles.csvButton}
                    onClick={handleExportCSV}
                    disabled={cachedPeriods.length === 0}
                    title="Download cached periods for this star as CSV file"
                  >
                    📥 Export Star {starNumber}
                  </button>
                  
                  <button
                    className={styles.csvButton}
                    onClick={() => {
                      if (confirm('Clear all cached periods for this star?')) {
                        PeriodCache.clearCacheForStar(starNumber);
                        loadCachedPeriods();
                      }
                    }}
                    disabled={cachedPeriods.length === 0}
                    title="Remove all cached periods for this star"
                  >
                    🗑️ Clear Star {starNumber}
                  </button>
                </div>

                <div className={styles.csvSectionHeader}>
                  <h4>All Stars ({totalCachedCount} periods across {starsWithCache.length} stars)</h4>
                  {starsWithCache.length > 0 && (
                    <p className={styles.starsWithCache}>
                      Stars with cached data: {starsWithCache.join(', ')}
                    </p>
                  )}
                </div>
                <div className={styles.csvActions}>
                  <button
                    className={styles.csvButton}
                    onClick={handleExportAllCSV}
                    disabled={totalCachedCount === 0}
                    title="Download all cached periods from all stars as CSV file"
                  >
                    📥 Export All Stars
                  </button>
                  
                  <label className={styles.csvButton}>
                    📤 Import from CSV
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleImportCSV}
                      style={{ display: 'none' }}
                    />
                  </label>
                  
                  <button
                    className={styles.csvButton}
                    onClick={handleClearAllCache}
                    disabled={totalCachedCount === 0}
                    title="Remove ALL cached periods for ALL stars"
                  >
                    🗑️ Clear All Stars
                  </button>
                </div>
                
                <div className={styles.csvInfo}>
                  <details className={styles.csvHelp}>
                    <summary>CSV Format Information</summary>
                    <div className={styles.csvHelpContent}>
                      <p><strong>Required columns:</strong></p>
                      <ul>
                        <li><code>star_number</code> - Integer star identifier</li>
                        <li><code>survey</code> - Survey name (hubble, kepler, tess)</li>
                        <li><code>campaign_id</code> - Integer campaign identifier</li>
                        <li><code>period</code> - Period in days (-9 for no period)</li>
                        <li><code>category</code> - Light curve category</li>
                        <li><code>is_primary</code> - true/false for primary period</li>
                        <li><code>timestamp</code> - Unix timestamp</li>
                      </ul>
                      <p><strong>Available categories:</strong></p>
                      <div className={styles.categoriesList}>
                        {categories.map(category => (
                          <span key={category} className={styles.categoryTag}>{category}</span>
                        ))}
                      </div>
                    </div>
                  </details>
                </div>
              </div>
            )}
          </div>
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
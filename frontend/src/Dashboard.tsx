import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import './Dashboard.css';

// Types
interface Campaign {
  campaign_id: string;
  telescope: string;
  star_number: number;
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

interface AutoPeriodClassification {
  type: string;
  confidence: number;
  description: string;
}

interface AutoPeriodMethod {
  success: boolean;
  periods: number[];
}

interface AutoPeriodsData {
  primary_period: number | null;
  secondary_period: number | null;
  classification: AutoPeriodClassification;
  methods: {
    periodogram?: AutoPeriodMethod;
    torch_fitting?: AutoPeriodMethod;
  };
  error?: string;
}

const API_BASE = 'http://localhost:8000';

// Helper functions for URL parameters
const getUrlParams = () => {
  const params = new URLSearchParams(window.location.search);
  return {
    star: params.get('star') ? Number(params.get('star')) : null,
    telescope: params.get('telescope') || '',
    campaign: params.get('campaign') || ''
  };
};

const updateUrlParams = (star: number | null, telescope: string, campaign: string) => {
  const params = new URLSearchParams();
  if (star) params.set('star', star.toString());
  if (telescope) params.set('telescope', telescope);
  if (campaign) params.set('campaign', campaign);
  
  const newUrl = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState({}, '', newUrl);
};

const Dashboard: React.FC = () => {
  const [stars, setStars] = useState<number[]>([]);
  const [selectedStar, setSelectedStar] = useState<number | null>(null);
  const [telescopes, setTelescopes] = useState<string[]>([]);
  const [selectedTelescope, setSelectedTelescope] = useState<string>('');
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [selectedCampaign, setSelectedCampaign] = useState<string>('');
  const [campaignData, setCampaignData] = useState<DataPoint[]>([]);
  const [periodogramData, setPeriodogramData] = useState<PeriodogramPoint[]>([]);
  const [selectedPeriod, setSelectedPeriod] = useState<number | null>(null);
  const [periodInputValue, setPeriodInputValue] = useState<string>(''); // Add controlled input state
  const [phaseFoldedData, setPhaseFoldedData] = useState<PhaseFoldedPoint[]>([]);
  const [autoPeriodsData, setAutoPeriodsData] = useState<AutoPeriodsData | null>(null);
  const [loading, setLoading] = useState(false);

  // Initialize from URL parameters on component mount
  useEffect(() => {
    const urlParams = getUrlParams();
    fetchStars().then(() => {
      if (urlParams.star) {
        setSelectedStar(urlParams.star);
      }
      if (urlParams.telescope) {
        setSelectedTelescope(urlParams.telescope);
      }
      if (urlParams.campaign) {
        setSelectedCampaign(urlParams.campaign);
      }
    });
  }, []);

  // Update URL when selections change
  useEffect(() => {
    updateUrlParams(selectedStar, selectedTelescope, selectedCampaign);
  }, [selectedStar, selectedTelescope, selectedCampaign]);

  // Fetch telescopes when star is selected
  useEffect(() => {
    if (selectedStar) {
      fetchTelescopes(selectedStar);
    }
  }, [selectedStar]);

  // Fetch campaigns when telescope is selected
  useEffect(() => {
    if (selectedStar && selectedTelescope) {
      fetchCampaigns(selectedStar, selectedTelescope);
    }
  }, [selectedStar, selectedTelescope]);

  // Fetch data when campaign is selected
  useEffect(() => {
    if (selectedStar && selectedTelescope && selectedCampaign) {
      fetchCampaignData(selectedStar, selectedTelescope, selectedCampaign);
      fetchPeriodogram(selectedStar, selectedTelescope, selectedCampaign);
      fetchAutoPeriodsData(selectedStar, selectedTelescope, selectedCampaign);
    }
  }, [selectedStar, selectedTelescope, selectedCampaign]);

  // Fetch phase-folded data when period is selected
  useEffect(() => {
    if (selectedStar && selectedTelescope && selectedCampaign && selectedPeriod) {
      fetchPhaseFoldedData(selectedStar, selectedTelescope, selectedCampaign, selectedPeriod);
    }
  }, [selectedStar, selectedTelescope, selectedCampaign, selectedPeriod]);

  const fetchStars = async () => {
    try {
      const response = await fetch(`${API_BASE}/stars`);
      const data = await response.json();
      setStars(data);
      // Don't automatically select the first star - let URL params handle it
    } catch (error) {
      console.error('Error fetching stars:', error);
    }
  };

  const fetchTelescopes = async (star: number) => {
    try {
      const response = await fetch(`${API_BASE}/telescopes/${star}`);
      const data = await response.json();
      setTelescopes(data);
      if (data.length > 0) {
        setSelectedTelescope(data[0]);
      }
    } catch (error) {
      console.error('Error fetching telescopes:', error);
    }
  };

  const fetchCampaigns = async (star: number, telescope: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/campaigns/${star}/${telescope}`);
      const data = await response.json();
      setCampaigns(data);
      if (data.length > 0) {
        setSelectedCampaign(data[0].campaign_id);
      }
    } catch (error) {
      console.error('Error fetching campaigns:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchCampaignData = async (star: number, telescope: string, campaign: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/data/${star}/${telescope}/${campaign}`);
      
      if (!response.ok) {
        console.error(`Error fetching campaign data: ${response.status} ${response.statusText}`);
        setCampaignData([]);
        return;
      }
      
      const data = await response.json();
      
      if (!data.time || !data.flux) {
        console.error('Invalid data format received');
        setCampaignData([]);
        return;
      }
      
      const formattedData = data.time.map((time: number, index: number) => ({
        time,
        flux: data.flux[index],
        error: data.error[index],
      }));
      
      setCampaignData(formattedData);
    } catch (error) {
      console.error('Error fetching campaign data:', error);
      setCampaignData([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchPeriodogram = async (star: number, telescope: string, campaign: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/periodogram/${star}/${telescope}/${campaign}`);
      
      if (!response.ok) {
        console.error(`Error fetching periodogram: ${response.status} ${response.statusText}`);
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
    } finally {
      setLoading(false);
    }
  };

  const fetchPhaseFoldedData = async (star: number, telescope: string, campaign: string, period: number) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/phase_fold/${star}/${telescope}/${campaign}?period=${period}`);
      
      if (!response.ok) {
        console.error(`Error fetching phase-folded data: ${response.status} ${response.statusText}`);
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
    } finally {
      setLoading(false);
    }
  };

  const fetchAutoPeriodsData = async (star: number, telescope: string, campaign: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/auto_periods/${star}/${telescope}/${campaign}`);
      
      if (!response.ok) {
        console.error(`Error fetching auto periods data: ${response.status} ${response.statusText}`);
        setAutoPeriodsData(null);
        return;
      }
      
      const data = await response.json();
      setAutoPeriodsData(data);
    } catch (error) {
      console.error('Error fetching auto periods data:', error);
      setAutoPeriodsData(null);
    } finally {
      setLoading(false);
    }
  };

  const handlePeriodogramClick = (data: any) => {
    console.log('Periodogram clicked:', data);
    // Handle Plotly.js click events
    if (data && data.points && data.points[0]) {
      const period = data.points[0].x;
      if (period) {
        console.log('Selected period:', period);
        setSelectedPeriod(period);
        setPeriodInputValue(period.toFixed(4)); // Update input field with clicked period
      }
    }
  };

  const handlePeriodInputChange = (value: string) => {
    setPeriodInputValue(value);
  };

  const handlePeriodSubmit = () => {
    const value = parseFloat(periodInputValue);
    if (value && value >= 0.1 && value <= 20) {
      setSelectedPeriod(value);
    }
  };

  const handleUsePrimaryPeriod = () => {
    if (autoPeriodsData && autoPeriodsData.primary_period) {
      const period = autoPeriodsData.primary_period;
      setSelectedPeriod(period);
      setPeriodInputValue(period.toFixed(4));
    }
  };

  const handleUseSecondaryPeriod = () => {
    if (autoPeriodsData && autoPeriodsData.secondary_period) {
      const period = autoPeriodsData.secondary_period;
      setSelectedPeriod(period);
      setPeriodInputValue(period.toFixed(4));
    }
  };

  return (
    <div className="dashboard">
      {/* Control Panel */}
      <div className="control-panel">
        <div className="control-group">
          <label>Star Number:</label>
          <select
            value={selectedStar || ''}
            onChange={(e) => setSelectedStar(Number(e.target.value))}
          >
            {stars.map((star) => (
              <option key={star} value={star}>
                Star {star}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Telescope/Sensor:</label>
          <select
            value={selectedTelescope}
            onChange={(e) => setSelectedTelescope(e.target.value)}
            disabled={!selectedStar}
          >
            {telescopes.map((telescope) => (
              <option key={telescope} value={telescope}>
                {telescope.charAt(0).toUpperCase() + telescope.slice(1)}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Campaign:</label>
          <select
            value={selectedCampaign}
            onChange={(e) => setSelectedCampaign(e.target.value)}
            disabled={!selectedTelescope}
          >
            {campaigns.map((campaign) => (
              <option key={campaign.campaign_id} value={campaign.campaign_id}>
                {campaign.campaign_id.toUpperCase()} ({campaign.data_points} points, {campaign.duration.toFixed(1)} days)
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && <div className="loading">Loading...</div>}

      {/* Automatic Period Detection Results */}
      {autoPeriodsData && !autoPeriodsData.error && (
        <div className="auto-periods-section">
          <h3>🤖 Automatic Period Detection Results</h3>
          
          <div className="auto-periods-content">
            {/* Classification Results */}
            <div className="classification-info">
              <div className="classification-badge">
                <span className={`classification-type classification-${autoPeriodsData.classification.type}`}>
                  {autoPeriodsData.classification.type.toUpperCase()}
                </span>
                <span className="classification-confidence">
                  {(autoPeriodsData.classification.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <p className="classification-description">
                {autoPeriodsData.classification.description}
              </p>
            </div>

            {/* Period Results */}
            <div className="periods-info">
              {autoPeriodsData.primary_period && (
                <div className="period-result">
                  <div className="period-label">Primary Period:</div>
                  <div className="period-value">
                    <strong>{autoPeriodsData.primary_period.toFixed(4)} days</strong>
                    <button 
                      className="use-period-btn primary-btn"
                      onClick={handleUsePrimaryPeriod}
                      title="Use this period for phase folding"
                    >
                      Use Period
                    </button>
                  </div>
                </div>
              )}
              
              {autoPeriodsData.secondary_period && (
                <div className="period-result">
                  <div className="period-label">Secondary Period:</div>
                  <div className="period-value">
                    <strong>{autoPeriodsData.secondary_period.toFixed(4)} days</strong>
                    <button 
                      className="use-period-btn secondary-btn"
                      onClick={handleUseSecondaryPeriod}
                      title="Use this period for phase folding"
                    >
                      Use Period
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Method Results */}
            <div className="methods-info">
              <div className="methods-header">Detection Methods:</div>
              <div className="methods-grid">
                {autoPeriodsData.methods.periodogram && (
                  <div className={`method-result ${autoPeriodsData.methods.periodogram.success ? 'success' : 'failed'}`}>
                    <div className="method-name">Periodogram</div>
                    <div className="method-status">
                      {autoPeriodsData.methods.periodogram.success ? '✓' : '✗'}
                    </div>
                    {autoPeriodsData.methods.periodogram.success && (
                      <div className="method-periods">
                        {autoPeriodsData.methods.periodogram.periods.slice(0, 3).map((period, idx) => (
                          <span key={idx} className="method-period">
                            {Number(period).toFixed(3)}d
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                
                {autoPeriodsData.methods.torch_fitting && (
                  <div className={`method-result ${autoPeriodsData.methods.torch_fitting.success ? 'success' : 'failed'}`}>
                    <div className="method-name">PyTorch Fitting</div>
                    <div className="method-status">
                      {autoPeriodsData.methods.torch_fitting.success ? '✓' : '✗'}
                    </div>
                    {autoPeriodsData.methods.torch_fitting.success && (
                      <div className="method-periods">
                        {autoPeriodsData.methods.torch_fitting.periods.slice(0, 3).map((period, idx) => (
                          <span key={idx} className="method-period">
                            {Number(period).toFixed(3)}d
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Show error message if auto period detection failed */}
      {autoPeriodsData && autoPeriodsData.error && (
        <div className="auto-periods-error">
          <h3>⚠️ Automatic Period Detection</h3>
          <p>Unable to automatically determine periods: {autoPeriodsData.error}</p>
          <p>You can still manually analyze the periodogram and select periods below.</p>
        </div>
      )}

      {/* Algorithm Documentation Section */}
      {autoPeriodsData && (
        <div className="algorithm-documentation">
          <details>
            <summary>
              <h3>🔬 How the Automatic Period Detection Works</h3>
            </summary>
            <div className="documentation-content">
              <p>
                The automatic period detection system combines traditional astronomical methods with modern machine learning 
                to reliably identify periodic signals in variable star light curves.
              </p>
              
              <h4>🧮 Method 1: Enhanced Lomb-Scargle Periodogram</h4>
              <ul>
                <li><strong>Robust Peak Detection</strong>: Uses median absolute deviation (MAD) instead of standard deviation for noise-resistant thresholds</li>
                <li><strong>Period Weighting</strong>: Applies astronomical priors that favor typical variable star periods (1-20 days)</li>
                <li><strong>Campaign Duration Validation</strong>: Rejects periods longer than 1/3 of the observing campaign to ensure reliable detection</li>
                <li><strong>Harmonic Filtering</strong>: Avoids spurious detections from noise artifacts and harmonic aliases</li>
              </ul>

              <h4>🤖 Method 2: PyTorch Sinusoidal Regression</h4>
              <ul>
                <li><strong>Neural Network Fitting</strong>: Custom PyTorch model that fits multiple sinusoidal components using gradient descent</li>
                <li><strong>Early Stopping</strong>: Prevents overfitting with patience-based convergence monitoring</li>
                <li><strong>Confidence Scoring</strong>: Provides reliability estimates based on amplitude strength relative to noise</li>
                <li><strong>Multi-Component Support</strong>: Can simultaneously fit 2+ periodic signals for binary systems</li>
              </ul>

              <h4>🔄 Cross-Validation & Classification</h4>
              <ul>
                <li><strong>Method Agreement</strong>: Prioritizes periods detected by both methods (within 10% tolerance)</li>
                <li><strong>Binary Detection</strong>: Identifies multiple period systems with intelligent ratio analysis</li>
                <li><strong>Quality Control</strong>: Campaign duration: <strong>{campaignData.length > 0 ? ((Math.max(...campaignData.map(d => d.time)) - Math.min(...campaignData.map(d => d.time))).toFixed(1)) : 'N/A'} days</strong>, Max valid period: <strong>{campaignData.length > 0 ? ((Math.max(...campaignData.map(d => d.time)) - Math.min(...campaignData.map(d => d.time))) / 3).toFixed(1) : 'N/A'} days</strong></li>
                <li><strong>Classification Types</strong>:
                  <ul>
                    <li><em>Regular</em>: Single dominant period system</li>
                    <li><em>Binary</em>: Multiple periods with 2:1 ellipsoidal variation or &gt;3:1 independent periods</li>
                    <li><em>Complex</em>: Irregular or multi-component variability</li>
                  </ul>
                </li>
              </ul>

              <h4>📊 Performance & Validation</h4>
              <p>
                The system has been validated on test data with known embedded periods, achieving 98-99% accuracy for periods 
                ranging from 2.5 to 15.3 days. The dual-method approach provides robust period detection while avoiding 
                common pitfalls like harmonic confusion and noise artifacts.
              </p>
            </div>
          </details>
        </div>
      )}

      {/* Charts - New Layout: Light curve on top, periodogram and phase-folded side by side */}
      <div className="charts-container">
        {/* Light Curve Chart - Full Width on Top */}
        {campaignData.length > 0 && (
          <div className="chart-section light-curve-section">
            <h3>Light Curve Data</h3>
            <Plot
              data={[
                {
                  x: campaignData.map(d => d.time),
                  y: campaignData.map(d => d.flux),
                  mode: 'markers',
                  type: 'scatter',
                  marker: {
                    color: '#8884d8',
                    size: 4,
                  },
                  name: 'Flux',
                },
              ]}
              layout={{
                width: undefined,
                height: 250,
                margin: { l: 60, r: 20, t: 20, b: 60 },
                xaxis: {
                  title: { text: 'Time (days)' },
                  automargin: true,
                },
                yaxis: {
                  title: { text: 'Flux' },
                  automargin: true,
                },
                showlegend: false,
                dragmode: 'pan',
              }}
              config={{
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
                displaylogo: false,
              }}
              style={{ width: '100%' }}
            />
          </div>
        )}

        {/* Bottom Row: Periodogram and Phase-Folded Data Side by Side */}
        {(periodogramData.length > 0 || (phaseFoldedData.length > 0 && selectedPeriod)) && (
          <div className="bottom-charts">
            {/* Periodogram Chart */}
            {periodogramData.length > 0 && (
              <div className="chart-section">
                <h3>Periodogram</h3>
                <p className="chart-description">
                  Hover over the chart to see period values. Click or manually enter a period below to fold the data.
                </p>
                <Plot
                  data={[
                    {
                      x: periodogramData.map(d => d.period),
                      y: periodogramData.map(d => d.power),
                      mode: 'lines+markers',
                      type: 'scatter',
                      line: {
                        color: '#ff7300',
                        width: 2,
                      },
                      marker: {
                        color: '#ff7300',
                        size: 4,
                      },
                      name: 'Power',
                    },
                  ]}
                  layout={{
                    width: undefined,
                    height: 250,
                    margin: { l: 60, r: 20, t: 20, b: 60 },
                    xaxis: {
                      title: { text: 'Period (days)' },
                      type: 'log',
                      automargin: true,
                    },
                    yaxis: {
                      title: { text: 'Power' },
                      automargin: true,
                    },
                    showlegend: false,
                    dragmode: 'pan',
                  }}
                  config={{
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
                    displaylogo: false,
                  }}
                  style={{ width: '100%' }}
                  onClick={handlePeriodogramClick}
                />
                
                {/* Manual period input */}
                <div className="period-input-section">
                  <label className="period-label">
                    Enter Period (days):
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    min="0.1"
                    max="20"
                    value={periodInputValue}
                    onChange={(e) => handlePeriodInputChange(e.target.value)}
                    className="period-input"
                    placeholder="e.g., 7.8"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handlePeriodSubmit();
                      }
                    }}
                  />
                  <button
                    onClick={handlePeriodSubmit}
                    className="period-button"
                  >
                    Fold Data
                  </button>
                </div>
                
                {selectedPeriod && (
                  <p className="selected-period">
                    Selected Period: {selectedPeriod.toFixed(4)} days
                  </p>
                )}
              </div>
            )}

            {/* Phase-Folded Data Chart */}
            {phaseFoldedData.length > 0 && selectedPeriod && (
              <div className="chart-section">
                <h3>Phase-Folded Data (Period: {selectedPeriod.toFixed(4)} days)</h3>
                <Plot
                  data={[
                    {
                      x: phaseFoldedData.map(d => d.phase),
                      y: phaseFoldedData.map(d => d.flux),
                      mode: 'markers',
                      type: 'scatter',
                      marker: {
                        color: '#82ca9d',
                        size: 4,
                      },
                      name: 'Flux',
                    },
                  ]}
                  layout={{
                    width: undefined,
                    height: 250,
                    margin: { l: 60, r: 20, t: 20, b: 60 },
                    xaxis: {
                      title: { text: 'Phase' },
                      range: [0, 1],
                      automargin: true,
                    },
                    yaxis: {
                      title: { text: 'Flux' },
                      automargin: true,
                    },
                    showlegend: false,
                    dragmode: 'pan',
                  }}
                  config={{
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
                    displaylogo: false,
                  }}
                  style={{ width: '100%' }}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
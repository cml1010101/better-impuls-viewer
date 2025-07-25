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

const API_BASE = 'http://localhost:8000';

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
  const [loading, setLoading] = useState(false);

  // Fetch available stars on component mount
  useEffect(() => {
    fetchStars();
  }, []);

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
      if (data.length > 0) {
        setSelectedStar(data[0]);
      }
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
      const data = await response.json();
      
      const formattedData = data.time.map((time: number, index: number) => ({
        time,
        flux: data.flux[index],
        error: data.error[index],
      }));
      
      setCampaignData(formattedData);
    } catch (error) {
      console.error('Error fetching campaign data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPeriodogram = async (star: number, telescope: string, campaign: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/periodogram/${star}/${telescope}/${campaign}`);
      const data = await response.json();
      
      const formattedData = data.periods.map((period: number, index: number) => ({
        period,
        power: data.powers[index],
      }));
      
      setPeriodogramData(formattedData);
    } catch (error) {
      console.error('Error fetching periodogram:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPhaseFoldedData = async (star: number, telescope: string, campaign: string, period: number) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/phase_fold/${star}/${telescope}/${campaign}?period=${period}`);
      const data = await response.json();
      
      const formattedData = data.phase.map((phase: number, index: number) => ({
        phase,
        flux: data.flux[index],
      }));
      
      setPhaseFoldedData(formattedData);
    } catch (error) {
      console.error('Error fetching phase-folded data:', error);
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
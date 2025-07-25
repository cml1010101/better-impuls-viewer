import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';
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
    // Try different ways to access the period data from Recharts onClick
    let period = null;
    
    if (data && data.activeLabel) {
      // activeLabel often contains the x-axis value (period in our case)
      period = data.activeLabel;
    } else if (data && data.activePayload && data.activePayload[0] && data.activePayload[0].payload) {
      period = data.activePayload[0].payload.period;
    }
    
    if (period) {
      console.log('Selected period:', period);
      setSelectedPeriod(period);
      setPeriodInputValue(period.toFixed(4)); // Update input field with clicked period
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

      {/* Charts */}
      <div className="charts-container">
        {/* Original Data Chart */}
        {campaignData.length > 0 && (
          <div className="chart-section">
            <h3>Light Curve Data</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart data={campaignData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="time" 
                  type="number" 
                  scale="linear"
                  domain={['dataMin', 'dataMax']}
                  label={{ value: 'Time (days)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  dataKey="flux"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  label={{ value: 'Flux', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value, name) => [Number(value).toFixed(6), name]}
                  labelFormatter={(value) => `Time: ${Number(value).toFixed(3)} days`}
                />
                <Scatter dataKey="flux" fill="#8884d8" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Periodogram Chart */}
        {periodogramData.length > 0 && (
          <div className="chart-section">
            <h3>Periodogram</h3>
            <p style={{ textAlign: 'center', marginBottom: '1rem', color: '#666' }}>
              Hover over the chart to see period values. Click or manually enter a period below to fold the data.
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={periodogramData} onClick={handlePeriodogramClick}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="period"
                  scale="log"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  label={{ value: 'Period (days)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'Power', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value, name) => [Number(value).toFixed(4), name]}
                  labelFormatter={(value) => `Period: ${Number(value).toFixed(4)} days`}
                />
                <Line 
                  type="monotone" 
                  dataKey="power" 
                  stroke="#ff7300" 
                  strokeWidth={2}
                  dot={{ fill: '#ff7300', strokeWidth: 2, r: 2 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
            
            {/* Manual period input */}
            <div style={{ textAlign: 'center', marginTop: '1rem' }}>
              <label style={{ marginRight: '0.5rem', fontWeight: '600' }}>
                Enter Period (days):
              </label>
              <input
                type="number"
                step="0.0001"
                min="0.1"
                max="20"
                value={periodInputValue}
                onChange={(e) => handlePeriodInputChange(e.target.value)}
                style={{
                  padding: '0.5rem',
                  border: '2px solid #e1e8ed',
                  borderRadius: '4px',
                  marginRight: '0.5rem',
                  width: '120px'
                }}
                placeholder="e.g., 7.8"
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handlePeriodSubmit();
                  }
                }}
              />
              <button
                onClick={handlePeriodSubmit}
                style={{
                  padding: '0.5rem 1rem',
                  background: '#667eea',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
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
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart data={phaseFoldedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="phase"
                  type="number"
                  domain={[0, 1]}
                  label={{ value: 'Phase', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  dataKey="flux"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  label={{ value: 'Flux', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value, name) => [Number(value).toFixed(6), name]}
                  labelFormatter={(value) => `Phase: ${Number(value).toFixed(3)}`}
                />
                <Scatter dataKey="flux" fill="#82ca9d" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
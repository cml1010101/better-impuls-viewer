import React, { useState, useEffect } from 'react';
import ControlPanel from '../ControlPanel/ControlPanel';
import AutoPeriodsSection from '../AutoPeriodsSection/AutoPeriodsSection';
import SEDImageSection from '../SEDImageSection/SEDImageSection';
import AlgorithmDocumentation from '../AlgorithmDocumentation/AlgorithmDocumentation';
import ChartsContainer from '../Charts/ChartsContainer';
import styles from './Dashboard.module.css';

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
    cnn_validation?: AutoPeriodMethod;
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
  const [periodInputValue, setPeriodInputValue] = useState<string>('');
  const [phaseFoldedData, setPhaseFoldedData] = useState<PhaseFoldedPoint[]>([]);
  const [autoPeriodsData, setAutoPeriodsData] = useState<AutoPeriodsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [sedImageAvailable, setSedImageAvailable] = useState<boolean>(false);
  const [sedImageLoading, setSedImageLoading] = useState<boolean>(false);

  useEffect(() => {
    document.title = selectedStar ? `${selectedStar} - ${selectedTelescope}` : 'Better IMPULS Viewer';
  }, [selectedStar, selectedTelescope]);

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

  // Check SED image availability when star changes
  useEffect(() => {
    if (selectedStar) {
      setSedImageLoading(true);
      setSedImageAvailable(false);
    }
  }, [selectedStar]);

  const fetchStars = async () => {
    try {
      const response = await fetch(`${API_BASE}/stars`);
      const data = await response.json();
      setStars(data);
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

  const handlePeriodogramClick = (data: Plotly.PlotMouseEvent) => {
    console.log('Periodogram clicked:', data);
    if (data && data.points && data.points[0]) {
      const period = data.points[0].x;
      if (period && typeof period === 'number') {
        console.log('Selected period:', period);
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

  const handleSedImageError = () => {
    setSedImageAvailable(false);
    setSedImageLoading(false);
  };

  const handleSedImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    console.log(`Successfully loaded SED image for Star ${selectedStar}`);
    e.currentTarget.style.display = 'block';
    setSedImageAvailable(true);
    setSedImageLoading(false);
  };

  return (
    <div className={styles.dashboard}>
      {/* Control Panel */}
      <div className={styles.fullWidthSection}>
        <ControlPanel
          stars={stars}
          selectedStar={selectedStar}
          setSelectedStar={setSelectedStar}
          telescopes={telescopes}
          selectedTelescope={selectedTelescope}
          setSelectedTelescope={setSelectedTelescope}
          campaigns={campaigns}
          selectedCampaign={selectedCampaign}
          setSelectedCampaign={setSelectedCampaign}
        />
      </div>

      {loading && <div className={styles.loading}>Loading...</div>}

      {/* Compact sections: AI view and SED side by side */}
      {(autoPeriodsData || (selectedStar && sedImageAvailable)) && (
        <div className={styles.compactSections}>
          {/* Automatic Period Detection Results */}
          {autoPeriodsData && (
            <AutoPeriodsSection
              autoPeriodsData={autoPeriodsData}
              onUsePrimaryPeriod={handleUsePrimaryPeriod}
              onUseSecondaryPeriod={handleUseSecondaryPeriod}
            />
          )}

          {/* SED Image Section - Only show if image is available */}
          {selectedStar && sedImageAvailable && (
            <SEDImageSection
              selectedStar={selectedStar}
              apiBase={API_BASE}
              onImageError={handleSedImageError}
              onImageLoad={handleSedImageLoad}
            />
          )}
        </div>
      )}

      {/* Algorithm Documentation Section */}
      {autoPeriodsData && (
        <div className={styles.fullWidthSection}>
          <AlgorithmDocumentation campaignData={campaignData} />
        </div>
      )}

      {/* Hidden image to test SED availability */}
      {selectedStar && sedImageLoading && (
        <img 
          src={`${API_BASE}/sed/${selectedStar}`} 
          alt=""
          className={styles.hiddenImage}
          onError={() => {
            setSedImageAvailable(false);
            setSedImageLoading(false);
          }}
          onLoad={() => {
            setSedImageAvailable(true);
            setSedImageLoading(false);
          }}
        />
      )}

      {/* Charts */}
      <div className={styles.fullWidthSection}>
        <ChartsContainer
          campaignData={campaignData}
          periodogramData={periodogramData}
          phaseFoldedData={phaseFoldedData}
          selectedPeriod={selectedPeriod}
          periodInputValue={periodInputValue}
          onPeriodogramClick={handlePeriodogramClick}
          onPeriodInputChange={handlePeriodInputChange}
          onPeriodSubmit={handlePeriodSubmit}
        />
      </div>
    </div>
  );
};

export default Dashboard;
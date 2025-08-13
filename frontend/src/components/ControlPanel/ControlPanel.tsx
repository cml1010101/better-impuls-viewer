import React, { useState, useEffect } from 'react';
import styles from './ControlPanel.module.css';

interface Campaign {
  campaign_id: string;
  telescope: string;
  star_number: number;
  data_points: number;
  duration: number;
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

interface DataSourceConfig {
  source: string;
  cache_enabled: boolean;
}

interface CacheInfo {
  source: string;
  cache_dir: string;
  cached_files: number;
  total_size_mb: number;
  last_updated?: string;
}

interface ControlPanelProps {
  stars: number[];
  selectedStar: number | null;
  setSelectedStar: (star: number) => void;
  telescopes: string[];
  selectedTelescope: string;
  setSelectedTelescope: (telescope: string) => void;
  campaigns: Campaign[];
  selectedCampaign: string;
  setSelectedCampaign: (campaign: string) => void;
  autoPeriodsData?: AutoPeriodsData | null;
  onUsePrimaryPeriod?: () => void;
  onUseSecondaryPeriod?: () => void;
  onDataSourceChange?: () => void; // Callback when data source changes
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  stars,
  selectedStar,
  setSelectedStar,
  telescopes,
  selectedTelescope,
  setSelectedTelescope,
  campaigns,
  selectedCampaign,
  setSelectedCampaign,
  autoPeriodsData,
  onUsePrimaryPeriod,
  onUseSecondaryPeriod,
  onDataSourceChange,
}) => {
  const [dataSourceConfig, setDataSourceConfig] = useState<DataSourceConfig>({ source: 'local', cache_enabled: true });
  const [cacheInfo, setCacheInfo] = useState<CacheInfo | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const API_BASE = 'http://localhost:8000';

  // Load initial data source config
  useEffect(() => {
    fetchDataSourceConfig();
    fetchCacheInfo();
  }, []);

  const fetchDataSourceConfig = async () => {
    try {
      const response = await fetch(`${API_BASE}/data_source/config`);
      if (response.ok) {
        const config = await response.json();
        setDataSourceConfig(config);
      }
    } catch (error) {
      console.error('Error fetching data source config:', error);
    }
  };

  const fetchCacheInfo = async () => {
    try {
      const response = await fetch(`${API_BASE}/data_source/cache/info`);
      if (response.ok) {
        const info = await response.json();
        setCacheInfo(info);
      }
    } catch (error) {
      console.error('Error fetching cache info:', error);
    }
  };

  const handleDataSourceToggle = async (newSource: string) => {
    if (newSource === dataSourceConfig.source) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/data_source/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: newSource, cache_enabled: true })
      });

      if (response.ok) {
        setDataSourceConfig({ source: newSource, cache_enabled: true });
        await fetchCacheInfo();
        
        // Notify parent component about the change
        if (onDataSourceChange) {
          onDataSourceChange();
        }
      } else {
        console.error('Failed to switch data source');
      }
    } catch (error) {
      console.error('Error switching data source:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/data_source/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          star_numbers: [1, 2, 3, 4, 5],
          telescopes: ['hubble', 'kepler', 'tess'],
          force_refresh: false
        })
      });

      if (response.ok) {
        await fetchCacheInfo();
        
        // Notify parent component to refresh data
        if (onDataSourceChange) {
          onDataSourceChange();
        }
      } else {
        console.error('Failed to download data');
      }
    } catch (error) {
      console.error('Error downloading data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearCache = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/data_source/cache/clear`, {
        method: 'POST'
      });

      if (response.ok) {
        await fetchCacheInfo();
        
        // Notify parent component to refresh data
        if (onDataSourceChange) {
          onDataSourceChange();
        }
      } else {
        console.error('Failed to clear cache');
      }
    } catch (error) {
      console.error('Error clearing cache:', error);
    } finally {
      setIsLoading(false);
    }
  };
  const handlePrevStar = () => {
    if (selectedStar) {
      const currentIndex = stars.indexOf(selectedStar);
      const prevIndex = currentIndex > 0 ? currentIndex - 1 : stars.length - 1;
      setSelectedStar(stars[prevIndex]);
    }
  };

  const handleNextStar = () => {
    if (selectedStar) {
      const currentIndex = stars.indexOf(selectedStar);
      const nextIndex = currentIndex < stars.length - 1 ? currentIndex + 1 : 0;
      setSelectedStar(stars[nextIndex]);
    }
  };

  const handlePrevTelescope = () => {
    if (selectedTelescope && telescopes.length > 0) {
      const currentIndex = telescopes.indexOf(selectedTelescope);
      const prevIndex = currentIndex > 0 ? currentIndex - 1 : telescopes.length - 1;
      setSelectedTelescope(telescopes[prevIndex]);
    }
  };

  const handleNextTelescope = () => {
    if (selectedTelescope && telescopes.length > 0) {
      const currentIndex = telescopes.indexOf(selectedTelescope);
      const nextIndex = currentIndex < telescopes.length - 1 ? currentIndex + 1 : 0;
      setSelectedTelescope(telescopes[nextIndex]);
    }
  };

  return (
    <div className={styles.controlPanel}>
      <div className={styles.controlGroup}>
        <label>Star Number:</label>
        <div className={styles.controlWithNav}>
          <button
            className={styles.navButton}
            onClick={handlePrevStar}
            disabled={!selectedStar || stars.length <= 1}
            title="Previous Star"
          >
            ←
          </button>
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
          <button
            className={styles.navButton}
            onClick={handleNextStar}
            disabled={!selectedStar || stars.length <= 1}
            title="Next Star"
          >
            →
          </button>
        </div>
      </div>

      <div className={styles.controlGroup}>
        <label>Telescope/Sensor:</label>
        <div className={styles.controlWithNav}>
          <button
            className={styles.navButton}
            onClick={handlePrevTelescope}
            disabled={!selectedStar || !selectedTelescope || telescopes.length <= 1}
            title="Previous Sensor"
          >
            ←
          </button>
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
          <button
            className={styles.navButton}
            onClick={handleNextTelescope}
            disabled={!selectedStar || !selectedTelescope || telescopes.length <= 1}
            title="Next Sensor"
          >
            →
          </button>
        </div>
      </div>

      <div className={styles.controlGroup}>
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

      {/* Data Source Toggle */}
      <div className={`${styles.controlGroup} ${styles.dataSourceGroup}`}>
        <label>Data Source:</label>
        <div className={styles.dataSourceToggle}>
          <div className={styles.toggleContainer}>
            <button
              className={`${styles.toggleOption} ${dataSourceConfig.source === 'local' ? styles.active : ''}`}
              onClick={() => handleDataSourceToggle('local')}
              disabled={isLoading}
            >
              Local
            </button>
            <button
              className={`${styles.toggleOption} ${dataSourceConfig.source === 'online' ? styles.active : ''}`}
              onClick={() => handleDataSourceToggle('online')}
              disabled={isLoading}
            >
              Online
            </button>
          </div>
        </div>
        
        {/* Data source indicator */}
        <div className={styles.dataSourceIndicator}>
          {isLoading ? (
            <span className={styles.loading}>Switching...</span>
          ) : (
            `Using ${dataSourceConfig.source} data`
          )}
        </div>

        {/* Cache controls for online mode */}
        {dataSourceConfig.source === 'online' && (
          <div className={styles.cacheControls}>
            <button
              className={`${styles.cacheButton} ${styles.download}`}
              onClick={handleDownloadData}
              disabled={isLoading}
              title="Download sample data from online sources"
            >
              {isLoading ? 'Downloading...' : 'Download'}
            </button>
            <button
              className={`${styles.cacheButton} ${styles.clear}`}
              onClick={handleClearCache}
              disabled={isLoading || !cacheInfo || cacheInfo.cached_files === 0}
              title="Clear cached online data"
            >
              {isLoading ? 'Clearing...' : 'Clear Cache'}
            </button>
          </div>
        )}

        {/* Cache info for online mode */}
        {dataSourceConfig.source === 'online' && cacheInfo && (
          <div className={styles.dataSourceIndicator}>
            {cacheInfo.cached_files} files cached ({cacheInfo.total_size_mb.toFixed(1)} MB)
          </div>
        )}
      </div>

      {/* Auto Periods integrated into header */}
      {autoPeriodsData && !autoPeriodsData.error && (
        <div className={styles.controlGroup}>
          <label>Auto Periods:</label>
          <div className={styles.autoPeriodsInline}>
            {autoPeriodsData.primary_period && (
              <div className={styles.inlinePeriod}>
                <span className={styles.periodValue}>
                  {autoPeriodsData.primary_period.toFixed(4)}d
                </span>
                <button 
                  className={`${styles.usePeriodBtn} ${styles.primaryBtn}`}
                  onClick={onUsePrimaryPeriod}
                  title="Use primary period"
                >
                  Use
                </button>
              </div>
            )}
            
            {autoPeriodsData.secondary_period && (
              <div className={styles.inlinePeriod}>
                <span className={styles.periodValue}>
                  {autoPeriodsData.secondary_period.toFixed(4)}d
                </span>
                <button 
                  className={`${styles.usePeriodBtn} ${styles.secondaryBtn}`}
                  onClick={onUseSecondaryPeriod}
                  title="Use secondary period"
                >
                  Use
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ControlPanel;
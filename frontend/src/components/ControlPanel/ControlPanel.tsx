import React from 'react';
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
}) => {
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
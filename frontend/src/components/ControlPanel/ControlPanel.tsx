import React from 'react';
import styles from './ControlPanel.module.css';

interface Campaign {
  campaign_id: string;
  telescope: string;
  star_number: number;
  data_points: number;
  duration: number;
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
}) => {
  return (
    <div className={styles.controlPanel}>
      <div className={styles.controlGroup}>
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

      <div className={styles.controlGroup}>
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
    </div>
  );
};

export default ControlPanel;
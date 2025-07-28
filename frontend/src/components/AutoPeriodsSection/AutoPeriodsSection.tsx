import React from 'react';
import styles from './AutoPeriodsSection.module.css';

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

interface AutoPeriodsSectionProps {
  autoPeriodsData: AutoPeriodsData | null;
  onUsePrimaryPeriod: () => void;
  onUseSecondaryPeriod: () => void;
}

const AutoPeriodsSection: React.FC<AutoPeriodsSectionProps> = ({
  autoPeriodsData,
  onUsePrimaryPeriod,
  onUseSecondaryPeriod,
}) => {
  if (!autoPeriodsData) return null;

  if (autoPeriodsData.error) {
    return (
      <div className={styles.autoPeriodsError}>
        <h3>⚠️ Auto Periods</h3>
        <p>Unable to detect periods automatically.</p>
      </div>
    );
  }

  return (
    <div className={styles.autoPeriodsSection}>
      <h3>🤖 Auto Periods</h3>
      
      <div className={styles.compactPeriodsInfo}>
        {autoPeriodsData.primary_period && (
          <div className={styles.compactPeriodResult}>
            <span className={styles.compactPeriodValue}>
              {autoPeriodsData.primary_period.toFixed(4)}d
            </span>
            <button 
              className={`${styles.compactUsePeriodBtn} ${styles.primaryBtn}`}
              onClick={onUsePrimaryPeriod}
              title="Use primary period"
            >
              Use
            </button>
          </div>
        )}
        
        {autoPeriodsData.secondary_period && (
          <div className={styles.compactPeriodResult}>
            <span className={styles.compactPeriodValue}>
              {autoPeriodsData.secondary_period.toFixed(4)}d
            </span>
            <button 
              className={`${styles.compactUsePeriodBtn} ${styles.secondaryBtn}`}
              onClick={onUseSecondaryPeriod}
              title="Use secondary period"
            >
              Use
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default AutoPeriodsSection;
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
        <h3>⚠️ Automatic Period Detection</h3>
        <p>Unable to automatically determine periods: {autoPeriodsData.error}</p>
        <p>You can still manually analyze the periodogram and select periods below.</p>
      </div>
    );
  }

  const getClassificationClassName = (type: string) => {
    const normalizedType = type.toLowerCase().replace(/\s+/g, '').replace(/_/g, '');
    return styles[`classification${normalizedType.charAt(0).toUpperCase() + normalizedType.slice(1)}`] || styles.classificationOther;
  };

  return (
    <div className={styles.autoPeriodsSection}>
      <h3>🤖 Automatic Period Detection Results</h3>
      
      <div className={styles.autoPeriodsContent}>
        {/* Classification Results */}
        <div className={styles.classificationInfo}>
          <div className={styles.classificationBadge}>
            <span className={`${styles.classificationType} ${getClassificationClassName(autoPeriodsData.classification.type)}`}>
              {autoPeriodsData.classification.type.toUpperCase()}
            </span>
            <span className={styles.classificationConfidence}>
              {(autoPeriodsData.classification.confidence * 100).toFixed(1)}% confidence
            </span>
          </div>
          <p className={styles.classificationDescription}>
            {autoPeriodsData.classification.description}
          </p>
        </div>

        {/* Period Results */}
        <div className={styles.periodsInfo}>
          {autoPeriodsData.primary_period && (
            <div className={styles.periodResult}>
              <div className={styles.periodLabel}>Primary Period:</div>
              <div className={styles.periodValue}>
                <strong>{autoPeriodsData.primary_period.toFixed(4)} days</strong>
                <button 
                  className={`${styles.usePeriodBtn} ${styles.primaryBtn}`}
                  onClick={onUsePrimaryPeriod}
                  title="Use this period for phase folding"
                >
                  Use Period
                </button>
              </div>
            </div>
          )}
          
          {autoPeriodsData.secondary_period && (
            <div className={styles.periodResult}>
              <div className={styles.periodLabel}>Secondary Period:</div>
              <div className={styles.periodValue}>
                <strong>{autoPeriodsData.secondary_period.toFixed(4)} days</strong>
                <button 
                  className={`${styles.usePeriodBtn} ${styles.secondaryBtn}`}
                  onClick={onUseSecondaryPeriod}
                  title="Use this period for phase folding"
                >
                  Use Period
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Method Results */}
        <div className={styles.methodsInfo}>
          <div className={styles.methodsHeader}>Detection Methods:</div>
          <div className={styles.methodsGrid}>
            {autoPeriodsData.methods.periodogram && (
              <div className={`${styles.methodResult} ${autoPeriodsData.methods.periodogram.success ? styles.success : styles.failed}`}>
                <div className={styles.methodName}>Periodogram</div>
                <div className={styles.methodStatus}>
                  {autoPeriodsData.methods.periodogram.success ? '✓' : '✗'}
                </div>
                {autoPeriodsData.methods.periodogram.success && (
                  <div className={styles.methodPeriods}>
                    {autoPeriodsData.methods.periodogram.periods.slice(0, 3).map((period, idx) => (
                      <span key={idx} className={styles.methodPeriod}>
                        {Number(period).toFixed(3)}d
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            {autoPeriodsData.methods.cnn_validation && (
              <div className={`${styles.methodResult} ${autoPeriodsData.methods.cnn_validation.success ? styles.success : styles.failed}`}>
                <div className={styles.methodName}>CNN Validation</div>
                <div className={styles.methodStatus}>
                  {autoPeriodsData.methods.cnn_validation.success ? '✓' : '✗'}
                </div>
                {autoPeriodsData.methods.cnn_validation.success && (
                  <div className={styles.methodPeriods}>
                    {autoPeriodsData.methods.cnn_validation.periods.slice(0, 3).map((period, idx) => (
                      <span key={idx} className={styles.methodPeriod}>
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
  );
};

export default AutoPeriodsSection;
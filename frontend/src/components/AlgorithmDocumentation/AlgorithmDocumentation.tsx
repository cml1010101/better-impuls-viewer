import React from 'react';
import styles from './AlgorithmDocumentation.module.css';

interface DataPoint {
  time: number;
  flux: number;
  error: number;
}

interface AlgorithmDocumentationProps {
  campaignData: DataPoint[];
}

const AlgorithmDocumentation: React.FC<AlgorithmDocumentationProps> = ({ campaignData }) => {
  const campaignDuration = campaignData.length > 0 
    ? (Math.max(...campaignData.map(d => d.time)) - Math.min(...campaignData.map(d => d.time))).toFixed(1)
    : 'N/A';

  return (
    <div className={styles.algorithmDocumentation}>
      <div className={styles.documentationContent}>
        <h3>📊 Analysis Summary</h3>
        <p>
          Campaign duration: <strong>{campaignDuration} days</strong>
        </p>
      </div>
    </div>
  );
};

export default AlgorithmDocumentation;
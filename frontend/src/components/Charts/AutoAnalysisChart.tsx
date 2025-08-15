import React from 'react';
import { type AutoAnalysisResult } from '../../utils/api';
import styles from './AutoAnalysisChart.module.css';

interface AutoAnalysisChartProps {
  autoAnalysis: AutoAnalysisResult;
}

const AutoAnalysisChart: React.FC<AutoAnalysisChartProps> = ({ autoAnalysis }) => {
  // Format confidence as percentage
  const confidencePercent = (autoAnalysis.class_confidence * 100).toFixed(1);
  
  // Sort candidate periods by rank for display
  const sortedCandidates = [...autoAnalysis.candidate_periods].sort((a, b) => a.rank - b.rank);

  return (
    <div className={styles.chartSection}>
      <h3>Automatic Analysis Results</h3>
      
      <div className={styles.mainResults}>
        <div className={styles.predictionCard}>
          <h4>ML Prediction</h4>
          <div className={styles.predictionValue}>
            <span className={styles.periodValue}>{autoAnalysis.predicted_period.toFixed(4)} days</span>
            <span className={styles.classValue}>{autoAnalysis.predicted_class}</span>
            <span className={styles.confidenceValue}>Confidence: {confidencePercent}%</span>
          </div>
        </div>
        
        <div className={styles.detectedCard}>
          <h4>Detected Period</h4>
          <div className={styles.detectedValue}>
            {autoAnalysis.detected_period.toFixed(4)} days
          </div>
        </div>
      </div>

      {sortedCandidates.length > 0 && (
        <div className={styles.candidatesSection}>
          <h4>Candidate Periods</h4>
          <div className={styles.candidatesGrid}>
            {sortedCandidates.slice(0, 5).map((candidate, index) => (
              <div key={index} className={styles.candidateCard}>
                <div className={styles.candidateRank}>#{candidate.rank}</div>
                <div className={styles.candidatePeriod}>{candidate.period.toFixed(4)}d</div>
                <div className={styles.candidateScore}>Score: {candidate.score.toFixed(3)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AutoAnalysisChart;
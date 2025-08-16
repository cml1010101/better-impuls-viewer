import React from 'react';
import { type AutoAnalysisResult } from '../../utils/api';
import styles from './AutoAnalysisChart.module.css';

interface AutoAnalysisChartProps {
  autoAnalysis: AutoAnalysisResult;
  onPeriodClick?: (period: number) => void;
}

const AutoAnalysisChart: React.FC<AutoAnalysisChartProps> = ({ autoAnalysis, onPeriodClick }) => {
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
            <button 
              className={styles.periodButton}
              onClick={() => onPeriodClick?.(autoAnalysis.predicted_period)}
              title="Click to use this period for phase folding"
            >
              {autoAnalysis.predicted_period.toFixed(4)} days
            </button>
            <span className={styles.classValue}>{autoAnalysis.predicted_class}</span>
            <span className={styles.confidenceValue}>Confidence: {confidencePercent}%</span>
          </div>
        </div>
        
        <div className={styles.detectedCard}>
          <h4>Detected Period</h4>
          <div className={styles.detectedValue}>
            <button 
              className={styles.periodButton}
              onClick={() => onPeriodClick?.(autoAnalysis.detected_period)}
              title="Click to use this period for phase folding"
            >
              {autoAnalysis.detected_period.toFixed(4)} days
            </button>
          </div>
        </div>
      </div>

      {sortedCandidates.length > 0 && (
        <div className={styles.candidatesSection}>
          <h4>Candidate Periods</h4>
          <div className={styles.candidatesGrid}>
            {sortedCandidates.slice(0, 5).map((candidate, index) => (
              <button 
                key={index} 
                className={styles.candidateCard}
                onClick={() => onPeriodClick?.(candidate.period)}
                title="Click to use this period for phase folding"
              >
                <div className={styles.candidateRank}>#{candidate.rank}</div>
                <div className={styles.candidatePeriod}>{candidate.period.toFixed(4)}d</div>
                <div className={styles.candidateScore}>Score: {candidate.score.toFixed(3)}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AutoAnalysisChart;
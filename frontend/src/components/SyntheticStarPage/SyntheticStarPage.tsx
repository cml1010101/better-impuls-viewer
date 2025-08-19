import React, { useState, useEffect } from 'react';
import LightCurveChart from '../Charts/LightCurveChart';
import styles from './SyntheticStarPage.module.css';

const API_BASE = 'http://localhost:8000/api';

interface SyntheticStarInfo {
  star_id: number;
  name: string;
  variability_class: string;
  primary_period?: number;
  secondary_period?: number;
  surveys: string[];
}

interface ProcessedData {
  time: number[];
  flux: number[];
  error: number[];
}

interface SyntheticStarPageProps {
  datasetName: string;
  starId: number;
  onBackToDataset: () => void;
}

const SyntheticStarPage: React.FC<SyntheticStarPageProps> = ({
  datasetName,
  starId,
  onBackToDataset
}) => {
  const [starInfo, setStarInfo] = useState<SyntheticStarInfo | null>(null);
  const [selectedSurvey, setSelectedSurvey] = useState<string>('');
  const [lightCurveData, setLightCurveData] = useState<ProcessedData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStarInfo();
  }, [datasetName, starId]);

  useEffect(() => {
    if (starInfo && starInfo.surveys.length > 0 && !selectedSurvey) {
      setSelectedSurvey(starInfo.surveys[0]);
    }
  }, [starInfo]);

  useEffect(() => {
    if (selectedSurvey) {
      fetchLightCurveData();
    }
  }, [selectedSurvey]);

  const fetchStarInfo = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/datasets/${datasetName}/stars`);
      if (!response.ok) {
        throw new Error(`Failed to fetch star info: ${response.status}`);
      }
      
      const stars = await response.json();
      const star = stars.find((s: SyntheticStarInfo) => s.star_id === starId);
      
      if (!star) {
        throw new Error(`Star ${starId} not found in dataset ${datasetName}`);
      }
      
      setStarInfo(star);
    } catch (error) {
      console.error('Error fetching star info:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch star info');
    } finally {
      setLoading(false);
    }
  };

  const fetchLightCurveData = async () => {
    if (!selectedSurvey) return;

    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(
        `${API_BASE}/datasets/${datasetName}/star/${starId}/survey/${selectedSurvey}`
      );
      
      if (!response.ok) {
        throw new Error(`Failed to fetch light curve data: ${response.status}`);
      }
      
      const data = await response.json();
      setLightCurveData(data);
    } catch (error) {
      console.error('Error fetching light curve data:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch light curve data');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !starInfo) {
    return (
      <div className={styles.syntheticStarPage}>
        <div className={styles.loading}>Loading star information...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.syntheticStarPage}>
        <div className={styles.header}>
          <button onClick={onBackToDataset} className={styles.backButton}>
            ← Back to Dataset
          </button>
          <h1>Error</h1>
        </div>
        <div className={styles.error}>{error}</div>
      </div>
    );
  }

  if (!starInfo) {
    return (
      <div className={styles.syntheticStarPage}>
        <div className={styles.header}>
          <button onClick={onBackToDataset} className={styles.backButton}>
            ← Back to Dataset
          </button>
          <h1>Star Not Found</h1>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.syntheticStarPage}>
      <div className={styles.header}>
        <button onClick={onBackToDataset} className={styles.backButton}>
          ← Back to Dataset
        </button>
        <div className={styles.titleSection}>
          <h1>{starInfo.name}</h1>
          <div className={styles.subtitle}>Dataset: {datasetName}</div>
        </div>
      </div>

      <div className={styles.starDetails}>
        <div className={styles.detailCard}>
          <h3>Star Information</h3>
          <div className={styles.detailRow}>
            <span className={styles.label}>Star ID:</span>
            <span className={styles.value}>#{starInfo.star_id}</span>
          </div>
          <div className={styles.detailRow}>
            <span className={styles.label}>Variability Class:</span>
            <span className={styles.value}>{starInfo.variability_class}</span>
          </div>
          {starInfo.primary_period && (
            <div className={styles.detailRow}>
              <span className={styles.label}>Primary Period:</span>
              <span className={styles.value}>{starInfo.primary_period.toFixed(6)} days</span>
            </div>
          )}
          {starInfo.secondary_period && (
            <div className={styles.detailRow}>
              <span className={styles.label}>Secondary Period:</span>
              <span className={styles.value}>{starInfo.secondary_period.toFixed(6)} days</span>
            </div>
          )}
        </div>

        <div className={styles.surveySelector}>
          <h3>Select Survey</h3>
          <div className={styles.surveyButtons}>
            {starInfo.surveys.map((survey) => (
              <button
                key={survey}
                className={`${styles.surveyButton} ${
                  selectedSurvey === survey ? styles.active : ''
                }`}
                onClick={() => setSelectedSurvey(survey)}
              >
                {survey.charAt(0).toUpperCase() + survey.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {selectedSurvey && (
        <div className={styles.chartSection}>
          <div className={styles.chartHeader}>
            <h2>Light Curve - {selectedSurvey.toUpperCase()}</h2>
            {lightCurveData && (
              <div className={styles.dataStats}>
                {lightCurveData.time.length.toLocaleString()} data points
              </div>
            )}
          </div>

          {loading && <div className={styles.loading}>Loading light curve data...</div>}
          {error && <div className={styles.error}>{error}</div>}

          {lightCurveData && !loading && (
            <div className={styles.chartContainer}>
              <LightCurveChart
                campaignData={lightCurveData.time.map((time, index) => ({
                  time,
                  flux: lightCurveData.flux[index],
                  error: lightCurveData.error[index]
                }))}
              />
            </div>
          )}
        </div>
      )}

      <div className={styles.syntheticInfo}>
        <div className={styles.infoCard}>
          <h3>About Synthetic Data</h3>
          <p>
            This is synthetic astronomical data generated using realistic variability models. 
            The light curve shows simulated observations with:
          </p>
          <ul>
            <li>Realistic photometric noise levels (0.001-0.003 magnitude)</li>
            <li>Time-series patterns characteristic of {starInfo.variability_class} variables</li>
            <li>Random observational cadence similar to real surveys</li>
            <li>Proper error arrays based on flux levels</li>
          </ul>
          <p>
            This data can be used for testing algorithms, training machine learning models, 
            and validating analysis pipelines.
          </p>
        </div>
      </div>
    </div>
  );
};

export default SyntheticStarPage;
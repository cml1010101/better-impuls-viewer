import React, { useState, useEffect } from 'react';
import styles from './TrainingDashboard.module.css';

const API_BASE = 'http://localhost:8000';

interface ModelStatus {
  model_available: boolean;
  message: string;
  model_path: string;
  model_info?: {
    classes: string[];
    training_metadata: {
      epochs_trained: number;
      final_loss: number;
      training_samples: number;
    };
  };
}

interface TrainingResult {
  success: boolean;
  message: string;
  model_path: string;
  epochs_trained: number;
  final_loss: number;
  training_samples: number;
}

const TrainingDashboard: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportResult, setExportResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      setError(null);
      const response = await fetch(`${API_BASE}/model_status`);
      const data = await response.json();
      setModelStatus(data);
    } catch (err) {
      setError('Failed to fetch model status');
      console.error('Error fetching model status:', err);
    }
  };

  const handleTrainModel = async () => {
    try {
      setError(null);
      setIsTraining(true);
      setTrainingResult(null);
      
      const response = await fetch(`${API_BASE}/train_model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          force_retrain: false,
          stars_to_extract: null
        }),
      });

      if (!response.ok) {
        throw new Error(`Training failed: ${response.statusText}`);
      }

      const result = await response.json();
      setTrainingResult(result);
      
      // Refresh model status after training
      await fetchModelStatus();
    } catch (err) {
      setError(`Training failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Error training model:', err);
    } finally {
      setIsTraining(false);
    }
  };

  const handleForceRetrain = async () => {
    try {
      setError(null);
      setIsTraining(true);
      setTrainingResult(null);
      
      const response = await fetch(`${API_BASE}/train_model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          force_retrain: true,
          stars_to_extract: null
        }),
      });

      if (!response.ok) {
        throw new Error(`Training failed: ${response.statusText}`);
      }

      const result = await response.json();
      setTrainingResult(result);
      
      // Refresh model status after training
      await fetchModelStatus();
    } catch (err) {
      setError(`Training failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Error training model:', err);
    } finally {
      setIsTraining(false);
    }
  };

  const handleExportData = async () => {
    try {
      setError(null);
      setIsExporting(true);
      setExportResult(null);
      
      const response = await fetch(`${API_BASE}/export_training_csv`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stars_to_extract: null,
          output_dir: 'training_data'
        }),
      });

      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }

      const result = await response.json();
      setExportResult(result.message);
    } catch (err) {
      setError(`Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Error exporting data:', err);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className={styles.trainingDashboard}>
      <h2>ML Training Dashboard</h2>
      
      {/* Model Status Section */}
      <div className={styles.section}>
        <h3>Model Status</h3>
        {modelStatus ? (
          <div className={styles.statusCard}>
            <div className={`${styles.statusIndicator} ${modelStatus.model_available ? styles.available : styles.unavailable}`}>
              {modelStatus.model_available ? '✓ Model Available' : '⚠ No Model'}
            </div>
            <p className={styles.statusMessage}>{modelStatus.message}</p>
            <p className={styles.modelPath}><strong>Path:</strong> {modelStatus.model_path}</p>
            
            {modelStatus.model_info && (
              <div className={styles.modelDetails}>
                <h4>Model Information</h4>
                <p><strong>Classes:</strong> {modelStatus.model_info.classes.join(', ')}</p>
                <p><strong>Epochs Trained:</strong> {modelStatus.model_info.training_metadata.epochs_trained}</p>
                <p><strong>Final Loss:</strong> {modelStatus.model_info.training_metadata.final_loss.toFixed(4)}</p>
                <p><strong>Training Samples:</strong> {modelStatus.model_info.training_metadata.training_samples}</p>
              </div>
            )}
          </div>
        ) : (
          <div className={styles.loading}>Loading model status...</div>
        )}
        
        <button onClick={fetchModelStatus} className={styles.refreshButton}>
          Refresh Status
        </button>
      </div>

      {/* Training Section */}
      <div className={styles.section}>
        <h3>Model Training</h3>
        <div className={styles.trainingControls}>
          <button 
            onClick={handleTrainModel} 
            disabled={isTraining}
            className={styles.trainButton}
          >
            {isTraining ? 'Training...' : 'Train Model'}
          </button>
          
          <button 
            onClick={handleForceRetrain} 
            disabled={isTraining}
            className={styles.retrainButton}
          >
            {isTraining ? 'Training...' : 'Force Retrain'}
          </button>
          
          <div className={styles.trainingInfo}>
            <p>• <strong>Train Model:</strong> Use existing model if available, train new one if not</p>
            <p>• <strong>Force Retrain:</strong> Train a new model even if one exists</p>
          </div>
        </div>

        {isTraining && (
          <div className={styles.trainingProgress}>
            <div className={styles.spinner}></div>
            <p>Training model... This may take several minutes.</p>
          </div>
        )}

        {trainingResult && (
          <div className={styles.trainingResult}>
            <h4>Training Result</h4>
            <div className={`${styles.resultStatus} ${trainingResult.success ? styles.success : styles.error}`}>
              {trainingResult.success ? '✓ Success' : '✗ Failed'}
            </div>
            <p>{trainingResult.message}</p>
            {trainingResult.success && (
              <div className={styles.resultDetails}>
                <p><strong>Epochs:</strong> {trainingResult.epochs_trained}</p>
                <p><strong>Final Loss:</strong> {trainingResult.final_loss.toFixed(4)}</p>
                <p><strong>Training Samples:</strong> {trainingResult.training_samples}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Data Export Section */}
      <div className={styles.section}>
        <h3>Training Data Export</h3>
        <div className={styles.exportControls}>
          <button 
            onClick={handleExportData} 
            disabled={isExporting}
            className={styles.exportButton}
          >
            {isExporting ? 'Exporting...' : 'Export Training Data to CSV'}
          </button>
          
          <div className={styles.exportInfo}>
            <p>Export training data from Google Sheets to CSV format for external analysis.</p>
          </div>
        </div>

        {isExporting && (
          <div className={styles.exportProgress}>
            <div className={styles.spinner}></div>
            <p>Exporting data...</p>
          </div>
        )}

        {exportResult && (
          <div className={styles.exportResult}>
            <div className={styles.resultStatus + ' ' + styles.success}>
              ✓ Export Complete
            </div>
            <p>{exportResult}</p>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  );
};

export default TrainingDashboard;
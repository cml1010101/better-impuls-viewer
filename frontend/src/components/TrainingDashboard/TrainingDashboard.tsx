import React, { useState, useEffect } from 'react';
import styles from './TrainingDashboard.module.css';

const API_BASE = 'http://localhost:8000';

interface ModelStatus {
  model_available: boolean;
  message: string;
  model_path: string;
  model_info?: {
    model_path: string;
    file_size_bytes: number;
    file_size_mb: number;
    last_modified: string;
    input_size: number;
    num_classes: number;
    has_label_encoder: boolean;
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
  const [error, setError] = useState<string | null>(null);
  const [csvFilePath, setCsvFilePath] = useState<string>('');
  const [selectedFileName, setSelectedFileName] = useState<string>('');
  const [starRange, setStarRange] = useState<string>('');

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const parseStarRange = (rangeStr: string): number[] => {
    if (!rangeStr.trim()) return [];
    
    const stars: number[] = [];
    const parts = rangeStr.split(',');
    
    for (const part of parts) {
      const trimmedPart = part.trim();
      if (trimmedPart.includes('-')) {
        // Handle range like "1-10"
        const [start, end] = trimmedPart.split('-').map(num => parseInt(num.trim()));
        if (!isNaN(start) && !isNaN(end) && start <= end) {
          for (let i = start; i <= end; i++) {
            stars.push(i);
          }
        }
      } else {
        // Handle single number
        const num = parseInt(trimmedPart);
        if (!isNaN(num)) {
          stars.push(num);
        }
      }
    }
    
    // Remove duplicates and sort
    return [...new Set(stars)].sort((a, b) => a - b);
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFileName(file.name);
      // In a real scenario, you'd need to upload the file to the server first
      // For now, we'll use the file name and assume the user places it in the correct location
      setCsvFilePath(file.name);
    }
  };

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
      
      // Build query parameters
      const params = new URLSearchParams();
      params.append('force_retrain', 'false');
      
      if (csvFilePath.trim()) {
        params.append('csv_file_path', csvFilePath.trim());
      }
      
      // Parse and add star range if provided
      const starsToExtract = parseStarRange(starRange);
      if (starsToExtract.length > 0) {
        // Add multiple query parameters for each star
        starsToExtract.forEach(star => {
          params.append('stars_to_extract', star.toString());
        });
      }
      
      const response = await fetch(`${API_BASE}/train_model?${params.toString()}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Training failed: ${response.status} ${response.statusText} - ${errorText}`);
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
      
      // Build query parameters
      const params = new URLSearchParams();
      params.append('force_retrain', 'true');
      
      if (csvFilePath.trim()) {
        params.append('csv_file_path', csvFilePath.trim());
      }
      
      // Parse and add star range if provided
      const starsToExtract = parseStarRange(starRange);
      if (starsToExtract.length > 0) {
        // Add multiple query parameters for each star
        starsToExtract.forEach(star => {
          params.append('stars_to_extract', star.toString());
        });
      }
      
      const response = await fetch(`${API_BASE}/train_model?${params.toString()}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Training failed: ${response.status} ${response.statusText} - ${errorText}`);
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
                <p><strong>File Size:</strong> {modelStatus.model_info.file_size_mb.toFixed(2)} MB</p>
                <p><strong>Last Modified:</strong> {new Date(modelStatus.model_info.last_modified).toLocaleString()}</p>
                <p><strong>Input Size:</strong> {modelStatus.model_info.input_size}</p>
                <p><strong>Number of Classes:</strong> {modelStatus.model_info.num_classes}</p>
                <p><strong>Has Label Encoder:</strong> {modelStatus.model_info.has_label_encoder ? 'Yes' : 'No'}</p>
                <p><strong>Training Metadata:</strong></p>
                <ul>
                  <li>Epochs Trained: {modelStatus.model_info.training_metadata.epochs_trained}</li>
                  <li>Final Loss: {modelStatus.model_info.training_metadata.final_loss.toFixed(4)}</li>
                  <li>Training Samples: {modelStatus.model_info.training_metadata.training_samples}</li>
                </ul>
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
        
        {/* CSV File Path Input */}
        <div className={styles.filePathSection}>
          <label htmlFor="csvFileInput" className={styles.filePathLabel}>
            CSV Training Data File (optional):
          </label>
          
          <div className={styles.fileInputContainer}>
            <input
              type="file"
              id="csvFileInput"
              accept=".csv"
              onChange={handleFileSelect}
              className={styles.fileInput}
              disabled={isTraining}
            />
            <label htmlFor="csvFileInput" className={styles.fileInputLabel}>
              {selectedFileName || 'Choose CSV File'}
            </label>
          </div>
          
          <input
            type="text"
            value={csvFilePath}
            onChange={(e) => setCsvFilePath(e.target.value)}
            placeholder="Or enter file path manually (leave empty for default)"
            className={styles.filePathInput}
            disabled={isTraining}
          />
          
          <div className={styles.filePathHelp}>
            Select a CSV file or enter the file path manually. Leave empty to use the default training dataset.
          </div>
        </div>

        {/* Star Range Input */}
        <div className={styles.starRangeSection}>
          <label htmlFor="starRange" className={styles.starRangeLabel}>
            Star Range (optional):
          </label>
          <input
            type="text"
            id="starRange"
            value={starRange}
            onChange={(e) => setStarRange(e.target.value)}
            placeholder="e.g., 1-100, 1,5,10-20, or leave empty for all stars"
            className={styles.starRangeInput}
            disabled={isTraining}
          />
          <div className={styles.starRangeHelp}>
            Specify which stars to include in training. Examples: "1-100" (range), "1,5,10-20" (mixed), or leave empty to use all available stars.
          </div>
        </div>
        
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
            <p>• <strong>Data Source:</strong> Training uses CSV data file for model training</p>
            <p>• <strong>Star Selection:</strong> Optionally specify which stars to include in training</p>
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
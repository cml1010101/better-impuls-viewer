import { useState, useEffect } from 'react';
import { API_BASE } from './api';
import './Training.css';

interface ModelStatus {
  model_available: boolean;
  model_info?: {
    model_path: string;
    training_metadata: {
      epochs_trained: number;
      final_loss: number;
      training_samples: number;
      training_date: string;
    };
  };
  message?: string;
}

interface TrainingResult {
  success: boolean;
  epochs_trained: number;
  final_loss: number;
  model_path: string;
  training_samples: number;
}

const Training: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [starRange, setStarRange] = useState('');
  const [forceRetrain, setForceRetrain] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error' | 'info' | ''>('');
  const [trainingProgress, setTrainingProgress] = useState('');

  useEffect(() => {
    loadModelStatus();
  }, []);

  const loadModelStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/model_status`);
      if (response.ok) {
        const status = await response.json();
        setModelStatus(status);
      }
    } catch (error) {
      console.error('Error loading model status:', error);
    }
  };

  const showMessage = (text: string, type: 'success' | 'error' | 'info') => {
    setMessage(text);
    setMessageType(type);
    setTimeout(() => {
      setMessage('');
      setMessageType('');
    }, 10000);
  };

  const parseStarRange = (range: string): number[] | null => {
    if (!range.trim()) return null;
    
    // Handle comma-separated list
    if (range.includes(',')) {
      try {
        return range.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
      } catch {
        return null;
      }
    }
    
    // Handle range like "30:50"
    if (range.includes(':')) {
      const [start, end] = range.split(':').map(s => parseInt(s.trim()));
      if (!isNaN(start) && !isNaN(end) && start <= end) {
        const result = [];
        for (let i = start; i <= end; i++) {
          result.push(i);
        }
        return result;
      }
      return null;
    }
    
    // Handle single number
    const num = parseInt(range.trim());
    return isNaN(num) ? null : [num];
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    setTrainingProgress('Initializing training...');
    
    try {
      const starsToExtract = parseStarRange(starRange);
      
      const response = await fetch(`${API_BASE}/train_model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stars_to_extract: starsToExtract,
          force_retrain: forceRetrain,
        }),
      });

      if (response.ok) {
        const result: TrainingResult = await response.json();
        showMessage(
          `Training completed! Epochs: ${result.epochs_trained}, Final Loss: ${result.final_loss.toFixed(4)}, Samples: ${result.training_samples}`,
          'success'
        );
        await loadModelStatus();
      } else {
        const error = await response.json();
        showMessage(`Training failed: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Training error: ${error}`, 'error');
    } finally {
      setIsTraining(false);
      setTrainingProgress('');
    }
  };

  const handleExportCSV = async () => {
    setIsExporting(true);
    
    try {
      const starsToExtract = parseStarRange(starRange);
      
      const response = await fetch(`${API_BASE}/export_training_csv`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stars_to_extract: starsToExtract,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        showMessage(
          `CSV export completed! File: ${result.csv_path}, Rows: ${result.total_rows}, Size: ${(result.file_size_bytes / 1024 / 1024).toFixed(2)} MB`,
          'success'
        );
      } else {
        const error = await response.json();
        showMessage(`Export failed: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Export error: ${error}`, 'error');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="training-container">
      <div className="training-header">
        <h2>Model Training & Management</h2>
        <p>Train machine learning models using Google Sheets data and manage training datasets.</p>
      </div>

      {message && (
        <div className={`training-message ${messageType}`}>
          {message}
        </div>
      )}

      {/* Model Status Section */}
      <div className="training-section">
        <h3>Current Model Status</h3>
        
        {modelStatus ? (
          <div className="model-status">
            <div className="status-indicator">
              <span className={`status-dot ${modelStatus.model_available ? 'green' : 'red'}`}></span>
              <span>Model Available: {modelStatus.model_available ? 'Yes' : 'No'}</span>
            </div>
            
            {modelStatus.model_available && modelStatus.model_info ? (
              <div className="model-info">
                <div className="info-grid">
                  <div className="info-item">
                    <label>Training Date:</label>
                    <span>{new Date(modelStatus.model_info.training_metadata.training_date).toLocaleString()}</span>
                  </div>
                  <div className="info-item">
                    <label>Epochs Trained:</label>
                    <span>{modelStatus.model_info.training_metadata.epochs_trained}</span>
                  </div>
                  <div className="info-item">
                    <label>Final Loss:</label>
                    <span>{modelStatus.model_info.training_metadata.final_loss.toFixed(4)}</span>
                  </div>
                  <div className="info-item">
                    <label>Training Samples:</label>
                    <span>{modelStatus.model_info.training_metadata.training_samples.toLocaleString()}</span>
                  </div>
                  <div className="info-item">
                    <label>Model Path:</label>
                    <span className="model-path">{modelStatus.model_info.model_path}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="no-model">
                <p>{modelStatus.message}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="loading">Loading model status...</div>
        )}
      </div>

      {/* Training Configuration Section */}
      <div className="training-section">
        <h3>Training Configuration</h3>
        
        <div className="form-group">
          <label htmlFor="starRange">Star Range (optional):</label>
          <input
            type="text"
            id="starRange"
            value={starRange}
            onChange={(e) => setStarRange(e.target.value)}
            placeholder="e.g., 30:50, 1,5,10, or 42 (leave empty for all stars)"
            disabled={isTraining || isExporting}
          />
          <div className="help-text">
            Specify which stars to include in training. Examples:
            <ul>
              <li><code>30:50</code> - Stars 30 through 50</li>
              <li><code>1,5,10</code> - Stars 1, 5, and 10</li>
              <li><code>42</code> - Only star 42</li>
              <li>Leave empty to use all available stars</li>
            </ul>
          </div>
        </div>

        <div className="checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={forceRetrain}
              onChange={(e) => setForceRetrain(e.target.checked)}
              disabled={isTraining || isExporting}
            />
            Force retrain (even if model already exists)
          </label>
        </div>
      </div>

      {/* Training Actions Section */}
      <div className="training-section">
        <h3>Training Actions</h3>
        
        <div className="action-buttons">
          <button
            onClick={handleTrainModel}
            disabled={isTraining || isExporting}
            className="action-button train-button"
          >
            {isTraining ? 'Training...' : 'Train Model'}
          </button>

          <button
            onClick={handleExportCSV}
            disabled={isTraining || isExporting}
            className="action-button export-button"
          >
            {isExporting ? 'Exporting...' : 'Export Training Data (CSV)'}
          </button>

          <button
            onClick={loadModelStatus}
            disabled={isTraining || isExporting}
            className="action-button refresh-button"
          >
            Refresh Status
          </button>
        </div>

        {(isTraining || isExporting) && (
          <div className="progress-indicator">
            <div className="spinner"></div>
            <span>{isTraining ? trainingProgress || 'Training in progress...' : 'Exporting data...'}</span>
          </div>
        )}
      </div>

      {/* Training Information Section */}
      <div className="training-section">
        <h3>Training Information</h3>
        
        <div className="training-info">
          <h4>Training Strategy</h4>
          <p>The system uses an enhanced 5-period strategy for each light curve:</p>
          <ul>
            <li><strong>1-2 correct periods</strong> (high confidence) - from Google Sheets data</li>
            <li><strong>2 periodogram peaks</strong> (medium confidence) - that are not correct periods</li>
            <li><strong>2 random periods</strong> (low confidence) - for comparison</li>
          </ul>
          
          <h4>Data Sources</h4>
          <p>Training data is extracted from multiple telescope/sensor sources:</p>
          <ul>
            <li>CDIPS, ELEANOR, QLP, SPOC</li>
            <li>TESS 16, TASOC, TGLC, EVEREST</li>
            <li>K2SC, K2SFF, K2VARCAT</li>
            <li>ZTF (R & G bands), WISE (W1 & W2)</li>
          </ul>

          <h4>Authentication Requirements</h4>
          <p>
            Training requires Google Sheets authentication. Make sure to:
          </p>
          <ul>
            <li>Configure Google Sheets URL in Settings</li>
            <li>Complete OAuth authentication</li>
            <li>Ensure access to the training data spreadsheet</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Training;
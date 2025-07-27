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

interface TrainingFile {
  filename: string;
  upload_time: number;
  size_bytes: number;
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
  const [isUploading, setIsUploading] = useState(false);
  const [starRange, setStarRange] = useState('');
  const [forceRetrain, setForceRetrain] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error' | 'info' | ''>('');
  const [trainingProgress, setTrainingProgress] = useState('');
  const [trainingFiles, setTrainingFiles] = useState<TrainingFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');

  useEffect(() => {
    loadModelStatus();
    loadTrainingFiles();
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

  const loadTrainingFiles = async () => {
    try {
      const response = await fetch(`${API_BASE}/training_data/files`);
      if (response.ok) {
        const data = await response.json();
        setTrainingFiles(data.files);
        // Select the first file by default if none selected
        if (data.files.length > 0 && !selectedFile) {
          setSelectedFile(data.files[0].filename);
        }
      }
    } catch (error) {
      console.error('Error loading training files:', error);
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

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      showMessage('Please select a CSV file', 'error');
      return;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/training_data/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        showMessage(`CSV uploaded successfully: ${data.message}`, 'success');
        await loadTrainingFiles();
        // Select the newly uploaded file
        if (data.filename) {
          setSelectedFile(data.filename);
        }
      } else {
        const error = await response.json();
        showMessage(`Error uploading CSV: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Error uploading CSV: ${error}`, 'error');
    } finally {
      setIsUploading(false);
      // Reset file input
      event.target.value = '';
    }
  };

  const handleDeleteFile = async (filename: string) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/training_data/files/${filename}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        showMessage('File deleted successfully', 'success');
        await loadTrainingFiles();
        // Clear selected file if it was deleted
        if (selectedFile === filename) {
          setSelectedFile('');
        }
      } else {
        const error = await response.json();
        showMessage(`Error deleting file: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Error deleting file: ${error}`, 'error');
    }
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
    if (!selectedFile) {
      showMessage('Please select a CSV file first', 'error');
      return;
    }

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
          csv_filename: selectedFile,
          stars_to_extract: starsToExtract,
          force_retrain: forceRetrain,
        }),
      });

      if (response.ok) {
        const result: TrainingResult = await response.json();
        
        if (result.success) {
          showMessage(
            `Training completed successfully! 
            Epochs: ${result.epochs_trained}, 
            Final Loss: ${result.final_loss.toFixed(4)}, 
            Training Samples: ${result.training_samples}`,
            'success'
          );
          await loadModelStatus();
        } else {
          showMessage('Training failed', 'error');
        }
      } else {
        const error = await response.json();
        showMessage(`Training failed: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Training failed: ${error}`, 'error');
    } finally {
      setIsTraining(false);
      setTrainingProgress('');
    }
  };

  const handleExportCSV = async () => {
    if (!selectedFile) {
      showMessage('Please select a CSV file first', 'error');
      return;
    }

    setIsExporting(true);
    
    try {
      const starsToExtract = parseStarRange(starRange);
      
      const response = await fetch(`${API_BASE}/export_training_csv`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          csv_filename: selectedFile,
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
      showMessage(`Export failed: ${error}`, 'error');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="training-container">
      <div className="training-header">
        <h2>Model Training & Management</h2>
        <p>Train machine learning models using CSV training data and manage training datasets.</p>
      </div>

      {message && (
        <div className={`training-message ${messageType}`}>
          {message}
        </div>
      )}

      {/* CSV File Management Section */}
      <div className="training-section">
        <h3>Training Data Files</h3>
        
        <div className="csv-upload-section">
          <div className="upload-area">
            <label htmlFor="csv-upload" className="upload-label">
              <div className="upload-content">
                <div className="upload-icon">📁</div>
                <div className="upload-text">
                  <strong>Upload CSV Training Data</strong>
                  <p>Select a CSV file containing period data for training</p>
                </div>
              </div>
            </label>
            <input
              id="csv-upload"
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              disabled={isUploading}
              style={{ display: 'none' }}
            />
            {isUploading && <div className="upload-progress">Uploading...</div>}
          </div>

          <div className="csv-requirements">
            <h4>CSV Format Requirements:</h4>
            <ul>
              <li><strong>Star</strong> - Star number/identifier</li>
              <li><strong>LC_Category</strong> - Light curve category</li>
              <li><strong>{'{Sensor}_period_1'}</strong> and <strong>{'{Sensor}_period_2'}</strong> - Period data for each sensor</li>
            </ul>
            <p><strong>Supported sensors:</strong> CDIPS, ELEANOR, QLP, SPOC, TESS16, TASOC, TGLC, EVEREST, K2SC, K2SFF, K2VARCAT, ZTF_R, ZTF_G, W1, W2</p>
          </div>
        </div>

        {trainingFiles.length > 0 && (
          <div className="training-files">
            <h4>Uploaded Files:</h4>
            <div className="file-list">
              {trainingFiles.map((file) => (
                <div key={file.filename} className={`file-item ${selectedFile === file.filename ? 'selected' : ''}`}>
                  <div className="file-info">
                    <div className="file-name">{file.filename}</div>
                    <div className="file-details">
                      Uploaded: {new Date(file.upload_time * 1000).toLocaleDateString()} | 
                      Size: {(file.size_bytes / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </div>
                  <div className="file-actions">
                    <button
                      onClick={() => setSelectedFile(file.filename)}
                      className={`select-btn ${selectedFile === file.filename ? 'selected' : ''}`}
                      disabled={isTraining || isExporting}
                    >
                      {selectedFile === file.filename ? 'Selected' : 'Select'}
                    </button>
                    <button
                      onClick={() => handleDeleteFile(file.filename)}
                      className="delete-btn"
                      disabled={isTraining || isExporting}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

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
            <li><strong>1-2 correct periods</strong> (high confidence) - from CSV training data</li>
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

          <h4>CSV File Requirements</h4>
          <p>
            Upload a CSV file with your training data. The file should contain:
          </p>
          <ul>
            <li>Star column with star identifiers</li>
            <li>LC_Category column with light curve classifications</li>
            <li>Period columns for each sensor (e.g., CDIPS_period_1, CDIPS_period_2)</li>
            <li>Any additional period data from available telescopes/sensors</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Training;
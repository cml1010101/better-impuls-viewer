import { useState, useEffect } from 'react';
import { API_BASE } from './api';
import './Settings.css';

interface CredentialsStatus {
  sed_service: {
    configured: boolean;
  };
  data_folder: {
    configured: boolean;
    current_path: string;
  };
}

interface SettingsProps {
  onClose: () => void;
}

const Settings: React.FC<SettingsProps> = ({ onClose }) => {
  const [credentialsStatus, setCredentialsStatus] = useState<CredentialsStatus | null>(null);
  const [sedUrl, setSedUrl] = useState('');
  const [sedUsername, setSedUsername] = useState('');
  const [sedPassword, setSedPassword] = useState('');
  const [dataFolderPath, setDataFolderPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error' | ''>('');

  useEffect(() => {
    loadCredentialsStatus();
  }, []);

  const loadCredentialsStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/credentials/status`);
      if (response.ok) {
        const status = await response.json();
        setCredentialsStatus(status);
      }
    } catch (error) {
      console.error('Error loading credentials status:', error);
    }
  };

  const showMessage = (text: string, type: 'success' | 'error') => {
    setMessage(text);
    setMessageType(type);
    setTimeout(() => {
      setMessage('');
      setMessageType('');
    }, 5000);
  };

  const handleSaveCredentials = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/credentials/configure`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sed_url: sedUrl || undefined,
          sed_username: sedUsername || undefined,
          sed_password: sedPassword || undefined,
          data_folder_path: dataFolderPath || undefined,
        }),
      });

      if (response.ok) {
        showMessage('Configuration saved successfully!', 'success');
        await loadCredentialsStatus();
        // Clear form
        setSedUrl('');
        setSedUsername('');
        setSedPassword('');
        setDataFolderPath('');
      } else {
        const error = await response.json();
        showMessage(`Error saving configuration: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Error saving configuration: ${error}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleBrowseDataFolder = async () => {
    // Check if we're running in Electron
    if (window.electronAPI && window.electronAPI.isElectron) {
      try {
        const selectedPath = await window.electronAPI.selectDataFolder();
        if (selectedPath) {
          setDataFolderPath(selectedPath);
        }
      } catch (error) {
        console.error('Error selecting folder in Electron:', error);
        showMessage('Error selecting folder. Please try typing the path manually.', 'error');
      }
    } else {
      // Fallback for web browsers - use webkitdirectory
      const input = document.createElement('input');
      input.type = 'file';
      input.webkitdirectory = true;
      input.style.display = 'none';
      
      input.onchange = (event: any) => {
        const files = event.target.files;
        if (files && files.length > 0) {
          // Get the directory path from the first file
          const firstFile = files[0];
          const fullPath = firstFile.webkitRelativePath;
          const folderPath = fullPath.substring(0, fullPath.lastIndexOf('/'));
          
          // Try to get full path if available
          if (firstFile.path) {
            const dirPath = firstFile.path.substring(0, firstFile.path.lastIndexOf('/'));
            setDataFolderPath(dirPath);
          } else {
            setDataFolderPath(folderPath);
            showMessage('Note: In web browsers, you may need to manually enter the full path.', 'error');
          }
        }
      };
      
      document.body.appendChild(input);
      input.click();
      document.body.removeChild(input);
    }
  };

  return (
    <div className="settings-overlay">
      <div className="settings-modal">
        <div className="settings-header">
          <h2>Application Settings</h2>
          <button onClick={onClose} className="close-btn">×</button>
        </div>
        
        <div className="settings-content">
          {message && (
            <div className={`message ${messageType}`}>
              {message}
            </div>
          )}
          
          {/* SED Service Configuration */}
          <section className="credentials-section">
            <h3>SED Service Configuration</h3>
            <div className="status-info">
              <span className={`status-indicator ${credentialsStatus?.sed_service.configured ? 'configured' : 'not-configured'}`}>
                {credentialsStatus?.sed_service.configured ? '✓' : '✗'}
              </span>
              <span>SED Service: {credentialsStatus?.sed_service.configured ? 'Configured' : 'Not Configured'}</span>
            </div>
            
            <div className="input-group">
              <label htmlFor="sed-url">SED Service URL:</label>
              <input
                id="sed-url"
                type="url"
                value={sedUrl}
                onChange={(e) => setSedUrl(e.target.value)}
                placeholder="https://sed.example.com/api"
              />
            </div>
            
            <div className="input-group">
              <label htmlFor="sed-username">Username:</label>
              <input
                id="sed-username"
                type="text"
                value={sedUsername}
                onChange={(e) => setSedUsername(e.target.value)}
                placeholder="your-username"
              />
            </div>
            
            <div className="input-group">
              <label htmlFor="sed-password">Password:</label>
              <input
                id="sed-password"
                type="password"
                value={sedPassword}
                onChange={(e) => setSedPassword(e.target.value)}
                placeholder="your-password"
              />
            </div>
          </section>

          {/* Data Folder Configuration */}
          <section className="credentials-section">
            <h3>Data Folder Configuration</h3>
            <div className="status-info">
              <span className={`status-indicator ${credentialsStatus?.data_folder.configured ? 'configured' : 'not-configured'}`}>
                {credentialsStatus?.data_folder.configured ? '✓' : '✗'}
              </span>
              <span>Data Folder: {credentialsStatus?.data_folder.configured ? 'Configured' : 'Not Configured'}</span>
              {credentialsStatus?.data_folder.current_path && (
                <div className="current-path">
                  Current path: {credentialsStatus.data_folder.current_path}
                </div>
              )}
            </div>
            
            <div className="input-group">
              <label htmlFor="data-folder">Data Folder Path:</label>
              <div className="folder-input-group">
                <input
                  id="data-folder"
                  type="text"
                  value={dataFolderPath}
                  onChange={(e) => setDataFolderPath(e.target.value)}
                  placeholder="/path/to/your/data/folder"
                />
                <button
                  type="button"
                  onClick={handleBrowseDataFolder}
                  className="browse-btn"
                >
                  Browse
                </button>
              </div>
              <small>Specify the folder containing your light curve data files.</small>
            </div>
          </section>

          {/* CSV Upload Information */}
          <section className="credentials-section">
            <h3>Training Data</h3>
            <div className="info-box">
              <h4>Using CSV Files for Training</h4>
              <p>
                This application now uses CSV files for training data instead of Google Sheets.
                To train models, you'll need to upload a CSV file containing period data for your stars.
              </p>
              <p><strong>Required CSV columns:</strong></p>
              <ul>
                <li><code>Star</code> - Star number/identifier</li>
                <li><code>LC_Category</code> - Light curve category</li>
                <li><code>{'{Sensor}_period_1'}</code> and <code>{'{Sensor}_period_2'}</code> - Period data for each sensor</li>
              </ul>
              <p><strong>Supported sensors:</strong> CDIPS, ELEANOR, QLP, SPOC, TESS16, TASOC, TGLC, EVEREST, K2SC, K2SFF, K2VARCAT, ZTF_R, ZTF_G, W1, W2</p>
              <p>
                Upload your CSV file in the Training page to begin model training.
              </p>
            </div>
          </section>

          <div className="form-actions">
            <button
              onClick={handleSaveCredentials}
              disabled={loading}
              className="save-btn"
            >
              {loading ? 'Saving...' : 'Save Configuration'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
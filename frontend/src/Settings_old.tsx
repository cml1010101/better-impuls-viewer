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

  const handleGoogleAuth = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/oauth/google/authorize`);
      if (response.ok) {
        const data = await response.json();
        // Open OAuth URL in a new window
        window.open(data.authorization_url, '_blank', 'width=500,height=600');
        showMessage('Please complete authentication in the new window', 'success');
        
        // Poll for authentication status
        const pollAuth = setInterval(async () => {
          await loadCredentialsStatus();
          if (credentialsStatus?.google_sheets.authenticated) {
            clearInterval(pollAuth);
            showMessage('Google authentication successful!', 'success');
          }
        }, 2000);
        
        // Clear polling after 2 minutes
        setTimeout(() => clearInterval(pollAuth), 120000);
      } else {
        const error = await response.json();
        showMessage(`Error starting authentication: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Error starting authentication: ${error}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleRevokeGoogleAuth = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/oauth/google/revoke`, {
        method: 'POST',
      });

      if (response.ok) {
        showMessage('Google authentication revoked successfully!', 'success');
        await loadCredentialsStatus();
      } else {
        const error = await response.json();
        showMessage(`Error revoking authentication: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Error revoking authentication: ${error}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleConfigureOAuth = async () => {
    if (!googleClientId.trim() || !googleClientSecret.trim()) {
      showMessage('Both Client ID and Client Secret are required', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/oauth/google/configure`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          client_id: googleClientId.trim(),
          client_secret: googleClientSecret.trim(),
        }),
      });

      if (response.ok) {
        showMessage('OAuth credentials configured successfully!', 'success');
        setGoogleClientId('');
        setGoogleClientSecret('');
        setShowOAuthSetup(false);
        // Force immediate reload of credentials status
        setTimeout(async () => {
          await loadCredentialsStatus();
        }, 500);
      } else {
        const error = await response.json();
        showMessage(`Error configuring OAuth: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Error configuring OAuth: ${error}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleClearOAuthConfig = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/oauth/google/configure`, {
        method: 'DELETE',
      });

      if (response.ok) {
        showMessage('OAuth configuration cleared successfully!', 'success');
        await loadCredentialsStatus();
      } else {
        const error = await response.json();
        showMessage(`Error clearing OAuth config: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Error clearing OAuth config: ${error}`, 'error');
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
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="settings-content">
          {message && (
            <div className={`message ${messageType}`}>
              {message}
            </div>
          )}

          {/* Google Sheets Configuration */}
          <div className="settings-section">
            <h3>Google Sheets Configuration</h3>
            
            <div className="status-indicator">
              <span className={`status-dot ${credentialsStatus?.google_sheets.url_configured ? 'green' : 'red'}`}></span>
              <span>URL Configured: {credentialsStatus?.google_sheets.url_configured ? 'Yes' : 'No'}</span>
            </div>
            
            <div className="status-indicator">
              <span className={`status-dot ${credentialsStatus?.google_sheets.oauth_configured ? 'green' : 'red'}`}></span>
              <span>OAuth Configured: {credentialsStatus?.google_sheets.oauth_configured ? 'Yes' : 'No'}</span>
              {credentialsStatus?.google_sheets.config_error && (
                <div className="config-error">{credentialsStatus.google_sheets.config_error}</div>
              )}
            </div>
            
            <div className="status-indicator">
              <span className={`status-dot ${credentialsStatus?.google_sheets.authenticated ? 'green' : 'red'}`}></span>
              <span>Authenticated: {credentialsStatus?.google_sheets.authenticated ? 'Yes' : 'No'}</span>
              {credentialsStatus?.google_sheets.user_info && (
                <span className="user-info"> ({credentialsStatus.google_sheets.user_info.email})</span>
              )}
            </div>

            {!credentialsStatus?.google_sheets.oauth_configured && (
              <div className="oauth-setup-section">
                <div className="oauth-setup-header">
                  <h4>OAuth Setup Required</h4>
                  <p>You need to set up Google OAuth credentials to authenticate with Google Sheets.</p>
                  <button 
                    onClick={() => setShowOAuthSetup(!showOAuthSetup)}
                    className="setup-toggle-button"
                  >
                    {showOAuthSetup ? 'Hide Setup Instructions' : 'Show Setup Instructions'}
                  </button>
                </div>

                {showOAuthSetup && (
                  <div className="oauth-setup-content">
                    <div className="setup-instructions">
                      <h5>Setup Instructions:</h5>
                      <ol>
                        <li>Go to <a href="https://console.cloud.google.com/" target="_blank" rel="noopener noreferrer">Google Cloud Console</a></li>
                        <li>Create a new project or select an existing one</li>
                        <li>Enable the Google Sheets API and Google Drive API</li>
                        <li>Go to "APIs & Services" → "Credentials" → "Create Credentials" → "OAuth client ID"</li>
                        <li>Choose "Web application" as the application type</li>
                        <li>Add this redirect URI: <code>http://localhost:8000/oauth/google/callback</code></li>
                        <li>Copy the Client ID and Client Secret and paste them below</li>
                      </ol>
                    </div>

                    <div className="oauth-form">
                      <div className="form-group">
                        <label htmlFor="googleClientId">Google OAuth Client ID:</label>
                        <input
                          type="text"
                          id="googleClientId"
                          value={googleClientId}
                          onChange={(e) => setGoogleClientId(e.target.value)}
                          placeholder="your-client-id.apps.googleusercontent.com"
                        />
                      </div>

                      <div className="form-group">
                        <label htmlFor="googleClientSecret">Google OAuth Client Secret:</label>
                        <input
                          type="password"
                          id="googleClientSecret"
                          value={googleClientSecret}
                          onChange={(e) => setGoogleClientSecret(e.target.value)}
                          placeholder="Your client secret"
                        />
                      </div>

                      <button 
                        onClick={handleConfigureOAuth}
                        disabled={loading || !googleClientId.trim() || !googleClientSecret.trim()}
                        className="auth-button configure-oauth"
                      >
                        {loading ? 'Configuring...' : 'Configure OAuth'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {credentialsStatus?.google_sheets.oauth_configured && (
              <div className="oauth-configured">
                <div className="oauth-status">
                  ✅ OAuth credentials configured successfully
                </div>
                <button 
                  onClick={handleClearOAuthConfig}
                  disabled={loading}
                  className="auth-button clear-oauth"
                >
                  {loading ? 'Clearing...' : 'Clear OAuth Configuration'}
                </button>
              </div>
            )}

            <div className="form-group">
              <label htmlFor="googleSheetsUrl">Google Sheets URL:</label>
              <input
                type="url"
                id="googleSheetsUrl"
                value={googleSheetsUrl}
                onChange={(e) => setGoogleSheetsUrl(e.target.value)}
                placeholder="https://docs.google.com/spreadsheets/d/..."
              />
            </div>

            <div className="auth-buttons">
              {credentialsStatus?.google_sheets.oauth_configured && (
                <>
                  {!credentialsStatus?.google_sheets.authenticated ? (
                    <button 
                      onClick={handleGoogleAuth} 
                      disabled={loading || !credentialsStatus?.google_sheets.url_configured}
                      className="auth-button google-auth"
                    >
                      {loading ? 'Authenticating...' : 'Authenticate with Google'}
                    </button>
                  ) : (
                    <button 
                      onClick={handleRevokeGoogleAuth} 
                      disabled={loading}
                      className="auth-button revoke-auth"
                    >
                      {loading ? 'Revoking...' : 'Revoke Authentication'}
                    </button>
                  )}
                </>
              )}
              {!credentialsStatus?.google_sheets.oauth_configured && (
                <div className="oauth-required-message">
                  Please configure OAuth credentials above to enable Google Sheets authentication.
                </div>
              )}
            </div>
          </div>

          {/* SED Service Configuration */}
          <div className="settings-section">
            <h3>SED Service Configuration</h3>
            
            <div className="status-indicator">
              <span className={`status-dot ${credentialsStatus?.sed_service.configured ? 'green' : 'red'}`}></span>
              <span>Configured: {credentialsStatus?.sed_service.configured ? 'Yes' : 'No'}</span>
            </div>

            <div className="form-group">
              <label htmlFor="sedUrl">SED Service URL:</label>
              <input
                type="text"
                id="sedUrl"
                value={sedUrl}
                onChange={(e) => setSedUrl(e.target.value)}
                placeholder="example.com:8080"
              />
            </div>

            <div className="form-group">
              <label htmlFor="sedUsername">Username:</label>
              <input
                type="text"
                id="sedUsername"
                value={sedUsername}
                onChange={(e) => setSedUsername(e.target.value)}
                placeholder="Username"
              />
            </div>

            <div className="form-group">
              <label htmlFor="sedPassword">Password:</label>
              <input
                type="password"
                id="sedPassword"
                value={sedPassword}
                onChange={(e) => setSedPassword(e.target.value)}
                placeholder="Password"
              />
            </div>
          </div>

          {/* Data Folder Configuration */}
          <div className="settings-section">
            <h3>Data Folder Configuration</h3>
            
            <div className="status-indicator">
              <span className={`status-dot ${credentialsStatus?.data_folder.configured ? 'green' : 'yellow'}`}></span>
              <span>Current Path: {credentialsStatus?.data_folder.current_path || 'Using default'}</span>
            </div>

            <div className="form-group">
              <label htmlFor="dataFolderPath">Data Folder Path:</label>
              <div className="folder-input-group">
                <input
                  type="text"
                  id="dataFolderPath"
                  value={dataFolderPath}
                  onChange={(e) => setDataFolderPath(e.target.value)}
                  placeholder="Select folder containing .tbl files"
                />
                <button 
                  type="button" 
                  onClick={handleBrowseDataFolder}
                  className="browse-button"
                  title="Browse for folder"
                >
                  📁
                </button>
              </div>
              <small className="form-help">
                Select the folder containing your star data files (.tbl format). 
                Files should be named like "001-telescope.tbl".
              </small>
            </div>
          </div>

          <div className="settings-actions">
            <button 
              onClick={handleSaveCredentials} 
              disabled={loading}
              className="save-button"
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
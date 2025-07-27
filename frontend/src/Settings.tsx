import { useState, useEffect } from 'react';
import { API_BASE } from './api';
import './Settings.css';

interface CredentialsStatus {
  google_sheets: {
    url_configured: boolean;
    authenticated: boolean;
    user_info?: {
      email: string;
      name: string;
    };
    has_refresh_token: boolean;
  };
  sed_service: {
    configured: boolean;
  };
}

interface SettingsProps {
  onClose: () => void;
}

const Settings: React.FC<SettingsProps> = ({ onClose }) => {
  const [credentialsStatus, setCredentialsStatus] = useState<CredentialsStatus | null>(null);
  const [googleSheetsUrl, setGoogleSheetsUrl] = useState('');
  const [sedUrl, setSedUrl] = useState('');
  const [sedUsername, setSedUsername] = useState('');
  const [sedPassword, setSedPassword] = useState('');
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
          google_sheets_url: googleSheetsUrl || undefined,
          sed_url: sedUrl || undefined,
          sed_username: sedUsername || undefined,
          sed_password: sedPassword || undefined,
        }),
      });

      if (response.ok) {
        showMessage('Credentials saved successfully!', 'success');
        await loadCredentialsStatus();
        // Clear form
        setGoogleSheetsUrl('');
        setSedUrl('');
        setSedUsername('');
        setSedPassword('');
      } else {
        const error = await response.json();
        showMessage(`Error saving credentials: ${error.detail}`, 'error');
      }
    } catch (error) {
      showMessage(`Error saving credentials: ${error}`, 'error');
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
              <span className={`status-dot ${credentialsStatus?.google_sheets.authenticated ? 'green' : 'red'}`}></span>
              <span>Authenticated: {credentialsStatus?.google_sheets.authenticated ? 'Yes' : 'No'}</span>
              {credentialsStatus?.google_sheets.user_info && (
                <span className="user-info"> ({credentialsStatus.google_sheets.user_info.email})</span>
              )}
            </div>

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
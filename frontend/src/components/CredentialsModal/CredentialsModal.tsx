import React, { useState, useEffect } from 'react';
import styles from './CredentialsModal.module.css';

interface CredentialsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (credentials: SEDCredentials) => void;
  apiBase: string;
}

interface SEDCredentials {
  sed_url: string;
  username: string;
  password: string;
}

const CredentialsModal: React.FC<CredentialsModalProps> = ({
  isOpen,
  onClose,
  onSave,
  apiBase,
}) => {
  const [credentials, setCredentials] = useState<SEDCredentials>({
    sed_url: '',
    username: '',
    password: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  // Load default configuration and cached credentials on mount
  useEffect(() => {
    if (isOpen) {
      loadDefaultConfig();
      loadCachedCredentials();
    }
  }, [isOpen, apiBase]);

  const loadDefaultConfig = async () => {
    try {
      const response = await fetch(`${apiBase}/sed/config/default`);
      if (response.ok) {
        const config = await response.json();
        setCredentials(prev => ({
          ...prev,
          sed_url: config.sed_url || prev.sed_url,
          username: config.username || prev.username,
          // Don't override password from server for security
        }));
      }
    } catch (err) {
      console.warn('Could not load default SED configuration:', err);
    }
  };

  const loadCachedCredentials = () => {
    try {
      const cached = localStorage.getItem('sed_credentials');
      if (cached) {
        const parsedCredentials = JSON.parse(cached);
        setCredentials(prev => ({
          ...prev,
          ...parsedCredentials,
        }));
      }
    } catch (err) {
      console.warn('Could not load cached credentials:', err);
    }
  };

  const handleInputChange = (field: keyof SEDCredentials, value: string) => {
    setCredentials(prev => ({
      ...prev,
      [field]: value,
    }));
    setError(''); // Clear error when user types
  };

  const handleSave = () => {
    setError('');
    
    // Validate required fields
    if (!credentials.sed_url.trim()) {
      setError('SED URL is required');
      return;
    }
    if (!credentials.username.trim()) {
      setError('Username is required');
      return;
    }
    if (!credentials.password.trim()) {
      setError('Password is required');
      return;
    }

    // Cache credentials in localStorage
    try {
      localStorage.setItem('sed_credentials', JSON.stringify(credentials));
    } catch (err) {
      console.warn('Could not cache credentials:', err);
    }

    onSave(credentials);
    onClose();
  };

  const handleTestConnection = async () => {
    setError('');
    setLoading(true);

    try {
      // Test with star number 1 (assuming it exists)
      const response = await fetch(`${apiBase}/sed/1`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      if (response.ok) {
        setError(''); // Clear any previous errors on success
        alert('Connection successful!');
      } else {
        const errorText = await response.text();
        setError(`Connection failed: ${errorText}`);
      }
    } catch (err) {
      setError(`Connection failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClearCache = () => {
    try {
      localStorage.removeItem('sed_credentials');
      setCredentials({
        sed_url: '',
        username: '',
        password: '',
      });
      loadDefaultConfig(); // Reload defaults
    } catch (err) {
      console.warn('Could not clear cache:', err);
    }
  };

  if (!isOpen) return null;

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>SED Server Credentials</h2>
          <button className={styles.closeButton} onClick={onClose}>
            ×
          </button>
        </div>

        <div className={styles.modalBody}>
          <p className={styles.description}>
            Enter credentials to access Spectral Energy Distribution (SED) images.
            These will be cached locally for future use.
          </p>

          {error && (
            <div className={styles.errorMessage}>
              {error}
            </div>
          )}

          <div className={styles.inputGroup}>
            <label htmlFor="sed_url">SED Server URL:</label>
            <input
              id="sed_url"
              type="text"
              value={credentials.sed_url}
              onChange={(e) => handleInputChange('sed_url', e.target.value)}
              placeholder="e.g., sed-server.example.com:8080"
              className={styles.input}
            />
          </div>

          <div className={styles.inputGroup}>
            <label htmlFor="username">Username:</label>
            <input
              id="username"
              type="text"
              value={credentials.username}
              onChange={(e) => handleInputChange('username', e.target.value)}
              placeholder="Enter username"
              className={styles.input}
            />
          </div>

          <div className={styles.inputGroup}>
            <label htmlFor="password">Password:</label>
            <input
              id="password"
              type="password"
              value={credentials.password}
              onChange={(e) => handleInputChange('password', e.target.value)}
              placeholder="Enter password"
              className={styles.input}
            />
          </div>
        </div>

        <div className={styles.modalFooter}>
          <button
            onClick={handleTestConnection}
            disabled={loading}
            className={styles.testButton}
          >
            {loading ? 'Testing...' : 'Test Connection'}
          </button>
          <button onClick={handleClearCache} className={styles.clearButton}>
            Clear Cache
          </button>
          <button onClick={onClose} className={styles.cancelButton}>
            Cancel
          </button>
          <button onClick={handleSave} className={styles.saveButton}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default CredentialsModal;
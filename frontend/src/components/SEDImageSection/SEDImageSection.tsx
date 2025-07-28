import React, { useState, useEffect } from 'react';
import CredentialsModal from '../CredentialsModal/CredentialsModal';
import styles from './SEDImageSection.module.css';

interface SEDImageSectionProps {
  selectedStar: number;
  apiBase: string;
  onImageError: () => void;
  onImageLoad: (event: React.SyntheticEvent<HTMLImageElement>) => void;
}

interface SEDCredentials {
  sed_url: string;
  username: string;
  password: string;
}

const SEDImageSection: React.FC<SEDImageSectionProps> = ({
  selectedStar,
  apiBase,
  onImageError,
  onImageLoad,
}) => {
  const [showCredentialsModal, setShowCredentialsModal] = useState(false);
  const [credentials, setCredentials] = useState<SEDCredentials | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [imageError, setImageError] = useState<string>('');
  const [loading, setLoading] = useState(false);

  // Load cached credentials on mount
  useEffect(() => {
    loadCachedCredentials();
  }, []);

  // Update image when star or credentials change
  useEffect(() => {
    updateImage();
  }, [selectedStar, credentials, apiBase]);

  const loadCachedCredentials = () => {
    try {
      const cached = localStorage.getItem('sed_credentials');
      if (cached) {
        const parsedCredentials = JSON.parse(cached);
        setCredentials(parsedCredentials);
      }
    } catch (err) {
      console.warn('Could not load cached credentials:', err);
    }
  };

  const updateImage = async () => {
    if (!credentials) {
      // Try fallback to environment variables (GET endpoint)
      setImageUrl(`${apiBase}/sed/${selectedStar}`);
      return;
    }

    setLoading(true);
    setImageError('');

    try {
      const response = await fetch(`${apiBase}/sed/${selectedStar}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      if (response.ok) {
        // Create a blob URL for the image
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setImageUrl(imageUrl);
        setImageError('');
      } else {
        const errorText = await response.text();
        setImageError(`Failed to load SED image: ${errorText}`);
        setImageUrl('');
      }
    } catch (err) {
      setImageError(`Error connecting to SED server: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setImageUrl('');
    } finally {
      setLoading(false);
    }
  };

  const handleCredentialsSave = (newCredentials: SEDCredentials) => {
    setCredentials(newCredentials);
  };

  const handleImageLoad = (event: React.SyntheticEvent<HTMLImageElement>) => {
    setImageError('');
    onImageLoad(event);
  };

  const handleImageError = () => {
    setImageError('Failed to load SED image. Please check your credentials.');
    onImageError();
  };

  const hasCredentials = credentials && credentials.sed_url && credentials.username && credentials.password;

  return (
    <div className={styles.sedSection}>
      <div className={styles.sedHeader}>
        <h3>Spectral Energy Distribution (SED)</h3>
        <div className={styles.sedControls}>
          <button 
            onClick={() => setShowCredentialsModal(true)}
            className={hasCredentials ? styles.configuredButton : styles.configureButton}
            title={hasCredentials ? 'Credentials configured' : 'Configure SED credentials'}
          >
            {hasCredentials ? '⚙️ Configured' : '⚙️ Configure'}
          </button>
          {hasCredentials && (
            <button 
              onClick={updateImage}
              className={styles.refreshButton}
              disabled={loading}
              title="Refresh SED image"
            >
              {loading ? '⟳' : '🔄'}
            </button>
          )}
        </div>
      </div>

      <div className={styles.sedImageContainer}>
        {imageError ? (
          <div className={styles.errorContainer}>
            <div className={styles.errorMessage}>
              {imageError}
            </div>
            <button 
              onClick={() => setShowCredentialsModal(true)}
              className={styles.configureButton}
            >
              Configure Credentials
            </button>
          </div>
        ) : loading ? (
          <div className={styles.loadingContainer}>
            <div className={styles.spinner}></div>
            <p>Loading SED image...</p>
          </div>
        ) : imageUrl ? (
          <img 
            src={imageUrl}
            alt={`SED for Star ${selectedStar}`}
            className={styles.sedImage}
            onError={handleImageError}
            onLoad={handleImageLoad}
          />
        ) : (
          <div className={styles.placeholderContainer}>
            <p>No SED image available</p>
            <button 
              onClick={() => setShowCredentialsModal(true)}
              className={styles.configureButton}
            >
              Configure Credentials
            </button>
          </div>
        )}
      </div>

      <CredentialsModal
        isOpen={showCredentialsModal}
        onClose={() => setShowCredentialsModal(false)}
        onSave={handleCredentialsSave}
        apiBase={apiBase}
      />
    </div>
  );
};

export default SEDImageSection;
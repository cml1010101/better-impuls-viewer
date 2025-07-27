import React from 'react';
import styles from './SEDImageSection.module.css';

interface SEDImageSectionProps {
  selectedStar: number;
  apiBase: string;
  onImageError: () => void;
  onImageLoad: (event: React.SyntheticEvent<HTMLImageElement>) => void;
}

const SEDImageSection: React.FC<SEDImageSectionProps> = ({
  selectedStar,
  apiBase,
  onImageError,
  onImageLoad,
}) => {
  return (
    <div className={styles.sedSection}>
      <h3>Spectral Energy Distribution (SED)</h3>
      <div className={styles.sedImageContainer}>
        <img 
          src={`${apiBase}/sed/${selectedStar}`} 
          alt={`SED for Star ${selectedStar}`}
          className={styles.sedImage}
          onError={onImageError}
          onLoad={onImageLoad}
        />
      </div>
    </div>
  );
};

export default SEDImageSection;
import React, { useState, useEffect } from 'react';
import styles from './StarList.module.css';

interface StarListProps {
  onSelectStar: (starNumber: number) => void;
}

const API_BASE = 'http://localhost:8000/api';

const StarList: React.FC<StarListProps> = ({ onSelectStar }) => {
  const [stars, setStars] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStars();
  }, []);

  const fetchStars = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/stars`);
      if (!response.ok) {
        throw new Error(`Failed to fetch stars: ${response.status}`);
      }
      const data = await response.json();
      setStars(data);
    } catch (error) {
      console.error('Error fetching stars:', error);
      // For demo purposes, show mock data when API is not available
      setStars([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
      setError(null);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className={styles.starList}>
        <div className={styles.header}>
          <h1>Better Impuls Viewer</h1>
          <p>Astronomical Data Analysis Dashboard</p>
        </div>
        <div className={styles.loading}>Loading stars...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.starList}>
        <div className={styles.header}>
          <h1>Better Impuls Viewer</h1>
          <p>Astronomical Data Analysis Dashboard</p>
        </div>
        <div className={styles.error}>
          <p>{error}</p>
          <button onClick={fetchStars} className={styles.retryButton}>
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.starList}>
      <div className={styles.header}>
        <h1>Better Impuls Viewer</h1>
        <p>Astronomical Data Analysis Dashboard</p>
      </div>
      
      <div className={styles.content}>
        <div className={styles.intro}>
          <h2>Select a Star to Analyze</h2>
          <p>Choose from {stars.length} available stars to explore their survey data, light curves, and periodograms.</p>
        </div>
        
        <div className={styles.starGrid}>
          {stars.map((starNumber) => (
            <button
              key={starNumber}
              className={styles.starCard}
              onClick={() => onSelectStar(starNumber)}
            >
              <div className={styles.starNumber}>
                {starNumber}
              </div>
              <div className={styles.starLabel}>
                Star {starNumber}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default StarList;
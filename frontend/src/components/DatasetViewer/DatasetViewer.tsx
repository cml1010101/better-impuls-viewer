import React, { useState, useEffect } from 'react';
import styles from './DatasetViewer.module.css';

const API_BASE = 'http://localhost:8000/api';

interface DatasetInfo {
  name: string;
  path: string;
  n_stars: number;
  n_files: number;
  surveys: string[];
  created_at?: string;
  total_data_points: number;
}

interface SyntheticStarInfo {
  star_id: number;
  name: string;
  variability_class: string;
  primary_period?: number;
  secondary_period?: number;
  surveys: string[];
}

interface DatasetViewerProps {
  onSelectSyntheticStar: (datasetName: string, starId: number) => void;
  onBackToMain: () => void;
}

const DatasetViewer: React.FC<DatasetViewerProps> = ({ onSelectSyntheticStar, onBackToMain }) => {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [stars, setStars] = useState<SyntheticStarInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showGenerator, setShowGenerator] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/datasets`);
      if (!response.ok) {
        throw new Error(`Failed to fetch datasets: ${response.status}`);
      }
      const data = await response.json();
      setDatasets(data);
    } catch (error) {
      console.error('Error fetching datasets:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch datasets');
    } finally {
      setLoading(false);
    }
  };

  const fetchStars = async (datasetName: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/datasets/${datasetName}/stars`);
      if (!response.ok) {
        throw new Error(`Failed to fetch stars: ${response.status}`);
      }
      const data = await response.json();
      setStars(data);
      setSelectedDataset(datasetName);
    } catch (error) {
      console.error('Error fetching stars:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch stars');
    } finally {
      setLoading(false);
    }
  };

  const deleteDataset = async (datasetName: string) => {
    if (!confirm(`Are you sure you want to delete dataset "${datasetName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/datasets/${datasetName}`, {
        method: 'DELETE'
      });
      if (!response.ok) {
        throw new Error(`Failed to delete dataset: ${response.status}`);
      }
      await fetchDatasets();
      if (selectedDataset === datasetName) {
        setSelectedDataset(null);
        setStars([]);
      }
    } catch (error) {
      console.error('Error deleting dataset:', error);
      setError(error instanceof Error ? error.message : 'Failed to delete dataset');
    }
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleString();
  };

  if (showGenerator) {
    return (
      <DatasetGenerator 
        onClose={() => setShowGenerator(false)}
        onDatasetCreated={fetchDatasets}
      />
    );
  }

  if (selectedDataset) {
    return (
      <div className={styles.datasetViewer}>
        <div className={styles.header}>
          <button 
            onClick={() => setSelectedDataset(null)}
            className={styles.backButton}
          >
            ← Back to Datasets
          </button>
          <h1>Dataset: {selectedDataset}</h1>
          <button 
            onClick={onBackToMain}
            className={styles.mainButton}
          >
            ← Back to Main
          </button>
        </div>

        {loading && <div className={styles.loading}>Loading stars...</div>}
        {error && <div className={styles.error}>{error}</div>}

        <div className={styles.starsGrid}>
          {stars.map((star) => (
            <button
              key={star.star_id}
              className={styles.starCard}
              onClick={() => onSelectSyntheticStar(selectedDataset, star.star_id)}
            >
              <div className={styles.starInfo}>
                <div className={styles.starNumber}>#{star.star_id}</div>
                <div className={styles.starName}>{star.name}</div>
                <div className={styles.variabilityClass}>{star.variability_class}</div>
                {star.primary_period && (
                  <div className={styles.period}>
                    Period: {star.primary_period.toFixed(4)} days
                  </div>
                )}
                <div className={styles.surveys}>
                  Surveys: {star.surveys.join(', ')}
                </div>
              </div>
              <div className={styles.arrow}>→</div>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.datasetViewer}>
      <div className={styles.header}>
        <button 
          onClick={onBackToMain}
          className={styles.backButton}
        >
          ← Back to Main
        </button>
        <h1>Synthetic Datasets</h1>
        <button 
          onClick={() => setShowGenerator(true)}
          className={styles.generateButton}
        >
          + Generate Dataset
        </button>
      </div>

      {loading && <div className={styles.loading}>Loading datasets...</div>}
      {error && <div className={styles.error}>{error}</div>}

      {datasets.length === 0 && !loading ? (
        <div className={styles.emptyState}>
          <h2>No Synthetic Datasets Found</h2>
          <p>Create your first synthetic dataset to get started with testing and training.</p>
          <button 
            onClick={() => setShowGenerator(true)}
            className={styles.generateButton}
          >
            Generate Dataset
          </button>
        </div>
      ) : (
        <div className={styles.datasetsGrid}>
          {datasets.map((dataset) => (
            <div key={dataset.name} className={styles.datasetCard}>
              <div className={styles.datasetInfo}>
                <h3>{dataset.name}</h3>
                <div className={styles.datasetStats}>
                  <div>⭐ {dataset.n_stars} stars</div>
                  <div>📁 {dataset.n_files} files</div>
                  <div>📊 {dataset.total_data_points.toLocaleString()} data points</div>
                </div>
                <div className={styles.surveys}>
                  Surveys: {dataset.surveys.join(', ')}
                </div>
                <div className={styles.created}>
                  Created: {formatDate(dataset.created_at)}
                </div>
              </div>
              <div className={styles.datasetActions}>
                <button 
                  onClick={() => fetchStars(dataset.name)}
                  className={styles.viewButton}
                >
                  View Stars
                </button>
                <button 
                  onClick={() => deleteDataset(dataset.name)}
                  className={styles.deleteButton}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Dataset Generator Component
interface DatasetGeneratorProps {
  onClose: () => void;
  onDatasetCreated: () => void;
}

const DatasetGenerator: React.FC<DatasetGeneratorProps> = ({ onClose, onDatasetCreated }) => {
  const [formData, setFormData] = useState({
    name: '',
    n_stars: 20,
    surveys: ['hubble', 'kepler', 'tess'],
    max_days: 50,
    min_days: 10,
    noise_level: 0.02,
    seed: 42
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      setError('Dataset name is required');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/datasets/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to generate dataset: ${response.status}`);
      }

      await response.json();
      onDatasetCreated();
      onClose();
    } catch (error) {
      console.error('Error generating dataset:', error);
      setError(error instanceof Error ? error.message : 'Failed to generate dataset');
    } finally {
      setLoading(false);
    }
  };

  const handleSurveyChange = (survey: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      surveys: checked 
        ? [...prev.surveys, survey]
        : prev.surveys.filter(s => s !== survey)
    }));
  };

  return (
    <div className={styles.generatorContainer}>
      <div className={styles.generator}>
        <div className={styles.generatorHeader}>
          <h2>Generate Synthetic Dataset</h2>
          <button onClick={onClose} className={styles.closeButton}>×</button>
        </div>

        {error && <div className={styles.error}>{error}</div>}

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.formGroup}>
            <label>Dataset Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              placeholder="e.g., training_dataset_v1"
              required
            />
          </div>

          <div className={styles.formGroup}>
            <label>Number of Stars</label>
            <input
              type="number"
              value={formData.n_stars}
              onChange={(e) => setFormData(prev => ({ ...prev, n_stars: parseInt(e.target.value) }))}
              min="1"
              max="1000"
            />
          </div>

          <div className={styles.formGroup}>
            <label>Surveys</label>
            <div className={styles.checkboxGroup}>
              {['hubble', 'kepler', 'tess'].map(survey => (
                <label key={survey} className={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={formData.surveys.includes(survey)}
                    onChange={(e) => handleSurveyChange(survey, e.target.checked)}
                  />
                  {survey.charAt(0).toUpperCase() + survey.slice(1)}
                </label>
              ))}
            </div>
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label>Min Days</label>
              <input
                type="number"
                value={formData.min_days}
                onChange={(e) => setFormData(prev => ({ ...prev, min_days: parseFloat(e.target.value) }))}
                min="1"
                step="0.1"
              />
            </div>
            <div className={styles.formGroup}>
              <label>Max Days</label>
              <input
                type="number"
                value={formData.max_days}
                onChange={(e) => setFormData(prev => ({ ...prev, max_days: parseFloat(e.target.value) }))}
                min="1"
                step="0.1"
              />
            </div>
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label>Noise Level</label>
              <input
                type="number"
                value={formData.noise_level}
                onChange={(e) => setFormData(prev => ({ ...prev, noise_level: parseFloat(e.target.value) }))}
                min="0.001"
                max="0.1"
                step="0.001"
              />
            </div>
            <div className={styles.formGroup}>
              <label>Random Seed</label>
              <input
                type="number"
                value={formData.seed}
                onChange={(e) => setFormData(prev => ({ ...prev, seed: parseInt(e.target.value) }))}
              />
            </div>
          </div>

          <div className={styles.formActions}>
            <button type="button" onClick={onClose} className={styles.cancelButton}>
              Cancel
            </button>
            <button type="submit" disabled={loading} className={styles.submitButton}>
              {loading ? 'Generating...' : 'Generate Dataset'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default DatasetViewer;
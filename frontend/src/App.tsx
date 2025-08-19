import { useState } from 'react';
import StarList from './components/StarList/StarList';
import StarPage from './components/StarPage/StarPage';
import DatasetViewer from './components/DatasetViewer/DatasetViewer';
import SyntheticStarPage from './components/SyntheticStarPage/SyntheticStarPage';
import styles from './App.module.css';

type ViewMode = 'starList' | 'starPage' | 'datasetViewer' | 'syntheticStarPage';

function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('starList');
  const [selectedStar, setSelectedStar] = useState<number | null>(null);
  const [syntheticDatasetName, setSyntheticDatasetName] = useState<string>('');
  const [syntheticStarId, setSyntheticStarId] = useState<number>(0);

  const handleSelectStar = (starNumber: number) => {
    setSelectedStar(starNumber);
    setViewMode('starPage');
  };

  const handleSelectSyntheticStar = (datasetName: string, starId: number) => {
    setSyntheticDatasetName(datasetName);
    setSyntheticStarId(starId);
    setViewMode('syntheticStarPage');
  };

  const handleBackToStarList = () => {
    setSelectedStar(null);
    setSyntheticDatasetName('');
    setSyntheticStarId(0);
    setViewMode('starList');
  };

  const handleViewDatasets = () => {
    setViewMode('datasetViewer');
  };

  const handleBackToDataset = () => {
    setViewMode('datasetViewer');
  };

  const handleBackToMain = () => {
    setViewMode('starList');
  };

  return (
    <div className={styles.app}>
      {viewMode === 'starList' && (
        <div>
          <div className={styles.navigation}>
            <button
              onClick={handleViewDatasets}
              className={styles.navButton}
            >
              📊 View Synthetic Datasets
            </button>
          </div>
          <StarList onSelectStar={handleSelectStar} />
        </div>
      )}

      {viewMode === 'starPage' && selectedStar && (
        <StarPage 
          starNumber={selectedStar} 
          onBackToStarList={handleBackToStarList}
        />
      )}

      {viewMode === 'datasetViewer' && (
        <DatasetViewer
          onSelectSyntheticStar={handleSelectSyntheticStar}
          onBackToMain={handleBackToMain}
        />
      )}

      {viewMode === 'syntheticStarPage' && syntheticDatasetName && (
        <SyntheticStarPage
          datasetName={syntheticDatasetName}
          starId={syntheticStarId}
          onBackToDataset={handleBackToDataset}
        />
      )}
    </div>
  );
}

export default App;

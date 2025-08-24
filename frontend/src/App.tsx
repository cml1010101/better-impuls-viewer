import { useState } from 'react';
import StarList from './components/StarList/StarList';
import StarPage from './components/StarPage/StarPage';
import styles from './App.module.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [selectedStar, setSelectedStar] = useState<number | null>(null);

  console.log('API_BASE:', API_BASE);

  const handleSelectStar = (starNumber: number) => {
    setSelectedStar(starNumber);
  };

  const handleBackToStarList = () => {
    setSelectedStar(null);
  };

  return (
    <div className={styles.app}>
      {selectedStar ? (
        <StarPage 
          starNumber={selectedStar} 
          onBackToStarList={handleBackToStarList}
        />
      ) : (
        <StarList onSelectStar={handleSelectStar} />
      )}
    </div>
  );
}

export default App;

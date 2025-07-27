import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Dashboard from './Dashboard';
import Training from './Training';
import Settings from './Settings';

function App() {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <Router>
      <div className="App">
        <header className="app-header">
          <div className="header-content">
            <div className="header-left">
              <h1>Better Impuls Viewer</h1>
              <p>Astronomical Data Analysis Dashboard</p>
            </div>
            <nav className="header-nav">
              <Link to="/" className="nav-link">Dashboard</Link>
              <Link to="/training" className="nav-link">Training</Link>
              <button 
                onClick={() => setShowSettings(true)} 
                className="nav-button settings-button"
              >
                Settings
              </button>
            </nav>
          </div>
        </header>

        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/training" element={<Training />} />
          </Routes>
        </main>

        {showSettings && (
          <Settings onClose={() => setShowSettings(false)} />
        )}
      </div>
    </Router>
  );
}

export default App;

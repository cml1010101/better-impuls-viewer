import styles from './App.module.css'
import Dashboard from './components/Dashboard/Dashboard'

function App() {
  return (
    <div className={styles.app}>
      <header className={styles.appHeader}>
        <h1>Better Impuls Viewer</h1>
        <p>Astronomical Data Analysis Dashboard</p>
      </header>
      <Dashboard />
    </div>
  )
}

export default App

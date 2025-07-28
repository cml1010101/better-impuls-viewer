import React from 'react';
import Plot from 'react-plotly.js';
import styles from './PeriodogramChart.module.css';

interface PeriodogramPoint {
  period: number;
  power: number;
}

interface PeriodogramChartProps {
  periodogramData: PeriodogramPoint[];
  selectedPeriod: number | null;
  periodInputValue: string;
  onPeriodogramClick: (data: Plotly.PlotMouseEvent) => void;
  onPeriodInputChange: (value: string) => void;
  onPeriodSubmit: () => void;
}

const PeriodogramChart: React.FC<PeriodogramChartProps> = ({
  periodogramData,
  selectedPeriod,
  periodInputValue,
  onPeriodogramClick,
  onPeriodInputChange,
  onPeriodSubmit,
}) => {
  if (periodogramData.length === 0) return null;

  return (
    <div className={styles.chartSection}>
      <h3>Periodogram</h3>
      <p className={styles.chartDescription}>
        Hover over the chart to see period values. Click or manually enter a period below to fold the data.
      </p>
      <Plot
        data={[
          {
            x: periodogramData.map(d => d.period),
            y: periodogramData.map(d => d.power),
            mode: 'lines+markers',
            type: 'scatter',
            line: {
              color: '#ff7300',
              width: 2,
            },
            marker: {
              color: '#ff7300',
              size: 4,
            },
            name: 'Power',
          },
        ]}
        layout={{
          width: undefined,
          height: 180,
          margin: { l: 50, r: 15, t: 20, b: 45 },
          xaxis: {
            title: { text: 'Period (days)' },
            type: 'log',
            automargin: true,
          },
          yaxis: {
            title: { text: 'Power' },
            automargin: true,
          },
          showlegend: false,
          dragmode: 'pan',
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
          displaylogo: false,
        }}
        style={{ width: '100%' }}
        onClick={onPeriodogramClick}
      />
      
      {/* Manual period input */}
      <div className={styles.periodInputSection}>
        <label className={styles.periodLabel}>
          Enter Period (days):
        </label>
        <input
          type="number"
          step="0.0001"
          min="0.1"
          max="20"
          value={periodInputValue}
          onChange={(e) => onPeriodInputChange(e.target.value)}
          className={styles.periodInput}
          placeholder="e.g., 7.8"
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              onPeriodSubmit();
            }
          }}
        />
        <button
          onClick={onPeriodSubmit}
          className={styles.periodButton}
        >
          Fold Data
        </button>
      </div>
      
      {selectedPeriod && (
        <p className={styles.selectedPeriod}>
          Selected Period: {selectedPeriod.toFixed(4)} days
        </p>
      )}
    </div>
  );
};

export default PeriodogramChart;
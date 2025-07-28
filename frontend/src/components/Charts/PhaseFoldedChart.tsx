import React from 'react';
import Plot from 'react-plotly.js';
import styles from './PhaseFoldedChart.module.css';

interface PhaseFoldedPoint {
  phase: number;
  flux: number;
}

interface PhaseFoldedChartProps {
  phaseFoldedData: PhaseFoldedPoint[];
  selectedPeriod: number;
}

const PhaseFoldedChart: React.FC<PhaseFoldedChartProps> = ({
  phaseFoldedData,
  selectedPeriod,
}) => {
  if (phaseFoldedData.length === 0 || !selectedPeriod) return null;

  return (
    <div className={styles.chartSection}>
      <h3>Phase-Folded Data (Period: {selectedPeriod.toFixed(4)} days)</h3>
      <Plot
        data={[
          {
            x: phaseFoldedData.map(d => d.phase),
            y: phaseFoldedData.map(d => d.flux),
            mode: 'markers',
            type: 'scatter',
            marker: {
              color: '#82ca9d',
              size: 4,
            },
            name: 'Flux',
          },
        ]}
        layout={{
          width: undefined,
          height: 180,
          margin: { l: 50, r: 15, t: 20, b: 45 },
          xaxis: {
            title: { text: 'Phase' },
            range: [0, 1],
            automargin: true,
          },
          yaxis: {
            title: { text: 'Flux' },
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
      />
    </div>
  );
};

export default PhaseFoldedChart;
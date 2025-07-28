import React from 'react';
import Plot from 'react-plotly.js';
import styles from './LightCurveChart.module.css';

interface DataPoint {
  time: number;
  flux: number;
  error: number;
}

interface LightCurveChartProps {
  campaignData: DataPoint[];
}

const LightCurveChart: React.FC<LightCurveChartProps> = ({ campaignData }) => {
  if (campaignData.length === 0) return null;

  return (
    <div className={`${styles.chartSection} ${styles.lightCurveSection}`}>
      <h3>Light Curve Data</h3>
      <Plot
        data={[
          {
            x: campaignData.map(d => d.time),
            y: campaignData.map(d => d.flux),
            mode: 'markers',
            type: 'scatter',
            marker: {
              color: '#8884d8',
              size: 4,
            },
            name: 'Flux',
          },
        ]}
        layout={{
          width: undefined,
          height: 220,
          margin: { l: 50, r: 15, t: 20, b: 45 },
          xaxis: {
            title: { text: 'Time (days)' },
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

export default LightCurveChart;
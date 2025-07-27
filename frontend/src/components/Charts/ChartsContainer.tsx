import React from 'react';
import LightCurveChart from './LightCurveChart';
import PeriodogramChart from './PeriodogramChart';
import PhaseFoldedChart from './PhaseFoldedChart';
import * as Plotly from 'plotly.js';
import styles from './ChartsContainer.module.css';

interface DataPoint {
  time: number;
  flux: number;
  error: number;
}

interface PeriodogramPoint {
  period: number;
  power: number;
}

interface PhaseFoldedPoint {
  phase: number;
  flux: number;
}

interface ChartsContainerProps {
  campaignData: DataPoint[];
  periodogramData: PeriodogramPoint[];
  phaseFoldedData: PhaseFoldedPoint[];
  selectedPeriod: number | null;
  periodInputValue: string;
  onPeriodogramClick: (data: Plotly.PlotMouseEvent) => void;
  onPeriodInputChange: (value: string) => void;
  onPeriodSubmit: () => void;
}

const ChartsContainer: React.FC<ChartsContainerProps> = ({
  campaignData,
  periodogramData,
  phaseFoldedData,
  selectedPeriod,
  periodInputValue,
  onPeriodogramClick,
  onPeriodInputChange,
  onPeriodSubmit,
}) => {
  return (
    <div className={styles.chartsContainer}>
      {/* Light Curve Chart - Full Width on Top */}
      <LightCurveChart campaignData={campaignData} />

      {/* Bottom Row: Periodogram and Phase-Folded Data Side by Side */}
      {(periodogramData.length > 0 || (phaseFoldedData.length > 0 && selectedPeriod)) && (
        <div className={styles.bottomCharts}>
          <PeriodogramChart
            periodogramData={periodogramData}
            selectedPeriod={selectedPeriod}
            periodInputValue={periodInputValue}
            onPeriodogramClick={onPeriodogramClick}
            onPeriodInputChange={onPeriodInputChange}
            onPeriodSubmit={onPeriodSubmit}
          />

          <PhaseFoldedChart
            phaseFoldedData={phaseFoldedData}
            selectedPeriod={selectedPeriod!}
          />
        </div>
      )}
    </div>
  );
};

export default ChartsContainer;
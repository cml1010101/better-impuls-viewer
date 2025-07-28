import React from 'react';
import LightCurveChart from './LightCurveChart';
import PeriodogramChart from './PeriodogramChart';
import PhaseFoldedChart from './PhaseFoldedChart';
import SEDImageSection from '../SEDImageSection/SEDImageSection';
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
  // SED props
  selectedStar: number | null;
  apiBase: string;
  sedImageAvailable: boolean;
  onSedImageError: () => void;
  onSedImageLoad: (event: React.SyntheticEvent<HTMLImageElement>) => void;
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
  selectedStar,
  apiBase,
  sedImageAvailable,
  onSedImageError,
  onSedImageLoad,
}) => {
  return (
    <div className={styles.chartsContainer}>
      {/* Top Row: Light Curve Chart and SED Image Side by Side */}
      {campaignData.length > 0 && (
        <div className={styles.topCharts}>
          <LightCurveChart campaignData={campaignData} />
          
          {/* SED Image Section - Only show if image is available and star is selected */}
          {selectedStar && sedImageAvailable && (
            <SEDImageSection
              selectedStar={selectedStar}
              apiBase={apiBase}
              onImageError={onSedImageError}
              onImageLoad={onSedImageLoad}
            />
          )}
        </div>
      )}

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
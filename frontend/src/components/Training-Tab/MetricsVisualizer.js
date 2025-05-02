import React, { useEffect, useRef } from 'react';
import { 
  Chart as ChartJS, 
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement
} from 'chart.js';
import { Line, Bar, Pie } from 'react-chartjs-2';
import './TrainingTab.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ArcElement
);

/**
 * MetricsVisualizer component for displaying training metrics visualizations
 * Shows loss curves, accuracy trends, and other relevant performance metrics
 * 
 * @param {Object} props - Component props
 * @param {Object} props.metrics - Training metrics data from model training
 */
const MetricsVisualizer = ({ metrics }) => {
  // Charts container reference for responsive sizing
  const chartsContainerRef = useRef(null);

  // Guard against invalid metrics data
  if (!metrics) {
    return null;
  }

  /**
   * Prepares data for the loss curve chart
   * @returns {Object} Chart.js data object for loss visualization
   */
  const getLossChartData = () => {
    // Safely access training history or return empty arrays if not available
    const history = metrics.history || {};
    const epochs = Array.from({ length: history.loss?.length || 0 }, (_, i) => i + 1);
    
    return {
      labels: epochs,
      datasets: [
        {
          label: 'Training Loss',
          data: history.loss || [],
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          fill: true,
          tension: 0.4,
        },
        {
          label: 'Validation Loss',
          data: history.val_loss || [],
          borderColor: 'rgb(54, 162, 235)',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          fill: true,
          tension: 0.4,
        }
      ]
    };
  };

  /**
   * Prepares data for the accuracy curve chart
   * @returns {Object} Chart.js data object for accuracy visualization
   */
  const getAccuracyChartData = () => {
    // Safely access training history or return empty arrays if not available
    const history = metrics.history || {};
    const epochs = Array.from({ length: history.accuracy?.length || 0 }, (_, i) => i + 1);
    
    return {
      labels: epochs,
      datasets: [
        {
          label: 'Training Accuracy',
          data: history.accuracy || [],
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          fill: true,
          tension: 0.4,
        },
        {
          label: 'Validation Accuracy',
          data: history.val_accuracy || [],
          borderColor: 'rgb(153, 102, 255)',
          backgroundColor: 'rgba(153, 102, 255, 0.1)',
          fill: true,
          tension: 0.4,
        }
      ]
    };
  };

  /**
   * Prepares data for class distribution chart
   * @returns {Object} Chart.js data object for class distribution visualization
   */
  const getClassDistributionData = () => {
    // If no class distribution data is available, return null
    if (!metrics.class_distribution) {
      return null;
    }
    
    const classNames = Object.keys(metrics.class_distribution);
    const counts = Object.values(metrics.class_distribution);
    
    // Generate colors for each class
    const backgroundColors = classNames.map((_, index) => {
      const hue = (index * 137) % 360; // Use golden ratio for color spacing
      return `hsla(${hue}, 70%, 60%, 0.7)`;
    });
    
    return {
      labels: classNames,
      datasets: [
        {
          data: counts,
          backgroundColor: backgroundColors,
          borderWidth: 1,
        }
      ]
    };
  };

  // Chart options
  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      y: {
        min: 0,
        ticks: {
          callback: (value) => value.toFixed(4)
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  // Accuracy chart options with y-axis limited to 0-1 range
  const accuracyChartOptions = {
    ...lineChartOptions,
    scales: {
      ...lineChartOptions.scales,
      y: {
        min: 0,
        max: 1,
        ticks: {
          callback: (value) => (value * 100).toFixed(1) + '%'
        }
      }
    }
  };

  // Class distribution chart options
  const pieChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          boxWidth: 12
        }
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const label = context.label || '';
            const value = context.raw || 0;
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            return `${label}: ${value} (${percentage}%)`;
          }
        }
      }
    }
  };

  // Determine if we have history data for charts
  const hasLossHistory = metrics.history?.loss && metrics.history.loss.length > 0;
  const hasAccuracyHistory = metrics.history?.accuracy && metrics.history.accuracy.length > 0;
  const hasClassDistribution = metrics.class_distribution && Object.keys(metrics.class_distribution).length > 0;

  return (
    <div className="metrics-visualizer" ref={chartsContainerRef}>
      <h3>Training Metrics Visualization</h3>
      
      <div className="metrics-charts-container">
        {/* Loss curve */}
        {hasLossHistory && (
          <div className="chart-container">
            <h4>Loss Curve</h4>
            <div className="chart-wrapper">
              <Line data={getLossChartData()} options={lineChartOptions} />
            </div>
          </div>
        )}
        
        {/* Accuracy curve */}
        {hasAccuracyHistory && (
          <div className="chart-container">
            <h4>Accuracy Curve</h4>
            <div className="chart-wrapper">
              <Line data={getAccuracyChartData()} options={accuracyChartOptions} />
            </div>
          </div>
        )}
      </div>
      
      {/* Performance metrics display */}
      <div className="performance-metrics">
        <h4>Model Performance</h4>
        <div className="metrics-grid">
          <div className="metric-item">
            <span className="metric-label">Test Accuracy:</span>
            <span className="metric-value">{(metrics.test_accuracy * 100).toFixed(2)}%</span>
          </div>
          
          {metrics.f1_score !== undefined && (
            <div className="metric-item">
              <span className="metric-label">F1 Score:</span>
              <span className="metric-value">{metrics.f1_score.toFixed(4)}</span>
            </div>
          )}
          
          {metrics.precision !== undefined && (
            <div className="metric-item">
              <span className="metric-label">Precision:</span>
              <span className="metric-value">{metrics.precision.toFixed(4)}</span>
            </div>
          )}
          
          {metrics.recall !== undefined && (
            <div className="metric-item">
              <span className="metric-label">Recall:</span>
              <span className="metric-value">{metrics.recall.toFixed(4)}</span>
            </div>
          )}
          
          {metrics.training_time && (
            <div className="metric-item">
              <span className="metric-label">Training Time:</span>
              <span className="metric-value">{metrics.training_time.toFixed(2)}s</span>
            </div>
          )}
        </div>
      </div>
      
      {/* Class distribution */}
      {hasClassDistribution && (
        <div className="chart-container class-distribution">
          <h4>Class Distribution</h4>
          <div className="chart-wrapper pie-chart-wrapper">
            <Pie data={getClassDistributionData()} options={pieChartOptions} />
          </div>
        </div>
      )}
      
      {/* Classification report if available */}
      {metrics.class_report && (
        <div className="classification-report">
          <h4>Classification Report</h4>
          <pre className="metric-report">{metrics.class_report}</pre>
        </div>
      )}
    </div>
  );
};

export default MetricsVisualizer;
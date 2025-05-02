import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import MetricsVisualizer from '../components/Training-Tab/MetricsVisualizer';

// Mock Chart.js to avoid canvas rendering issues in tests
jest.mock('chart.js', () => ({
  Chart: {
    register: jest.fn()
  },
  CategoryScale: jest.fn(),
  LinearScale: jest.fn(),
  PointElement: jest.fn(),
  LineElement: jest.fn(),
  BarElement: jest.fn(),
  Title: jest.fn(),
  Tooltip: jest.fn(),
  Legend: jest.fn(),
  Filler: jest.fn(),
  ArcElement: jest.fn()
}));

// Mock react-chartjs-2 components
jest.mock('react-chartjs-2', () => ({
  Line: () => <div data-testid="mocked-line-chart">Mocked Line Chart</div>,
  Bar: () => <div data-testid="mocked-bar-chart">Mocked Bar Chart</div>,
  Pie: () => <div data-testid="mocked-pie-chart">Mocked Pie Chart</div>
}));

describe('MetricsVisualizer Component', () => {
  // Test data for various scenarios
  const completeMetricsData = {
    test_accuracy: 0.85,
    f1_score: 0.84,
    precision: 0.86,
    recall: 0.82,
    training_time: 120.5,
    history: {
      loss: [0.8, 0.6, 0.4, 0.3, 0.25],
      val_loss: [0.85, 0.7, 0.5, 0.4, 0.35],
      accuracy: [0.5, 0.6, 0.7, 0.8, 0.85],
      val_accuracy: [0.45, 0.55, 0.65, 0.75, 0.8]
    },
    class_distribution: {
      'Class A': 100,
      'Class B': 150,
      'Class C': 80
    },
    class_report: 'Precision: 0.86, Recall: 0.82, F1-score: 0.84'
  };

  const partialMetricsData = {
    test_accuracy: 0.80,
    training_time: 90.2,
    history: {
      loss: [0.9, 0.7, 0.5],
      val_loss: [0.95, 0.8, 0.6],
      accuracy: [0.4, 0.5, 0.6],
      val_accuracy: [0.35, 0.45, 0.55]
    }
  };

  const singleEpochMetricsData = {
    test_accuracy: 0.60,
    history: {
      loss: [0.8],
      val_loss: [0.85],
      accuracy: [0.6],
      val_accuracy: [0.55]
    }
  };

  const largeEpochsMetricsData = {
    test_accuracy: 0.95,
    history: {
      loss: Array(100).fill(0).map((_, i) => 1 - i/100),
      val_loss: Array(100).fill(0).map((_, i) => 1.1 - i/100),
      accuracy: Array(100).fill(0).map((_, i) => i/100),
      val_accuracy: Array(100).fill(0).map((_, i) => (i/100) - 0.05)
    }
  };

  const emptyMetricsData = {
    test_accuracy: 0,
    history: {
      loss: [],
      val_loss: [],
      accuracy: [],
      val_accuracy: []
    }
  };

  // 1. Basic Rendering Tests
  describe('Basic Rendering', () => {
    test('renders without crashing', () => {
      render(<MetricsVisualizer metrics={completeMetricsData} />);
      expect(screen.getByText('Training Metrics Visualization')).toBeInTheDocument();
    });

    test('renders all chart sections with complete data', () => {
      render(<MetricsVisualizer metrics={completeMetricsData} />);
      
      // Check for chart headings
      expect(screen.getByText('Loss Curve')).toBeInTheDocument();
      expect(screen.getByText('Accuracy Curve')).toBeInTheDocument();
      expect(screen.getByText('Class Distribution')).toBeInTheDocument();

      // Check for chart components
      expect(screen.getAllByTestId('mocked-line-chart').length).toBe(2); // Loss and accuracy charts
      expect(screen.getByTestId('mocked-pie-chart')).toBeInTheDocument(); // Class distribution chart
    });

    test('renders metrics display with expected values', () => {
      render(<MetricsVisualizer metrics={completeMetricsData} />);
      
      // Check for performance metrics heading
      expect(screen.getByText('Model Performance')).toBeInTheDocument();
      
      // Check for metric values
      expect(screen.getByText('85.00%')).toBeInTheDocument(); // Test accuracy
      expect(screen.getByText('0.8400')).toBeInTheDocument(); // F1 score
      expect(screen.getByText('0.8600')).toBeInTheDocument(); // Precision
      expect(screen.getByText('0.8200')).toBeInTheDocument(); // Recall
      expect(screen.getByText('120.50s')).toBeInTheDocument(); // Training time
      
      // Check for classification report
      expect(screen.getByText('Classification Report')).toBeInTheDocument();
      expect(screen.getByText('Precision: 0.86, Recall: 0.82, F1-score: 0.84')).toBeInTheDocument();
    });
  });

  // 2. Data Handling Tests
  describe('Data Handling', () => {
    test('handles complete metrics data correctly', () => {
      render(<MetricsVisualizer metrics={completeMetricsData} />);
      
      // Check for all parts of complete data
      expect(screen.getByText('85.00%')).toBeInTheDocument();
      expect(screen.getByText('0.8400')).toBeInTheDocument();
      expect(screen.getByText('Class Distribution')).toBeInTheDocument();
      expect(screen.getByText('Classification Report')).toBeInTheDocument();
      
      // Verify all expected metric items are present
      expect(screen.getByText('Test Accuracy:')).toBeInTheDocument();
      expect(screen.getByText('F1 Score:')).toBeInTheDocument();
      expect(screen.getByText('Precision:')).toBeInTheDocument();
      expect(screen.getByText('Recall:')).toBeInTheDocument();
      expect(screen.getByText('Training Time:')).toBeInTheDocument();
    });

    test('handles partial metrics data gracefully', () => {
      render(<MetricsVisualizer metrics={partialMetricsData} />);
      
      // Should show available metrics
      expect(screen.getByText('80.00%')).toBeInTheDocument(); // Test accuracy
      expect(screen.getByText('90.20s')).toBeInTheDocument(); // Training time
      
      // Should not show unavailable metrics
      expect(screen.queryByText('F1 Score:')).not.toBeInTheDocument();
      expect(screen.queryByText('Precision:')).not.toBeInTheDocument();
      expect(screen.queryByText('Recall:')).not.toBeInTheDocument();
      expect(screen.queryByText('Class Distribution')).not.toBeInTheDocument();
      expect(screen.queryByText('Classification Report')).not.toBeInTheDocument();
      
      // Charts should still be present in the DOM
      expect(screen.getAllByTestId('mocked-line-chart').length).toBe(2); // Only loss and accuracy charts
      expect(screen.queryByTestId('mocked-pie-chart')).not.toBeInTheDocument(); // No pie chart
    });

    test('displays class distribution when available', () => {
      render(<MetricsVisualizer metrics={completeMetricsData} />);
      
      // Class distribution should be shown
      expect(screen.getByText('Class Distribution')).toBeInTheDocument();
      
      // Should render the pie chart
      expect(screen.getByTestId('mocked-pie-chart')).toBeInTheDocument();
    });

    test('returns null when metrics is undefined', () => {
      const { container } = render(<MetricsVisualizer metrics={undefined} />);
      expect(container).toBeEmptyDOMElement();
    });
  });

  // 3. Chart Configuration Tests
  describe('Chart Configurations', () => {
    test('chart containers are properly rendered', () => {
      render(<MetricsVisualizer metrics={completeMetricsData} />);
      
      // Verify the chart containers exist
      expect(screen.getByText('Loss Curve')).toBeInTheDocument();
      expect(screen.getByText('Accuracy Curve')).toBeInTheDocument();
      expect(screen.getByText('Class Distribution')).toBeInTheDocument();
      
      // Verify all chart headings are present
      expect(screen.getByText('Loss Curve')).toBeInTheDocument();
      expect(screen.getByText('Accuracy Curve')).toBeInTheDocument();
      expect(screen.getByText('Class Distribution')).toBeInTheDocument();
      
      // Verify chart components are present
      expect(screen.getAllByTestId('mocked-line-chart').length).toBe(2);
      expect(screen.getByTestId('mocked-pie-chart')).toBeInTheDocument();
    });
  });

  // 4. Edge Cases Tests
  describe('Edge Cases', () => {
    test('handles empty metrics data gracefully', () => {
      render(<MetricsVisualizer metrics={emptyMetricsData} />);
      
      // Should show test accuracy of 0%
      expect(screen.getByText('0.00%')).toBeInTheDocument();
      
      // Should not show charts for empty history data
      expect(screen.queryByTestId('mocked-line-chart')).not.toBeInTheDocument();
      expect(screen.queryByText('Loss Curve')).not.toBeInTheDocument();
      expect(screen.queryByText('Accuracy Curve')).not.toBeInTheDocument();
    });

    test('handles single epoch of data correctly', () => {
      render(<MetricsVisualizer metrics={singleEpochMetricsData} />);
      
      // Test accuracy should be displayed correctly
      expect(screen.getByText('60.00%')).toBeInTheDocument();
      
      // Verify chart containers exist for single epoch data
      expect(screen.getByText('Loss Curve')).toBeInTheDocument();
      expect(screen.getByText('Accuracy Curve')).toBeInTheDocument();
      
      // Both chart components should be present
      expect(screen.getAllByTestId('mocked-line-chart').length).toBe(2); // Loss and accuracy charts only
    });

    test('handles very large number of epochs efficiently', () => {
      render(<MetricsVisualizer metrics={largeEpochsMetricsData} />);
      
      // Test accuracy should be displayed correctly
      expect(screen.getByText('95.00%')).toBeInTheDocument();
      
      // Verify chart containers exist for large epoch data
      expect(screen.getByText('Loss Curve')).toBeInTheDocument();
      expect(screen.getByText('Accuracy Curve')).toBeInTheDocument();
      
      // Both chart components should be present
      expect(screen.getAllByTestId('mocked-line-chart').length).toBe(2); // Loss and accuracy charts only
    });
  });
});
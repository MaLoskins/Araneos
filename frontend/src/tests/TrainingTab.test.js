// TrainingTab.test.js
import React from 'react';
import { render, screen, fireEvent, act, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import TrainingTab from '../components/Training-Tab/TrainingTab';
import { GraphDataProvider } from '../context/GraphDataContext';
import { trainModel } from '../api';

// Mock the API module
jest.mock('../api', () => ({
  trainModel: jest.fn()
}));

// Mock Chart.js components
jest.mock('react-chartjs-2', () => ({
  Line: () => <div data-testid="mock-line-chart">Line Chart Mock</div>,
  Bar: () => <div data-testid="mock-bar-chart">Bar Chart Mock</div>,
  Pie: () => <div data-testid="mock-pie-chart">Pie Chart Mock</div>,
  Doughnut: () => <div data-testid="mock-doughnut-chart">Doughnut Chart Mock</div>,
}));

// Mock the MetricsVisualizer component
jest.mock('../components/Training-Tab/MetricsVisualizer', () => {
  return function MockMetricsVisualizer({ metrics }) {
    return <div data-testid="mock-metrics-visualizer">Metrics Visualizer Mock</div>;
  };
});

// Mock localStorage and BroadcastChannel for test environment
beforeAll(() => {
  const localStorageMock = (function() {
    let store = {};
    return {
      getItem(key) { return store[key] || null; },
      setItem(key, value) { store[key] = value.toString(); },
      removeItem(key) { delete store[key]; },
      clear() { store = {}; }
    };
  })();
  Object.defineProperty(window, 'localStorage', { value: localStorageMock });

  class BroadcastChannelMock {
    constructor() { this.onmessage = null; }
    postMessage() {}
    close() {}
    addEventListener() {}
    removeEventListener() {}
  }
  window.BroadcastChannel = BroadcastChannelMock;
});

// Helper to render TrainingTab with GraphDataProvider and custom context state
function renderWithGraphProvider(graphState) {
  // Patch localStorage for initial state
  window.localStorage.setItem(
    'graphData',
    JSON.stringify(graphState)
  );
  return render(
    <BrowserRouter>
      <GraphDataProvider>
        <TrainingTab />
      </GraphDataProvider>
    </BrowserRouter>
  );
}

describe('TrainingTab Component (with GraphDataProvider)', () => {
  const mockGraphState = {
    nodes: [
      { id: 'node1', label: 'A' },
      { id: 'node2', label: 'B' }
    ],
    edges: [{ source: 'node1', target: 'node2' }],
    lastSync: Date.now()
  };

  const mockUnlabeledGraphState = {
    nodes: [
      { id: 'node1' },
      { id: 'node2' }
    ],
    edges: [{ source: 'node1', target: 'node2' }],
    lastSync: Date.now()
  };

  const mockEmptyGraphState = {
    nodes: [],
    edges: [],
    lastSync: Date.now()
  };

  beforeEach(() => {
    jest.clearAllMocks();
    const mockCancel = jest.fn();
    trainModel.mockReturnValue({ cancel: mockCancel });
    window.localStorage.clear();
  });

  test('renders model selection and hyperparameter inputs', () => {
    renderWithGraphProvider(mockGraphState);
    expect(screen.getByText('Model Architecture:')).toBeInTheDocument();
    expect(screen.getByText('Hyperparameters')).toBeInTheDocument();
    expect(screen.getByLabelText('Hidden Channels:')).toBeInTheDocument();
    expect(screen.getByLabelText('Learning Rate:')).toBeInTheDocument();
    expect(screen.getByLabelText('Epochs:')).toBeInTheDocument();
    expect(screen.getByLabelText('Dropout:')).toBeInTheDocument();
    expect(screen.getByText('Start Training')).toBeInTheDocument();
    expect(screen.getByText('Stop Training')).toBeInTheDocument();
  });

  test('renders training logs area', () => {
    renderWithGraphProvider(mockGraphState);
    expect(screen.getByText('Training Logs')).toBeInTheDocument();
    expect(screen.getByText('Training logs will appear here...')).toBeInTheDocument();
  });

  test('renders graph summary with correct node and edge counts', () => {
    renderWithGraphProvider(mockGraphState);
    expect(screen.getByText('Graph Summary')).toBeInTheDocument();
    expect(screen.getByText('Nodes:')).toBeInTheDocument();
    expect(screen.getByText('Edges:')).toBeInTheDocument();
    expect(screen.getByText('2', { selector: '.stat-value' })).toBeInTheDocument();
    expect(screen.getByText('1', { selector: '.stat-value' })).toBeInTheDocument();
  });

  test('renders label information when graph has labels', () => {
    renderWithGraphProvider(mockGraphState);
    expect(screen.getByText('Has Labels:')).toBeInTheDocument();
    expect(screen.getByText('Yes')).toBeInTheDocument();
    // Check Available Labels heading and section
    const labelsSection = screen.getByTestId('available-labels-section');
    expect(labelsSection).toBeInTheDocument();
    expect(within(labelsSection).getByText('Available Labels')).toBeInTheDocument();
    expect(within(labelsSection).getByText('A')).toBeInTheDocument();
    expect(within(labelsSection).getByText('B')).toBeInTheDocument();
  });

  test('renders "Last Updated:" label and section', () => {
    renderWithGraphProvider(mockGraphState);
    const lastUpdatedSection = screen.getByTestId('last-updated-section');
    expect(lastUpdatedSection).toBeInTheDocument();
    expect(within(lastUpdatedSection).getByText('Last Updated:')).toBeInTheDocument();
  });

  test('displays message when no graph data is available', () => {
    renderWithGraphProvider(mockEmptyGraphState);
    expect(screen.getByText('No Graph Data Available')).toBeInTheDocument();
    expect(screen.getByText('You need to create a graph before you can train a model.')).toBeInTheDocument();
    expect(screen.getByText('Go to GraphNet Tab')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Go to GraphNet Tab' })).toHaveAttribute('href', '/');
  });

  test('start training button is disabled when no graph data is available', () => {
    renderWithGraphProvider(mockEmptyGraphState);
    const startButton = screen.getByText('Start Training');
    expect(startButton).toBeDisabled();
  });

  test('start training button is disabled when graph has no labels', () => {
    renderWithGraphProvider(mockUnlabeledGraphState);
    const startButton = screen.getByText('Start Training');
    expect(startButton).toBeDisabled();
    // Check warning with exact text and data-testid
    const warning = screen.getByTestId('validation-warning');
    expect(warning).toBeInTheDocument();
    expect(within(warning).getByText('Your graph data is missing labels required for training')).toBeInTheDocument();
  });

  test('start training button is enabled when graph has labels', () => {
    renderWithGraphProvider(mockGraphState);
    const startButton = screen.getByText('Start Training');
    expect(startButton).not.toBeDisabled();
  });

  test('initiates training when start button is clicked', () => {
    renderWithGraphProvider(mockGraphState);
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    expect(trainModel).toHaveBeenCalledTimes(1);
    expect(trainModel.mock.calls[0][0]).toEqual({
      nodes: mockGraphState.nodes,
      links: mockGraphState.edges
    });
  });

  test('displays training logs when messages are received', async () => {
    trainModel.mockImplementation((graphData, config, onMessage) => {
      act(() => {
        onMessage({
          type: 'log',
          message: 'Training started',
          timestamp: new Date().toISOString()
        });
      });
      return { cancel: jest.fn() };
    });
    renderWithGraphProvider(mockGraphState);
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    // Check for log entry and message with data-testid and exact text
    const logEntry = screen.getByTestId('training-log-entry');
    expect(logEntry).toBeInTheDocument();
    const logMessage = within(logEntry).getByTestId('training-log-message');
    expect(logMessage).toBeInTheDocument();
    expect(logMessage).toHaveTextContent('Training started');
    // Also check the logs container renders the logs
    expect(screen.getByText('Training started')).toBeInTheDocument();
  });

  test('displays metrics when training completes', async () => {
    trainModel.mockImplementation((graphData, config, onMessage) => {
      act(() => {
        onMessage({
          type: 'metrics',
          data: {
            test_accuracy: 0.85,
            class_report: 'Precision: 0.84, Recall: 0.82',
            training_time: 10.5
          }
        });
      });
      return { cancel: jest.fn() };
    });
    renderWithGraphProvider(mockGraphState);
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    expect(screen.getByText('Training Results')).toBeInTheDocument();
    expect(screen.getByText('85.00%')).toBeInTheDocument();
    expect(screen.getByText('Precision: 0.84, Recall: 0.82')).toBeInTheDocument();
    expect(screen.getByText('10.50s')).toBeInTheDocument();
    expect(screen.getByTestId('mock-metrics-visualizer')).toBeInTheDocument();
  });

  test('handles training errors gracefully', async () => {
    trainModel.mockImplementation((graphData, config, onMessage, onError) => {
      act(() => {
        onError(new Error('Network error during training'));
      });
      return { cancel: jest.fn() };
    });
    renderWithGraphProvider(mockGraphState);
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    expect(screen.getByText(/Error: Network error during training/i)).toBeInTheDocument();
  });

  test('stops training when stop button is clicked', () => {
    const mockCancel = jest.fn();
    trainModel.mockReturnValue({ cancel: mockCancel });
    renderWithGraphProvider(mockGraphState);
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    const stopButton = screen.getByText('Stop Training');
    fireEvent.click(stopButton);
    expect(mockCancel).toHaveBeenCalledTimes(1);
  });

  test('changes model configuration when different model is selected', () => {
    renderWithGraphProvider(mockGraphState);
    const modelSelect = screen.getByLabelText(/Model Architecture:/i);
    fireEvent.change(modelSelect, { target: { value: 'GAT' } });
    expect(screen.getByLabelText(/Attention Heads:/i)).toBeInTheDocument();
    fireEvent.change(modelSelect, { target: { value: 'ChebConv' } });
    expect(screen.getByLabelText(/Chebyshev Filter Size/i)).toBeInTheDocument();
  });

  test('resets training logs and results when model changes', () => {
    trainModel.mockImplementation((graphData, config, onMessage) => {
      act(() => {
        onMessage({
          type: 'metrics',
          data: {
            test_accuracy: 0.85,
            class_report: 'Precision: 0.84, Recall: 0.82',
            training_time: 10.5
          }
        });
      });
      return { cancel: jest.fn() };
    });
    renderWithGraphProvider(mockGraphState);
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    expect(screen.getByText('Training Results')).toBeInTheDocument();
    const modelSelect = screen.getByLabelText(/Model Architecture:/i);
    fireEvent.change(modelSelect, { target: { value: 'GraphSAGE' } });
    expect(screen.queryByText('Training Results')).not.toBeInTheDocument();
  });
});

// Additional tests for UI element visibility in all graph states
describe('TrainingTab UI element visibility and data-testid attributes', () => {
  const mockGraphState = {
    nodes: [
      { id: 'node1', label: 'A' },
      { id: 'node2', label: 'B' }
    ],
    edges: [{ source: 'node1', target: 'node2' }],
    lastSync: Date.now()
  };

  const mockUnlabeledGraphState = {
    nodes: [
      { id: 'node1' },
      { id: 'node2' }
    ],
    edges: [{ source: 'node1', target: 'node2' }],
    lastSync: Date.now()
  };

  const mockEmptyGraphState = {
    nodes: [],
    edges: [],
    lastSync: Date.now()
  };

  test('all key UI elements and data-testid attributes present with valid graph', () => {
    renderWithGraphProvider(mockGraphState);
    expect(screen.getByTestId('available-labels-section')).toBeInTheDocument();
    expect(screen.getByTestId('last-updated-section')).toBeInTheDocument();
    // No warning in valid state
    expect(screen.queryByTestId('validation-warning')).not.toBeInTheDocument();
    // Logs area always present
    expect(screen.getByText('Training Logs')).toBeInTheDocument();
  });

  test('all key UI elements and data-testid attributes present with unlabeled graph', () => {
    renderWithGraphProvider(mockUnlabeledGraphState);
    expect(screen.getByTestId('available-labels-section')).toBeInTheDocument();
    expect(screen.getByTestId('last-updated-section')).toBeInTheDocument();
    expect(screen.getByTestId('validation-warning')).toBeInTheDocument();
    expect(screen.getByText('Training Logs')).toBeInTheDocument();
  });

  test('all key UI elements present with empty graph', () => {
    renderWithGraphProvider(mockEmptyGraphState);
    // No available-labels-section or last-updated-section in empty state
    expect(screen.queryByTestId('available-labels-section')).not.toBeInTheDocument();
    expect(screen.queryByTestId('last-updated-section')).not.toBeInTheDocument();
    // No validation-warning in empty state
    expect(screen.queryByTestId('validation-warning')).not.toBeInTheDocument();
    expect(screen.getByText('Training Logs')).toBeInTheDocument();
  });
describe('TrainingTab validation warning with links property', () => {
  const mockGraphStateLinks = {
    nodes: [
      { id: 'node1', label: 'A' },
      { id: 'node2', label: 'B' }
    ],
    links: [{ source: 'node1', target: 'node2' }],
    lastSync: Date.now()
  };

  const mockUnlabeledGraphStateLinks = {
    nodes: [
      { id: 'node1' },
      { id: 'node2' }
    ],
    links: [{ source: 'node1', target: 'node2' }],
    lastSync: Date.now()
  };

  test('does not show validation warning for valid graph with links property', () => {
    renderWithGraphProvider(mockGraphStateLinks);
    expect(screen.queryByTestId('validation-warning')).not.toBeInTheDocument();
    expect(screen.getByText('Training Logs')).toBeInTheDocument();
  });

  test('shows validation warning for unlabeled graph with links property', () => {
    renderWithGraphProvider(mockUnlabeledGraphStateLinks);
    const warning = screen.getByTestId('validation-warning');
    expect(warning).toBeInTheDocument();
    expect(warning).toHaveTextContent('missing labels');
    expect(screen.getByText('Training Logs')).toBeInTheDocument();
  });
});
});
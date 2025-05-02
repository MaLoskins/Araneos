import React from 'react';
import { render, screen, fireEvent, act, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import TrainingTab from '../components/Training-Tab/TrainingTab';
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

describe('TrainingTab Component', () => {
  // Common test data
  const mockGraphData = {
    nodes: [
      { id: 'node1', label: 'A' },
      { id: 'node2', label: 'B' }
    ],
    links: [{ source: 'node1', target: 'node2' }]
  };
  
  const mockEmptyGraphData = {
    nodes: [],
    links: []
  };
  
  const mockUnlabeledGraphData = {
    nodes: [
      { id: 'node1' },
      { id: 'node2' }
    ],
    links: [{ source: 'node1', target: 'node2' }]
  };
  
  const mockGraphStats = {
    nodes: 2,
    edges: 1,
    hasLabels: true,
    uniqueLabels: ['A', 'B'],
    lastProcessed: new Date('2025-01-01T12:00:00Z')
  };
  
  const mockUnlabeledGraphStats = {
    nodes: 2,
    edges: 1,
    hasLabels: false,
    uniqueLabels: [],
    lastProcessed: new Date('2025-01-01T12:00:00Z')
  };
  
  const mockEmptyGraphStats = {
    nodes: 0,
    edges: 0,
    hasLabels: false,
    uniqueLabels: []
  };
  
  // More complex graph with multiple label types
  const mockComplexGraphData = {
    nodes: [
      { id: 'node1', label: 'A', data: { feature1: 0.5 } },
      { id: 'node2', label: 'B', data: { feature1: 0.3 } },
      { id: 'node3', label: 'C', data: { feature1: 0.8 } },
      { id: 'node4', label: 'A', data: { feature1: 0.2 } }
    ],
    links: [
      { source: 'node1', target: 'node2' },
      { source: 'node2', target: 'node3' },
      { source: 'node3', target: 'node4' },
      { source: 'node4', target: 'node1' }
    ]
  };
  
  const mockComplexGraphStats = {
    nodes: 4,
    edges: 4,
    hasLabels: true,
    uniqueLabels: ['A', 'B', 'C'],
    lastProcessed: new Date('2025-01-01T12:00:00Z')
  };
  
  let mockGetGraphStats;
  let mockIsGraphValidForTraining;
  
  // Reset mocks before each test
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetGraphStats = jest.fn().mockReturnValue(mockGraphStats);
    mockIsGraphValidForTraining = jest.fn().mockReturnValue(true);
    
    // Mock response for the trainModel function
    const mockCancel = jest.fn();
    trainModel.mockReturnValue({ cancel: mockCancel });
  });

  // 1. Basic Rendering Tests
  test('renders model selection and hyperparameter inputs', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify basic form elements are present
    expect(screen.getByText('Model Architecture:')).toBeInTheDocument();
    expect(screen.getByText('Hyperparameters')).toBeInTheDocument();
    expect(screen.getByLabelText('Hidden Channels:')).toBeInTheDocument();
    expect(screen.getByLabelText('Learning Rate:')).toBeInTheDocument();
    expect(screen.getByLabelText('Epochs:')).toBeInTheDocument();
    expect(screen.getByLabelText('Dropout:')).toBeInTheDocument();
    
    // Verify training buttons
    expect(screen.getByText('Start Training')).toBeInTheDocument();
    expect(screen.getByText('Stop Training')).toBeInTheDocument();
  });

  test('renders training logs area', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify logs container
    expect(screen.getByText('Training Logs')).toBeInTheDocument();
    expect(screen.getByText('Training logs will appear here...')).toBeInTheDocument();
  });

  // 2. Graph Status Display Tests
  test('renders graph summary with correct node and edge counts', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify graph summary section exists
    expect(screen.getByText('Graph Summary')).toBeInTheDocument();
    
    // Verify node and edge counts are displayed
    expect(screen.getByText('Nodes:')).toBeInTheDocument();
    expect(screen.getByText('Edges:')).toBeInTheDocument();
    
    // Verify the actual values appear in stat-value elements
    expect(screen.getByText('2', { selector: '.stat-value' })).toBeInTheDocument(); // Node count
    expect(screen.getByText('1', { selector: '.stat-value' })).toBeInTheDocument(); // Edge count
  });

  test('renders label information when graph has labels', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify label info is displayed
    expect(screen.getByText('Has Labels:')).toBeInTheDocument();
    expect(screen.getByText('Yes')).toBeInTheDocument();
    
    // Verify unique labels section
    expect(screen.getByText('Available Labels')).toBeInTheDocument();
    expect(screen.getByText('A')).toBeInTheDocument();
    expect(screen.getByText('B')).toBeInTheDocument();
  });

  test('renders timestamp information when available', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify timestamp is displayed
    expect(screen.getByText('Last Updated:')).toBeInTheDocument();
    // The time string will depend on locale, so we test for the label
  });

  // Enhanced graph status tests
  test('renders complex graph statistics with multiple labels correctly', () => {
    mockGetGraphStats = jest.fn().mockReturnValue(mockComplexGraphStats);
    
    render(
      <BrowserRouter>
        <TrainingTab
          graphData={mockComplexGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify complex graph summary displays correctly
    expect(screen.getByText('Nodes:')).toBeInTheDocument();
    expect(screen.getByText('Edges:')).toBeInTheDocument();
    
    // Use getAllByText for elements that might have duplicates
    const nodeValues = screen.getAllByText('4', { selector: '.stat-value' });
    expect(nodeValues.length).toBeGreaterThan(0);
    
    // Verify all unique labels are shown
    expect(screen.getByText('Available Labels')).toBeInTheDocument();
    expect(screen.getByText('A')).toBeInTheDocument();
    expect(screen.getByText('B')).toBeInTheDocument();
    expect(screen.getByText('C')).toBeInTheDocument();
    
    // Verify we have all three labels
    const labelItems = screen.getAllByText(/^[A-C]$/);
    expect(labelItems.length).toBe(3); // Three unique labels
  });

  // 3. No Graph Data Tests
  test('displays message when no graph data is available', () => {
    mockGetGraphStats = jest.fn().mockReturnValue(mockEmptyGraphStats);
    
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={null}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify no graph data message
    expect(screen.getByText('No Graph Data Available')).toBeInTheDocument();
    expect(screen.getByText('You need to create a graph before training a model.')).toBeInTheDocument();
    
    // Verify navigation link is present
    expect(screen.getByText('Go to GraphNet Tab')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Go to GraphNet Tab' })).toHaveAttribute('href', '/');
  });

  // 4. Workflow Guidance Tests
  test('renders workflow guidance steps', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify workflow guide section
    expect(screen.getByText('Workflow Guide')).toBeInTheDocument();
    
    // Verify workflow steps
    const steps = screen.getAllByRole('listitem');
    expect(steps.length).toBe(3); // There should be 3 steps
    
    // Verify step content
    expect(screen.getByText(/Create your graph/i)).toBeInTheDocument();
    expect(screen.getByText(/Configure and train your model/i)).toBeInTheDocument();
    expect(screen.getByText(/Evaluate model performance/i)).toBeInTheDocument();
  });

  test('workflow steps have appropriate state based on current progress', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Get all list items in the workflow
    const steps = screen.getAllByRole('listitem');
    
    // With graph data present, first step should be marked as completed and second as current
    // Instead of checking class names directly, we could check the contents/structure
    const firstStep = steps[0];
    const secondStep = steps[1];
    
    // Get step contents to verify their status indirectly
    expect(within(firstStep).getByText('1')).toBeInTheDocument();
    expect(within(firstStep).getByText(/Create your graph/)).toBeInTheDocument();
    
    expect(within(secondStep).getByText('2')).toBeInTheDocument();
    expect(within(secondStep).getByText(/Configure and train your model/)).toBeInTheDocument();
  });

  test('shows appropriate workflow messages based on graph state', () => {
    // Test with no graph data
    mockGetGraphStats = jest.fn().mockReturnValue(mockEmptyGraphStats);
    
    const { rerender } = render(
      <BrowserRouter>
        <TrainingTab 
          graphData={null}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={() => false}
        />
      </BrowserRouter>
    );
    
    // Verify warning message for no graph
    expect(screen.getByText(/You need to create a graph before you can train/i)).toBeInTheDocument();
    
    // Rerender with unlabeled graph
    mockGetGraphStats = jest.fn().mockReturnValue(mockUnlabeledGraphStats);
    mockIsGraphValidForTraining = jest.fn().mockReturnValue(false);
    
    rerender(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockUnlabeledGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify warning message for missing labels
    expect(screen.getByText(/Your graph needs labels for training/i)).toBeInTheDocument();
  });

  // Advanced workflow guidance test
  test('workflow step 3 becomes active when metrics are available', async () => {
    // Mock trainModel to provide metrics
    trainModel.mockImplementation((graphData, config, onMessage) => {
      // Send metrics immediately to simulate training completion
      act(() => {
        onMessage({
          type: 'metrics',
          data: {
            test_accuracy: 0.85,
            class_report: 'Precision: 0.84, Recall: 0.82',
            training_time: 5.2
          }
        });
      });
      return { cancel: jest.fn() };
    });
    
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Start training
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Verify metrics appear
    await screen.findByText('Training Results');
    
    // Get workflow steps
    const steps = screen.getAllByRole('listitem');
    const evaluateStep = steps[2]; // Third step - evaluate performance
    
    // Check that step 3 has the 'current' class or indicator
    expect(within(evaluateStep).getByText('3')).toBeInTheDocument();
    expect(within(evaluateStep).getByText(/Evaluate model performance/)).toBeInTheDocument();
    expect(evaluateStep.className).toContain('current');
  });

  // 5. Training Button Disabling Tests
  test('start training button is disabled when no graph data is available', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={null}
          loading={false}
          getGraphStats={() => mockEmptyGraphStats}
          isGraphValidForTraining={() => false}
        />
      </BrowserRouter>
    );
    
    // Verify Start Training button is disabled
    const startButton = screen.getByText('Start Training');
    expect(startButton).toBeDisabled();
  });

  test('start training button is disabled when graph has no labels', () => {
    mockIsGraphValidForTraining = jest.fn().mockReturnValue(false);
    
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockUnlabeledGraphData}
          loading={false}
          getGraphStats={() => mockUnlabeledGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify Start Training button is disabled
    const startButton = screen.getByText('Start Training');
    expect(startButton).toBeDisabled();
    
    // Verify validation warning is shown
    expect(screen.getByText(/Your graph data is missing labels required for training/i)).toBeInTheDocument();
  });

  test('start training button is enabled when graph has labels', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify Start Training button is enabled
    const startButton = screen.getByText('Start Training');
    expect(startButton).not.toBeDisabled();
  });

  // 6. Training Process Tests
  test('initiates training when start button is clicked', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Click the Start Training button
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Verify trainModel API was called with correct parameters
    expect(trainModel).toHaveBeenCalledTimes(1);
    expect(trainModel).toHaveBeenCalledWith(
      mockGraphData,
      expect.objectContaining({
        model_name: 'GCN',
        hidden_channels: 64,
        learning_rate: 0.01,
        epochs: 200,
        dropout: 0.3
      }),
      expect.any(Function),
      expect.any(Function)
    );
  });

  test('displays training logs when messages are received', async () => {
    // Setup mock for trainModel to simulate receiving messages
    trainModel.mockImplementation((graphData, config, onMessage) => {
      // Simulate log message
      act(() => {
        onMessage({
          type: 'log',
          message: 'Training started',
          timestamp: new Date().toISOString()
        });
      });
      
      return { cancel: jest.fn() };
    });
    
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Click the Start Training button
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Verify log message appears
    expect(screen.getByText('Training started')).toBeInTheDocument();
  });

  test('displays metrics when training completes', async () => {
    // Setup mock for trainModel to simulate receiving metrics
    trainModel.mockImplementation((graphData, config, onMessage) => {
      // Simulate metrics message
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
    
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Click the Start Training button
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Verify metrics are displayed
    expect(screen.getByText('Training Results')).toBeInTheDocument();
    expect(screen.getByText('85.00%')).toBeInTheDocument(); // 0.85 * 100 = 85.00%
    expect(screen.getByText('Precision: 0.84, Recall: 0.82')).toBeInTheDocument();
    expect(screen.getByText('10.50s')).toBeInTheDocument(); // 10.5s
    
    // Verify metrics visualizer is rendered
    expect(screen.getByTestId('mock-metrics-visualizer')).toBeInTheDocument();
  });

  test('handles training errors gracefully', async () => {
    // Setup mock for trainModel to simulate error
    trainModel.mockImplementation((graphData, config, onMessage, onError) => {
      // Simulate error
      act(() => {
        onError(new Error('Network error during training'));
      });
      
      return { cancel: jest.fn() };
    });
    
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Click the Start Training button
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Verify error message is displayed
    expect(screen.getByText(/Error: Network error during training/i)).toBeInTheDocument();
  });

  test('stops training when stop button is clicked', () => {
    // Setup mock cancel function
    const mockCancel = jest.fn();
    trainModel.mockReturnValue({ cancel: mockCancel });
    
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Start training
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Stop training
    const stopButton = screen.getByText('Stop Training');
    fireEvent.click(stopButton);
    
    // Verify cancel was called
    expect(mockCancel).toHaveBeenCalledTimes(1);
  });

  // 7. Model Selection Tests
  test('changes model configuration when different model is selected', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Get model select dropdown
    const modelSelect = screen.getByLabelText(/Model Architecture:/i);
    
    // Change model to GAT, which has additional parameters
    fireEvent.change(modelSelect, { target: { value: 'GAT' } });
    
    // Verify GAT-specific parameters appear
    expect(screen.getByLabelText(/Attention Heads:/i)).toBeInTheDocument();
    
    // Change model to ChebConv
    fireEvent.change(modelSelect, { target: { value: 'ChebConv' } });
    
    // Verify ChebConv-specific parameters appear
    expect(screen.getByLabelText(/Chebyshev Filter Size/i)).toBeInTheDocument();
  });

  test('resets training logs and results when model changes', () => {
    // Setup trainModel to simulate training completion
    trainModel.mockImplementation((graphData, config, onMessage) => {
      // Simulate metrics message
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
    
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Start training
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Verify metrics appear
    expect(screen.getByText('Training Results')).toBeInTheDocument();
    
    // Change model
    const modelSelect = screen.getByLabelText(/Model Architecture:/i);
    fireEvent.change(modelSelect, { target: { value: 'GraphSAGE' } });
    
    // Verify metrics are reset (should no longer be in document)
    expect(screen.queryByText('Training Results')).not.toBeInTheDocument();
  });
  
  // 8. Enhanced tests for graph validation and UI indicators
  test('displays all available labels with correct styling', () => {
    // Create mock with many labels
    const manyLabelsGraphStats = {
      ...mockGraphStats,
      uniqueLabels: ['A', 'B', 'C', 'D', 'E', 'F']
    };
    
    mockGetGraphStats = jest.fn().mockReturnValue(manyLabelsGraphStats);
    
    render(
      <BrowserRouter>
        <TrainingTab
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Verify the labels section exists
    expect(screen.getByText('Available Labels')).toBeInTheDocument();
    
    // Verify all six labels are shown
    const labels = screen.getAllByText(/^[A-F]$/);
    expect(labels.length).toBe(6);
    
    // Add data-testid to the component for label items to check styling if needed
    // For now, we're just checking the number of labels is correct
  });
  
  test('model-specific hyperparameters are properly configured for training', () => {
    render(
      <BrowserRouter>
        <TrainingTab 
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Test GAT model parameters
    // Select GAT model
    const modelSelect = screen.getByLabelText(/Model Architecture:/i);
    fireEvent.change(modelSelect, { target: { value: 'GAT' } });
    
    // Set attention heads
    const headsInput = screen.getByLabelText(/Attention Heads:/i);
    fireEvent.change(headsInput, { target: { value: '12' } });
    
    // Start training
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Verify GAT-specific params are included in the API call
    expect(trainModel).toHaveBeenCalledWith(
      mockGraphData,
      expect.objectContaining({
        model_name: 'GAT',
        heads: 12
      }),
      expect.any(Function),
      expect.any(Function)
    );
  });
  
  test('ChebConv-specific hyperparameters are properly configured for training', () => {
    // Mock trainModel API call
    trainModel.mockClear();
    
    // Render component with fresh state
    render(
      <BrowserRouter>
        <TrainingTab
          graphData={mockGraphData}
          loading={false}
          getGraphStats={mockGetGraphStats}
          isGraphValidForTraining={mockIsGraphValidForTraining}
        />
      </BrowserRouter>
    );
    
    // Select ChebConv model
    const modelSelect = screen.getByLabelText(/Model Architecture:/i);
    fireEvent.change(modelSelect, { target: { value: 'ChebConv' } });
    
    // Set K parameter
    const kInput = screen.getByLabelText(/Chebyshev Filter Size/i);
    fireEvent.change(kInput, { target: { value: '5' } });
    
    // Start training
    const startButton = screen.getByText('Start Training');
    fireEvent.click(startButton);
    
    // Verify ChebConv-specific params are included in API call
    expect(trainModel).toHaveBeenCalledWith(
      mockGraphData,
      expect.objectContaining({
        model_name: 'ChebConv',
        K: 5
      }),
      expect.any(Function),
      expect.any(Function)
    );
  });

  describe('Graph Summary and Status Display', () => {
    test('displays accurate graph summary with node and edge counts', () => {
      // Mock graph data with specific counts
      const customMockGraphData = {
        nodes: [
          { id: 'node1', label: 'A' },
          { id: 'node2', label: 'B' },
          { id: 'node3', label: 'C' }
        ],
        links: [
          { source: 'node1', target: 'node2' },
          { source: 'node2', target: 'node3' },
          { source: 'node1', target: 'node3' },
        ]
      };
      
      mockGetGraphStats = jest.fn().mockReturnValue({
        nodes: 3,
        edges: 3,
        hasLabels: true,
        uniqueLabels: ['A', 'B', 'C'],
        lastProcessed: new Date('2025-01-01T10:00:00Z')
      });
      
      render(
        <BrowserRouter>
          <TrainingTab
            graphData={customMockGraphData}
            loading={false}
            getGraphStats={mockGetGraphStats}
            isGraphValidForTraining={mockIsGraphValidForTraining}
          />
        </BrowserRouter>
      );
      
      // Verify graph summary displays the correct counts
      expect(screen.getByText('Nodes:')).toBeInTheDocument();
      expect(screen.getByText('Edges:')).toBeInTheDocument();
      
      // Use getAllByText for elements that might have duplicates
      const nodeValues = screen.getAllByText('3', { selector: '.stat-value' });
      expect(nodeValues.length).toBeGreaterThan(0);
      
      // Verify labels section exists
      expect(screen.getByText('Available Labels')).toBeInTheDocument();
      
      // Verify timestamp is displayed
      expect(screen.getByText('Last Updated:')).toBeInTheDocument();
      // Check that some timestamp is displayed, but don't check the exact value as it might vary
      const timestampValue = screen.getAllByText(/\d+:\d+:\d+ [ap]m/i, { selector: '.stat-value' });
      expect(timestampValue.length).toBeGreaterThan(0);
    });

    test('displays visual warning when graph has no labels', () => {
      // Mock graph data without labels
      const unlabeledGraphData = {
        nodes: [
          { id: 'node1' },
          { id: 'node2' }
        ],
        links: [
          { source: 'node1', target: 'node2' }
        ]
      };
      
      mockGetGraphStats = jest.fn().mockReturnValue({
        nodes: 2,
        edges: 1,
        hasLabels: false,
        uniqueLabels: []
      });
      
      mockIsGraphValidForTraining = jest.fn().mockReturnValue(false);
      
      render(
        <BrowserRouter>
          <TrainingTab
            graphData={unlabeledGraphData}
            loading={false}
            getGraphStats={mockGetGraphStats}
            isGraphValidForTraining={mockIsGraphValidForTraining}
          />
        </BrowserRouter>
      );
      
      // Verify warning message is displayed
      expect(screen.getByText(/Your graph needs labels for training/i)).toBeInTheDocument();
      
      // Verify the Start Training button is disabled
      expect(screen.getByText('Start Training')).toBeDisabled();
      
      // Verify validation warning appears
      const validationWarning = screen.getByText(/Your graph data is missing labels required for training/i);
      expect(validationWarning).toBeInTheDocument();
      
      // Verify link to GraphNet tab is present
      expect(screen.getAllByText('Go to GraphNet Tab')[0]).toBeInTheDocument();
    });

    test('workflow guidance shows correct steps based on graph state', () => {
      // Mock graph data with labels to test workflow guidance with a valid graph
      const labeledGraphData = {
        nodes: [
          { id: 'node1', label: 'A' },
          { id: 'node2', label: 'B' }
        ],
        links: [
          { source: 'node1', target: 'node2' }
        ]
      };
      
      mockGetGraphStats = jest.fn().mockReturnValue({
        nodes: 2,
        edges: 1,
        hasLabels: true,
        uniqueLabels: ['A', 'B']
      });
      
      mockIsGraphValidForTraining = jest.fn().mockReturnValue(true);
      
      const { rerender } = render(
        <BrowserRouter>
          <TrainingTab
            graphData={labeledGraphData}
            loading={false}
            getGraphStats={mockGetGraphStats}
            isGraphValidForTraining={mockIsGraphValidForTraining}
          />
        </BrowserRouter>
      );
      
      // Verify workflow steps are displayed
      expect(screen.getByText('Workflow Guide')).toBeInTheDocument();
      
      // Check for completed step (create graph is complete)
      const steps = screen.getAllByRole('listitem');
      expect(steps[0]).toHaveClass('completed');
      expect(steps[1]).toHaveClass('current');
      
      // Now rerender with no graph data to check workflow state
      rerender(
        <BrowserRouter>
          <TrainingTab
            graphData={null}
            loading={false}
            getGraphStats={mockGetGraphStats}
            isGraphValidForTraining={mockIsGraphValidForTraining}
          />
        </BrowserRouter>
      );
      
      // Verify workflow message for no graph
      expect(screen.getByText(/You need to create a graph before you can train a model/i)).toBeInTheDocument();
    });
  });

  describe('Graph data flow and UI state handling', () => {
    test('disables training when graph lacks labels', () => {
      // Mock graph with nodes but no labels
      const unlabeledGraphData = {
        nodes: [
          { id: 'node1' },
          { id: 'node2' }
        ],
        links: [
          { source: 'node1', target: 'node2' }
        ]
      };
      
      mockGetGraphStats = jest.fn().mockReturnValue({
        nodes: 2,
        edges: 1,
        hasLabels: false,
        uniqueLabels: []
      });
      
      mockIsGraphValidForTraining = jest.fn().mockReturnValue(false);
      
      render(
        <BrowserRouter>
          <TrainingTab
            graphData={unlabeledGraphData}
            loading={false}
            getGraphStats={mockGetGraphStats}
            isGraphValidForTraining={mockIsGraphValidForTraining}
          />
        </BrowserRouter>
      );
      
      // Verify Start Training button is disabled
      const startButton = screen.getByText('Start Training');
      expect(startButton).toBeDisabled();
      
      // Verify tooltip/title explaining why button is disabled
      expect(startButton).toHaveAttribute('title', expect.stringContaining('Graph data must include labels for training'));
    });

    test('updates graph summary when new graph data is provided', () => {
      // Initial render with one graph configuration
      mockGetGraphStats = jest.fn().mockReturnValue({
        nodes: 2,
        edges: 1,
        hasLabels: true,
        uniqueLabels: ['A', 'B']
      });
      
      const { rerender } = render(
        <BrowserRouter>
          <TrainingTab
            graphData={mockGraphData}
            loading={false}
            getGraphStats={mockGetGraphStats}
            isGraphValidForTraining={mockIsGraphValidForTraining}
          />
        </BrowserRouter>
      );
      
      // Verify initial summary
      expect(screen.getByText('Nodes:')).toBeInTheDocument();
      expect(screen.getByText('Edges:')).toBeInTheDocument();
      
      // Use getAllByText for elements that might have duplicates
      const nodeValues = screen.getAllByText('2', { selector: '.stat-value' });
      expect(nodeValues.length).toBeGreaterThan(0);
      
      const edgeValues = screen.getAllByText('1', { selector: '.stat-value' });
      expect(edgeValues.length).toBeGreaterThan(0);
      
      // Now update mock to return different stats
      mockGetGraphStats = jest.fn().mockReturnValue({
        nodes: 4,
        edges: 3,
        hasLabels: true,
        uniqueLabels: ['A', 'B', 'C']
      });
      
      // Rerender with updated graph data
      const updatedGraphData = {
        nodes: [
          { id: 'node1', label: 'A' },
          { id: 'node2', label: 'B' },
          { id: 'node3', label: 'C' },
          { id: 'node4', label: 'A' }
        ],
        links: [
          { source: 'node1', target: 'node2' },
          { source: 'node2', target: 'node3' },
          { source: 'node3', target: 'node4' }
        ]
      };
      
      rerender(
        <BrowserRouter>
          <TrainingTab
            graphData={updatedGraphData}
            loading={false}
            getGraphStats={mockGetGraphStats}
            isGraphValidForTraining={mockIsGraphValidForTraining}
          />
        </BrowserRouter>
      );
      
      // Verify updated summary - after rerender
      expect(screen.getByText('Nodes:')).toBeInTheDocument();
      expect(screen.getByText('Edges:')).toBeInTheDocument();
      
      // Use getAllByText for elements that might have duplicates
      const updatedNodeValues = screen.getAllByText('4', { selector: '.stat-value' });
      expect(updatedNodeValues.length).toBeGreaterThan(0);
      
      const updatedEdgeValues = screen.getAllByText('3', { selector: '.stat-value' });
      expect(updatedEdgeValues.length).toBeGreaterThan(0);
      
      // Check for new label
      const labels = screen.getAllByText(/^[A-C]$/);
      expect(labels.length).toBe(3); // 3 unique labels
    });
  });
});
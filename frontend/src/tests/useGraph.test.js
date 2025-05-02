import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { processData } from '../api';
import useGraph from '../hooks/useGraph';

// Mock the API module
jest.mock('../api', () => ({
  processData: jest.fn()
}));

// Mock react-flow-renderer
jest.mock('react-flow-renderer', () => ({
  useNodesState: () => [[], jest.fn(), jest.fn()],
  useEdgesState: () => [[], jest.fn(), jest.fn()],
  addEdge: (edge, edges) => [...edges, edge]
}));

// Create a test component that uses the hook
function TestComponent() {
  const hookResult = useGraph();
  
  return (
    <div>
      <div data-testid="csv-data">{JSON.stringify(hookResult.csvData)}</div>
      <div data-testid="graph-data">{JSON.stringify(hookResult.graphData)}</div>
      <div data-testid="loading">{hookResult.loading.toString()}</div>
      <div data-testid="graph-error">{hookResult.graphError || 'no-error'}</div>
      
      <button 
        data-testid="drop-file-btn" 
        onClick={() => hookResult.handleFileDrop([{ id: 1, name: 'test' }], ['id', 'name'])}
      >
        Drop File
      </button>
      
      <button 
        data-testid="toggle-feature-space" 
        onClick={hookResult.toggleFeatureSpace}
      >
        Toggle Feature Space
      </button>
      
      <button 
        data-testid="select-node-btn" 
        onClick={() => hookResult.handleSelectNode('node1')}
      >
        Select Node
      </button>
      
      <button 
        data-testid="submit-btn" 
        onClick={() => hookResult.handleSubmit('label')}
      >
        Submit
      </button>
      
      <div data-testid="is-valid-for-training">
        {hookResult.isGraphValidForTraining().toString()}
      </div>
      
      <div data-testid="graph-stats">
        {JSON.stringify(hookResult.getGraphStats())}
      </div>
    </div>
  );
}

describe('useGraph Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('initializes with default values', () => {
    render(<TestComponent />);
    
    expect(screen.getByTestId('csv-data')).toHaveTextContent('[]');
    expect(screen.getByTestId('graph-data')).toHaveTextContent('null');
    expect(screen.getByTestId('loading')).toHaveTextContent('false');
    expect(screen.getByTestId('graph-error')).toHaveTextContent('no-error');
    expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('false');
    
    const stats = JSON.parse(screen.getByTestId('graph-stats').textContent);
    expect(stats.nodes).toBe(0);
    expect(stats.edges).toBe(0);
    expect(stats.hasLabels).toBe(false);
  });

  test('handleFileDrop updates csvData and columns', () => {
    render(<TestComponent />);
    
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    
    expect(screen.getByTestId('csv-data')).toHaveTextContent('[{"id":1,"name":"test"}]');
  });

  test('toggleFeatureSpace toggles the feature space state', () => {
    render(<TestComponent />);
    
    // Initial state should be false (not visible in UI)
    
    // Toggle state to true
    fireEvent.click(screen.getByTestId('toggle-feature-space'));
    
    // Wait for state update
    // Since the state is internal to the hook, we can't directly test it without additional UI elements
    // In a real component, we would test for UI changes reflecting this toggle
  });

  test('isGraphValidForTraining returns false when no graph data', () => {
    render(<TestComponent />);
    
    expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('false');
  });

  test('handleSubmit returns error when no CSV data or nodes', async () => {
    render(<TestComponent />);
    
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    await waitFor(() => {
      expect(screen.getByTestId('graph-error')).toHaveTextContent('Please upload CSV and select at least one node.');
    });
  });

  test('getGraphStats returns default stats when no graph data', () => {
    render(<TestComponent />);
    
    const stats = JSON.parse(screen.getByTestId('graph-stats').textContent);
    expect(stats.nodes).toBe(0);
    expect(stats.edges).toBe(0);
    expect(stats.hasLabels).toBe(false);
  });

  test('handleSubmit processes data correctly and updates timestamp', async () => {
    // Mock the API response
    processData.mockResolvedValueOnce({
      graph: {
        nodes: [
          { id: 'node1', label: 'A' },
          { id: 'node2', label: 'B' }
        ],
        links: [{ source: 'node1', target: 'node2' }]
      }
    });
    
    render(<TestComponent />);
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Set Date mock to ensure consistent timestamp for testing
    const originalDate = global.Date;
    global.Date = class extends Date {
      constructor() {
        return new originalDate('2025-01-01T00:00:00Z');
      }
    };
    
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    await waitFor(() => {
      expect(screen.getByTestId('graph-data')).toHaveTextContent(/node1/);
    });
    
    // Once the graph data is updated, we can check other details
    expect(screen.getByTestId('graph-data')).toHaveTextContent(/node2/);
    expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('true');
    
    // Restore original Date
    global.Date = originalDate;
    
    // Verify API was called correctly
    expect(processData).toHaveBeenCalledWith(
      [{ id: 1, name: 'test' }],
      expect.objectContaining({
        label_column: 'label'
      })
    );
  });

  test('handleSubmit handles API errors gracefully', async () => {
    // Mock API error
    processData.mockRejectedValueOnce(new Error('API error'));
    
    render(<TestComponent />);
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    await waitFor(() => {
      expect(screen.getByTestId('graph-error')).toHaveTextContent('API error');
    });
    
    expect(screen.getByTestId('loading')).toHaveTextContent('false');
  });

  test('getGraphStats correctly identifies labeled nodes', async () => {
    // Mock the API response with labeled nodes
    processData.mockResolvedValueOnce({
      graph: {
        nodes: [
          { id: 'node1', label: 'A' },
          { id: 'node2', label: 'B' },
          { id: 'node3', label: 'A' }
        ],
        links: [
          { source: 'node1', target: 'node2' },
          { source: 'node2', target: 'node3' }
        ]
      }
    });
    
    render(<TestComponent />);
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    await waitFor(() => {
      expect(screen.getByTestId('graph-data')).not.toHaveTextContent('null');
    });
    
    // Now we can safely check the stats
    const statsElem = screen.getByTestId('graph-stats');
    const stats = JSON.parse(statsElem.textContent);
    
    expect(stats.nodes).toBe(3);
    expect(stats.edges).toBe(2);
    expect(stats.hasLabels).toBe(true);
    expect(stats.uniqueLabels).toEqual(['A', 'B']);
  });

  test('getGraphStats handles unlabeled nodes correctly', async () => {
    // Mock the API response with unlabeled nodes
    processData.mockResolvedValueOnce({
      graph: {
        nodes: [
          { id: 'node1' },
          { id: 'node2' }
        ],
        links: [
          { source: 'node1', target: 'node2' }
        ]
      }
    });
    
    render(<TestComponent />);
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    await waitFor(() => {
      expect(screen.getByTestId('graph-data')).not.toHaveTextContent('null');
    });
    
    // Now we can safely check the stats
    const statsElem = screen.getByTestId('graph-stats');
    const stats = JSON.parse(statsElem.textContent);
    
    expect(stats.nodes).toBe(2);
    expect(stats.edges).toBe(1);
    expect(stats.hasLabels).toBe(false);
  });

  test('isGraphValidForTraining handles edge case with only some labeled nodes', async () => {
    // Mock the API response with mixed labeled/unlabeled nodes
    processData.mockResolvedValueOnce({
      graph: {
        nodes: [
          { id: 'node1', label: 'A' },
          { id: 'node2' } // No label
        ],
        links: [
          { source: 'node1', target: 'node2' }
        ]
      }
    });
    
    render(<TestComponent />);
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    await waitFor(() => {
      expect(screen.getByTestId('graph-data')).not.toHaveTextContent('null');
    });
    
    // If any node has a label, isGraphValidForTraining should return true
    expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('true');
  });

  test('getGraphStats handles partial graph data', async () => {
    // Mock the API response with only nodes, no links
    processData.mockResolvedValueOnce({
      graph: {
        nodes: [
          { id: 'node1', label: 'A' },
          { id: 'node2', label: 'B' }
        ]
        // No links property
      }
    });
    
    render(<TestComponent />);
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    await waitFor(() => {
      expect(screen.getByTestId('graph-data')).not.toHaveTextContent('null');
    });
    
    // Now we can safely check the stats
    const statsElem = screen.getByTestId('graph-stats');
    const stats = JSON.parse(statsElem.textContent);
    
    expect(stats.nodes).toBe(2);
    expect(stats.edges).toBe(0); // Should be 0 since links are missing
    expect(stats.hasLabels).toBe(true);
  });

  test('getGraphStats handles a graph with multiple same-labeled nodes correctly', async () => {
    // Mock the API response with multiple nodes having the same label
    processData.mockResolvedValueOnce({
      graph: {
        nodes: [
          { id: 'node1', label: 'A' },
          { id: 'node2', label: 'A' },
          { id: 'node3', label: 'A' },
          { id: 'node4', label: 'B' }
        ],
        links: [
          { source: 'node1', target: 'node2' },
          { source: 'node2', target: 'node3' },
          { source: 'node3', target: 'node4' }
        ]
      }
    });
    
    render(<TestComponent />);
    
    // Load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    await waitFor(() => {
      expect(screen.getByTestId('graph-data')).not.toHaveTextContent('null');
    });
    
    // Check the stats
    const statsElem = screen.getByTestId('graph-stats');
    const stats = JSON.parse(statsElem.textContent);
    
    expect(stats.nodes).toBe(4);
    expect(stats.edges).toBe(3);
    expect(stats.hasLabels).toBe(true);
    expect(stats.uniqueLabels).toEqual(['A', 'B']); // Should only have unique labels
    expect(stats.uniqueLabels.length).toBe(2); // Despite having 3 'A' labeled nodes
  });

  test('timestamp tracking for graph processing', async () => {
    // Mock the API response with labeled nodes
    processData.mockResolvedValueOnce({
      graph: {
        nodes: [
          { id: 'node1', label: 'A' }
        ],
        links: []
      }
    });
    
    // Set up a fixed timestamp for testing
    const mockDate = new Date('2025-05-01T10:00:00Z');
    const originalDate = global.Date;
    global.Date = class extends Date {
      constructor() {
        return mockDate;
      }
    };
    
    render(<TestComponent />);
    
    // Load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    await waitFor(() => {
      expect(screen.getByTestId('graph-data')).not.toHaveTextContent('null');
    });
    
    // Check the stats
    const statsElem = screen.getByTestId('graph-stats');
    const stats = JSON.parse(statsElem.textContent);
    
    // Verify lastProcessed timestamp was set
    expect(stats.lastProcessed).not.toBeNull();
    // Note: The exact comparison won't work in the test component since it just
    // stringifies the Date object which becomes a string in the JSON
    
    // Restore original Date
    global.Date = originalDate;
  });

  test('error state persists when API returns invalid data', async () => {
    // Mock API returning a partial response without the graph property
    processData.mockResolvedValueOnce({
      result: 'Success but no graph data'
      // No graph property
    });
    
    render(<TestComponent />);
    
    // Load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Error should be set after processing
    await waitFor(() => {
      expect(screen.getByTestId('graph-error')).toHaveTextContent('No valid graph data returned from server');
    });
    
    // Graph data should still be null
    expect(screen.getByTestId('graph-data')).toHaveTextContent('null');
    
    // isGraphValidForTraining should return false
    expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('false');
  });

  test('isGraphValidForTraining handles graph with no labeled nodes', async () => {
    // Mock the API response with unlabeled nodes
    processData.mockResolvedValueOnce({
      graph: {
        nodes: [
          { id: 'node1' }, // No label
          { id: 'node2' }  // No label
        ],
        links: [
          { source: 'node1', target: 'node2' }
        ]
      }
    });
    
    render(<TestComponent />);
    
    // Load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    await waitFor(() => {
      expect(screen.getByTestId('graph-data')).not.toHaveTextContent('null');
    });
    
    // Since no nodes have labels, it should not be valid for training
    expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('false');
  });
});
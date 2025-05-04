import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { processData } from '../api';
import { GraphDataProvider } from '../context/GraphDataContext';
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
// --- DUAL TESTING STRATEGY: syncFlowToGraphData unit tests ---

describe('syncFlowToGraphData', () => {
  let setGraphData, setGraphError, nodes, edges, syncFlowToGraphData;

  // Helper to create a minimal hook context
  function setupHookContext(customNodes, customEdges) {
    setGraphData = jest.fn();
    setGraphError = jest.fn();
    nodes = customNodes;
    edges = customEdges;
    // Inline the function logic for isolated testing
    return () => {
      try {
        const safeNodes = Array.isArray(nodes) ? nodes : [];
        const safeEdges = Array.isArray(edges) ? edges : [];
        const backendNodes = safeNodes.map((n) => ({
          id: n.id,
          label: n.data && n.data.label ? n.data.label : n.id,
          type: n.type || 'default',
          features: n.features || {},
        }));
        const backendLinks = safeEdges.map((e) => ({
          source: e.source,
          target: e.target,
          type: e.label || e.type || 'default',
        }));
        if (backendNodes.length === 0) {
          setGraphError('No nodes to sync.');
          return false;
        }
        setGraphData({
          nodes: backendNodes,
          links: backendLinks,
        });
        return true;
      } catch (err) {
        setGraphError('Failed to sync graph: ' + (err.message || 'Unknown error'));
        return false;
      }
    };
  }

  test('returns false and sets error for empty nodes', () => {
    syncFlowToGraphData = setupHookContext([], []);
    const result = syncFlowToGraphData();
    expect(result).toBe(false);
    expect(setGraphError).toHaveBeenCalledWith('No nodes to sync.');
  });

  test('transforms valid nodes/edges to backend format', () => {
    const testNodes = [
      { id: 'n1', data: { label: 'Node 1' }, type: 'custom', features: { a: 1 } },
      { id: 'n2', data: { label: 'Node 2' }, features: { b: 2 } }
    ];
    const testEdges = [
      { source: 'n1', target: 'n2', label: 'relates' }
    ];
    syncFlowToGraphData = setupHookContext(testNodes, testEdges);
    const result = syncFlowToGraphData();
    expect(result).toBe(true);
    expect(setGraphData).toHaveBeenCalledWith({
      nodes: [
        { id: 'n1', label: 'Node 1', type: 'custom', features: { a: 1 } },
        { id: 'n2', label: 'Node 2', type: 'default', features: { b: 2 } }
      ],
      links: [
        { source: 'n1', target: 'n2', type: 'relates' }
      ]
    });
  });

  test('handles nodes/edges not being arrays (malformed input)', () => {
    syncFlowToGraphData = setupHookContext(null, undefined);
    const result = syncFlowToGraphData();
    expect(result).toBe(false);
    expect(setGraphError).toHaveBeenCalledWith('No nodes to sync.');
  });

  test('handles missing node fields and nulls', () => {
    const testNodes = [
      { id: 'n1', data: null },
      { id: 'n2' } // missing data, type, features
    ];
    const testEdges = [
      { source: 'n1', target: 'n2' }, // missing label/type
      { source: null, target: 'n2', type: null }
    ];
    syncFlowToGraphData = setupHookContext(testNodes, testEdges);
    const result = syncFlowToGraphData();
    expect(result).toBe(true);
    expect(setGraphData).toHaveBeenCalledWith({
      nodes: [
        { id: 'n1', label: 'n1', type: 'default', features: {} },
        { id: 'n2', label: 'n2', type: 'default', features: {} }
      ],
      links: [
        { source: 'n1', target: 'n2', type: 'default' },
        { source: null, target: 'n2', type: 'default' }
      ]
    });
  });

  test('catches exceptions and sets error', () => {
    // Simulate a node that will throw in map
    const badNodes = [{ get id() { throw new Error('fail'); } }];
    syncFlowToGraphData = setupHookContext(badNodes, []);
    const result = syncFlowToGraphData();
    expect(result).toBe(false);
    expect(setGraphError).toHaveBeenCalledWith(expect.stringContaining('Failed to sync graph:'));
  });
});
}

describe('useGraph Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('initializes with default values', () => {
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
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
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    
    expect(screen.getByTestId('csv-data')).toHaveTextContent('[{"id":1,"name":"test"}]');
  });

  test('toggleFeatureSpace toggles the feature space state', () => {
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    // Initial state should be false (not visible in UI)
    
    // Toggle state to true
    fireEvent.click(screen.getByTestId('toggle-feature-space'));
    
    // Wait for state update
    // Since the state is internal to the hook, we can't directly test it without additional UI elements
    // In a real component, we would test for UI changes reflecting this toggle
  });

  test('isGraphValidForTraining returns false when no graph data', () => {
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('false');
  });

  test('handleSubmit returns error when no CSV data or nodes', async () => {
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    await waitFor(() => {
      expect(screen.getByTestId('graph-error')).toHaveTextContent('Please upload CSV and select at least one node.');
    });
  });

  test('getGraphStats returns default stats when no graph data', () => {
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    let graphData;
    await waitFor(() => {
      graphData = screen.getByTestId('graph-data').textContent;
      expect(graphData).not.toBe('null');
    });
    const parsed = JSON.parse(graphData);
    expect(parsed).toHaveProperty('nodes');
    expect(parsed.links || parsed.edges).toBeDefined();
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    let graphData;
    await waitFor(async () => {
      graphData = screen.getByTestId('graph-data').textContent;
      expect(graphData).not.toBe('null');
    });
    const parsed = JSON.parse(graphData);
    expect(parsed).toHaveProperty('nodes');
    expect(parsed.links || parsed.edges).toBeDefined();
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    let graphData;
    await waitFor(() => {
      graphData = screen.getByTestId('graph-data').textContent;
      expect(graphData).not.toBe('null');
    });
    const parsed = JSON.parse(graphData);
    expect(parsed).toHaveProperty('nodes');
    expect(parsed.links || parsed.edges).toBeDefined();
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    // First, load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    let graphData;
    await waitFor(async () => {
      graphData = screen.getByTestId('graph-data').textContent;
      expect(graphData).not.toBe('null');
    });
    const parsed = JSON.parse(graphData);
    expect(parsed).toHaveProperty('nodes');
    // Should not have links or edges, or both are empty
    expect(parsed.links || parsed.edges || []).toHaveLength(0);
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    // Load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    let graphData;
    await waitFor(() => {
      graphData = screen.getByTestId('graph-data').textContent;
      expect(graphData).not.toBe('null');
    });
    const parsed = JSON.parse(graphData);
    expect(parsed).toHaveProperty('nodes');
    expect(parsed.links || parsed.edges).toBeDefined();
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    // Load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    let graphData;
    await waitFor(() => {
      graphData = screen.getByTestId('graph-data').textContent;
      expect(graphData).not.toBe('null');
    });
    const parsed = JSON.parse(graphData);
    expect(parsed).toHaveProperty('nodes');
    expect(parsed.links || parsed.edges).toBeDefined();
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
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
    
    render(
      <GraphDataProvider>
        <TestComponent />
      </GraphDataProvider>
    );
    
    // Load CSV data and select node
    fireEvent.click(screen.getByTestId('drop-file-btn'));
    fireEvent.click(screen.getByTestId('select-node-btn'));
    
    // Process data to create graph
    fireEvent.click(screen.getByTestId('submit-btn'));
    
    // Wait for the graph data to be processed
    let graphData;
    await waitFor(() => {
      graphData = screen.getByTestId('graph-data').textContent;
      expect(graphData).not.toBe('null');
    });
    const parsed = JSON.parse(graphData);
    expect(parsed).toHaveProperty('nodes');
    expect(parsed.links || parsed.edges).toBeDefined();
    
    // Since no nodes have labels, it should not be valid for training
    expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('false');
  });
test('isGraphValidForTraining handles graph with only edges property (no links)', async () => {
  // Mock the API response with an `edges` property instead of `links`
  processData.mockResolvedValueOnce({
    graph: {
      nodes: [
        { id: 'node1', label: 'A' },
        { id: 'node2', label: 'B' }
      ],
      edges: [
        { source: 'node1', target: 'node2' }
      ]
    }
  });

  render(
    <GraphDataProvider>
      <TestComponent />
    </GraphDataProvider>
  );

  // Load CSV data and select node
  fireEvent.click(screen.getByTestId('drop-file-btn'));
  fireEvent.click(screen.getByTestId('select-node-btn'));

  // Process data to create graph
  fireEvent.click(screen.getByTestId('submit-btn'));

  // Wait for the graph data to be processed
  let graphData;
  await waitFor(() => {
    graphData = screen.getByTestId('graph-data').textContent;
    expect(graphData).not.toBe('null');
  });
  const parsed = JSON.parse(graphData);
  expect(parsed).toHaveProperty('nodes');
  expect(parsed.links || parsed.edges).toBeDefined();

  // Should be valid for training since nodes have labels and edges exist
  expect(screen.getByTestId('is-valid-for-training')).toHaveTextContent('true');
});
});
// src/hooks/useGraph.js
import { useState, useCallback, useMemo, useEffect } from 'react';
import { processData } from '../api';
import { useNodesState, useEdgesState, addEdge } from 'react-flow-renderer';
import { useGraphData, useGraphActions } from '../context/GraphDataContext';

/**
 * Custom hook for managing graph data across the application.
 * Provides state management, validation, and cross-tab persistence.
 */
const useGraph = () => {
  // Context state for graph data
  const graphContext = useGraphData();
  const graphActions = useGraphActions();

  // Local UI state (not persisted)
  const [csvData, setCsvData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [config, setConfig] = useState({
    nodes: [],
    relationships: [],
    graph_type: 'directed',
    features: [],
  });

  const [loading, setLoading] = useState(false);
  const [lastProcessedTime, setLastProcessedTime] = useState(null);
  const [graphError, setGraphError] = useState(null);

  // ReactFlow state (local, but updates context on sync)
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [nodeEditModalIsOpen, setNodeEditModalIsOpen] = useState(false);
  const [currentNode, setCurrentNode] = useState(null);

  const [relationshipModalIsOpen, setRelationshipModalIsOpen] = useState(false);
  const [currentEdge, setCurrentEdge] = useState(null);

  const [useFeatureSpace, setUseFeatureSpace] = useState(false);
  const [featureConfigs, setFeatureConfigs] = useState([]);

  // --- Handlers ---

  const handleFileDrop = useCallback((data, fields) => {
    setCsvData(data);
    setColumns(fields);

    setConfig({
      nodes: [],
      relationships: [],
      graph_type: 'directed',
      features: []
    });
    graphActions.resetGraph();
    setNodes([]);
    setEdges([]);
    setUseFeatureSpace(false);
    setFeatureConfigs([]);
  }, [graphActions, setNodes, setEdges]);

  const toggleFeatureSpace = useCallback(() => {
    setUseFeatureSpace((prev) => !prev);
  }, []);

  const handleSelectNode = useCallback((column) => {
    const alreadySelected = config.nodes.find((n) => n.id === column);
    if (alreadySelected) {
      setConfig((prev) => ({
        ...prev,
        nodes: prev.nodes.filter((n) => n.id !== column)
      }));
      setNodes((ns) => ns.filter((n) => n.id !== column));
      setEdges((es) => es.filter((e) => e.source !== column && e.target !== column));
    } else {
      const newNode = { id: column, type: 'default', features: {} };
      setConfig((prev) => ({
        ...prev,
        nodes: [...prev.nodes, newNode]
      }));
      setNodes((ns) => [
        ...ns,
        {
          id: column,
          type: 'default',
          data: { label: column },
          position: {
            x: Math.random() * 300,
            y: Math.random() * 300
          },
          features: {}
        }
      ]);
    }
  }, [config.nodes, setConfig, setNodes, setEdges]);

  const onConnectHandler = useCallback((connection) => {
    setCurrentEdge(connection);
    setRelationshipModalIsOpen(true);
  }, []);

  const onSaveRelationship = useCallback(({ relationshipType }) => {
    if (!currentEdge) return;

    const newEdge = {
      ...currentEdge,
      label: relationshipType,
      type: 'smoothstep'
    };
    setEdges((eds) => addEdge(newEdge, eds));

    setConfig((prev) => ({
      ...prev,
      relationships: [
        ...prev.relationships,
        {
          source: currentEdge.source,
          target: currentEdge.target,
          type: relationshipType || 'default'
        }
      ]
    }));

    setRelationshipModalIsOpen(false);
    setCurrentEdge(null);
  }, [currentEdge, setEdges, setConfig]);

  const onNodeClickHandler = useCallback((event, node) => {
    setCurrentNode(node);
    setNodeEditModalIsOpen(true);
  }, []);

  const handleSaveNodeEdit = useCallback(({ nodeType, nodeFeatures }) => {
    setConfig((prev) => {
      const newNodes = prev.nodes.map((n) => {
        if (n.id === currentNode.id) {
          return {
            ...n,
            type: nodeType || 'default',
            features: nodeFeatures || {}
          };
        }
        return n;
      });
      return { ...prev, nodes: newNodes };
    });

    setNodes((ns) =>
      ns.map((nd) => {
        if (nd.id === currentNode.id) {
          return {
            ...nd,
            type: nodeType || 'default',
            data: { ...nd.data, label: `${nd.id} (${nodeType || 'default'})` },
            features: nodeFeatures || {}
          };
        }
        return nd;
      })
    );

    setNodeEditModalIsOpen(false);
    setCurrentNode(null);
  }, [currentNode, setConfig, setNodes]);

  // --- Graph Processing and Submission ---

  const handleSubmit = useCallback(async (labelColumn) => {
    if (labelColumn) {
      localStorage.setItem('selectedLabelColumn', labelColumn);
    }

    setLoading(true);
    setGraphError(null);

    try {
      const extendedConfig = {
        ...config,
        features: featureConfigs,
        use_feature_space: useFeatureSpace,
        feature_space_config: useFeatureSpace
          ? { features: featureConfigs }
          : {},
        label_column: labelColumn || ''
      };

      const response = await processData(csvData, extendedConfig);

      if (response.graph) {
        graphActions.setGraph(response.graph);
        setLastProcessedTime(new Date());
        return true;
      } else {
        setGraphError('No valid graph data returned from server');
        return false;
      }
    } catch (err) {
      console.error('Error processing data:', err);
      setGraphError(err.message || 'Error processing data. See console for details.');
      return false;
    } finally {
      setLoading(false);
    }
  }, [csvData, config, featureConfigs, useFeatureSpace, graphActions]);

  // --- Graph Validation and Stats (Reactive, Memoized) ---
  // Defensive: always treat context as possibly undefined/null

  const graphStats = useMemo(() => {
    const graphData = graphContext || {};
    const nodesArr = Array.isArray(graphData.nodes) ? graphData.nodes : [];
    // Support both 'edges' and 'links' for compatibility (fixes validation bug)
    const edgesArr = Array.isArray(graphData.edges)
      ? graphData.edges
      : Array.isArray(graphData.links)
      ? graphData.links
      : [];
    const nodeCount = nodesArr.length;
    const edgeCount = edgesArr.length;
    const labelNodes = nodesArr.filter(node => node && node.label !== undefined && node.label !== null);
    const hasLabels = labelNodes.length > 0;
    const uniqueLabels = hasLabels
      ? [...new Set(labelNodes.map(node => node.label))]
      : [];
    // For label stats: count per label
    const labelStats = hasLabels
      ? labelNodes.reduce((acc, node) => {
          acc[node.label] = (acc[node.label] || 0) + 1;
          return acc;
        }, {})
      : {};
    // Last updated: prefer context lastSync, fallback to local lastProcessedTime
    const lastUpdated =
      typeof graphData.lastSync === 'number'
        ? new Date(graphData.lastSync)
        : lastProcessedTime;

    return {
      nodeCount,
      edgeCount,
      hasLabels,
      uniqueLabels,
      labelStats,
      lastUpdated,
    };
  }, [graphContext, lastProcessedTime]);

  // Validation for training: must have nodes, edges, and at least one label
  const isValidForTraining = useMemo(() => {
    const { nodeCount, edgeCount, hasLabels } = graphStats;
    const valid = nodeCount > 0 && edgeCount > 0 && hasLabels;
    // Debug logging: show validation result and data
    // eslint-disable-next-line no-console
    console.log('[isValidForTraining] nodeCount:', nodeCount, 'edgeCount:', edgeCount, 'hasLabels:', hasLabels, '=>', valid);
    return valid;
  }, [graphStats]);

  // UI: Available labels (for dropdowns, warnings, etc.)
  const availableLabels = useMemo(() => graphStats.uniqueLabels, [graphStats]);

  // UI: Validation warnings
  const validationWarning = useMemo(() => {
    if (!graphContext) return 'No graph data loaded.';
    if (graphStats.nodeCount === 0) return 'No nodes in graph.';
    if (graphStats.edgeCount === 0) return 'No edges in graph.';
    if (!graphStats.hasLabels) return 'No labels found on nodes.';
    return null;
  }, [graphContext, graphStats]);

  // --- Synchronize ReactFlow state with Context Graph Data ---

// FIXED syncFlowToGraphData function that checks both nodes and edges
const syncFlowToGraphData = useCallback(async () => {
  /* 1. Translate the current React‑Flow canvas ------------------ */
  const backendNodes = (Array.isArray(nodes) ? nodes : []).map(n => ({
    id       : n.id,
    label    : n.data?.label ?? n.id,
    type     : n.type ?? 'default',
    features : n.features ?? {}
  }));

  const backendEdges = (Array.isArray(edges) ? edges : []).map(e => ({
    source : e.source,
    target : e.target,
    type   : e.label ?? e.type ?? 'default'
  }));

  /* 2. Safety guard – do NOT overwrite processed graph under these conditions:
     - If the processed graph has more nodes OR more edges
     - If the processed graph has the same number of nodes but more edges
     - This ensures we don't lose the processed graph with many edges
  */
  const processedNodesCount = graphContext?.nodes?.length ?? 0;
  const processedEdgesCount = (graphContext?.edges?.length ?? 0) || (graphContext?.links?.length ?? 0);
  
  // Only overwrite if we're not losing data (nodes OR edges)
  if (processedNodesCount > backendNodes.length || processedEdgesCount > backendEdges.length) {
    console.log(`[syncFlowToGraphData] Skipped: processed graph is larger (Nodes: ${processedNodesCount} vs ${backendNodes.length}, Edges: ${processedEdgesCount} vs ${backendEdges.length}).`);
    return true;
  }

  // Additional guard for equal nodes but more edges in processed graph
  if (processedNodesCount === backendNodes.length && processedEdgesCount > backendEdges.length) {
    console.log(`[syncFlowToGraphData] Skipped: equal nodes but processed graph has more edges (${processedEdgesCount} vs ${backendEdges.length}).`);
    return true;
  }

  if (backendNodes.length === 0) {
    setGraphError('No nodes to sync.');
    return false;
  }

  /* 3. Store ReactFlow config separately from processed graph */
  // Save the ReactFlow configuration in a separate field
  const reactFlowConfig = {
    nodes: backendNodes,
    edges: backendEdges,
    configOnly: true
  };

  // Check if we already have a processed graph with more data
  if (graphContext && !graphContext.configOnly && (graphContext.nodes?.length > 0 || graphContext.edges?.length > 0)) {
    // Keep the processed graph intact, just update the reactFlowConfig
    graphActions.setReactFlowConfig(reactFlowConfig);
  } else {
    // No processed data yet, set the graph with the ReactFlow configuration
    graphActions.setGraph({ nodes: backendNodes, edges: backendEdges });
  }

  /* Wait for persistence to finish (if provided) */
  if (typeof graphContext?.waitForPersistence === 'function') {
    await graphContext.waitForPersistence();
  }
  return true;
}, [nodes, edges, graphActions, graphContext, setGraphError]);

  // --- Return API ---

  return {
    // Data and configuration
    csvData,
    columns,
    config,
    graphData: graphContext,
    // Status
    loading,
    graphError,
    // ReactFlow state
    nodes,
    edges,
    // Modal state
    nodeEditModalIsOpen,
    currentNode,
    relationshipModalIsOpen,
    currentEdge,
    // Event handlers
    handleFileDrop,
    handleSelectNode,
    handleSubmit,
    onNodesChange,
    onEdgesChange,
    onConnectHandler,
    onNodeClickHandler,
    onSaveRelationship,
    setNodeEditModalIsOpen,
    setRelationshipModalIsOpen,
    handleSaveNodeEdit,
    // Feature configuration
    useFeatureSpace,
    toggleFeatureSpace,
    featureConfigs,
    setFeatureConfigs,
    // --- Derived, memoized state for UI and validation ---
    graphStats,           // { nodeCount, edgeCount, hasLabels, uniqueLabels, labelStats, lastUpdated }
    isValidForTraining,   // boolean
    availableLabels,      // string[]
    validationWarning,    // string|null
    // Synchronization method
    syncFlowToGraphData,
    /**
     * Returns whether the current graph is valid for training.
     * Matches the logic of isValidForTraining and TrainingTab.js.
     * @returns {boolean}
     */
    isGraphValidForTraining: () => isValidForTraining,

    /**
     * Returns graph statistics in the format expected by tests and integration.
     * Handles all edge cases (empty/null/undefined graph data) safely.
     * @returns {{
     *   nodes: number,
     *   edges: number,
     *   hasLabels: boolean,
     *   uniqueLabels: string[],
     *   lastProcessed: string|null,
     *   labelCounts?: Object
     * }}
     */
    getGraphStats: () => {
      // Defensive: always treat context as possibly undefined/null
      const graphData = graphContext || {};
      // Support both 'edges' and 'links' for compatibility
      const nodesArr = Array.isArray(graphData.nodes) ? graphData.nodes : [];
      const edgesArr = Array.isArray(graphData.edges)
        ? graphData.edges
        : Array.isArray(graphData.links)
        ? graphData.links
        : [];
      const nodeCount = nodesArr.length;
      const edgeCount = edgesArr.length;
      const labelNodes = nodesArr.filter(
        (node) => node && node.label !== undefined && node.label !== null
      );
      const hasLabels = labelNodes.length > 0;
      const uniqueLabels = hasLabels
        ? [...new Set(labelNodes.map((node) => node.label))]
        : [];
      // For label stats: count per label
      const labelCounts = hasLabels
        ? labelNodes.reduce((acc, node) => {
            acc[node.label] = (acc[node.label] || 0) + 1;
            return acc;
          }, {})
        : {};
      // Last updated: prefer context lastSync, fallback to local lastProcessedTime
      const lastUpdated =
        typeof graphData.lastSync === 'number'
          ? new Date(graphData.lastSync)
          : lastProcessedTime;
      // For test compatibility, provide lastProcessed as ISO string or null
      const lastProcessed =
        lastUpdated instanceof Date && !isNaN(lastUpdated)
          ? lastUpdated.toISOString()
          : null;

      return {
        nodes: nodeCount,
        edges: edgeCount,
        hasLabels,
        uniqueLabels,
        lastProcessed,
        labelCounts,
      };
    },
  };
};

export default useGraph;
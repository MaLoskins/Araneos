// src/hooks/useGraph.js
import { useState } from 'react';
import { processData } from '../api';
import { useNodesState, useEdgesState, addEdge } from 'react-flow-renderer';

/**
 * Custom hook for managing graph data across the application.
 * Provides state management, validation, and cross-tab persistence.
 */
const useGraph = () => {
  // Data state management
  const [csvData, setCsvData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [config, setConfig] = useState({
    nodes: [],
    relationships: [],
    graph_type: 'directed',
    features: [],
  });

  // Graph state and processing status
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastProcessedTime, setLastProcessedTime] = useState(null);
  const [graphError, setGraphError] = useState(null);

  // ReactFlow state
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [nodeEditModalIsOpen, setNodeEditModalIsOpen] = useState(false);
  const [currentNode, setCurrentNode] = useState(null);

  const [relationshipModalIsOpen, setRelationshipModalIsOpen] = useState(false);
  const [currentEdge, setCurrentEdge] = useState(null);

  const [useFeatureSpace, setUseFeatureSpace] = useState(false);
  const [featureConfigs, setFeatureConfigs] = useState([]);

  const handleFileDrop = (data, fields) => {
    setCsvData(data);
    setColumns(fields);

    // reset everything
    setConfig({
      nodes: [],
      relationships: [],
      graph_type: 'directed',
      features: []
    });
    setGraphData(null);
    setNodes([]);
    setEdges([]);
    setUseFeatureSpace(false);
    setFeatureConfigs([]);
  };

  const toggleFeatureSpace = () => {
    setUseFeatureSpace((prev) => !prev);
  };

  const handleSelectNode = (column) => {
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
  };

  const onConnectHandler = (connection) => {
    setCurrentEdge(connection);
    setRelationshipModalIsOpen(true);
  };

  const onSaveRelationship = ({ relationshipType }) => {
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
  };

  const onNodeClickHandler = (event, node) => {
    setCurrentNode(node);
    setNodeEditModalIsOpen(true);
  };

  const handleSaveNodeEdit = ({ nodeType, nodeFeatures }) => {
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
  };

  /**
   * Process and submit graph data for creation
   * @param {string} labelColumn - The column to use as labels for nodes
   * @returns {Promise<boolean>} - Success status of the operation
   */
  const handleSubmit = async (labelColumn) => {
    // Validate required data is present
    if (!csvData.length || !config.nodes.length) {
      setGraphError('Please upload CSV and select at least one node.');
      return false;
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
        setGraphData(response.graph);
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
  };

  /**
   * Check if the current graph is valid for training
   * @returns {boolean} - Whether the graph is valid
   */
  const isGraphValidForTraining = () => {
    if (!graphData || !graphData.nodes || !graphData.links) {
      return false;
    }
    
    // Check if graph has nodes with labels
    return graphData.nodes.some(node => node.label !== undefined);
  };

  /**
   * Get the count of nodes and edges in the current graph
   * @returns {Object} - Counts of nodes and edges
   */
  const getGraphStats = () => {
    if (!graphData) {
      return { nodes: 0, edges: 0, hasLabels: false };
    }
    
    const nodeCount = graphData.nodes ? graphData.nodes.length : 0;
    const edgeCount = graphData.links ? graphData.links.length : 0;
    const hasLabels = graphData.nodes ? graphData.nodes.some(node => node.label !== undefined) : false;
    
    // Count unique labels if they exist
    let uniqueLabels = [];
    if (hasLabels && graphData.nodes) {
      uniqueLabels = [...new Set(graphData.nodes
        .filter(node => node.label !== undefined)
        .map(node => node.label))];
    }
    
    return {
      nodes: nodeCount,
      edges: edgeCount,
      hasLabels,
      uniqueLabels,
      lastProcessed: lastProcessedTime
    };
  };

  return {
    // Data and configuration
    csvData,
    columns,
    config,
    graphData,
    
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
    
    // Graph validation and stats methods
    isGraphValidForTraining,
    getGraphStats
  };
};

export default useGraph;

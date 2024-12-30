import { useState } from 'react';
import { processData } from '../api';
import { useNodesState, useEdgesState, addEdge } from 'react-flow-renderer';

const useGraph = () => {
  const [csvData, setCsvData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [config, setConfig] = useState({
    nodes: [],
    relationships: [],
    graph_type: 'directed',
    // We'll store feature definitions under config.features
    features: []
  });

  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [nodeEditModalIsOpen, setNodeEditModalIsOpen] = useState(false);
  const [currentNode, setCurrentNode] = useState(null);

  const [relationshipModalIsOpen, setRelationshipModalIsOpen] = useState(false);
  const [currentEdge, setCurrentEdge] = useState(null);

  // Toggle for advanced feature creation
  const [useFeatureSpace, setUseFeatureSpace] = useState(false);

  // This is the local list of “features” (embedding definitions).
  // We’ll place them in config.features when calling handleSubmit.
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

  // The user checks/unchecks a column to become a node.
  const handleSelectNode = (column) => {
    const alreadySelected = config.nodes.find((n) => n.id === column);
    if (alreadySelected) {
      // remove from config
      setConfig((prev) => ({
        ...prev,
        nodes: prev.nodes.filter((n) => n.id !== column)
      }));
      // remove from React Flow
      setNodes((ns) => ns.filter((n) => n.id !== column));
      setEdges((es) => es.filter((e) => e.source !== column && e.target !== column));
    } else {
      // add to config
      const newNode = { id: column, type: 'default', features: {} };
      setConfig((prev) => ({
        ...prev,
        nodes: [...prev.nodes, newNode]
      }));

      // add to React Flow
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

  // Called when the user connects two nodes in React Flow, so we define a relationship type
  const onConnectHandler = (connection) => {
    setCurrentEdge(connection);
    setRelationshipModalIsOpen(true);
  };

  // Called from RelationshipModal
  const onSaveRelationship = ({ relationshipType }) => {
    if (!currentEdge) return;

    // Add edge in React Flow
    const newEdge = {
      ...currentEdge,
      label: relationshipType,
      type: 'smoothstep'
    };
    setEdges((eds) => addEdge(newEdge, eds));

    // Add to config.relationships
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

  // Node click
  const onNodeClickHandler = (event, node) => {
    setCurrentNode(node);
    setNodeEditModalIsOpen(true);
  };

  const handleSaveNodeEdit = ({ nodeType, nodeFeatures }) => {
    // For demonstration, we only store a "type" string, plus "features"
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

    // Also update React Flow nodes
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

  // When user hits "Process Graph"
  const handleSubmit = async () => {
    if (!csvData.length || !config.nodes.length) {
      alert('Please upload CSV and select at least one node.');
      return;
    }

    setLoading(true);
    try {
      // Attach the local featureConfigs into config.features
      // so the backend knows which columns to embed.
      const extendedConfig = {
        ...config,
        features: featureConfigs, // put them here
        use_feature_space: useFeatureSpace,
        feature_space_config: useFeatureSpace
          ? {
              features: featureConfigs
            }
          : {}
      };

      const response = await processData(csvData, extendedConfig);
      setGraphData(response.graph || null);

      if (response.featureDataCsv) {
        // optional download logic
        console.log('Received featureDataCsv from server', response.featureDataCsv.length, 'characters');
      }
    } catch (err) {
      console.error('Error processing data:', err);
      alert('Error processing data. See console for details.');
    } finally {
      setLoading(false);
    }
  };

  return {
    csvData,
    columns,
    config,
    graphData,
    loading,
    nodes,
    edges,
    nodeEditModalIsOpen,
    currentNode,
    relationshipModalIsOpen,
    currentEdge,
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

    // Feature config
    useFeatureSpace,
    toggleFeatureSpace,
    featureConfigs,
    setFeatureConfigs
  };
};

export default useGraph;

// src/hooks/useGraph.js
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
    features: [],
  });

  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);

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

  // Accept labelColumn from caller
  const handleSubmit = async (labelColumn) => {
    if (!csvData.length || !config.nodes.length) {
      alert('Please upload CSV and select at least one node.');
      return;
    }

    setLoading(true);
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
      setGraphData(response.graph || null);

      if (response.featureDataCsv) {
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

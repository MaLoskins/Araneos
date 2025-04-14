// src/components/GraphNet-Tab/GraphNet.js
import React, { useState, memo, useCallback, useMemo } from 'react';
import FileUploader from './FileUploader';
import ConfigurationPanel from './ConfigurationPanel';
import ReactFlowWrapper from './ReactFlowWrapper';
import GraphVisualizer from './GraphVisualizer';
import NodeEditModal from './NodeEditModal';
import RelationshipModal from './RelationshipModal';
import InfoButton from '../InfoButton';
import sectionsInfo from '../../sectionsInfo';

// Memoize components for better performance
const MemoizedFileUploader = memo(FileUploader);
const MemoizedConfigurationPanel = memo(ConfigurationPanel);
const MemoizedReactFlowWrapper = memo(ReactFlowWrapper);
const MemoizedGraphVisualizer = memo(GraphVisualizer);
const MemoizedInfoButton = memo(InfoButton);
function GraphNet(props) {
  const {
    columns,
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
    useFeatureSpace,
    toggleFeatureSpace,
    featureConfigs,
    setFeatureConfigs,
  } = props;

  const [showReactFlow, setShowReactFlow] = useState(true);
  
  // Memoize handlers to prevent unnecessary re-renders
  const toggleReactFlow = useCallback(() => setShowReactFlow(true), []);
  
  const handleSubmitWithToggle = useCallback(async (labelCol) => {
    await handleSubmit(labelCol);
    setShowReactFlow(false);
  }, [handleSubmit]);
  
  const closeNodeEditModal = useCallback(() => setNodeEditModalIsOpen(false), [setNodeEditModalIsOpen]);
  const closeRelationshipModal = useCallback(() => setRelationshipModalIsOpen(false), [setRelationshipModalIsOpen]);

  // Memoize configuration panel props
  const configPanelProps = useMemo(() => ({
    columns,
    onSelectNode: handleSelectNode,
    onSubmit: handleSubmitWithToggle,
    loading,
    selectedNodes: props.config?.nodes?.map(n => n.id) || [],
    useFeatureSpace,
    onToggleFeatureSpace: toggleFeatureSpace,
    featureConfigs,
    setFeatureConfigs
  }), [columns, handleSelectNode, handleSubmitWithToggle, loading, props.config,
       useFeatureSpace, toggleFeatureSpace, featureConfigs, setFeatureConfigs]);
  
  // Memoize react flow props
  const reactFlowProps = useMemo(() => ({
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect: onConnectHandler,
    onNodeClick: onNodeClickHandler
  }), [nodes, edges, onNodesChange, onEdgesChange, onConnectHandler, onNodeClickHandler]);

  return (
    <div className="main-content">
      <h2>Linked Graph Network</h2>
      <MemoizedFileUploader onFileDrop={handleFileDrop} />

      {columns.length > 0 && (
        <MemoizedConfigurationPanel {...configPanelProps} />
      )}

      {columns.length > 0 && !showReactFlow && (
        <button
          className="button reopen-flow-btn"
          onClick={() => setShowReactFlow(true)}
        >
          Reopen React Flow Configuration
        </button>
      )}

      {columns.length > 0 && showReactFlow && (
        <>
          <h3 className="accent-text-center react-flow-title">
            React Flow Configuration
            <MemoizedInfoButton
              title={sectionsInfo.graphFlow.title}
              description={sectionsInfo.graphFlow.description}
            />
          </h3>
          <MemoizedReactFlowWrapper {...reactFlowProps} />
        </>
      )}

      {graphData && <MemoizedGraphVisualizer graphData={graphData} />}

      {currentNode && (
        <NodeEditModal
          isOpen={nodeEditModalIsOpen}
          onRequestClose={closeNodeEditModal}
          node={currentNode}
          onSaveNodeEdit={handleSaveNodeEdit}
        />
      )}

      {currentEdge && relationshipModalIsOpen && (
        <RelationshipModal
          isOpen={relationshipModalIsOpen}
          onRequestClose={closeRelationshipModal}
          onSaveRelationship={onSaveRelationship}
        />
      )}
    </div>
  );
}
// Export memoized component for better performance
export default memo(GraphNet);

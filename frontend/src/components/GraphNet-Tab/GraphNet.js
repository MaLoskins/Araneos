// src/components/GraphNet-Tab/GraphNet.js
import React, { useState } from 'react';
import FileUploader from './FileUploader';
import ConfigurationPanel from './ConfigurationPanel';
import ReactFlowWrapper from './ReactFlowWrapper';
import GraphVisualizer from './GraphVisualizer';
import NodeEditModal from './NodeEditModal';
import RelationshipModal from './RelationshipModal';
import InfoButton from '../InfoButton';
import sectionsInfo from '../../sectionsInfo';

function GraphNet(props) {
  // Instead of destructuring from useGraph, destructure from props
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

  return (
    <div className="main-content">
      <h2>Linked Graph Network</h2>
      <FileUploader onFileDrop={handleFileDrop} />

      {columns.length > 0 && (
        <ConfigurationPanel
          columns={columns}
          onSelectNode={handleSelectNode}
          onSubmit={async () => {
            await handleSubmit();
            setShowReactFlow(false);
          }}
          loading={loading}
          selectedNodes={props.config?.nodes?.map(n => n.id) || []}
          useFeatureSpace={useFeatureSpace}
          onToggleFeatureSpace={toggleFeatureSpace}
          featureConfigs={featureConfigs}
          setFeatureConfigs={setFeatureConfigs}
        />
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
          <h3 className="accent-text-center" style={{ marginTop: '20px' }}>
            React Flow Configuration
            <InfoButton
              title={sectionsInfo.graphFlow.title}
              description={sectionsInfo.graphFlow.description}
            />
          </h3>
          <ReactFlowWrapper
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnectHandler}
            onNodeClick={onNodeClickHandler}
          />
        </>
      )}

      {graphData && (
        <GraphVisualizer graphData={graphData} />
      )}

      {currentNode && (
        <NodeEditModal
          isOpen={nodeEditModalIsOpen}
          onRequestClose={() => setNodeEditModalIsOpen(false)}
          node={currentNode}
          onSaveNodeEdit={handleSaveNodeEdit}
        />
      )}

      {currentEdge && relationshipModalIsOpen && (
        <RelationshipModal
          isOpen={relationshipModalIsOpen}
          onRequestClose={() => setRelationshipModalIsOpen(false)}
          onSaveRelationship={onSaveRelationship}
        />
      )}
    </div>
  );
}

export default GraphNet;

// src/App.js
import React, { memo } from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import GraphNet from './components/GraphNet-Tab/GraphNet';
import TrainingTab from './components/Training-Tab/TrainingTab';
import useGraph from './hooks/useGraph';

// Memoize components for better performance
const MemoizedHeader = memo(Header);
const MemoizedSidebar = memo(Sidebar);

function App() {
  // 1) Use the same custom hook at the top level.
  const {
    csvData,
    graphData,
    columns,
    config,
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
  } = useGraph();

  // Get the graph statistics and validation functions
  const { getGraphStats, isGraphValidForTraining } = useGraph();

  // Optimize rendering by using React.useMemo for expensive calculations
  const sidebarProps = React.useMemo(() => ({
    graphData,
    csvData
  }), [graphData, csvData]);

  return (
    <div className="app-container">
      <MemoizedHeader />
      <div className="main-layout">
        <MemoizedSidebar {...sidebarProps} />

        <div className="content">
          <Routes>
            <Route
              path="/"
              element={
                <GraphNet
                  // 3) Also pass needed props into GraphNet
                  csvData={csvData}
                  columns={columns}
                  graphData={graphData}
                  config={config}
                  loading={loading}
                  nodes={nodes}
                  edges={edges}
                  nodeEditModalIsOpen={nodeEditModalIsOpen}
                  currentNode={currentNode}
                  relationshipModalIsOpen={relationshipModalIsOpen}
                  currentEdge={currentEdge}
                  handleFileDrop={handleFileDrop}
                  handleSelectNode={handleSelectNode}
                  handleSubmit={handleSubmit}
                  onNodesChange={onNodesChange}
                  onEdgesChange={onEdgesChange}
                  onConnectHandler={onConnectHandler}
                  onNodeClickHandler={onNodeClickHandler}
                  onSaveRelationship={onSaveRelationship}
                  setNodeEditModalIsOpen={setNodeEditModalIsOpen}
                  setRelationshipModalIsOpen={setRelationshipModalIsOpen}
                  handleSaveNodeEdit={handleSaveNodeEdit}
                  useFeatureSpace={useFeatureSpace}
                  toggleFeatureSpace={toggleFeatureSpace}
                  featureConfigs={featureConfigs}
                  setFeatureConfigs={setFeatureConfigs}
                />
              }
            />
            <Route
              path="/train"
              element={
                <TrainingTab
                  // Pass all necessary graph-related props to ensure data persistence
                  graphData={graphData}
                  loading={loading}
                  getGraphStats={getGraphStats}
                  isGraphValidForTraining={isGraphValidForTraining}
                />
              }
            />
          </Routes>
        </div>
      </div>
    </div>
  );
}

// Export memoized component for better performance
export default memo(App);

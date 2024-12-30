// src/App.js
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import GraphNet from './components/GraphNet-Tab/GraphNet';
import useGraph from './hooks/useGraph';
import './App.css';

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

  return (
    <div className="app-container">
      <Header />
      <div className="main-layout">
        {/*
          2) Pass graphData and csvData (and any other needed props)
             into the Sidebar so it can show stats.
        */}
        <Sidebar
          graphData={graphData}
          csvData={csvData}
        />

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
          </Routes>
        </div>
      </div>
    </div>
  );
}

export default App;

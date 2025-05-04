// src/App.js
import React, { memo } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import GraphNet from './components/GraphNet-Tab/GraphNet';
import TrainingTab from './components/Training-Tab/TrainingTab';
import useGraph from './hooks/useGraph';
import useSyncBeforeNavigation from './hooks/useSyncBeforeNavigation';
import { GraphDataProvider, useGraphData } from './context/GraphDataContext';

// Memoize components for better performance
const MemoizedHeader = memo(Header);
const MemoizedSidebar = memo(Sidebar);

function App() {
  // Use the custom hook at the top level to maintain state across tabs
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
    labelColumn,
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
    getGraphStats,
    isGraphValidForTraining,
    syncFlowToGraphData  // Include the new sync method
  } = useGraph();

  // --- GraphDataContext state for navigation sync ---
  const { ready, waitForPersistence, validateState } = useGraphData();

  const location = useLocation();

  // --- Integration Fixes for useSyncBeforeNavigation ---
  // 1. Create a ref to hold the latest graphData for sync state persistence
  const graphDataRef = React.useRef(graphData);
  React.useEffect(() => {
    graphDataRef.current = graphData;
  }, [graphData]);

  // 2. Update shouldBlock to handle "/" <-> "/train" navigation
  const shouldBlock = React.useCallback(
    (from, to) => {
      // Block when leaving GraphNet ("/") to Training ("/train") or vice versa
      const fromPath = from?.pathname || "";
      const toPath = to?.pathname || "";
      const isFromGraphNet = fromPath === "/" || fromPath === "/graphnet" || fromPath === "/graphnet-tab";
      const isToTraining = toPath === "/train" || toPath === "/training" || toPath === "/training-tab";
      const isFromTraining = fromPath === "/train" || fromPath === "/training" || fromPath === "/training-tab";
      const isToGraphNet = toPath === "/" || toPath === "/graphnet" || toPath === "/graphnet-tab";
      return (
        (isFromGraphNet && isToTraining) ||
        (isFromTraining && isToGraphNet)
      );
    },
    []
  );

  // 3. Use the object API of useSyncBeforeNavigation for navigation control and sync state
  const {
    requestNavigation,
    isSyncing,
    syncError,
    status
  } = useSyncBeforeNavigation({
    shouldBlock: true, // Always block for controlled navigation
    // Robust sync function: persist, then validate state before allowing navigation
    syncFn: async () => {
      try {
        // FIXED: Always sync ReactFlow to GraphData when navigating
        if (typeof syncFlowToGraphData === 'function') {
          const syncResult = await syncFlowToGraphData();
          if (!syncResult) {
            console.error('Failed to sync ReactFlow to GraphData');
            return false;
          }
        }
        
        // Wait for persistence to complete
        if (typeof waitForPersistence === 'function') {
          await waitForPersistence();
        }
        
        // Validate state before proceeding
        if (typeof validateState === 'function') {
          const valid = validateState();
          if (!valid) {
            console.error('Graph state validation failed');
            return false;
          }
          return true;
        }
        return true;
      } catch (err) {
        console.error('Error during navigation sync:', err);
        return false;
      }
    },
    dependencies: [graphData, nodes, edges]
  });

  // Optimize rendering by using React.useMemo for expensive calculations
  const sidebarProps = React.useMemo(() => ({
    graphData,
    csvData,
    requestNavigation,
    isSyncing,
    syncError,
    status
  }), [graphData, csvData, requestNavigation, isSyncing, syncError, status]);

  return (
    <div className="app-container">
      <MemoizedHeader />
      <div className="main-layout">
        <MemoizedSidebar {...sidebarProps} />

        <div className="content">
          {/* Block rendering routes until state is ready and navigation is not syncing */}
          {(!ready || isSyncing) ? (
            <div data-testid="graph-data-loading" style={{ padding: 32, textAlign: 'center' }}>
              {isSyncing ? "Synchronizing graph data before navigation..." : "Loading graph data..."}
            </div>
          ) : (
            <Routes>
              <Route
                path="/"
                element={
                  <GraphNet
                    // Pass all needed props into GraphNet
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
                    labelColumn={labelColumn}
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
                    syncFlowToGraphData={syncFlowToGraphData}  // Pass the sync method
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
          )}
        </div>
      </div>
    </div>
  );
}

// Export memoized component for better performance
export default memo(App);
import React, { useState, useEffect, useRef } from 'react';
import { trainModel } from '../../api';
import MetricsVisualizer from './MetricsVisualizer';
import { Link } from 'react-router-dom';
import './TrainingTab.css';
import { useGraphData, useGraphActions } from '../../context/GraphDataContext';
import useGraph from '../../hooks/useGraph';

// Debug function to inspect graph data structure
function debugGraphData(graphContext) {
  if (!graphContext) return {};
  
  // Safe access to potentially missing properties
  const hasNodes = Array.isArray(graphContext.nodes);
  const hasEdges = Array.isArray(graphContext.edges);
  const hasLinks = Array.isArray(graphContext.links);
  
  // Check all possible properties that might contain edge data
  const edgeProps = Object.keys(graphContext).filter(key => 
    Array.isArray(graphContext[key]) && 
    graphContext[key].length > 0 &&
    graphContext[key][0] && 
    (graphContext[key][0].source || graphContext[key][0].target)
  );
  
  return {
    nodeCount: hasNodes ? graphContext.nodes.length : 0,
    edgeCount: hasEdges ? graphContext.edges.length : 0,
    linkCount: hasLinks ? graphContext.links.length : 0,
    possibleEdgeProps: edgeProps,
    keys: Object.keys(graphContext)
  };
}

/**
 * Training tab component that allows users to train GNN models on graph data
 * Provides model selection, hyperparameter configuration, and training management
 */
const TrainingTab = (props) => {
  // Use props if provided (from App.js), otherwise fallback to context
  const contextGraph = useGraphData();
  const graphActions = useGraphActions();
  const graphContext = props.graphData || contextGraph;
  const ready = contextGraph.ready !== undefined ? contextGraph.ready : true;

  // Enhanced function to find edge data regardless of where it's stored
  function getAllEdges(graphData) {
    if (!graphData) return [];
    
    // Debug output to console
    console.log('Graph data debug:', debugGraphData(graphData));
    
    // Check for edges directly in graphData
    if (Array.isArray(graphData.edges) && graphData.edges.length > 0) {
      return graphData.edges;
    }
    
    // Check for links directly in graphData
    if (Array.isArray(graphData.links) && graphData.links.length > 0) {
      return graphData.links;
    }
    
    // Check if the data might be nested in a 'graph' property
    if (graphData.graph) {
      if (Array.isArray(graphData.graph.edges) && graphData.graph.edges.length > 0) {
        return graphData.graph.edges;
      }
      if (Array.isArray(graphData.graph.links) && graphData.graph.links.length > 0) {
        return graphData.graph.links;
      }
    }
    
    // For processed data, check if it might be in another property
    for (const key of Object.keys(graphData)) {
      const value = graphData[key];
      if (typeof value === 'object' && value !== null) {
        // Check if this property has edges or links
        if (Array.isArray(value.edges) && value.edges.length > 0) {
          return value.edges;
        }
        if (Array.isArray(value.links) && value.links.length > 0) {
          return value.links;
        }
      }
    }
    
    // Check if there's any array property that looks like edge data
    for (const key of Object.keys(graphData)) {
      const value = graphData[key];
      if (Array.isArray(value) && 
          value.length > 0 && 
          value[0] && 
          typeof value[0] === 'object' &&
          (value[0].source !== undefined || value[0].target !== undefined)) {
        return value;
      }
    }
    
    // No edge data found
    return [];
  }

  // Use the canonical graph validation from useGraph hook
  const { isValidForTraining } = useGraph();

  // Allow prop override for validation if provided (for testability/backward compatibility)
  const graphIsValidForTraining = typeof props.isGraphValidForTraining === "function"
    ? props.isGraphValidForTraining()
    : isValidForTraining;

  // ENHANCED: More robust graph stats computation
  const graphStats = React.useMemo(() => {
    if (!graphContext || !Array.isArray(graphContext.nodes)) {
      return {
        nodes: 0,
        edges: 0,
        hasLabels: false,
        uniqueLabels: [],
        lastProcessed: null,
      };
    }
    
    // Get nodes from the context
    const nodeCount = graphContext.nodes.length;
    
    // IMPROVED: Get edges using the enhanced detection function
    const edgesData = getAllEdges(graphContext);
    const edgeCount = edgesData.length;
    
    // Check for node labels
    const hasLabels = graphContext.nodes.some(
      (node) =>
        node.label !== undefined &&
        node.label !== null &&
        node.label !== ""
    );
    
    // Get unique labels
    let uniqueLabels = [];
    if (hasLabels) {
      uniqueLabels = [
        ...new Set(
          graphContext.nodes
            .filter(
              (node) =>
                node.label !== undefined &&
                node.label !== null &&
                node.label !== ""
            )
            .map((node) => node.label)
        ),
      ];
    }
    
    return {
      nodes: nodeCount,
      edges: edgeCount,
      hasLabels,
      uniqueLabels,
      lastProcessed: graphContext.lastSync
        ? new Date(graphContext.lastSync)
        : null,
    };
  }, [graphContext]);

  // ENHANCED: Better graph data preparation for training
  const graphData = React.useMemo(() => {
    if (!graphContext || !Array.isArray(graphContext.nodes)) return null;
    
    // Get edges using the enhanced detection function
    const edgesData = getAllEdges(graphContext);
    
    // Only proceed if we have both nodes and edges
    if (graphContext.nodes.length > 0 && edgesData.length > 0) {
      return { 
        nodes: graphContext.nodes,
        links: edgesData // Always use 'links' for consistency with the API
      };
    }
    
    return null;
  }, [graphContext]);

  // Ensure UI updates on context changes (logs, warnings, lastSync)
  React.useEffect(() => {
    // This effect ensures the component re-renders when any relevant context state changes
  }, [graphContext.trainingLogs, graphContext.validationWarning, graphContext.lastSync]);
  
  // Training configuration state
  const [modelConfig, setModelConfig] = useState({
    model_name: 'GCN',
    hidden_channels: 64,
    learning_rate: 0.01,
    epochs: 200,
    dropout: 0.3,
    heads: 8,          // For GAT
    K: 3               // For ChebConv
  });
  
  // Training status and logs
  const [isTraining, setIsTraining] = useState(false);
  const trainingLogs = graphContext.trainingLogs || [];
  const [metrics, setMetrics] = useState(null);
  const [trainingError, setTrainingError] = useState(null);
  const trainingRequestRef = useRef(null);
  const logsContainerRef = useRef(null);
  
  // Reset logs when model changes
  useEffect(() => {
    graphActions.clearTrainingLogs();
    setMetrics(null);
    setTrainingError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelConfig.model_name]);
  
  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [trainingLogs]);
  
  /**
   * Handle form input changes
   */
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    // Convert numeric values from strings
    let parsedValue = value;
    if (name === 'hidden_channels' || name === 'epochs' || name === 'heads' || name === 'K') {
      parsedValue = parseInt(value, 10);
      // Input validation: ensure values are positive integers
      if (isNaN(parsedValue) || parsedValue <= 0) return;
    } else if (name === 'learning_rate' || name === 'dropout') {
      parsedValue = parseFloat(value);
      // Input validation: ensure values are positive floats
      if (isNaN(parsedValue) || parsedValue < 0) return;
      // Additional validation for dropout (must be between 0 and 1)
      if (name === 'dropout' && parsedValue > 1) return;
    }
    
    setModelConfig(prev => ({
      ...prev,
      [name]: parsedValue
    }));
  };
  
  /**
   * Handle model selection change
   */
  const handleModelChange = (e) => {
    const model = e.target.value;
    // Update model name while preserving other config settings
    setModelConfig(prev => ({
      ...prev,
      model_name: model
    }));
  };
  
  /**
   * Start the training process
   */
  const handleStartTraining = () => {
    // Validate graph data exists
    if (!graphData || !graphData.nodes || !graphData.links) {
      setTrainingError("No valid graph data available. Please create a graph first.");
      return;
    }

    // Debug output of graph data being used for training
    console.log('Training with graph data:', {
      nodes: graphData.nodes.length,
      links: graphData.links.length
    });

    // Clear previous logs and errors
    graphActions.clearTrainingLogs();
    setMetrics(null);
    setTrainingError(null);
    setIsTraining(true);

    // Prepare the configuration object based on selected model
    const configToSend = {
      model_name: modelConfig.model_name,
      hidden_channels: modelConfig.hidden_channels,
      learning_rate: modelConfig.learning_rate,
      epochs: modelConfig.epochs,
      dropout: modelConfig.dropout
    };

    // Add model-specific parameters if needed
    if (modelConfig.model_name === 'GAT') {
      configToSend.heads = modelConfig.heads;
    } else if (modelConfig.model_name === 'ChebConv') {
      configToSend.K = modelConfig.K;
    }

    // Message handler for training updates
    const handleTrainingMessage = (message) => {
      // Prevent duplicate "Training started" log if backend also sends it
      if (message.type === 'log' && message.message === 'Training started') {
        // Only append if not already present in logs
        if (
          !graphContext.trainingLogs ||
          !graphContext.trainingLogs.some(
            (log) => log.message === 'Training started'
          )
        ) {
          graphActions.appendTrainingLog(message);
        }
      } else if (message.type === 'log') {
        graphActions.appendTrainingLog(message);
      } else if (message.type === 'metrics') {
        setMetrics(message.data);
        setIsTraining(false);
      } else if (message.type === 'complete') {
        setIsTraining(false);
      }
    };

    // Error handler for training errors
    const handleTrainingError = (error) => {
      console.error("Training error:", error);
      // Normalize error message for network/API errors
      let errorMsg = "An unknown error occurred during training";
      if (error && typeof error === "object") {
        if (error.message) {
          errorMsg = error.message;
        } else if (error.response && error.response.data && error.response.data.error) {
          errorMsg = error.response.data.error;
        }
      } else if (typeof error === "string") {
        errorMsg = error;
      }
      setTrainingError(errorMsg);
      setIsTraining(false);
    };

    try {
      // Call the API to train the model
      trainingRequestRef.current = trainModel(
        graphData,
        configToSend,
        handleTrainingMessage,
        handleTrainingError
      );
    } catch (error) {
      handleTrainingError(error);
    }
  };
  
  /**
   * Stop the ongoing training process
   */
  const handleStopTraining = () => {
    if (trainingRequestRef.current) {
      trainingRequestRef.current.cancel?.();
      setIsTraining(false);
      graphActions.appendTrainingLog({
        type: 'log',
        message: 'Training canceled by user',
        timestamp: new Date().toISOString()
      });
    }
  };
  
  /**
   * Render model-specific parameters based on selected model
   */
  const renderModelSpecificParams = () => {
    switch(modelConfig.model_name) {
      case 'GAT':
        return (
          <div className="form-group">
            <label htmlFor="heads">
              Attention Heads:
              <input
                type="number"
                id="heads"
                name="heads"
                value={modelConfig.heads}
                onChange={handleInputChange}
                min="1"
                max="16"
              />
            </label>
          </div>
        );
      case 'ChebConv':
        return (
          <div className="form-group">
            <label htmlFor="K">
              Chebyshev Filter Size (K):
              <input
                type="number"
                id="K"
                name="K"
                value={modelConfig.K}
                onChange={handleInputChange}
                min="1"
                max="10"
              />
            </label>
          </div>
        );
      default:
        return null;
    }
  };
  
  /**
   * Render training metrics if available
   */
  const renderMetrics = () => {
    if (!metrics) return null;
    
    return (
      <div className="metrics-container" data-testid="metrics-container">
        <h3>Training Results</h3>
        <div className="metrics-grid">
          <div className="metric-item">
            <span className="metric-label">Test Accuracy:</span>
            <span className="metric-value">{(metrics.test_accuracy * 100).toFixed(2)}%</span>
          </div>
          {metrics.class_report && (
            <div className="metric-item">
              <span className="metric-label">Classification Report:</span>
              <pre className="metric-report">{metrics.class_report}</pre>
            </div>
          )}
          {metrics.training_time && (
            <div className="metric-item">
              <span className="metric-label">Training Time:</span>
              <span className="metric-value">{metrics.training_time.toFixed(2)}s</span>
            </div>
          )}
        </div>
      </div>
    );
  };
  
  /**
   * Render metrics visualizer if metrics data is available
   */
  const renderMetricsVisualizer = () => {
    if (!metrics) return null;
    
    return <MetricsVisualizer metrics={metrics} />;
  };
  
  /**
   * Render the graph summary section
   */
  const renderGraphSummary = () => {
    return (
      <div className="graph-summary" data-testid="graph-summary">
        <h3>Graph Summary</h3>
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label" data-testid="stat-nodes">Nodes:</span>
            <span className="stat-value">{graphStats.nodes}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label" data-testid="stat-edges">Edges:</span>
            <span className="stat-value">{graphStats.edges}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label" data-testid="stat-has-labels">Has Labels:</span>
            <span className="stat-value">{graphStats.hasLabels ? 'Yes' : 'No'}</span>
          </div>
        </div>

        <div className="labels-section" data-testid="available-labels-section">
          <h3>Available Labels</h3>
          <div className="labels-list">
            {graphStats.hasLabels && graphStats.uniqueLabels && graphStats.uniqueLabels.length > 0 ? (
              graphStats.uniqueLabels.map((label, index) => (
                <span key={index} className="label-item">{label}</span>
              ))
            ) : (
              <span className="label-item label-empty">(none)</span>
            )}
          </div>
        </div>
      </div>
    );
  };

  /**
   * Render the workflow guidance section
   */
  const renderWorkflowGuidance = () => {
    return (
      <div className="workflow-guidance">
        <h3>Workflow Guide</h3>
        <ol className="workflow-steps">
          <li className={graphData ? 'completed' : 'current'}>
            <span className="step-number">1</span>
            <span className="step-description">Create your graph in the <Link to="/">GraphNet tab</Link></span>
          </li>
          <li className={graphData && graphStats.hasLabels ? 'current' : ''}>
            <span className="step-number">2</span>
            <span className="step-description">Configure and train your model</span>
          </li>
          <li className={metrics ? 'current' : ''}>
            <span className="step-number">3</span>
            <span className="step-description">Evaluate model performance</span>
          </li>
        </ol>
        {!graphData && (
          <div className="workflow-message warning">
            <p>You need to create a graph before you can train a model.</p>
          </div>
        )}
        {graphData && !graphStats.hasLabels && (
          <div className="workflow-message warning">
            <p>Your graph needs labels for training. Return to the GraphNet tab and add a label column.</p>
          </div>
        )}
      </div>
    );
  };

  // Only render main content if ready and graphContext is valid
  if (!ready || !graphContext || !Array.isArray(graphContext.nodes)) {
    return (
      <div className="training-tab" data-testid="training-tab-loading" style={{ padding: 32, textAlign: 'center' }}>
        Loading training data...
      </div>
    );
  }

  // IMPROVED: Check for both nodes and edges using our enhanced edge detection
  const hasGraphData = graphContext && 
                      Array.isArray(graphContext.nodes) && 
                      graphContext.nodes.length > 0 &&
                      getAllEdges(graphContext).length > 0;

  // Show "No Graph Data Available" message if no graph data or no edges
  if (!hasGraphData) {
    return (
      <div className="training-tab" data-testid="training-tab-root">
        <div className="training-container">
          <div className="training-config-panel">
            <h2>Model Training</h2>
            <p>No Graph Data Available</p>
            <Link to="/" className="link-button" data-testid="nav-graphnet">Go to GraphNet Tab</Link>
          </div>
          <div className="training-output-panel">
            <h3>Training Logs</h3>
            <div className="logs-container" ref={logsContainerRef}>
              <p className="empty-logs">Training logs will appear here...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="training-tab" data-testid="training-tab-root">
      <div className="training-container">
        <div className="training-config-panel">
          <h2>Model Training</h2>

          {/* Model Architecture Section - always visible */}
          <div
            className="model-architecture-section"
            data-testid="model-architecture-section"
            style={{ marginBottom: 12 }}
          >
            <span style={{ fontWeight: 600 }}>Model Architecture:</span>{" "}
            {modelConfig && modelConfig.model_name
              ? modelConfig.model_name
              : "(none)"}
          </div>

          {/* Last Updated Section - always visible */}
          <div
            className="last-updated-section"
            data-testid="last-updated-section"
            style={{ marginBottom: 12 }}
          >
            <span style={{ fontWeight: 600 }}>Last Updated:</span>{" "}
            {graphStats && graphStats.lastProcessed
              ? graphStats.lastProcessed.toLocaleString()
              : "(not available)"}
          </div>

          {/* Graph Status Panel */}
          {renderGraphSummary()}

          {/* Workflow Guidance */}
          {renderWorkflowGuidance()}

          {/* Model Selection */}
          <div className="form-group">
            <label htmlFor="model-select">
              <span style={{ display: "none" }}>Model Architecture:</span>
              <select
                id="model-select"
                value={modelConfig.model_name}
                onChange={handleModelChange}
              >
                <option value="GCN">Graph Convolutional Network (GCN)</option>
                <option value="ResidualGCN">Residual GCN</option>
                <option value="GraphSAGE">GraphSAGE</option>
                <option value="GAT">Graph Attention Network (GAT)</option>
                <option value="GIN">Graph Isomorphism Network (GIN)</option>
                <option value="ChebConv">Chebyshev Spectral CNN</option>
                <option value="NaiveBayes">Naive Bayes (Baseline)</option>
              </select>
            </label>
          </div>
          
          {/* Common Hyperparameters */}
          <div className="form-section">
            <h3>Hyperparameters</h3>
            <div className="form-group">
              <label htmlFor="hidden_channels">
                Hidden Channels:
                <input
                  type="number"
                  id="hidden_channels"
                  name="hidden_channels"
                  value={modelConfig.hidden_channels}
                  onChange={handleInputChange}
                  min="8"
                  max="256"
                  disabled={modelConfig.model_name === "NaiveBayes"}
                />
              </label>
            </div>
            
            <div className="form-group">
              <label htmlFor="learning_rate">
                Learning Rate:
                <input
                  type="number"
                  id="learning_rate"
                  name="learning_rate"
                  value={modelConfig.learning_rate}
                  onChange={handleInputChange}
                  step="0.001"
                  min="0.0001"
                  max="0.1"
                  disabled={modelConfig.model_name === "NaiveBayes"}
                />
              </label>
            </div>
            
            <div className="form-group">
              <label htmlFor="epochs">
                Epochs:
                <input
                  type="number"
                  id="epochs"
                  name="epochs"
                  value={modelConfig.epochs}
                  onChange={handleInputChange}
                  min="1"
                  max="1000"
                  disabled={modelConfig.model_name === "NaiveBayes"}
                />
              </label>
            </div>
            
            <div className="form-group">
              <label htmlFor="dropout">
                Dropout:
                <input
                  type="number"
                  id="dropout"
                  name="dropout"
                  value={modelConfig.dropout}
                  onChange={handleInputChange}
                  step="0.1"
                  min="0"
                  max="0.9"
                  disabled={modelConfig.model_name === "NaiveBayes"}
                />
              </label>
            </div>
            
            {/* Model-specific parameters */}
            {renderModelSpecificParams()}
          </div>
          
          {/* Training Controls */}
          <div className="training-controls">
            <button
              className="start-button"
              onClick={handleStartTraining}
              disabled={isTraining || !graphIsValidForTraining}
              title={
                isTraining
                  ? "Training is already in progress"
                  : !graphIsValidForTraining
                    ? "Graph data is missing, invalid, or lacks labels required for training"
                    : ""
              }
            >
              Start Training
            </button>
            <button
              className="stop-button"
              onClick={handleStopTraining}
              disabled={!isTraining}
            >
              Stop Training
            </button>
          </div>
          
          {/* Training Validation Warning */}
          {!graphIsValidForTraining && (
            <div
              className="validation-warning"
              data-testid="validation-warning"
            >
              <p>
                Your graph data is invalid or missing required labels, nodes, or edges. Please check your graph in the GraphNet tab.
              </p>
              <Link to="/" className="link-button">Go to GraphNet Tab</Link>
            </div>
          )}
          
          {/* Training Error Display */}
          {trainingError && (
            <div className="training-error" role="alert">
              <p>Error: {trainingError}</p>
            </div>
          )}
        </div>
        
        {/* Training Logs */}
        <div className="training-output-panel">
          <h3>Training Logs</h3>
          <div className="logs-container" ref={logsContainerRef}>
            {trainingLogs.length === 0 ? (
              <p className="empty-logs">Training logs will appear here...</p>
            ) : (
              trainingLogs.map((log, index) => (
                <div
                  key={index}
                  className="log-entry"
                  data-testid="training-log-entry"
                >
                  <span className="log-time">
                    {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : ""}
                  </span>
                  <span className="log-message" data-testid="training-log-message">
                    {log.message}
                  </span>
                </div>
              ))
            )}
          </div>
          
          {/* Basic Metrics Display */}
          {renderMetrics()}
          
          {/* Metrics Visualizations */}
          {renderMetricsVisualizer()}
        </div>
      </div>
    </div>
  );
};

export default TrainingTab;
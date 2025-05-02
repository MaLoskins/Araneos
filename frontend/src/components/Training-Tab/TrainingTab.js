import React, { useState, useEffect, useRef } from 'react';
import { trainModel } from '../../api';
import MetricsVisualizer from './MetricsVisualizer';
import { Link } from 'react-router-dom';
import './TrainingTab.css';

/**
 * Training tab component that allows users to train GNN models on graph data
 * Provides model selection, hyperparameter configuration, and training management
 */
const TrainingTab = ({ graphData, loading, getGraphStats, isGraphValidForTraining }) => {
  // Get graph statistics
  const graphStats = getGraphStats();
  // Training configuration state
  const [modelConfig, setModelConfig] = useState({
    model_name: 'GCN',
    hidden_channels: 64,
    learning_rate: 0.01,
    epochs: 200,
    dropout: 0.3,
    // Optional model-specific parameters
    heads: 8,          // For GAT
    K: 3               // For ChebConv
  });
  
  // Training status and logs
  const [isTraining, setIsTraining] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [trainingError, setTrainingError] = useState(null);
  
  // Reference to cancel training if needed
  const trainingRequestRef = useRef(null);
  
  // Auto-scroll for logs
  const logsContainerRef = useRef(null);
  
  // Reset logs when model changes
  useEffect(() => {
    setTrainingLogs([]);
    setMetrics(null);
    setTrainingError(null);
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
    
    // Clear previous logs and errors
    setTrainingLogs([]);
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
      if (message.type === 'log') {
        setTrainingLogs(prev => [...prev, message]);
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
      setTrainingError(error.message || "An unknown error occurred during training");
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
      setTrainingLogs(prev => [...prev, { 
        type: 'log', 
        message: 'Training canceled by user',
        timestamp: new Date().toISOString()
      }]);
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
      <div className="metrics-container">
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
    if (!graphData) {
      return (
        <div className="graph-summary no-graph">
          <h3>No Graph Data Available</h3>
          <p>You need to create a graph before training a model.</p>
          <Link to="/" className="link-button">Go to GraphNet Tab</Link>
        </div>
      );
    }

    return (
      <div className="graph-summary">
        <h3>Graph Summary</h3>
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label">Nodes:</span>
            <span className="stat-value">{graphStats.nodes}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Edges:</span>
            <span className="stat-value">{graphStats.edges}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Has Labels:</span>
            <span className="stat-value">{graphStats.hasLabels ? 'Yes' : 'No'}</span>
          </div>
          {graphStats.lastProcessed && (
            <div className="stat-item">
              <span className="stat-label">Last Updated:</span>
              <span className="stat-value">{graphStats.lastProcessed.toLocaleTimeString()}</span>
            </div>
          )}
        </div>

        {graphStats.hasLabels && graphStats.uniqueLabels && graphStats.uniqueLabels.length > 0 && (
          <div className="labels-section">
            <h4>Available Labels</h4>
            <div className="labels-list">
              {graphStats.uniqueLabels.map((label, index) => (
                <span key={index} className="label-item">{label}</span>
              ))}
            </div>
          </div>
        )}
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

  return (
    <div className="training-tab">
      <div className="training-container">
        <div className="training-config-panel">
          <h2>Model Training</h2>
          
          {/* Graph Status Panel */}
          {renderGraphSummary()}
          
          {/* Workflow Guidance */}
          {renderWorkflowGuidance()}
          
          {/* Model Selection */}
          <div className="form-group">
            <label htmlFor="model-select">
              Model Architecture:
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
              disabled={isTraining || !graphData || !isGraphValidForTraining()}
              title={!isGraphValidForTraining() ? "Graph data must include labels for training" : ""}
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
          {graphData && !isGraphValidForTraining() && (
            <div className="validation-warning">
              <p>Your graph data is missing labels required for training. Go back to the GraphNet tab to add labels.</p>
              <Link to="/" className="link-button">Go to GraphNet Tab</Link>
            </div>
          )}
          
          {/* Training Error Display */}
          {trainingError && (
            <div className="training-error">
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
                <div key={index} className="log-entry">
                  <span className="log-time">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="log-message">{log.message}</span>
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
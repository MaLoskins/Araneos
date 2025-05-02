import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

/**
 * Processes data using the backend API
 * @param {Object} data - The data to process
 * @param {Object} config - Configuration for processing
 * @returns {Promise<Object>} - The processed data
 */
export const processData = async (data, config) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/process-data`, { data, config });
    return response.data;
  } catch (error) {
    throw error;
  }
};

/**
 * Trains a GNN model on the provided graph data and streams back results
 * @param {Object} graph - The graph data in node-link format
 * @param {Object} modelConfig - Configuration for the GNN model
 * @param {Function} onMessage - Callback for each training update
 * @param {Function} onError - Callback for error handling
 * @returns {Object} - The axios request object for potential cancellation
 */
export const trainModel = (graph, modelConfig, onMessage, onError) => {
  // Input validation
  if (!graph || !graph.nodes || !graph.links) {
    const error = new Error('Invalid graph data: Must contain nodes and links');
    onError(error);
    return Promise.reject(error);
  }
  
  if (!modelConfig || !modelConfig.model_name) {
    const error = new Error('Invalid model configuration: Missing required parameters');
    onError(error);
    return Promise.reject(error);
  }
  
  if (typeof onMessage !== 'function') {
    const error = new Error('onMessage must be a function');
    onError(error);
    return Promise.reject(error);
  }
  
  if (typeof onError !== 'function') {
    console.error('onError must be a function, using default error handler');
    onError = (err) => console.error('Training error:', err);
  }

  try {
    // Create the request configuration
    const config = {
      url: `${API_BASE_URL}/train-gnn`,
      method: 'POST',
      data: {
        graph: graph,
        configuration: modelConfig
      },
      responseType: 'stream',
      
      // Handle streaming response with event handlers
      onDownloadProgress: (progressEvent) => {
        // Parse the raw response text and handle each chunk
        const rawText = progressEvent.currentTarget.response;
        if (!rawText) return;
        
        // Split the text by newlines to get individual JSON messages
        const messages = rawText.split('\n').filter(line => line.trim());
        
        // Process only new messages since last update
        const lastProcessedIndex = progressEvent.lastProcessedIndex || 0;
        const newMessages = messages.slice(lastProcessedIndex);
        
        // Process each new message
        newMessages.forEach(message => {
          try {
            const parsedMessage = JSON.parse(message);
            onMessage(parsedMessage);
          } catch (parseError) {
            console.warn('Error parsing training message:', parseError);
          }
        });
        
        // Update the last processed index
        progressEvent.lastProcessedIndex = messages.length;
      }
    };

    // Execute the request
    return axios(config).catch(error => {
      // Handle request errors
      onError(error);
      throw error;
    });
  } catch (error) {
    // Handle any synchronous errors during request setup
    onError(error);
    return Promise.reject(error);
  }
};

// Import the function to test
import { trainModel } from '../api';

// Create a mock for axios
jest.mock('axios', () => {
  // Create a mock function with chainable methods
  const mockAxios = jest.fn(() => {
    return {
      catch: jest.fn().mockImplementation(fn => {
        // Store the callback to use later when simulating errors
        mockAxios.errorCallback = fn;
        return Promise.resolve();
      })
    };
  });
  
  // Return status of the mock
  mockAxios.mockClear = jest.fn();
  
  // Return our configured mock
  return mockAxios;
});

// Import axios again after mocking
const axios = require('axios');

describe('trainModel function', () => {
  // Setup test variables
  let mockOnMessage;
  let mockOnError;
  let validGraph;
  let validModelConfig;

  // Reset mocks and set up test data before each test
  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();
    
    // Setup mock callbacks
    mockOnMessage = jest.fn();
    mockOnError = jest.fn();
    
    // Setup valid test data
    validGraph = {
      nodes: [{ id: 1 }, { id: 2 }],
      links: [{ source: 1, target: 2 }]
    };
    
    validModelConfig = {
      model_name: 'GCN',
      epochs: 100,
      learning_rate: 0.01
    };
  });

  // #1 - Test input validation
  describe('Input validation', () => {
    test('rejects when graph is missing nodes', async () => {
      const invalidGraph = { links: [] };
      
      await expect(trainModel(invalidGraph, validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Invalid graph data');
      
      expect(mockOnError).toHaveBeenCalled();
      expect(axios).not.toHaveBeenCalled();
    });
    
    test('rejects when graph is missing links', async () => {
      const invalidGraph = { nodes: [] };
      
      await expect(trainModel(invalidGraph, validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Invalid graph data');
      
      expect(mockOnError).toHaveBeenCalled();
      expect(axios).not.toHaveBeenCalled();
    });
    
    test('rejects when modelConfig is missing model_name', async () => {
      const invalidConfig = { epochs: 100 };
      
      await expect(trainModel(validGraph, invalidConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Invalid model configuration');
      
      expect(mockOnError).toHaveBeenCalled();
      expect(axios).not.toHaveBeenCalled();
    });
    
    test('rejects when onMessage is not a function', async () => {
      await expect(trainModel(validGraph, validModelConfig, 'not a function', mockOnError))
        .rejects.toThrow('onMessage must be a function');
      
      expect(mockOnError).toHaveBeenCalled();
      expect(axios).not.toHaveBeenCalled();
    });
    
    test('uses default error handler when onError is not a function', async () => {
      // Mock console.error
      const originalConsoleError = console.error;
      console.error = jest.fn();
      
      try {
        // Set up axios to resolve successfully
        axios.mockImplementation(() => Promise.resolve({}));
        
        // Call with invalid onError
        await trainModel(validGraph, validModelConfig, mockOnMessage, 'not a function');
        
        // Check that console.error was called
        expect(console.error).toHaveBeenCalled();
      } finally {
        // Restore console.error
        console.error = originalConsoleError;
      }
    });
  });

  // #2 - Test request configuration
  describe('Request configuration', () => {
    test('creates correct request configuration', async () => {
      // Set up axios to resolve successfully
      axios.mockImplementation(() => Promise.resolve({}));
      
      // Call the function
      await trainModel(validGraph, validModelConfig, mockOnMessage, mockOnError);
      
      // Verify axios was called with correct parameters
      expect(axios).toHaveBeenCalledWith(expect.objectContaining({
        url: 'http://localhost:8000/train-gnn',
        method: 'POST',
        data: {
          graph: validGraph,
          configuration: validModelConfig
        },
        responseType: 'stream',
      }));
      
      // Verify onDownloadProgress handler is included
      const config = axios.mock.calls[0][0];
      expect(config).toHaveProperty('onDownloadProgress');
      expect(typeof config.onDownloadProgress).toBe('function');
    });
  });

  // #3 - Test streaming response handling
  describe('Streaming response handling', () => {
    test('processes streaming data correctly', async () => {
      // Set up axios to resolve successfully
      axios.mockImplementation(() => Promise.resolve({}));
      
      // Call the function
      await trainModel(validGraph, validModelConfig, mockOnMessage, mockOnError);
      
      // Get the onDownloadProgress handler from the axios call
      const config = axios.mock.calls[0][0];
      const onDownloadProgress = config.onDownloadProgress;
      
      // Create mock progressEvent with training data
      const progressEvent = {
        currentTarget: {
          response: '{"epoch":1,"loss":0.5}\n{"epoch":2,"loss":0.3}'
        }
      };
      
      // Call the handler with the progress event
      onDownloadProgress(progressEvent);
      
      // Verify onMessage was called twice with the correct data
      expect(mockOnMessage).toHaveBeenCalledTimes(2);
      expect(mockOnMessage).toHaveBeenNthCalledWith(1, { epoch: 1, loss: 0.5 });
      expect(mockOnMessage).toHaveBeenNthCalledWith(2, { epoch: 2, loss: 0.3 });
    });
    
    test('handles empty response', async () => {
      // Set up axios to resolve successfully
      axios.mockImplementation(() => Promise.resolve({}));
      
      // Call the function
      await trainModel(validGraph, validModelConfig, mockOnMessage, mockOnError);
      
      // Get the onDownloadProgress handler
      const config = axios.mock.calls[0][0];
      const onDownloadProgress = config.onDownloadProgress;
      
      // Create mock progressEvent with empty response
      const progressEvent = {
        currentTarget: {
          response: ''
        }
      };
      
      // Call the handler
      onDownloadProgress(progressEvent);
      
      // Verify onMessage was not called
      expect(mockOnMessage).not.toHaveBeenCalled();
    });
    
    test('processes only new messages since last update', async () => {
      // Set up axios to resolve successfully
      axios.mockImplementation(() => Promise.resolve({}));
      
      // Call the function
      await trainModel(validGraph, validModelConfig, mockOnMessage, mockOnError);
      
      // Get the onDownloadProgress handler
      const config = axios.mock.calls[0][0];
      const onDownloadProgress = config.onDownloadProgress;
      
      // First progress event with initial data
      const initialProgressEvent = {
        currentTarget: {
          response: '{"epoch":1,"loss":0.5}\n{"epoch":2,"loss":0.3}'
        },
        lastProcessedIndex: 0
      };
      
      // Call the handler with initial data
      onDownloadProgress(initialProgressEvent);
      
      // Verify onMessage was called twice
      expect(mockOnMessage).toHaveBeenCalledTimes(2);
      
      // Reset the mock to count new calls only
      mockOnMessage.mockClear();
      
      // Second progress event with additional data
      const updatedProgressEvent = {
        currentTarget: {
          response: '{"epoch":1,"loss":0.5}\n{"epoch":2,"loss":0.3}\n{"epoch":3,"loss":0.2}'
        },
        lastProcessedIndex: 2
      };
      
      // Call the handler with updated data
      onDownloadProgress(updatedProgressEvent);
      
      // Verify onMessage was called once with only the new message
      expect(mockOnMessage).toHaveBeenCalledTimes(1);
      expect(mockOnMessage).toHaveBeenCalledWith({ epoch: 3, loss: 0.2 });
    });
    
    test('handles invalid JSON in the response', async () => {
      // Mock console.warn
      const originalConsoleWarn = console.warn;
      console.warn = jest.fn();
      
      try {
        // Set up axios to resolve
        axios.mockImplementation(() => Promise.resolve({}));
        
        // Call the function
        await trainModel(validGraph, validModelConfig, mockOnMessage, mockOnError);
        
        // Get the onDownloadProgress handler
        const config = axios.mock.calls[0][0];
        const onDownloadProgress = config.onDownloadProgress;
        
        // Create mock progressEvent with invalid JSON
        const progressEvent = {
          currentTarget: {
            response: '{"epoch":1,"loss":0.5}\nNOT JSON\n{"epoch":3,"loss":0.2}'
          }
        };
        
        // Call the handler
        onDownloadProgress(progressEvent);
        
        // Verify onMessage was called only for valid JSON
        expect(mockOnMessage).toHaveBeenCalledTimes(2);
        expect(mockOnMessage).toHaveBeenNthCalledWith(1, { epoch: 1, loss: 0.5 });
        expect(mockOnMessage).toHaveBeenNthCalledWith(2, { epoch: 3, loss: 0.2 });
        
        // Verify console.warn was called for the invalid JSON
        expect(console.warn).toHaveBeenCalled();
      } finally {
        // Restore console.warn
        console.warn = originalConsoleWarn;
      }
    });
  });

  // #4 - Test error handling
  describe('Error handling', () => {
    test('handles network errors', async () => {
      // Create a network error
      const networkError = new Error('Network Error');
      
      // Set up axios to reject with the error
      axios.mockImplementation(() => {
        return {
          catch: jest.fn().mockImplementation(fn => {
            // Execute the error callback immediately
            fn(networkError);
            return Promise.reject(networkError);
          })
        };
      });
      
      // Call the function and expect it to reject
      await expect(trainModel(validGraph, validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Network Error');
      
      // Verify onError was called with the network error
      expect(mockOnError).toHaveBeenCalledWith(networkError);
    });
    
    test('handles server errors (4xx/5xx)', async () => {
      // Create a server error
      const serverError = new Error('Internal Server Error');
      serverError.response = { status: 500, data: { message: 'Server failed' } };
      
      // Set up axios to reject with server error
      axios.mockImplementation(() => {
        return {
          catch: jest.fn().mockImplementation(fn => {
            // Execute the error callback immediately
            fn(serverError);
            return Promise.reject(serverError);
          })
        };
      });
      
      // Call the function and expect it to reject
      await expect(trainModel(validGraph, validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Internal Server Error');
      
      // Verify onError was called with the server error
      expect(mockOnError).toHaveBeenCalledWith(serverError);
    });
    
    test('handles synchronous errors during request setup', async () => {
      // Set up axios to throw a synchronous error
      const syncError = new Error('Synchronous Error');
      axios.mockImplementation(() => { throw syncError; });
      
      // Call the function and expect it to reject
      await expect(trainModel(validGraph, validModelConfig, mockOnMessage, mockOnError))
        .rejects.toThrow('Synchronous Error');
      
      // Verify onError was called
      expect(mockOnError).toHaveBeenCalledWith(syncError);
    });
  });
});
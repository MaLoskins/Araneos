import React, { createContext, useContext, useReducer, useEffect, useCallback, useMemo } from 'react';

// --- Custom Serialization/Deserialization Utilities ---

// Helper to get object path (for circular refs) - simplified for acyclic graphs
function graphReplacer(key, value) {
  // Remove non-serializable fields (functions, DOM refs, etc.)
  if (typeof value === 'function') return undefined;
  if (value instanceof Map || value instanceof Set) return Array.from(value);
  if (value && value.$$typeof) return undefined; // React elements
  return value;
}

function graphReviver(key, value) {
  // No special handling needed for acyclic, serializable graph data
  return value;
}

// --- Initial State ---

const LOCAL_STORAGE_KEY = 'graphData';
const REACTFLOW_CONFIG_KEY = 'reactFlowConfig';

const initialGraphState = {
  nodes: [],
  edges: [],
  links: [], // Always keep both edges and links for compatibility
  // Add other serializable graph properties as needed
  lastSync: null, // for synchronization markers
  trainingLogs: [],
  validationWarning: null,
  lastUpdated: null,
  // New field to differentiate between ReactFlow config and processed graph
  configOnly: false,
  // Separate storage for ReactFlow configuration
  reactFlowConfig: null,
};

// --- Reducer for Immutable Updates ---

function graphReducer(state, action) {
  switch (action.type) {
    case 'SET_GRAPH':
      // ENHANCED: Always make sure both edges and links are set
      const payload = { ...action.payload };
      
      // Make sure we have both edges and links (for compatibility)
      if (Array.isArray(payload.edges) && !Array.isArray(payload.links)) {
        payload.links = [...payload.edges];
      } else if (Array.isArray(payload.links) && !Array.isArray(payload.edges)) {
        payload.edges = [...payload.links];
      }
      
      return { 
        ...state, 
        ...payload, 
        lastSync: Date.now(), 
        lastUpdated: Date.now(),
        // Clear configOnly flag when setting real graph data
        configOnly: action.payload.configOnly || false
      };
    case 'SET_REACTFLOW_CONFIG':
      return { 
        ...state, 
        reactFlowConfig: action.payload,
        lastSync: Date.now(),
        lastUpdated: Date.now() 
      };
    case 'UPDATE_NODES':
      return { ...state, nodes: action.payload, lastSync: Date.now(), lastUpdated: Date.now() };
    case 'UPDATE_EDGES':
      // ENHANCED: Update both edges and links
      return { 
        ...state, 
        edges: action.payload, 
        links: action.payload, // Keep links in sync with edges
        lastSync: Date.now(), 
        lastUpdated: Date.now() 
      };
    case 'RESET_GRAPH':
      return { ...initialGraphState, lastSync: Date.now(), lastUpdated: Date.now() };
    case 'APPEND_TRAINING_LOG':
      return { ...state, trainingLogs: [...(state.trainingLogs || []), action.payload], lastUpdated: Date.now() };
    case 'CLEAR_TRAINING_LOGS':
      return { ...state, trainingLogs: [], lastUpdated: Date.now() };
    case 'SET_VALIDATION_WARNING':
      return { ...state, validationWarning: action.payload, lastUpdated: Date.now() };
    default:
      throw new Error(`Unhandled action type: ${action.type}`);
  }
}

// --- Debug Function to log graph state ---
function debugGraphState(state) {
  return {
    nodeCount: state?.nodes?.length || 0,
    edgeCount: state?.edges?.length || 0,
    linkCount: state?.links?.length || 0,
    hasReactFlowConfig: !!state?.reactFlowConfig,
    configOnly: !!state?.configOnly,
    keys: Object.keys(state || {})
  };
}

// --- Contexts ---

const GraphDataContext = createContext();
const GraphDispatchContext = createContext();

// --- Provider ---

export function GraphDataProvider({ children }) {
  // --- Synchronous State Initialization from localStorage ---
  const getInitialState = () => {
    try {
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
      const storedConfig = localStorage.getItem(REACTFLOW_CONFIG_KEY);
      
      let state = { ...initialGraphState };
      
      if (stored) {
        const parsed = JSON.parse(stored, graphReviver);
        
        // ENHANCED: Handle both edges and links correctly
        let edges = [];
        let links = [];
        
        if (parsed) {
          if (Array.isArray(parsed.edges)) {
            edges = parsed.edges;
          }
          if (Array.isArray(parsed.links)) {
            links = parsed.links;
          }
          
          // Use whichever is non-empty, prefer edges
          if (edges.length > 0) {
            links = edges;
          } else if (links.length > 0) {
            edges = links;
          }
        }
        
        if (parsed && Array.isArray(parsed.nodes)) {
          state = { 
            ...state, 
            ...parsed, 
            edges, 
            links, // Keep both for compatibility
          };
        }
      }
      
      // Load ReactFlow config if available
      if (storedConfig) {
        try {
          const parsedConfig = JSON.parse(storedConfig, graphReviver);
          state.reactFlowConfig = parsedConfig;
        } catch (e) {
          console.error('Failed to parse ReactFlow config:', e);
        }
      }
      
      // Debug log of initialized state
      console.log('GraphDataContext initialized with:', debugGraphState(state));
      
      return state;
    } catch (e) {
      console.error('Failed to load graph data from storage:', e);
    }
    return initialGraphState;
  };

  const [state, dispatch] = useReducer(graphReducer, undefined, getInitialState);

  // --- Hydration/Ready State ---
  const [ready, setReady] = React.useState(false);

  // --- State ref for stable callbacks ---
  const stateRef = React.useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // --- Persistence sync promise for race-free navigation ---
  const persistPromiseRef = React.useRef(Promise.resolve());
  useEffect(() => {
    let resolver;
    persistPromiseRef.current = new Promise((resolve) => {
      resolver = resolve;
    });
    try {
      // ENHANCED: Always include both edges and links
      // Make sure both are present and contain the same data
      let edges = Array.isArray(state.edges) ? state.edges : [];
      let links = Array.isArray(state.links) ? state.links : [];
      
      // If either is empty but the other isn't, sync them
      if (edges.length === 0 && links.length > 0) {
        edges = links;
      } else if (links.length === 0 && edges.length > 0) {
        links = edges;
      }
      
      const persistObj = {
        ...state,
        edges,
        links,
      };
      
      const serialized = JSON.stringify(persistObj, graphReplacer);
      localStorage.setItem(LOCAL_STORAGE_KEY, serialized);
      
      // Store ReactFlow config separately if available
      if (state.reactFlowConfig) {
        const configSerialized = JSON.stringify(state.reactFlowConfig, graphReplacer);
        localStorage.setItem(REACTFLOW_CONFIG_KEY, configSerialized);
      }
      
      // Debug log of persisted state
      console.log('GraphDataContext persisted:', debugGraphState(persistObj));
    } catch (e) {
      console.error('Failed to save graph data to storage:', e);
    }
    if (resolver) resolver();
    // eslint-disable-next-line
  }, [state]);

  // --- Hydration effect: Mark ready after first mount ---
  useEffect(() => {
    setReady(true);
  }, []);

  // --- Cross-tab Sync: BroadcastChannel (Observer Pattern) ---
  // Channel is created once for the provider's lifetime
  const channelRef = React.useRef(null);

  useEffect(() => {
    try {
      channelRef.current = new window.BroadcastChannel('graph_sync');
    } catch (e) {
      console.error('Failed to create BroadcastChannel:', e);
      channelRef.current = null;
      return;
    }

    const handleMessage = (event) => {
      if (event.data?.type === 'GRAPH_UPDATE') {
        try {
          const incoming = JSON.parse(event.data.payload, graphReviver);
          // Only update if incoming is newer
          if (!state.lastSync || (incoming.lastSync && incoming.lastSync > state.lastSync)) {
            dispatch({ type: 'SET_GRAPH', payload: incoming });
          }
        } catch (e) {
          console.error('Failed to process cross-tab graph update:', e);
        }
      } else if (event.data?.type === 'REACTFLOW_CONFIG_UPDATE') {
        try {
          const incomingConfig = JSON.parse(event.data.payload, graphReviver);
          dispatch({ type: 'SET_REACTFLOW_CONFIG', payload: incomingConfig });
        } catch (e) {
          console.error('Failed to process cross-tab ReactFlow config update:', e);
        }
      }
    };

    const channel = channelRef.current;
    if (channel) {
      channel.onmessage = handleMessage;
    }

    return () => {
      if (channelRef.current) {
        channelRef.current.close();
        channelRef.current = null;
      }
    };
    // Only run on mount/unmount
    // eslint-disable-next-line
  }, []);

  // --- Keep event handler in sync with latest state ---
  useEffect(() => {
    // This effect ensures the handler always sees the latest state
    if (!channelRef.current) return;
    const channel = channelRef.current;
    const handler = (event) => {
      if (event.data?.type === 'GRAPH_UPDATE') {
        try {
          const incoming = JSON.parse(event.data.payload, graphReviver);
          if (!state.lastSync || (incoming.lastSync && incoming.lastSync > state.lastSync)) {
            dispatch({ type: 'SET_GRAPH', payload: incoming });
          }
        } catch (e) {
          console.error('Failed to process cross-tab graph update:', e);
        }
      } else if (event.data?.type === 'REACTFLOW_CONFIG_UPDATE') {
        try {
          const incomingConfig = JSON.parse(event.data.payload, graphReviver);
          dispatch({ type: 'SET_REACTFLOW_CONFIG', payload: incomingConfig });
        } catch (e) {
          console.error('Failed to process cross-tab ReactFlow config update:', e);
        }
      }
    };
    channel.onmessage = handler;
    return () => {
      if (channel.onmessage === handler) channel.onmessage = null;
    };
  }, [state.lastSync]); // Only update handler when lastSync changes

  // --- Broadcast changes to other tabs ---
  const broadcastGraph = useCallback((newState) => {
    if (!channelRef.current) return;
    try {
      channelRef.current.postMessage({
        type: 'GRAPH_UPDATE',
        payload: JSON.stringify(newState, graphReplacer),
      });
    } catch (e) {
      console.error('Failed to broadcast graph data:', e);
    }
  }, []);

  // New method to broadcast ReactFlow config
  const broadcastReactFlowConfig = useCallback((configData) => {
    if (!channelRef.current) return;
    try {
      channelRef.current.postMessage({
        type: 'REACTFLOW_CONFIG_UPDATE',
        payload: JSON.stringify(configData, graphReplacer),
      });
    } catch (e) {
      console.error('Failed to broadcast ReactFlow config:', e);
    }
  }, []);

  // --- Actions ---

  // Use stateRef for stable callbacks, so these functions don't change on every render
  const setGraph = useCallback((graph) => {
    // ENHANCED: Ensure both edges and links are set
    const enhancedGraph = { ...graph };
    
    if (Array.isArray(enhancedGraph.edges) && !Array.isArray(enhancedGraph.links)) {
      enhancedGraph.links = [...enhancedGraph.edges];
    } else if (Array.isArray(enhancedGraph.links) && !Array.isArray(enhancedGraph.edges)) {
      enhancedGraph.edges = [...enhancedGraph.links];
    }
    
    const next = { 
      ...stateRef.current, 
      ...enhancedGraph, 
      lastSync: Date.now(), 
      lastUpdated: Date.now() 
    };
    
    dispatch({ type: 'SET_GRAPH', payload: enhancedGraph });
    broadcastGraph(next);
    
    // Debug log
    console.log('GraphDataContext setGraph:', debugGraphState(next));
  }, [broadcastGraph]);

  // New action to set ReactFlow config separately
  const setReactFlowConfig = useCallback((config) => {
    dispatch({ type: 'SET_REACTFLOW_CONFIG', payload: config });
    broadcastReactFlowConfig(config);
  }, [broadcastReactFlowConfig]);

  const updateNodes = useCallback((nodes) => {
    const next = { ...stateRef.current, nodes, lastSync: Date.now(), lastUpdated: Date.now() };
    dispatch({ type: 'UPDATE_NODES', payload: nodes });
    broadcastGraph(next);
  }, [broadcastGraph]);

  const updateEdges = useCallback((edges) => {
    // ENHANCED: Keep links in sync with edges
    const next = { 
      ...stateRef.current, 
      edges, 
      links: edges, // Keep links in sync with edges
      lastSync: Date.now(), 
      lastUpdated: Date.now() 
    };
    dispatch({ type: 'UPDATE_EDGES', payload: edges });
    broadcastGraph(next);
  }, [broadcastGraph]);

  const resetGraph = useCallback(() => {
    const next = { ...initialGraphState, lastSync: Date.now(), lastUpdated: Date.now() };
    dispatch({ type: 'RESET_GRAPH' });
    broadcastGraph(next);
  }, [broadcastGraph]);
  
  // Ensure log/warning actions update lastSync and broadcast for persistence/sync
  const appendTrainingLog = useCallback((logEntry) => {
    const next = {
      ...stateRef.current,
      trainingLogs: [...(stateRef.current.trainingLogs || []), logEntry],
      lastSync: Date.now(),
      lastUpdated: Date.now(),
    };
    dispatch({ type: 'APPEND_TRAINING_LOG', payload: logEntry });
    broadcastGraph(next);
  }, [broadcastGraph]);
  
  const clearTrainingLogs = useCallback(() => {
    const next = {
      ...stateRef.current,
      trainingLogs: [],
      lastSync: Date.now(),
      lastUpdated: Date.now(),
    };
    dispatch({ type: 'CLEAR_TRAINING_LOGS' });
    broadcastGraph(next);
  }, [broadcastGraph]);
  
  const setValidationWarning = useCallback((warning) => {
    const next = {
      ...stateRef.current,
      validationWarning: warning,
      lastSync: Date.now(),
      lastUpdated: Date.now(),
    };
    dispatch({ type: 'SET_VALIDATION_WARNING', payload: warning });
    broadcastGraph(next);
  }, [broadcastGraph]);

  // --- Selectors ---

  const getTrainingData = useCallback(() => {
    // ENHANCED: Correctly handle both edges and links
    let edges = [];
    
    // Check all possible locations for edge data
    if (Array.isArray(state.edges) && state.edges.length > 0) {
      edges = state.edges;
    } else if (Array.isArray(state.links) && state.links.length > 0) {
      edges = state.links;
    } else if (state.graph) {
      if (Array.isArray(state.graph.edges) && state.graph.edges.length > 0) {
        edges = state.graph.edges;
      } else if (Array.isArray(state.graph.links) && state.graph.links.length > 0) {
        edges = state.graph.links;
      }
    }
    
    return {
      nodes: Array.isArray(state.nodes) ? state.nodes : [],
      edges: edges
    };
  }, [state]);

  // --- Memoized Context Values ---

  // --- Persistence Waiter ---
  const waitForPersistence = useCallback(() => {
    // Returns a promise that resolves when the latest state is persisted
    return persistPromiseRef.current || Promise.resolve();
  }, []);

  // --- State Validator ---
  const validateState = useCallback(() => {
    // ENHANCED: Check for both edges and links
    const hasEdges = (Array.isArray(state.edges) && state.edges.length > 0) || 
                   (Array.isArray(state.links) && state.links.length > 0);
    
    return (
      state &&
      Array.isArray(state.nodes) &&
      state.nodes.length >= 0 &&
      hasEdges
    );
  }, [state]);

  const dataValue = useMemo(() => ({
    ...state,
    getTrainingData,
    waitForPersistence,
    validateState,
  }), [state, getTrainingData, waitForPersistence, validateState]);

  const actionsValue = useMemo(() => ({
    setGraph,
    updateNodes,
    updateEdges,
    resetGraph,
    appendTrainingLog,
    clearTrainingLogs,
    setValidationWarning,
    setReactFlowConfig, // Add the new action
  }), [setGraph, updateNodes, updateEdges, resetGraph, appendTrainingLog, clearTrainingLogs, setValidationWarning, setReactFlowConfig]);

  // --- Provide loading fallback until ready ---
  return (
    <GraphDataContext.Provider value={{ ...dataValue, ready }}>
      <GraphDispatchContext.Provider value={actionsValue}>
        {!ready ? (
          <div data-testid="graph-data-loading" style={{ padding: 32, textAlign: 'center' }}>
            Loading graph data...
          </div>
        ) : (
          children
        )}
      </GraphDispatchContext.Provider>
    </GraphDataContext.Provider>
  );
}

// --- Hooks for Consumers ---

export function useGraphData() {
  const context = useContext(GraphDataContext);
  if (context === undefined) {
    throw new Error('useGraphData must be used within a GraphDataProvider');
  }
  
  // ENHANCED: Ensure we return both edges and links
  const edges = Array.isArray(context.edges) ? context.edges : [];
  const links = Array.isArray(context.links) ? context.links : [];
  
  // If one is empty but the other isn't, use the non-empty one
  const finalEdges = edges.length > 0 ? edges : links;
  const finalLinks = links.length > 0 ? links : edges;
  
  return {
    ...context,
    nodes: Array.isArray(context.nodes) ? context.nodes : [],
    edges: finalEdges,
    links: finalLinks,
  };
}

export function useGraphActions() {
  const context = useContext(GraphDispatchContext);
  if (context === undefined) {
    throw new Error('useGraphActions must be used within a GraphDataProvider');
  }
  return context;
}
# Model Training Documentation

## 1. Overview of the Model Training Tab

### Purpose and Functionality

The Model Training tab provides a comprehensive interface for training Graph Neural Network (GNN) models on graph data within the GraphNet application. This feature enables users to apply state-of-the-art deep learning techniques to graph-structured data for tasks such as node classification, graph classification, and link prediction.

The primary purpose of the Model Training tab is to:
- Allow users to select from various GNN architectures
- Configure model hyperparameters
- Train models on graph data
- Visualize training progress in real-time
- Analyze model performance through comprehensive metrics

### Integration with GraphNet

The Model Training feature seamlessly integrates with the existing GraphNet functionality:

1. **Graph Data Flow**: The training tab consumes graph data created and manipulated in the GraphNet tab. It accesses this data through the `useGraph` hook, ensuring that any changes made to the graph in the GraphNet tab are automatically available for training.

2. **Component Architecture**: The training tab follows the same component-based architecture as the rest of the application, with clear separation of concerns:
   - `TrainingTab.js` - Main container component that manages state and coordinates the training process
   - `MetricsVisualizer.js` - Specialized component for visualizing training metrics and results

3. **API Integration**: The training feature communicates with the backend through the application's central API module, maintaining consistency with other data operations in the application.

## 2. Workflow & Data Flow

The GNN Application is designed with a clear workflow that guides users from graph creation to model training. Understanding this data flow is essential for effective use of the application.

### From GraphNet to Training Tab

The application maintains graph state across tabs through the `useGraph` hook, which serves as a central state management system:

1. **Graph Creation in GraphNet Tab**:
   - Users upload CSV data and define nodes and relationships
   - Node attributes and features are configured through the node editing interface
   - Graph structure is visualized and validated
   - When satisfied with the graph structure, users process the data to create the final graph

2. **Data Persistence**:
   - The processed graph data is stored in the `graphData` state within the `useGraph` hook
   - This state is accessible to all components that consume the hook
   - Any updates to the graph in the GraphNet tab are immediately reflected in the Training tab

3. **Data Validation for Training**:
   - Before training can begin, the system validates that the graph contains all required elements
   - Most importantly, nodes must have labels assigned for supervised learning tasks
   - The `isGraphValidForTraining()` method checks for the presence of these labels

### Importance of Node Labels

Labels are a critical requirement for supervised learning with GNNs:

- **Adding Labels**: In the GraphNet tab, labels can be assigned to nodes by:
  - Selecting a label column during graph creation
  - Editing individual nodes to add label attributes
  - Ensuring that node features include at least one categorical feature marked as a label

- **Label Verification**: The Graph Summary section in the Training tab shows:
  - Whether labels are present in the graph
  - The unique label values available for training
  - The distribution of these labels across nodes

- **Impact on Training**: Without labels, supervised learning models cannot be trained properly, and the application will display appropriate validation warnings

### Verifying Graph Readiness for Training

Before starting the training process, users should verify that their graph is ready:

1. **Check the Graph Summary panel** which displays:
   - Total number of nodes and edges
   - Label availability status
   - List of unique label values
   - Last time the graph was processed

2. **Review any validation warnings** that appear in the Training tab:
   - Missing labels warning
   - Data structure issues
   - Potential training problems

3. **Follow the Workflow Guide** which indicates the current stage of the process and what steps remain

## 3. Using the Training Interface

### Step-by-Step Guide

To train a Graph Neural Network model on your data:

1. **Prepare Your Graph**
   - First, create and configure your graph in the GraphNet tab
   - Ensure nodes have appropriate feature attributes
   - Assign class labels to nodes for supervised learning tasks
   - Verify the Graph Summary shows your graph has labels

2. **Configure Your Model**
   - Navigate to the Training tab
   - Review the Graph Summary to confirm your data is ready
   - Select a model architecture from the dropdown menu
   - Adjust hyperparameters as needed for your specific task

3. **Start Training**
   - If validation warnings appear, address the issues before continuing
   - Click the "Start Training" button to begin the training process
   - Monitor the training logs for real-time progress updates
   - Observe the metrics visualizations as they appear during training

4. **Analyze Results**
   - After training completes, review the performance metrics
   - Examine the loss and accuracy curves to understand training dynamics
   - Analyze the class distribution and classification report for detailed insights
   - If needed, adjust hyperparameters and retrain to improve performance

### Understanding the Graph Summary

The Graph Summary section provides essential information about your graph data:

- **Node and Edge Count**: Shows the total number of nodes and edges in your graph
- **Label Status**: Indicates whether your graph has node labels for supervised learning
- **Available Labels**: Lists the unique label values present in your graph
- **Last Updated**: Shows when the graph was last processed or modified

This information helps you understand your data before training and diagnose potential issues if training results are unexpected.

### Interpreting Validation Warnings

The Training tab may display validation warnings when issues are detected:

- **Missing Labels Warning**: Appears when your graph lacks node labels, indicating you need to return to the GraphNet tab to add them
- **Invalid Graph Structure**: Displayed when the graph structure is inconsistent or incomplete
- **Single Class Warning**: Shows when all nodes have the same label, which prevents meaningful classification

These warnings help prevent common issues and guide you toward creating a valid training dataset.

### Workflow Guidance

The Training tab includes a Workflow Guide that:

- Shows your current position in the graph creation and training process
- Highlights completed steps and indicates the next action
- Provides direct links to other tabs when you need to complete prerequisite steps
- Displays context-specific messages based on your graph's current state

When no valid graph is available, the workflow guidance will direct you to first create a graph in the GraphNet tab before attempting to train a model.

### Model Types Explanation

The training interface supports several state-of-the-art GNN architectures:

| Model | Description | Best Used For |
|-------|-------------|--------------|
| **GCN** (Graph Convolutional Network) | Classic architecture that applies convolutional operations to graph-structured data | General-purpose node classification with homogeneous graphs |
| **ResidualGCN** | GCN with residual connections | Deeper networks, helps prevent gradient vanishing |
| **GraphSAGE** | Inductive learning approach that samples and aggregates features from a node's neighborhood | Large graphs with heterogeneous features |
| **GAT** (Graph Attention Network) | Applies attention mechanisms to learn node relationships | When relationships between nodes have varying importance |
| **GIN** (Graph Isomorphism Network) | Designed to maximize expressiveness in distinguishing graph structures | Graph-level classification tasks |
| **ChebConv** (Chebyshev Spectral CNN) | Spectral-based approach using Chebyshev polynomials | Problems that benefit from spectral graph theory |
| **NaiveBayes** | Traditional probabilistic classifier (non-neural baseline) | Baseline comparison or when neural approaches overfit |

### Hyperparameters and Their Effects

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **Hidden Channels** | Number of features in the hidden layers | Higher values increase model capacity but may lead to overfitting |
| **Learning Rate** | Step size for gradient descent optimization | Too high: unstable training; too low: slow convergence |
| **Epochs** | Number of complete passes through the training data | More epochs allow better learning but risk overfitting |
| **Dropout** | Probability of randomly ignoring neurons during training | Reduces overfitting but too high values may impair learning |
| **Attention Heads** (GAT only) | Number of parallel attention mechanisms | More heads capture different aspects of node relationships |
| **Chebyshev Filter Size (K)** (ChebConv only) | Order of the Chebyshev polynomial | Higher orders capture more complex spectral patterns |

### Interpreting Training Logs

The training logs provide real-time feedback on the training process:

```
12:45:23 Starting training for GCN model with 64 hidden channels
12:45:24 Epoch 1/200: Train Loss: 0.6932, Val Loss: 0.6824
12:45:24 Epoch 2/200: Train Loss: 0.6854, Val Loss: 0.6721
...
12:46:15 Epoch 200/200: Train Loss: 0.2145, Val Loss: 0.2356
12:46:16 Training completed in 53.21s
12:46:16 Test Accuracy: 0.8745
```

Key information to look for:
- **Decreasing loss values** indicate the model is learning
- **Gap between train and validation loss** helps identify overfitting
- **Plateauing loss values** suggest learning has stagnated
- **Training time** provides insight into computational efficiency

## 4. Understanding the Metrics

### Explanation of Charts

#### Loss Curve
The loss curve displays how the training and validation loss values change over epochs:
- **Downward trend**: Model is learning successfully
- **Validation loss > Training loss**: Potential overfitting
- **Plateaus**: Learning has stagnated, consider adjusting hyperparameters
- **Spikes**: Potential issues with learning rate or batch size

#### Accuracy Curve
The accuracy curve shows the classification accuracy on training and validation sets:
- **Upward trend**: Model is improving its predictive ability
- **Gap between train/validation**: Indicates degree of generalization
- **Flat sections**: Learning has plateaued, consider early stopping

### Class Distribution
The class distribution chart visualizes how instances are distributed across different classes:
- **Balanced distribution**: Ideal for most models
- **Imbalanced distribution**: May require techniques like class weighting or oversampling
- **Missing classes**: Data quality issue that must be addressed

### Key Performance Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Test Accuracy** | Percentage of correctly classified nodes in the test set | Higher is better, but consider class balance |
| **F1 Score** | Harmonic mean of precision and recall | Better metric for imbalanced classes |
| **Precision** | Ratio of true positives to all positive predictions | Higher values mean fewer false positives |
| **Recall** | Ratio of true positives to all actual positives | Higher values mean fewer false negatives |
| **Training Time** | Total time taken to train the model | Useful for computational efficiency comparisons |

The **Classification Report** provides a detailed breakdown of precision, recall, and F1-score for each class, offering insights into where the model performs well or poorly.

## 5. Best Practices

### Recommended Settings for Different Types of Graphs

| Graph Type | Recommended Model | Suggested Hyperparameters |
|------------|-------------------|--------------------|
| **Small homogeneous graphs** (< 1,000 nodes) | GCN | Hidden channels: 64-128<br>Learning rate: 0.01<br>Epochs: 200<br>Dropout: 0.3 |
| **Large homogeneous graphs** (> 10,000 nodes) | GraphSAGE | Hidden channels: 128-256<br>Learning rate: 0.005<br>Epochs: 100<br>Dropout: 0.5 |
| **Graphs with heterogeneous node types** | GAT | Hidden channels: 64<br>Learning rate: 0.005<br>Epochs: 300<br>Dropout: 0.4<br>Heads: 8 |
| **Graphs with complex structure** | ChebConv | Hidden channels: 64<br>Learning rate: 0.01<br>Epochs: 200<br>Dropout: 0.3<br>K: 3-5 |
| **Highly connected graphs** | ResidualGCN | Hidden channels: 128<br>Learning rate: 0.001<br>Epochs: 500<br>Dropout: 0.2 |

### Tips for Improving Model Performance

1. **Feature Engineering**
   - Include meaningful node features that correlate with the target classes
   - Normalize numerical features to stabilize training
   - Use embedding techniques for categorical or text data

2. **Hyperparameter Optimization**
   - Start with the recommended settings above
   - Increase model capacity (hidden channels) if underfitting
   - Increase dropout if overfitting
   - Decrease learning rate if training is unstable

3. **Training Strategies**
   - Use early stopping to prevent overfitting
   - For imbalanced classes, consider class weights
   - Try different optimizers (Adam is the default but others may work better)
   - Consider learning rate scheduling

4. **Graph Structure**
   - Ensure graph connectivity (isolated nodes may not learn properly)
   - Consider adding self-loops for node feature preservation
   - For sparse graphs, consider higher-order connections

### Troubleshooting Common Issues

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| **Training doesn't start** | No valid graph data | Create graph data in GraphNet tab first |
| **Constant loss values** | Learning rate too low<br>Gradient vanishing | Increase learning rate<br>Try ResidualGCN |
| **Unstable training** | Learning rate too high | Decrease learning rate |
| **Poor test accuracy** | Overfitting<br>Underfitting<br>Class imbalance | Increase dropout<br>Increase model capacity<br>Use class weights |
| **Training too slow** | Complex model<br>Large graph | Reduce hidden channels<br>Try GraphSAGE for sampling |
| **"Single class" error** | No variation in labels | Ensure graph has multiple classes |

## 6. Technical Details

### How the Feature Works Under the Hood

The Model Training feature is implemented as a React component that communicates with a PyTorch Geometric backend for GNN training:

1. **Frontend Architecture**:
   - `TrainingTab.js`: Main component that handles user interaction, configuration, and displays results
   - `MetricsVisualizer.js`: Component for visualizing training metrics using Chart.js
   - `api.js`: Module containing API functions for communicating with the backend

2. **Data Flow**:
   ```
   User Input → TrainingTab → API Request → Backend Training →
   Streaming Response → Real-time Updates → Results Display
   ```

3. **Training Process**:
   - User configures and initiates training 
   - Graph data and configuration are sent to the backend
   - Backend constructs PyTorch Geometric data objects
   - Selected GNN model is instantiated with specified parameters
   - Training loop runs with periodic updates sent to frontend
   - After completion, final metrics are computed and sent to frontend

4. **Real-time Updates**:
   - The backend uses a streaming response to send progress updates
   - Frontend processes these updates to display logs and intermediate metrics
   - This approach provides immediate feedback during long training processes

### Cross-Tab State Management with useGraph

The `useGraph` hook serves as the central state management system for graph data:

1. **Shared State**:
   - The hook maintains a `graphData` state that contains the processed graph structure
   - This state is made available to both GraphNet and Training tabs
   - Changes in one tab are immediately reflected in the other, ensuring consistency

2. **Graph Validation**:
   - The hook provides the `isGraphValidForTraining()` method that checks:
     - If a valid graph structure exists
     - If nodes have label attributes assigned
     - If the graph meets minimal requirements for training
   - This validation is used to enable/disable training controls and display appropriate warnings

3. **Graph Statistics**:
   - The `getGraphStats()` method provides detailed information about the graph:
     - Node and edge counts
     - Label presence and unique label values
     - Last processing timestamp
   - These statistics are displayed in the Graph Summary panel

4. **Event Handling**:
   - The hook centralizes graph manipulation events
   - Updates are propagated to all consuming components
   - This ensures that any changes made in the GraphNet tab (e.g., adding labels) are immediately available for training

### Integration with TorchGeometricGraphBuilder

The feature leverages the `TorchGeometricGraphBuilder` to convert generic graph data into the specialized data structures required by PyTorch Geometric:

1. **Data Conversion**:
   - Graph nodes and links are converted to PyTorch tensors
   - Node features are extracted and normalized
   - Edge indices are constructed in the COO format required by PyG
   - Labels are encoded as class indices

2. **Data Splitting**:
   - The graph is split into training, validation, and test sets
   - Masks are created to identify nodes in each set
   - This enables proper evaluation of model performance

3. **Model Construction**:
   - Based on the selected architecture, appropriate PyG models are instantiated
   - Model parameters are configured based on user settings
   - Models inherit from PyTorch's `nn.Module` for compatibility

## SAPPO Patterns and Problems Considered

### Patterns Applied

1. **:DelegatedResponsibility**
   - TrainingTab delegates visualization to MetricsVisualizer
   - API module handles communication details, allowing components to focus on UI
   - Each model type encapsulates its specific implementation details

2. **:ComposableInterface**
   - Training interface integrates seamlessly with the GraphNet interface
   - Components use consistent props and state management
   - Metrics visualizer accepts standardized data format regardless of model type

3. **:EventDrivenUpdates**
   - Training process uses streaming responses for real-time updates
   - UI reacts to training events (epochs, completion, errors)
   - State changes trigger appropriate UI updates

4. **:IsolatedDataTransformation**
   - Graph data transformation happens in TorchGeometricGraphBuilder
   - Training configuration is sanitized before API calls
   - Metrics data is processed for visualization separately from display logic

5. **:SharedStateManagement**
   - useGraph hook provides controlled access to graph state across tabs
   - Graph validation logic is centralized in the hook
   - Components react to state changes in a predictable manner

### Problems Addressed

1. **:DataInconsistency**
   - Validation checks ensure graph data has required structure
   - Error handling for missing labels or invalid feature formats
   - Guards against single-class datasets that can't be meaningfully trained

2. **:UIPerformanceIssue**
   - Auto-scrolling logs implement useRef to avoid unnecessary rerenders
   - Chart rendering optimization with memoization
   - Streaming response processing to handle large training outputs

3. **:IntegrationGap**
   - Consistent data formats between frontend and backend
   - Proper error propagation from backend to frontend
   - Graceful handling of training cancellation

4. **:TestingBlindSpot**
   - Comprehensive test suite covering component rendering, interactions, and edge cases
   - Mock implementations to test API integration without backend
   - Backend tests to ensure endpoint behavior

5. **:WorkflowDisruption**
   - Clear guidance when prerequisites aren't met
   - Contextual warnings based on current graph state
   - Intuitive progression through the application's workflow

## Dual Testing Strategy

The Model Training feature implements a comprehensive dual testing approach:

### Cumulative Tests

Cumulative tests ensure that the component works correctly in isolation and integrates properly with the rest of the application:

1. **Component Tests**:
   - `TrainingTab.test.js` verifies that the training tab renders correctly, handles user input, and manages training state
   - `MetricsVisualizer.test.js` confirms that visualizations correctly display various metrics scenarios

2. **API Tests**:
   - `api.test.js` validates that API calls are formatted correctly
   - Tests for error handling and response processing

3. **Integration Tests**:
   - Tests that ensure the training tab properly accesses graph data from the shared context
   - Verification of component interaction (e.g., TrainingTab using MetricsVisualizer correctly)

4. **Cross-Tab Communication Tests**:
   - `useGraph.test.js` validates that state changes propagate correctly between tabs
   - Tests for proper validation of graph data for training
   - Verification that graph statistics are calculated correctly

### Recursive Tests

Recursive tests focus on specific behaviors and edge cases to ensure robust implementation:

1. **Behavioral Tests**:
   - Tests for UI behavior when models are changed
   - Tests for disabling controls appropriately based on state
   - Validation of form input handling

2. **Edge Cases**:
   - Tests with empty metrics data
   - Tests with single-epoch training data
   - Tests with large numbers of epochs
   - Tests with various graph structures and label distributions

3. **Error Handling**:
   - Tests for missing graph data
   - Tests for API error scenarios
   - Tests for recovery from training interruption
   - Tests for handling invalid label data

4. **Workflow Validation**:
   - Tests that verify appropriate guidance is shown based on graph state
   - Tests for correct warnings when prerequisites aren't met
   - Tests that confirm proper state transitions during the workflow

The testing strategy ensures that:
1. Components behave correctly individually (unit tests)
2. Components work together properly (integration tests)
3. The feature handles edge cases gracefully (robustness tests)
4. Changes don't break existing functionality (regression tests)

## Conclusion

The Model Training feature provides a powerful yet accessible interface for applying Graph Neural Networks to your data. By following the guidelines in this documentation, you can effectively train models, interpret results, and iteratively improve performance.

For best results:
1. Start with recommended settings for your graph type
2. Monitor training metrics to identify issues early
3. Adjust hyperparameters based on observed performance
4. Use the visualization tools to gain deeper insights into model behavior

Happy model training!
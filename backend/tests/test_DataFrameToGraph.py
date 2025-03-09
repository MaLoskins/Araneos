#!/usr/bin/env python
"""
test_DataFrameToGraph.py

This script provides robust tests for the DataFrameToGraph module.
It covers:
  - Basic graph creation from a valid DataFrame.
  - Handling of duplicate nodes.
  - Missing node IDs (verifying logged warnings).
  - Invalid configuration handling.
  
Note:
Due to row‐by‐row processing (nodes are added before edges are processed),
some edges may be skipped if the target node isn’t yet present.
"""

import io
import logging
import pandas as pd
from DataFrameToGraph import DataFrameToGraph

def test_basic_graph_creation():
    # Create a sample DataFrame where the order causes only one valid edge to be created.
    data = {
        "id": ["1", "2", "3"],
        "source": ["1", "2", "3"],
        "target": ["2", "3", "1"],
        "text_embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    }
    df = pd.DataFrame(data)
    
    config = {
        "nodes": [
            {"id": "id", "type": "person"}
        ],
        "relationships": [
            {"source": "source", "target": "target", "type": "friend"}
        ]
    }
    
    converter = DataFrameToGraph(df, config, graph_type="directed")
    graph = converter.get_graph()
    
    # In this row order, only the edge in the last row is added.
    assert graph.number_of_nodes() == 3, "Expected 3 nodes in the graph"
    assert graph.number_of_edges() == 1, f"Expected 1 edge in the graph but got {graph.number_of_edges()}"
    print("test_basic_graph_creation passed.")

def test_duplicate_nodes():
    # DataFrame with duplicate node IDs.
    data = {
        "id": ["1", "1", "2"],
        "source": ["1", "1", "1"],
        "target": ["2", "2", "1"],
        "text_embedding": [[0.1, 0.2], [0.1, 0.2], [0.3, 0.4]]
    }
    df = pd.DataFrame(data)
    config = {
        "nodes": [
            {"id": "id", "type": "entity"}
        ],
        "relationships": [
            {"source": "source", "target": "target", "type": "link"}
        ]
    }
    converter = DataFrameToGraph(df, config, graph_type="undirected")
    graph = converter.get_graph()
    # Only unique nodes should be added.
    assert graph.number_of_nodes() == 2, "Expected 2 unique nodes in the graph"
    print("test_duplicate_nodes passed.")

def test_missing_node_id():
    # DataFrame where one row has an empty node id.
    data = {
        "id": ["1", "", "3"],
        "source": ["1", "3", "3"],
        "target": ["3", "1", "1"],
        "text_embedding": [[0.1, 0.2], [0.5, 0.6], [0.7, 0.8]]
    }
    df = pd.DataFrame(data)
    config = {
        "nodes": [
            {"id": "id", "type": "node"}
        ],
        "relationships": [
            {"source": "source", "target": "target", "type": "connects"}
        ]
    }
    
    # Set up a stream to capture log output.
    log_stream = io.StringIO()
    logger = logging.getLogger()  # Using the root logger (or you can get the module-specific one)
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)
    
    # Process the DataFrame.
    converter = DataFrameToGraph(df, config, graph_type="directed")
    graph = converter.get_graph()
    
    # Remove our temporary handler.
    logger.removeHandler(stream_handler)
    log_contents = log_stream.getvalue()
    log_stream.close()
    
    # Assert that the log contains a message about missing or empty node ID.
    assert "Missing or empty node ID" in log_contents, "Expected log warning for missing node ID"
    
    # Only 2 nodes should be added.
    assert graph.number_of_nodes() == 2, "Expected 2 nodes due to missing id in one row"
    print("test_missing_node_id passed.")

def test_invalid_config():
    # Invalid configuration (missing required "relationships" key).
    data = {
        "id": ["1", "2"],
        "source": ["1", "2"],
        "target": ["2", "1"],
    }
    df = pd.DataFrame(data)
    invalid_config = {
        "nodes": [
            {"id": "id"}
        ]
        # "relationships" key is missing.
    }
    try:
        _ = DataFrameToGraph(df, invalid_config, graph_type="directed")
    except KeyError as e:
        assert "relationships" in str(e), "Expected KeyError for missing 'relationships'"
        print("test_invalid_config passed.")
    else:
        raise AssertionError("Expected KeyError due to invalid configuration.")

def run_all_tests():
    test_basic_graph_creation()
    test_duplicate_nodes()
    test_missing_node_id()
    test_invalid_config()
    print("All DataFrameToGraph tests passed.")

if __name__ == "__main__":
    run_all_tests()




#----------------------------------------------------#
#----------- EXAMPLE FORMATTING ---------------------#
#----------------------------------------------------#

"""
## 1. DataFrameToGraph

### JSON Configuration Template

This JSON structure defines how to extract nodes and relationships from a DataFrame:

```json
{
  "nodes": [
    {
      "id": "id",          
      "type": "person"       
    }
  ],
  "relationships": [
    {
      "source": "source",  
      "target": "target",  
      "type": "friend"      
    }
  ]
}
```

**Field Limitations:**

- **Nodes:**
  - `"id"`: Must match a DataFrame column name containing non-empty unique identifiers.
  - `"type"`: Optional string; if omitted, defaults to `"default"`.

- **Relationships:**
  - `"source"` and `"target"`: Must match DataFrame column names. If a row’s source or target is missing (or empty), that edge is skipped.
  - `"type"`: Optional string; if omitted, defaults to `"default"`.

### Example Code

Below is an example of how to instantiate and call the `DataFrameToGraph` module:

```python
import pandas as pd
from DataFrameToGraph import DataFrameToGraph

# Create an example DataFrame
df = pd.DataFrame({
    "id": ["1", "2", "3"],
    "source": ["1", "2", "3"],
    "target": ["2", "3", "1"],
    "text_embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
})

# JSON configuration for DataFrameToGraph (could be read from a file)
config = {
    "nodes": [
        {"id": "id", "type": "person"}
    ],
    "relationships": [
        {"source": "source", "target": "target", "type": "friend"}
    ]
}

# Instantiate DataFrameToGraph (graph_type: "directed" or "undirected")
graph_builder = DataFrameToGraph(df, config, graph_type="directed")
graph = graph_builder.get_graph()

# Use the generated graph (e.g., print nodes and edges)
print("Nodes:")
print(graph.nodes(data=True))
print("\nEdges:")
print(graph.edges(data=True, keys=True))
```
"""
#----------------------------------------------------#
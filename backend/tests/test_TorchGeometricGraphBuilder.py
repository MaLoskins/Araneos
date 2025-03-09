#!/usr/bin/env python
"""
test_TorchGeometricGraphBuilder.py

This script provides robust tests for the TorchGeometricGraphBuilder module and its helper functions.
Tests include:
  - Building a PyG Data object from valid JSON.
  - Handling nodes with missing features.
  - Adding structural features and verifying that original feature columns remain normalized.
  - Reducing feature dimensions via PCA.
  - Splitting data into train/val/test masks.
"""

import numpy as np
import torch
from TorchGeometricGraphBuilder import (
    TorchGeometricGraphBuilder,
    add_structural_features,
    reduce_feature_dimensions,
    split_data
)
from torch_geometric.data import Data

def test_basic_graph_build():
    data_json = {
        "nodes": [
            {
                "id": "1",
                "features": {
                    "user_followers_count_feature": "10",
                    "text_embedding": [0.1, 0.2, 0.3],
                    "label": "A"
                }
            },
            {
                "id": "2",
                "features": {
                    "user_followers_count_feature": "20",
                    "text_embedding": [0.4, 0.5, 0.6],
                    "label": "B"
                }
            },
            {
                "id": "3",
                "features": {
                    "user_followers_count_feature": "15",
                    "text_embedding": [0.7, 0.8, 0.9],
                    "label": "A"
                }
            }
        ],
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}},
            {"source": {"id": "2"}, "target": {"id": "3"}},
            {"source": {"id": "3"}, "target": {"id": "1"}}
        ]
    }
    builder = TorchGeometricGraphBuilder(data_json)
    data = builder.build_data()
    
    assert data.num_nodes == 3, "Expected 3 nodes in the PyG Data object"
    assert data.edge_index.shape[0] == 2, "edge_index should have shape (2, num_edges)"
    print("test_basic_graph_build passed.")

def test_dummy_feature_handling():
    data_json = {
        "nodes": [
            {"id": "1", "features": {}},
            {"id": "2", "features": {"user_followers_count_feature": "5"}}
        ],
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}}
        ]
    }
    builder = TorchGeometricGraphBuilder(data_json)
    data = builder.build_data()
    assert data.x.shape[1] >= 1, "Expected at least one feature dimension even if none provided"
    print("test_dummy_feature_handling passed.")

def test_structural_features_and_normalization():
    data_json = {
        "nodes": [
            {"id": "1", "features": {"user_followers_count_feature": "10", "text_embedding": [0.1, 0.2]}},
            {"id": "2", "features": {"user_followers_count_feature": "20", "text_embedding": [0.3, 0.4]}},
            {"id": "3", "features": {"user_followers_count_feature": "30", "text_embedding": [0.5, 0.6]}}
        ],
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}},
            {"source": {"id": "2"}, "target": {"id": "3"}}
        ]
    }
    builder = TorchGeometricGraphBuilder(data_json)
    data = builder.build_data()
    
    original_feature_dim = data.x.shape[1]
    original_x = data.x.clone()  # This should be normalized per build_data.
    
    add_structural_features(data)
    assert data.x.shape[1] == original_feature_dim + 1, "Expected one additional column for node degree"
    
    # Check normalization on the original features.
    normalized_part = data.x[:, :original_feature_dim]
    mean = normalized_part.mean(dim=0)
    std = normalized_part.std(dim=0)
    # Tolerance set to 1e-1.
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1), "Original features not normalized to zero mean"
    assert torch.allclose(std, torch.ones_like(std), atol=1e-1), "Original features not normalized to unit variance"
    
    degree_col = data.x[:, -1]
    assert torch.all(degree_col >= 0), "Degree column should be non-negative"
    print("test_structural_features_and_normalization passed.")

def test_reduce_feature_dimensions():
    x = torch.randn(10, 5)
    x_reduced = reduce_feature_dimensions(x, n_components=2)
    assert x_reduced.shape[1] == 2, "Expected reduced feature dimensions to be 2"
    print("test_reduce_feature_dimensions passed.")

def test_split_data_masks():
    x = torch.randn(10, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    data = Data(x=x, y=y)
    data.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    
    data = split_data(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    total_mask = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
    assert total_mask == x.shape[0], "Masks do not cover all nodes"
    print("test_split_data_masks passed.")

def run_all_tests():
    test_basic_graph_build()
    test_dummy_feature_handling()
    test_structural_features_and_normalization()
    test_reduce_feature_dimensions()
    test_split_data_masks()
    print("All TorchGeometricGraphBuilder tests passed.")

if __name__ == "__main__":
    run_all_tests()


#----------------------------------------------------#
#----------- EXAMPLE FORMATTING ---------------------#
#----------------------------------------------------#

"""
## 3. TorchGeometricGraphBuilder

### JSON Configuration Template

This JSON structure follows a node–link schema:

```json
{
  "nodes": [
    {
      "id": "1",
      "features": {
        "user_followers_count_feature": "10",
        "text_embedding": [0.1, 0.2, 0.3],
        "label": "A"
      }
    },
    {
      "id": "2",
      "features": {
        "user_followers_count_feature": "20",
        "text_embedding": [0.4, 0.5, 0.6],
        "label": "B"
      }
    },
    {
      "id": "3",
      "features": {
        "user_followers_count_feature": "15",
        "text_embedding": [0.7, 0.8, 0.9],
        "label": "A"
      }
    }
  ],
  "links": [
    {
      "source": {"id": "1"},
      "target": {"id": "2"}
    },
    {
      "source": {"id": "2"},
      "target": {"id": "3"}
    },
    {
      "source": {"id": "3"},
      "target": {"id": "1"}
    }
  ]
}
```

**Field Limitations:**

- **Nodes:**
  - `"id"`: Must be a non-empty, unique string.
  - `"features"`:  
    - For numeric features (e.g., `"user_followers_count_feature"`), the value should be convertible to a float.
    - For embeddings, it must be an array of numbers.
  - `"label"`: Optional; if provided, used for class mapping.

- **Links:**
  - Both `"source"` and `"target"` must be dictionaries containing an `"id"` that corresponds to an existing node.

### Example Code

Below is an example of how to instantiate and call the `TorchGeometricGraphBuilder` module:

```python
import json
from TorchGeometricGraphBuilder import TorchGeometricGraphBuilder

# Define the JSON configuration as a Python dictionary (or load from a JSON file)
data_json = {
    "nodes": [
        {
            "id": "1",
            "features": {
                "user_followers_count_feature": "10",
                "text_embedding": [0.1, 0.2, 0.3],
                "label": "A"
            }
        },
        {
            "id": "2",
            "features": {
                "user_followers_count_feature": "20",
                "text_embedding": [0.4, 0.5, 0.6],
                "label": "B"
            }
        },
        {
            "id": "3",
            "features": {
                "user_followers_count_feature": "15",
                "text_embedding": [0.7, 0.8, 0.9],
                "label": "A"
            }
        }
    ],
    "links": [
        {"source": {"id": "1"}, "target": {"id": "2"}},
        {"source": {"id": "2"}, "target": {"id": "3"}},
        {"source": {"id": "3"}, "target": {"id": "1"}}
    ]
}

# Instantiate TorchGeometricGraphBuilder with the JSON configuration
builder = TorchGeometricGraphBuilder(data_json)
data = builder.build_data()

# Display the PyTorch Geometric Data object
print("PyTorch Geometric Data Object:")
print(data)
```

"""
#-----------------------------------------------------#
#!/usr/bin/env python
"""
test_FeatureSpaceCreator.py

This script provides robust tests for the FeatureSpaceCreator module.
Tests include:
  - Processing a DataFrame with both text and numeric features.
  - Handling a missing text column.
  - Verifying numeric standardization.
Note:
  - PCA target dimension is set to 2 (instead of 5) to satisfy the PCA constraint given the small sample size.
"""

import os
import pandas as pd
import numpy as np
from FeatureSpaceCreator import FeatureSpaceCreator

def test_basic_feature_space():
    # Create a sample DataFrame with text and numeric columns.
    data = {
        "text": [
            "This is a test sentence.",
            "Another test sentence for embedding.",
            "Yet another example."
        ],
        "num": [1.0, 2.0, 3.0]
    }
    df = pd.DataFrame(data)

    config = {
        "features": [
            {
                "column_name": "text",
                "type": "text",
                "embedding_method": "bert",
                "dim_reduction": {
                    "method": "pca",
                    "target_dim": 2
                },
                "additional_params": {
                    "bert_batch_size": 1,
                    "bert_model_name": "bert-base-uncased"
                }
            },
            {
                "column_name": "num",
                "type": "numeric",
                "processing": "standardize",
                "projection": {"method": "none", "target_dim": 1}
            }
        ]
    }

    log_file = "logs/test_feature_space_creator.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fsc = FeatureSpaceCreator(config=config, device="cpu", log_file=log_file)
    feature_space = fsc.process(df)
    
    # Check that expected columns are present.
    assert "text_embedding" in feature_space.columns, "Expected 'text_embedding' column in feature space."
    assert "num_feature" in feature_space.columns, "Expected 'num_feature' column in feature space."
    
    sample_embedding = feature_space["text_embedding"].iloc[0]
    assert isinstance(sample_embedding, (list, np.ndarray)), "Text embedding should be a list or numpy array."
    
    sample_numeric = feature_space["num_feature"].iloc[0]
    if not np.isscalar(sample_numeric):
        sample_numeric = np.array(sample_numeric)
        assert sample_numeric.size >= 1, "Numeric feature not processed correctly."
    print("test_basic_feature_space passed.")

def test_missing_text_column():
    # Create a DataFrame missing the expected text column.
    data = {"num": [1.0, 2.0, 3.0]}
    df = pd.DataFrame(data)
    config = {
        "features": [
            {
                "column_name": "text",
                "type": "text",
                "embedding_method": "bert",
                "additional_params": {
                    "bert_batch_size": 1,
                    "bert_model_name": "bert-base-uncased"
                }
            }
        ]
    }
    fsc = FeatureSpaceCreator(config=config, device="cpu", log_file="logs/test_feature_space_creator.log")
    try:
        _ = fsc.process(df)
    except ValueError as e:
        assert "Text column 'text' not found" in str(e), "Expected ValueError for missing text column."
        print("test_missing_text_column passed.")
    else:
        raise AssertionError("Expected ValueError due to missing text column.")

def test_numeric_standardization():
    # Create a DataFrame with a numeric column.
    data = {
        "text": ["Dummy text", "Dummy text", "Dummy text"],
        "num": [10, 20, 30]
    }
    df = pd.DataFrame(data)
    config = {
        "features": [
            {
                "column_name": "num",
                "type": "numeric",
                "processing": "standardize",
                "projection": {"method": "none", "target_dim": 1}
            }
        ]
    }
    fsc = FeatureSpaceCreator(config=config, device="cpu", log_file="logs/test_feature_space_creator.log")
    feature_space = fsc.process(df)
    
    num_features = np.array(feature_space["num_feature"].tolist())
    mean_val = np.mean(num_features)
    std_val = np.std(num_features)
    assert np.abs(mean_val) < 1e-6, f"Numeric feature mean ({mean_val}) not zero after standardization."
    assert np.abs(std_val - 1) < 1e-6, f"Numeric feature std ({std_val}) not one after standardization."
    print("test_numeric_standardization passed.")

def run_all_tests():
    test_basic_feature_space()
    test_missing_text_column()
    test_numeric_standardization()
    print("All FeatureSpaceCreator tests passed.")

if __name__ == "__main__":
    run_all_tests()


#----------------------------------------------------#
#----------- EXAMPLE FORMATTING ---------------------#
#----------------------------------------------------#

"""
## 2. FeatureSpaceCreator

### JSON Configuration Template

This configuration tells the module how to process text and numeric columns:

```json
{
  "features": [
    {
      "column_name": "text",
      "type": "text",
      "embedding_method": "bert",
      "dim_reduction": {
        "method": "pca",
        "target_dim": 2
      },
      "additional_params": {
        "bert_batch_size": 1,
        "bert_model_name": "bert-base-uncased"
      }
    },
    {
      "column_name": "num",
      "type": "numeric",
      "processing": "standardize",
      "projection": {
        "method": "none",
        "target_dim": 1
      }
    }
  ]
}
```

**Field Limitations:**

- **Text Features:**
  - `"column_name"`: Must exist in the DataFrame.
  - `"embedding_method"`: Must be one of `"bert"`, `"glove"`, or `"word2vec"`.
  - `"dim_reduction"`:  
    - `"method"`: Either `"pca"`, `"umap"`, or `"none"`.
    - `"target_dim"`: A positive integer that does not exceed the minimum of (n_samples, original embedding dimension).

- **Numeric Features:**
  - `"processing"`: Options are `"standardize"`, `"normalize"`, or `"none"`.
  - `"projection"`:  
    - If using `"linear"`, `"target_dim"` must be ≥ 1.

### Example Code

Below is an example of how to instantiate and call the `FeatureSpaceCreator` module:

```python
import pandas as pd
from FeatureSpaceCreator import FeatureSpaceCreator

# Create an example DataFrame with text and numeric data
df = pd.DataFrame({
    "text": [
        "This is a test sentence.",
        "Another sentence for embedding.",
        "Yet another example."
    ],
    "num": [1.0, 2.0, 3.0]
})

# JSON configuration for FeatureSpaceCreator (could be read from a file)
config = {
    "features": [
        {
            "column_name": "text",
            "type": "text",
            "embedding_method": "bert",
            "dim_reduction": {
                "method": "pca",
                "target_dim": 2
            },
            "additional_params": {
                "bert_batch_size": 1,
                "bert_model_name": "bert-base-uncased"
            }
        },
        {
            "column_name": "num",
            "type": "numeric",
            "processing": "standardize",
            "projection": {"method": "none", "target_dim": 1}
        }
    ]
}

# Instantiate FeatureSpaceCreator (specify device and log file)
fsc = FeatureSpaceCreator(config=config, device="cpu", log_file="logs/feature_space_creator.log")
feature_space = fsc.process(df)

# Display the generated feature space DataFrame
print("Feature Space DataFrame:")
print(feature_space.head())
```
"""
#-----------------------------------------------------#
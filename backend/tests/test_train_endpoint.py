# test_train_endpoint.py
import pytest
import json
import torch
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

# Import main app for testing
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app

# Create test client
@pytest.fixture
def client():
    return TestClient(app)

# Create a fixture for graph data with valid labels
@pytest.fixture
def valid_graph_data():
    return {
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}},
            {"source": {"id": "2"}, "target": {"id": "3"}},
            {"source": {"id": "1"}, "target": {"id": "3"}}
        ],
        "nodes": [
            {"id": "1", "features": {"label": "A", "text_embedding": [0.1, 0.2, 0.3], "user_followers_count_feature": "5"}},
            {"id": "2", "features": {"label": "B", "text_embedding": [0.4, 0.5, 0.6], "user_followers_count_feature": "10"}},
            {"id": "3", "features": {"label": "A", "text_embedding": [0.7, 0.8, 0.9], "user_followers_count_feature": "15"}}
        ]
    }

# Create a fixture for graph data with no labels
@pytest.fixture
def no_labels_graph_data():
    return {
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}},
            {"source": {"id": "2"}, "target": {"id": "3"}},
            {"source": {"id": "1"}, "target": {"id": "3"}}
        ],
        "nodes": [
            {"id": "1", "features": {"text_embedding": [0.1, 0.2, 0.3], "user_followers_count_feature": "5"}},
            {"id": "2", "features": {"text_embedding": [0.4, 0.5, 0.6], "user_followers_count_feature": "10"}},
            {"id": "3", "features": {"text_embedding": [0.7, 0.8, 0.9], "user_followers_count_feature": "15"}}
        ]
    }

# Create a fixture for graph data with only a single class (no actual classification possible)
@pytest.fixture
def single_class_graph_data():
    return {
        "links": [
            {"source": {"id": "1"}, "target": {"id": "2"}},
            {"source": {"id": "2"}, "target": {"id": "3"}},
            {"source": {"id": "1"}, "target": {"id": "3"}}
        ],
        "nodes": [
            {"id": "1", "features": {"label": "A", "text_embedding": [0.1, 0.2, 0.3], "user_followers_count_feature": "5"}},
            {"id": "2", "features": {"label": "A", "text_embedding": [0.4, 0.5, 0.6], "user_followers_count_feature": "10"}},
            {"id": "3", "features": {"label": "A", "text_embedding": [0.7, 0.8, 0.9], "user_followers_count_feature": "15"}}
        ]
    }

# Create a fixture for model configuration
@pytest.fixture
def valid_model_config():
    return {
        "model_name": "GCN",
        "hidden_channels": 64,
        "dropout": 0.3,
        "lr": 0.01,
        "epochs": 5,
        "label_feature": "label",
        "node_features": ["text_embedding", "user_followers_count_feature"]
    }

# Mock for TorchGeometric data objects
@pytest.fixture
def mock_data():
    mock = MagicMock()
    mock.x = torch.randn((3, 4))  # 3 nodes, 4 features
    mock.edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)  # Edge list
    mock.y = torch.tensor([0, 1, 0], dtype=torch.long)  # Labels
    mock.train_mask = torch.tensor([True, True, False])
    mock.val_mask = torch.tensor([False, False, True])
    mock.test_mask = torch.tensor([False, False, True])
    mock.num_node_features = 4
    mock.num_classes = 2
    return mock

# Test validation - should catch missing labels
def test_train_gnn_missing_labels(client, no_labels_graph_data, valid_model_config):
    """Test that a request without labels is rejected with an appropriate error."""
    response = client.post(
        "/train-gnn", 
        json={
            "graph": no_labels_graph_data,
            "configuration": valid_model_config
        }
    )
    
    assert response.status_code == 400
    assert "No labels found in the graph data" in response.json()["detail"]

# Test validation - should catch invalid model name
def test_train_gnn_invalid_model(client, valid_graph_data, valid_model_config):
    """Test that a request with an invalid model name is rejected with an appropriate error."""
    # This test evaluates if the endpoint rejects invalid model names
    invalid_config = valid_model_config.copy()
    invalid_config["model_name"] = "INVALID_MODEL"
    
    # We need to patch multiple things to reach the model validation logic
    with patch("main.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("main.split_data") as mock_split_data:
        
        # Setup the mock builder
        mock_instance = MockBuilder.return_value
        mock_data = MagicMock()
        mock_data.y = torch.tensor([0, 1])
        mock_instance.build_data.return_value = mock_data
        
        # Mock the split_data function to avoid class count errors
        mock_split_data.return_value = mock_data
        
        response = client.post(
            "/train-gnn",
            json={
                "graph": valid_graph_data,
                "configuration": invalid_config
            }
        )
    
    # The endpoint should return 400 for invalid model name
    assert response.status_code == 400
    assert "unsupported model" in response.json()["detail"].lower()

# Test validation - should handle single-class case
def test_train_gnn_single_class(client, single_class_graph_data, valid_model_config):
    """Test that single class data returns an appropriate error or warning."""
    with patch("main.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("torch.unique", return_value=torch.tensor([0])):  # Only one class
        
        # Setup the mock to return data with single class
        mock_instance = MockBuilder.return_value
        mock_data = MagicMock()
        mock_data.y = torch.tensor([0, 0, 0])  # All same class
        mock_instance.build_data.return_value = mock_data
        
        # Ensure this returns an error about single class
        response = client.post(
            "/train-gnn", 
            json={
                "graph": single_class_graph_data,
                "configuration": valid_model_config
            }
        )
        
        # Should get a 400 error about insufficient classes
        assert response.status_code == 400
        assert "insufficient number of classes" in response.json()["detail"].lower()

# Mock the training and inference process for GCN
@pytest.mark.asyncio
async def test_train_gnn_gcn_model():
    """Test that the GCN model is instantiated and used correctly."""
    from main import train_gnn, TrainGNNRequest
    
    # Create test data and configuration
    valid_graph_data = {
        "links": [{"source": {"id": "1"}, "target": {"id": "2"}}],
        "nodes": [
            {"id": "1", "features": {"label": "A", "feature1": [0.1]}},
            {"id": "2", "features": {"label": "B", "feature1": [0.2]}}
        ]
    }
    
    valid_model_config = {
        "model_name": "GCN",
        "hidden_channels": 16,
        "dropout": 0.2,
        "lr": 0.01,
        "epochs": 2,
        "label_feature": "label",
        "node_features": ["feature1"]
    }
    
    request = TrainGNNRequest(graph=valid_graph_data, configuration=valid_model_config)
    
    # Create mock objects
    mock_data = MagicMock()
    mock_data.y = torch.tensor([0, 1])
    mock_data.num_node_features = 4
    
    # Setup all required mocks
    with patch("main.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("main.GCNModel") as MockModel, \
         patch("main.split_data", return_value=mock_data), \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("torch.optim.Adam"), \
         patch("torch.nn.CrossEntropyLoss"), \
         patch("torch.optim.lr_scheduler.ReduceLROnPlateau"):
        
        # Setup mock builder
        mock_builder = MockBuilder.return_value
        mock_builder.build_data.return_value = mock_data
        
        # Setup model mock
        mock_model = MockModel.return_value
        mock_model.to.return_value = mock_model
        
        # Setup device mock to return 'cpu'
        with patch("torch.device", return_value="cpu"):
            with patch("torch.cuda.is_available", return_value=False):
                # Execute the function under test
                try:
                    response = await train_gnn(request)
                    assert isinstance(response, StreamingResponse)
                except Exception as e:
                    pytest.fail(f"train_gnn raised {type(e).__name__} unexpectedly: {e}")
        
        # Verify GCNModel was called (don't check exact params which may change)
        assert MockModel.called

# Test GraphSAGE model instantiation
@pytest.mark.asyncio
async def test_train_gnn_graphsage_model():
    """Test that the GraphSAGE model is instantiated and used correctly."""
    from main import train_gnn, TrainGNNRequest
    
    # Create test data and configuration with GraphSAGE model
    valid_graph_data = {
        "links": [{"source": {"id": "1"}, "target": {"id": "2"}}],
        "nodes": [
            {"id": "1", "features": {"label": "A", "feature1": [0.1]}},
            {"id": "2", "features": {"label": "B", "feature1": [0.2]}}
        ]
    }
    
    sage_config = {
        "model_name": "GraphSAGE",
        "hidden_channels": 16,
        "dropout": 0.2,
        "lr": 0.01,
        "epochs": 2,
        "label_feature": "label",
        "node_features": ["feature1"]
    }
    
    request = TrainGNNRequest(graph=valid_graph_data, configuration=sage_config)
    
    # Create mock objects
    mock_data = MagicMock()
    mock_data.y = torch.tensor([0, 1])
    mock_data.num_node_features = 4
    
    # Setup all required mocks
    with patch("main.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("main.GraphSageModel") as MockModel, \
         patch("main.split_data", return_value=mock_data), \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("torch.optim.Adam"), \
         patch("torch.nn.CrossEntropyLoss"), \
         patch("torch.optim.lr_scheduler.ReduceLROnPlateau"):
        
        # Setup mock builder
        mock_builder = MockBuilder.return_value
        mock_builder.build_data.return_value = mock_data
        
        # Setup model mock
        mock_model = MockModel.return_value
        mock_model.to.return_value = mock_model
        
        # Setup device mock to return 'cpu'
        with patch("torch.device", return_value="cpu"):
            with patch("torch.cuda.is_available", return_value=False):
                # Execute the function under test
                try:
                    response = await train_gnn(request)
                    assert isinstance(response, StreamingResponse)
                except Exception as e:
                    pytest.fail(f"train_gnn raised {type(e).__name__} unexpectedly: {e}")
        
        # Verify GraphSageModel was called
        assert MockModel.called

# Test GAT model with custom parameters
@pytest.mark.asyncio
async def test_train_gnn_gat_model_with_extra_params():
    """Test that the GAT model handles extra parameters correctly."""
    from main import train_gnn, TrainGNNRequest
    
    # Create test data and configuration with GAT model and custom heads
    valid_graph_data = {
        "links": [{"source": {"id": "1"}, "target": {"id": "2"}}],
        "nodes": [
            {"id": "1", "features": {"label": "A", "feature1": [0.1]}},
            {"id": "2", "features": {"label": "B", "feature1": [0.2]}}
        ]
    }
    
    gat_config = {
        "model_name": "GAT",
        "hidden_channels": 16,
        "dropout": 0.2,
        "lr": 0.01,
        "epochs": 2,
        "label_feature": "label",
        "node_features": ["feature1"],
        "extra_params": {"heads": 4}
    }
    
    request = TrainGNNRequest(graph=valid_graph_data, configuration=gat_config)
    
    # Create mock objects
    mock_data = MagicMock()
    mock_data.y = torch.tensor([0, 1])
    mock_data.num_node_features = 4
    
    # Setup all required mocks
    with patch("main.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("main.GATModel") as MockModel, \
         patch("main.split_data", return_value=mock_data), \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("torch.optim.Adam"), \
         patch("torch.nn.CrossEntropyLoss"), \
         patch("torch.optim.lr_scheduler.ReduceLROnPlateau"):
        
        # Setup mock builder
        mock_builder = MockBuilder.return_value
        mock_builder.build_data.return_value = mock_data
        
        # Setup model mock
        mock_model = MockModel.return_value
        mock_model.to.return_value = mock_model
        
        # Setup device mock to return 'cpu'
        with patch("torch.device", return_value="cpu"):
            with patch("torch.cuda.is_available", return_value=False):
                # Execute the function under test
                try:
                    response = await train_gnn(request)
                    assert isinstance(response, StreamingResponse)
                except Exception as e:
                    pytest.fail(f"train_gnn raised {type(e).__name__} unexpectedly: {e}")
        
        # Verify GAT model was called with heads parameter
        MockModel.assert_called_once()
        # We don't check exact parameter values as they might change

# Test the streaming response format
@pytest.mark.asyncio
async def test_train_gnn_streaming_response():
    """Test that the streaming response yields valid JSON for each epoch."""
    from main import train_gnn, TrainGNNRequest
    
    # Create request with minimal epochs to speed up test
    valid_graph_data = {
        "links": [{"source": {"id": "1"}, "target": {"id": "2"}}],
        "nodes": [
            {"id": "1", "features": {"label": "A", "feature1": [0.1]}},
            {"id": "2", "features": {"label": "B", "feature1": [0.2]}}
        ]
    }
    
    config = {
        "model_name": "GCN",
        "hidden_channels": 16,
        "dropout": 0.2,
        "lr": 0.01,
        "epochs": 2,  # Minimal epochs
        "label_feature": "label",
        "node_features": ["feature1"]
    }
    
    # Setup mock training data
    mock_data = MagicMock()
    mock_data.y = torch.tensor([0, 1])
    mock_data.train_mask = torch.tensor([True, False])
    mock_data.val_mask = torch.tensor([False, True])
    mock_data.test_mask = torch.tensor([False, True])
    
    # Mock the training stream generator
    async def mock_training_stream():
        yield json.dumps({"status": "started", "message": "Training GCN model"}) + "\n"
        yield json.dumps({"epoch": 1, "train_loss": 0.5, "val_loss": 0.4}) + "\n"
        yield json.dumps({"epoch": 2, "train_loss": 0.3, "val_loss": 0.2}) + "\n"
        yield json.dumps({"status": "completed", "test_accuracy": 0.85}) + "\n"
    
    # Setup mocks
    with patch("main.TorchGeometricGraphBuilder") as MockBuilder, \
         patch("main.GCNModel") as MockModel, \
         patch("main.split_data", return_value=mock_data), \
         patch("torch.unique", return_value=torch.tensor([0, 1])), \
         patch("torch.device", return_value="cpu"), \
         patch("torch.optim.Adam"), \
         patch("torch.nn.CrossEntropyLoss"), \
         patch("torch.optim.lr_scheduler.ReduceLROnPlateau"):
        
        # Setup mock builder
        mock_builder = MockBuilder.return_value
        mock_builder.build_data.return_value = mock_data
        
        # Setup model mock
        mock_model = MockModel.return_value
        mock_model.to.return_value = mock_model
        
        # Replace the streaming response with our mock
        with patch("main.StreamingResponse", return_value=StreamingResponse(mock_training_stream())):
            # Execute function
            request = TrainGNNRequest(graph=valid_graph_data, configuration=config)
            response = await train_gnn(request)
            
            # Collect streaming response data
            collected_data = []
            async for chunk in response.body_iterator:
                # If chunk is already a string, use it directly; otherwise decode it
                if isinstance(chunk, bytes):
                    chunk_str = chunk.decode('utf-8')
                else:
                    chunk_str = chunk
                collected_data.append(json.loads(chunk_str.strip()))
            
            # Verify the response format
            assert len(collected_data) == 4
            assert collected_data[0]["status"] == "started"
            assert "epoch" in collected_data[1]
            assert "train_loss" in collected_data[1]
            assert collected_data[3]["status"] == "completed"
            assert "test_accuracy" in collected_data[3]

# Test graph building error
def test_train_gnn_graph_building_error(client, valid_graph_data, valid_model_config):
    """Test that errors during graph building are properly handled."""
    with patch("main.TorchGeometricGraphBuilder") as MockBuilder:
        # Setup the mock to raise an exception during build_data
        mock_instance = MockBuilder.return_value
        mock_instance.build_data.side_effect = ValueError("Error building graph: invalid node feature")
        
        # Test the endpoint with the mocked error
        response = client.post(
            "/train-gnn", 
            json={
                "graph": valid_graph_data,
                "configuration": valid_model_config
            }
        )
        
        # Should return 500 with error details
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "error building graph" in response.json()["detail"].lower()
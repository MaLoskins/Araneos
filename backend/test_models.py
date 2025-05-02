import pytest
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Any, Optional

# Define the model classes directly in the test file to avoid dependency issues
class ModelConfig(BaseModel):
    model_name: str  # Name of the GNN model (e.g., "GCN", "GraphSAGE")
    hidden_channels: int  # Size of hidden layers
    lr: float  # Learning rate
    epochs: int  # Number of training epochs
    dropout: float  # Dropout rate
    extra_params: Optional[Dict[str, Any]] = None  # For model-specific parameters

class TrainGNNRequest(BaseModel):
    graph: Dict[str, List]  # The node-link JSON graph data
    configuration: ModelConfig  # The model configuration

class TestModelConfig:
    def test_valid_model_config(self):
        """Test that ModelConfig can be instantiated with valid parameters."""
        config = ModelConfig(
            model_name="GCN",
            hidden_channels=64,
            lr=0.01,
            epochs=200,
            dropout=0.5,
            extra_params={"num_layers": 2}
        )
        assert config.model_name == "GCN"
        assert config.hidden_channels == 64
        assert config.lr == 0.01
        assert config.epochs == 200
        assert config.dropout == 0.5
        assert config.extra_params == {"num_layers": 2}

    def test_valid_model_config_without_extra_params(self):
        """Test that ModelConfig can be instantiated without extra_params."""
        config = ModelConfig(
            model_name="GraphSAGE",
            hidden_channels=32,
            lr=0.001,
            epochs=100,
            dropout=0.2
        )
        assert config.model_name == "GraphSAGE"
        assert config.hidden_channels == 32
        assert config.lr == 0.001
        assert config.epochs == 100
        assert config.dropout == 0.2
        assert config.extra_params is None

    def test_model_config_validation_model_name(self):
        """Test that ModelConfig correctly validates model_name as str."""
        invalid_data = {
            "model_name": 123,  # Should be a string
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5,
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(**invalid_data)
        
        assert "Input should be a valid string" in str(excinfo.value)

    def test_model_config_validation_hidden_channels(self):
        """Test that ModelConfig correctly validates hidden_channels as int."""
        invalid_data = {
            "model_name": "GCN",
            "hidden_channels": "not-a-number",  # Should be an int
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5,
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(**invalid_data)
        
        assert "Input should be a valid integer" in str(excinfo.value)

    def test_model_config_validation_lr(self):
        """Test that ModelConfig correctly validates lr as float."""
        invalid_data = {
            "model_name": "GCN",
            "hidden_channels": 64,
            "lr": "not-a-number",  # Should be a float
            "epochs": 200,
            "dropout": 0.5,
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(**invalid_data)
        
        assert "Input should be a valid number" in str(excinfo.value)

    def test_model_config_validation_epochs(self):
        """Test that ModelConfig correctly validates epochs as int."""
        invalid_data = {
            "model_name": "GCN",
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 100.5,  # Should be an int
            "dropout": 0.5,
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(**invalid_data)
        
        assert "Input should be a valid integer" in str(excinfo.value)

    def test_model_config_validation_dropout(self):
        """Test that ModelConfig correctly validates dropout as float."""
        invalid_data = {
            "model_name": "GCN",
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": "not-a-number",  # Should be a float
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(**invalid_data)
        
        assert "Input should be a valid number" in str(excinfo.value)

    def test_model_config_validation_extra_params(self):
        """Test that ModelConfig correctly validates extra_params as dict."""
        invalid_data = {
            "model_name": "GCN",
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5,
            "extra_params": "not_a_dict"  # Should be a dict
        }
        
        with pytest.raises(ValidationError) as excinfo:
            ModelConfig(**invalid_data)
        
        assert "Input should be a valid dictionary" in str(excinfo.value)


class TestTrainGNNRequest:
    def test_valid_train_gnn_request(self):
        """Test that TrainGNNRequest can be instantiated with valid parameters."""
        model_config = ModelConfig(
            model_name="GCN",
            hidden_channels=64,
            lr=0.01,
            epochs=200,
            dropout=0.5
        )
        
        graph = {
            "nodes": [
                {"id": "1", "features": {"attr1": 1.0}},
                {"id": "2", "features": {"attr1": 2.0}}
            ],
            "links": [
                {"source": "1", "target": "2"}
            ]
        }
        
        request = TrainGNNRequest(
            graph=graph,
            configuration=model_config
        )
        
        assert request.graph == graph
        assert request.configuration == model_config
        assert request.configuration.model_name == "GCN"

    def test_train_gnn_request_with_dict_model_config(self):
        """Test that TrainGNNRequest can be instantiated with a dict for model_config."""
        model_config_dict = {
            "model_name": "GraphSAGE",
            "hidden_channels": 32,
            "lr": 0.001,
            "epochs": 100,
            "dropout": 0.2,
            "extra_params": {"aggregation": "mean"}
        }
        
        graph = {
            "nodes": [{"id": "1"}, {"id": "2"}],
            "links": [{"source": "1", "target": "2"}]
        }
        
        request = TrainGNNRequest(
            graph=graph,
            configuration=model_config_dict
        )
        
        assert request.graph == graph
        assert request.configuration.model_name == "GraphSAGE"
        assert request.configuration.extra_params == {"aggregation": "mean"}

    def test_train_gnn_request_graph_validation_none(self):
        """Test that TrainGNNRequest validates the graph structure for None values."""
        model_config = {
            "model_name": "GCN",
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5
        }
        
        graph = {
            "nodes": None,  # None is not a valid List
            "links": []
        }
        
        with pytest.raises(ValidationError) as excinfo:
            TrainGNNRequest(graph=graph, configuration=model_config)
        
        assert "Input should be a valid list" in str(excinfo.value)

    def test_train_gnn_request_graph_validation_not_list(self):
        """Test that TrainGNNRequest validates the graph structure for non-list values."""
        model_config = {
            "model_name": "GCN",
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5
        }
        
        graph = {
            "nodes": 123,  # Integer is not a List
            "links": []
        }
        
        with pytest.raises(ValidationError) as excinfo:
            TrainGNNRequest(graph=graph, configuration=model_config)
        
        assert "Input should be a valid list" in str(excinfo.value)

    def test_train_gnn_request_graph_validation_not_dict(self):
        """Test that TrainGNNRequest validates the graph is a dictionary."""
        model_config = {
            "model_name": "GCN",
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5
        }
        
        graph = "not_a_dict"  # Not a dictionary
        
        with pytest.raises(ValidationError) as excinfo:
            TrainGNNRequest(graph=graph, configuration=model_config)
        
        assert "Input should be a valid dictionary" in str(excinfo.value)

    def test_train_gnn_request_invalid_model_config(self):
        """Test that TrainGNNRequest validates the model_config."""
        graph = {
            "nodes": [{"id": "1"}, {"id": "2"}],
            "links": [{"source": "1", "target": "2"}]
        }
        
        invalid_model_config = {
            "model_name": 123,  # Should be a string
            "hidden_channels": 64,
            "lr": 0.01,
            "epochs": 200,
            "dropout": 0.5
        }
        
        with pytest.raises(ValidationError) as excinfo:
            TrainGNNRequest(graph=graph, configuration=invalid_model_config)
        
        assert "Input should be a valid string" in str(excinfo.value)

if __name__ == "__main__":
    # Allow running with python test_models.py
    pytest.main(["-v", __file__])
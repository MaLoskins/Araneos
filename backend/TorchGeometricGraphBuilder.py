# TorchGeometricGraphBuilder.py

"""
TorchGeometricGraphBuilder
==========================
- Parses JSON node-link data into a PyTorch Geometric Data object.
- Defines multiple GNN architectures (GCN, GraphSAGE, GAT, GIN, ChebConv, ResidualGCN).
- Splits data into training, validation, and test sets.
- Trains each GNN model with early stopping, class weighting, and learning rate scheduling.
- Evaluates model performance with detailed classification reports.
- Visualizes node embeddings.
- Analyzes misclassifications.
- Implements ensemble methods for improved performance.
"""

import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, ChebConv
import torch.nn as nn
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------------- Reproducibility Setup ------------------------- #

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ------------------------ Device Configuration --------------------------- #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------- GNN Model Classes --------------------------- #

class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) for Node Classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Edge indices.

        Returns:
            Tensor: Logits for each class.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # No activation; CrossEntropyLoss expects raw logits

class ResidualGCNModel(nn.Module):
    """
    Residual Graph Convolutional Network (Residual GCN) for Node Classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super(ResidualGCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        
        # Linear layer to match dimensions for residual connection
        if in_channels != out_channels:
            self.residual_transform = nn.Linear(in_channels, out_channels)
        else:
            self.residual_transform = None

    def forward(self, x, edge_index):
        """
        Forward pass with residual connections.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Edge indices.

        Returns:
            Tensor: Logits for each class.
        """
        residual = x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        if self.residual_transform:
            residual = self.residual_transform(residual)

        x += residual  # Residual connection
        return x


class GraphSageModel(nn.Module):
    """
    GraphSAGE for Node Classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super(GraphSageModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Edge indices.

        Returns:
            Tensor: Logits for each class.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) for Node Classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 8, dropout: float = 0.6):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Output layer: 1 head, no concat to get out_channels
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Edge indices.

        Returns:
            Tensor: Logits for each class.
        """
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GINModel(nn.Module):
    """
    Graph Isomorphism Network (GIN) for Node Classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super(GINModel, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)
        
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Edge indices.

        Returns:
            Tensor: Logits for each class.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        return x

class ChebConvModel(nn.Module):
    """
    Chebyshev Convolution Network for Node Classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, K: int = 3, dropout: float = 0.3):
        super(ChebConvModel, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Edge indices.

        Returns:
            Tensor: Logits for each class.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ------------------------ Graph Builder Class ---------------------------- #

class TorchGeometricGraphBuilder:
    """
    Builds a PyG Data object from node-link JSON data.
    """
    def __init__(self, data_json: dict):
        self.data_json = data_json
        self.node_id_map = {}

    def build_data(self) -> Data:
        """
        Converts JSON node-link data into a torch_geometric.data.Data object.

        Returns:
            Data: PyG Data object with x, edge_index, edge_attr, and y.
        """
        # 1) Map node IDs to integers
        all_nodes = self.data_json.get("nodes", [])
        for idx, node_obj in enumerate(all_nodes):
            node_id = str(node_obj["id"])
            self.node_id_map[node_id] = idx

        # 2) Collect node features & labels
        node_features_list = []
        labels_list = []
        for node_obj in all_nodes:
            feats_dict = node_obj.get("features", {})
            feats_vector = []

            # Extract numerical features
            if "user_followers_count_feature" in feats_dict:
                try:
                    feats_vector.append(float(feats_dict["user_followers_count_feature"]))
                except ValueError:
                    print(f"Invalid value for 'user_followers_count_feature' in node {node_obj['id']}. Using 0.0.")
                    feats_vector.append(0.0)
            if "text_embedding" in feats_dict:
                try:
                    feats_vector.extend([float(val) for val in feats_dict["text_embedding"]])
                except ValueError:
                    print(f"Invalid values in 'text_embedding' for node {node_obj['id']}. Using zeros.")
                    feats_vector.extend([0.0] * len(feats_dict["text_embedding"]))

            if not feats_vector:
                feats_vector.append(0.0)  # Default feature if none provided

            # Check for label
            label_val = feats_dict.get("label", None)
            labels_list.append(label_val)

            node_features_list.append(feats_vector)

        x = self._to_tensor(node_features_list, dtype=torch.float)

        # 3) Normalize node features
        x = self._normalize_features(x)

        # 4) Build edge_index
        all_links = self.data_json.get("links", [])
        source_indices = []
        target_indices = []
        for link_obj in all_links:
            source_dict = link_obj.get("source", {})
            target_dict = link_obj.get("target", {})
            s_id = str(source_dict.get("id", ""))
            t_id = str(target_dict.get("id", ""))
            s_idx = self.node_id_map.get(s_id)
            t_idx = self.node_id_map.get(t_id)

            if s_idx is None or t_idx is None:
                print(f"Warning: Edge from '{s_id}' to '{t_id}' contains undefined node IDs. Skipping this edge.")
                continue

            source_indices.append(s_idx)
            target_indices.append(t_idx)

        if not source_indices or not target_indices:
            raise ValueError("No valid edges found in the dataset.")

        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

        # 5) Build edge_attr (if applicable)
        # Example: If your JSON includes edge attributes like 'relation_type'
        # Since we're removing edge_attr, we'll skip this part
        edge_attr = None  # Set to None as we're not using edge attributes

        # 6) Build y (labels)
        unique_labels = sorted(set(lbl for lbl in labels_list if lbl is not None))
        if unique_labels:
            label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
            y_data = []
            for lbl in labels_list:
                if lbl is None:
                    y_data.append(-1)  # Sentinel for unlabeled
                else:
                    y_data.append(label_to_idx[lbl])
            y = torch.tensor(y_data, dtype=torch.long)
        else:
            y = None

        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    @staticmethod
    def _to_tensor(list_of_lists, dtype=torch.float):
        """
        Pads lists with zeros to ensure uniform length and converts to a Tensor.

        Args:
            list_of_lists (List[List[float]]): List of feature vectors.
            dtype (torch.dtype): Data type of the tensor.

        Returns:
            Tensor: Padded feature matrix.
        """
        max_len = max(len(row) for row in list_of_lists)
        padded = []
        for row in list_of_lists:
            row_padded = row + [0.0]*(max_len - len(row))
            padded.append(row_padded)
        return torch.tensor(padded, dtype=dtype)

    @staticmethod
    def _normalize_features(x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes node features to have zero mean and unit variance.

        Args:
            x (Tensor): Node feature matrix.

        Returns:
            Tensor: Normalized node feature matrix.
        """
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-6  # Prevent division by zero
        return (x - mean) / std

# -------------------------- Data Splitting Function ------------------------ #

def split_data(data: Data, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Splits data into training, validation, and test sets.

    Args:
        data (Data): PyG Data object.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.

    Returns:
        Data: PyG Data object with masks.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    labels = data.y.numpy()

    # Exclude unlabeled nodes (-1) from the splitting
    labeled_mask = labels != -1
    labeled_indices = indices[labeled_mask]
    labeled_labels = labels[labeled_mask]

    if len(labeled_indices) == 0:
        raise ValueError("No labeled nodes found in the dataset.")

    # First split into train and temp
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        labeled_indices, labeled_labels, stratify=labeled_labels,
        test_size=(1 - train_ratio), random_state=42
    )

    # Then split temp into val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, stratify=y_temp,
        test_size=(1 - val_size), random_state=42
    )

    # Initialize all masks to False
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Assign True to the respective indices
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Assign masks to data
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

# ---------------------- Training and Evaluation -------------------------- #

def train(model: nn.Module, data: Data, optimizer, criterion, epoch: int, edge_drop_prob: float = 0.0):
    """
    Trains the GNN model for one epoch.

    Args:
        model (nn.Module): The GNN model to train.
        data (Data): PyG Data object.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        epoch (int): Current epoch number.
        edge_drop_prob (float): Probability of dropping each edge for augmentation.

    Returns:
        float: Training loss.
    """
    model.train()
    optimizer.zero_grad()

    # Edge Dropping Augmentation (optional)
    if edge_drop_prob > 0.0:
        edge_mask = torch.rand(data.edge_index.size(1)) > edge_drop_prob
        edge_index = data.edge_index[:, edge_mask]
    else:
        edge_index = data.edge_index

    out = model(data.x, edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # Gradient Clipping (optional)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def validate(model: nn.Module, data: Data, criterion):
    """
    Validates the GNN model.

    Args:
        model (nn.Module): The GNN model to validate.
        data (Data): PyG Data object.
        criterion (nn.Module): Loss function.

    Returns:
        float: Validation loss.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return loss.item()

def test(model: nn.Module, data: Data):
    """
    Tests the GNN model.

    Args:
        model (nn.Module): The trained GNN model.
        data (Data): PyG Data object.

    Returns:
        float: Test accuracy.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        test_mask = data.test_mask.cpu().numpy()
        y_pred = pred[test_mask]
        y_true_test = y_true[test_mask]
        
        acc = accuracy_score(y_true_test, y_pred)
        report = classification_report(y_true_test, y_pred, digits=4)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    return acc

# -------------------------- Visualization Function ------------------------ #

def visualize_embeddings(data: Data, model: nn.Module, title: str = "Node Embeddings"):
    """
    Visualizes node embeddings using UMAP.

    Args:
        data (Data): PyG Data object.
        model (nn.Module): Trained GNN model.
        title (str): Title for the plot.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device)).cpu().numpy()
    
    # Reduce dimensions using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=data.y.cpu().numpy(), cmap='viridis', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()

# -------------------------- Class Distribution Function -------------------- #

def print_class_distribution(data: Data):
    """
    Prints the distribution of classes in the dataset.

    Args:
        data (Data): PyG Data object.
    """
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    unique, counts = np.unique(y[train_mask], return_counts=True)
    print("Training Set Class Distribution:", dict(zip(unique, counts)))
    
    unique, counts = np.unique(y[val_mask], return_counts=True)
    print("Validation Set Class Distribution:", dict(zip(unique, counts)))
    
    unique, counts = np.unique(y[test_mask], return_counts=True)
    print("Test Set Class Distribution:", dict(zip(unique, counts)))

# -------------------------- Misclassification Analysis --------------------- #

def analyze_misclassifications(model: nn.Module, data: Data, model_name: str):
    """
    Analyzes misclassified nodes.

    Args:
        model (nn.Module): Trained GNN model.
        data (Data): PyG Data object.
        model_name (str): Name of the model.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        test_mask = data.test_mask.cpu().numpy()
        misclassified = np.where((pred != y_true) & test_mask)[0]
        
        print(f"\n--- Misclassifications for {model_name} ---")
        print(f"Number of Misclassified Nodes: {len(misclassified)}")
        if len(misclassified) > 0:
            print("Sample Misclassifications (node indices):", misclassified[:10])
            # Further analysis can be done here, such as inspecting features or graph structure

# -------------------------- Ensemble Method Function ----------------------- #

def ensemble_predictions(models: dict, data: Data):
    """
    Aggregates predictions from multiple models via majority voting.

    Args:
        models (dict): Dictionary of model names and model instances.
        data (Data): PyG Data object.

    Returns:
        np.array: Ensemble predictions.
    """
    model_preds = []
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            try:
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1).cpu().numpy()
                model_preds.append(pred)
            except Exception as e:
                print(f"Error getting predictions from model {name}: {e}")
                continue  # Skip this model if error occurs

    if not model_preds:
        raise ValueError("No model predictions available for ensemble.")

    # Convert list to numpy array for easier manipulation
    model_preds = np.array(model_preds)  # Shape: [num_models, num_nodes]

    # Majority voting: for each node, find the most common prediction among models
    ensemble_preds = []
    for i in range(model_preds.shape[1]):
        counts = np.bincount(model_preds[:, i])
        ensemble_preds.append(np.argmax(counts))
    ensemble_preds = np.array(ensemble_preds)

    return ensemble_preds

# -------------------------- PCA Function ----------------------- #

def reduce_feature_dimensions(x: torch.Tensor, n_components: int = 50) -> torch.Tensor:
    """
    Reduces feature dimensions using PCA.

    Args:
        x (Tensor): Original node feature matrix.
        n_components (int): Number of principal components.

    Returns:
        Tensor: Reduced node feature matrix.
    """
    try:
        pca = PCA(n_components=n_components, random_state=42)
        x_np = x.cpu().numpy()
        x_reduced = pca.fit_transform(x_np)
        return torch.tensor(x_reduced, dtype=torch.float).to(x.device)
    except Exception as e:
        print(f"Error during PCA dimensionality reduction: {e}")
        raise e

# ------------------------ Structural Features Function -------------------- #

def add_structural_features(data: Data):
    """
    Adds node degree as an additional feature.

    Args:
        data (Data): PyG Data object.
    """
    deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
    deg = deg.unsqueeze(1)  # Shape: [num_nodes, 1]
    data.x = torch.cat([data.x, deg], dim=1)

# -------------------------- Main Execution Flow --------------------------- #

def main():
    """
    Main function to build graph data, define models, train, and evaluate.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train multiple GNN models on graph data.")
    parser.add_argument('--json_path', type=str, default='test_graph_data.json',
                        help='Path to the JSON file containing graph data.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs for each model.')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels for GNN models.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for optimizers.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate for GNN models.')
    parser.add_argument('--edge_drop_prob', type=float, default=0.0,
                        help='Probability of dropping edges during training for augmentation.')
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components for feature dimensionality reduction.')
    args = parser.parse_args()

    # 1. Load JSON data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, args.json_path)

    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}. Please provide a valid path.")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # 2. Build PyG Data object
    builder = TorchGeometricGraphBuilder(data_json)
    try:
        data = builder.build_data()
    except Exception as e:
        print(f"Error building data: {e}")
        return

    # 3. Add Structural Features (Node Degree)
    try:
        add_structural_features(data)
    except Exception as e:
        print(f"Error adding structural features: {e}")
        return

    # 4. Apply PCA for Dimensionality Reduction
    try:
        data.x = reduce_feature_dimensions(data.x, n_components=args.pca_components)
    except Exception as e:
        print(f"Error applying PCA: {e}")
        return

    # 5. Split data
    try:
        data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return

    data = data.to(device)

    print("\n==== Torch Geometric Data ====")
    print(data)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"x shape: {data.x.shape}")

    if data.y is not None:
        print("\ny:", data.y)
        print("Unique label IDs:", torch.unique(data.y))
        print_class_distribution(data)
    else:
        print("No labels found. Exiting.")
        return

    # 6. Define GNN models
    # Exclude the sentinel label (-1) from class count if present
    unique_labels = torch.unique(data.y)
    num_classes = len(unique_labels) - (1 if -1 in unique_labels else 0)
    if num_classes < 2:
        print("Insufficient number of classes for classification. Exiting.")
        return

    in_channels = data.num_node_features
    hidden_channels = args.hidden_channels

    models = {
        "GCN": GCNModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, dropout=args.dropout),
        "GraphSAGE": GraphSageModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, dropout=args.dropout),
        "GAT": GATModel(in_channels=in_channels, hidden_channels=hidden_channels//8, out_channels=num_classes, heads=8, dropout=args.dropout),
        "GIN": GINModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, dropout=args.dropout),
        "ChebConv": ChebConvModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, K=3, dropout=args.dropout),
        "ResidualGCN": ResidualGCNModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=num_classes, dropout=args.dropout)  # Updated Model
    }


    # 7. Move models to device
    for model in models.values():
        model.to(device)

    # 8. Define optimizer and loss for each model
    optimizers = {}
    criterions = {}
    for name, model in models.items():
        optimizers[name] = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        criterions[name] = nn.CrossEntropyLoss()

    # 9. Calculate class weights (if needed)
    y_train = data.y[data.train_mask].cpu().numpy()
    classes = np.unique(y_train)
    if len(classes) > 1:
        try:
            class_weights_array = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weights = torch.tensor(class_weights_array, dtype=torch.float).to(device)
            for name in models.keys():
                criterions[name] = nn.CrossEntropyLoss(weight=class_weights)
        except Exception as e:
            print(f"Error computing class weights: {e}")
            return
    else:
        print("Only one class present in training data. Exiting.")
        return

    # 10. Define schedulers for learning rate
    schedulers = {}
    for name, optimizer in optimizers.items():
        schedulers[name] = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 11. Training with Early Stopping
    from collections import defaultdict

    best_val_losses = defaultdict(lambda: float('inf'))
    patience_counters = defaultdict(int)
    best_models = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        optimizer = optimizers[name]
        criterion = criterions[name]
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, args.epochs + 1):
            try:
                loss = train(model, data, optimizer, criterion, epoch, edge_drop_prob=args.edge_drop_prob)
                val_loss = validate(model, data, criterion)
            except Exception as e:
                print(f"Error during training epoch {epoch} for {name}: {e}")
                break

            # Calculate validation accuracy
            model.eval()
            with torch.no_grad():
                try:
                    out = model(data.x, data.edge_index)
                    pred = out[data.val_mask].argmax(dim=1)
                    val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred.cpu())
                except Exception as e:
                    print(f"Error calculating validation accuracy for {name} at epoch {epoch}: {e}")
                    val_acc = 0.0

            print(f"Epoch {epoch:03d}/{args.epochs:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early Stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping triggered for {name} at epoch {epoch}.")
                break

            # Step the scheduler
            schedulers[name].step(val_loss)

        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            best_models[name] = model

    # 12. Testing each model
    print("\n=== Testing Models ===")
    accuracies = {}
    for name, model in best_models.items():
        print(f"\n--- Testing {name} ---")
        try:
            acc = test(model, data)
            accuracies[name] = acc
        except Exception as e:
            print(f"Error testing model {name}: {e}")
            accuracies[name] = 0.0

    # 13. Display Accuracies
    print("\n=== Model Accuracies ===")
    for name, acc in accuracies.items():
        print(f"{name}: {acc:.4f}")

    # 14. Visualization: Visualize embeddings from the best model
    if accuracies:
        # Identify the best model based on test accuracy
        best_model_name = max(accuracies, key=accuracies.get)
        best_model = best_models[best_model_name]
        print(f"\nVisualizing embeddings from the best model: {best_model_name}")
        try:
            visualize_embeddings(data, best_model, title=f"{best_model_name} Node Embeddings")
        except Exception as e:
            print(f"Error visualizing embeddings for {best_model_name}: {e}")

    # 15. Analyze Misclassifications
    print("\n=== Analyzing Misclassifications ===")
    for name, model in best_models.items():
        try:
            analyze_misclassifications(model, data, name)
        except Exception as e:
            print(f"Error analyzing misclassifications for {name}: {e}")

    # 16. Implement Ensemble Method
    print("\n=== Ensemble Method ===")
    try:
        ensemble_preds = ensemble_predictions(best_models, data)
        # Only evaluate on test set
        test_mask = data.test_mask.cpu().numpy()
        ensemble_acc = accuracy_score(data.y[test_mask].cpu(), ensemble_preds[test_mask])
        print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
        report = classification_report(data.y[test_mask].cpu(), ensemble_preds[test_mask], digits=4)
        print("Ensemble Classification Report:")
        print(report)
    except Exception as e:
        print(f"Error during ensemble prediction: {e}")

# ------------------------ Structural Features Function -------------------- #

def add_structural_features(data: Data):
    """
    Adds node degree as an additional feature.

    Args:
        data (Data): PyG Data object.
    """
    deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
    deg = deg.unsqueeze(1)  # Shape: [num_nodes, 1]
    data.x = torch.cat([data.x, deg], dim=1)

# -------------------------- Ensemble Method Function ----------------------- #

def ensemble_predictions(models: dict, data: Data):
    """
    Aggregates predictions from multiple models via majority voting.

    Args:
        models (dict): Dictionary of model names and model instances.
        data (Data): PyG Data object.

    Returns:
        np.array: Ensemble predictions.
    """
    model_preds = []
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            try:
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1).cpu().numpy()
                model_preds.append(pred)
            except Exception as e:
                print(f"Error getting predictions from model {name}: {e}")
                continue  # Skip this model if error occurs

    if not model_preds:
        raise ValueError("No model predictions available for ensemble.")

    # Convert list to numpy array for easier manipulation
    model_preds = np.array(model_preds)  # Shape: [num_models, num_nodes]

    # Majority voting: for each node, find the most common prediction among models
    ensemble_preds = []
    for i in range(model_preds.shape[1]):
        counts = np.bincount(model_preds[:, i])
        ensemble_preds.append(np.argmax(counts))
    ensemble_preds = np.array(ensemble_preds)

    return ensemble_preds

# -------------------------- PCA Function ----------------------- #

def reduce_feature_dimensions(x: torch.Tensor, n_components: int = 50) -> torch.Tensor:
    """
    Reduces feature dimensions using PCA.

    Args:
        x (Tensor): Original node feature matrix.
        n_components (int): Number of principal components.

    Returns:
        Tensor: Reduced node feature matrix.
    """
    try:
        pca = PCA(n_components=n_components, random_state=42)
        x_np = x.cpu().numpy()
        x_reduced = pca.fit_transform(x_np)
        return torch.tensor(x_reduced, dtype=torch.float).to(x.device)
    except Exception as e:
        print(f"Error during PCA dimensionality reduction: {e}")
        raise e

# -------------------------- Main Execution Flow --------------------------- #

if __name__ == "__main__":
    main()

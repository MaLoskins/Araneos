# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Generator
import pandas as pd, uvicorn, logging, os, numpy as np, json, torch
from networkx.readwrite import json_graph
from DataFrameToGraph import DataFrameToGraph
from FeatureSpaceCreator import FeatureSpaceCreator
from TorchGeometricGraphBuilder import (
    TorchGeometricGraphBuilder,
    split_data,
    GCNModel,
    GraphSageModel,
    GATModel,
    GINModel,
    ChebConvModel,
    ResidualGCNModel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ProcessDataRequest(BaseModel):
    data: List[Dict[str, Any]]
    config: Dict[str, Any]

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

@app.post("/process-data")
def process_data(req:ProcessDataRequest):
    df=pd.DataFrame(req.data);config=req.config
    
    # Extract config values
    nodes_config=config.get("nodes",[])
    relationships=config.get("relationships",[])
    graph_type=config.get("graph_type","directed")
    label_column=config.get("label_column","")
    use_feature_space=config.get("use_feature_space",False)
    feature_space_config=config.get("feature_space_config",{})
    user_features=config.get("features",[])

    # Build initial graph
    graph_config={"nodes":nodes_config,"relationships":relationships,"graph_type":graph_type}
    graph_data=json_graph.node_link_data(DataFrameToGraph(df,graph_config,graph_type=graph_type).get_graph())

    # Attach user-chosen label if valid
    if label_column and label_column in df.columns:
        for node in graph_data["nodes"]:
            node_id=str(node["id"])
            for nc in nodes_config:
                matching_rows=df.loc[df[nc["id"]].astype(str)==node_id]
                if not matching_rows.empty:
                    "features" not in node and node.update({"features":{}})
                    node["features"]["label"]=str(matching_rows[label_column].values[0])
                    break

    # Generate embeddings if requested
    feature_data=None
    if use_feature_space and feature_space_config:
        logger.info("Generating embeddings with FeatureSpaceCreator.")
        feature_data=FeatureSpaceCreator(config=feature_space_config,device="cuda").process(df)
        
        # Process each feature
        for feat in user_features:
            node_id_col,col_name=feat.get("node_id_column"),feat.get("column_name")
            feat_type=feat.get("type","text").lower()
            
            # Skip if missing required fields
            if not node_id_col or not col_name:
                logger.warning(f"Skipping feature {feat} - missing node_id_column or column_name.")
                continue
                
            # Determine feature column name
            feature_col_name=f"{col_name}_{'embedding' if feat_type=='text' else 'feature'}"
            if feature_col_name not in feature_data.columns:
                logger.warning(f"Feature column '{feature_col_name}' not found. Skipping.")
                continue
                
            # Ensure node_id_col exists in feature_data
            if node_id_col not in feature_data.columns:
                if node_id_col not in df.columns:
                    logger.error(f"Column '{node_id_col}' not in CSV. Cannot attach features.")
                    continue
                feature_data[node_id_col]=df[node_id_col]

            # Attach features to nodes
            for _,row in feature_data.iterrows():
                if pd.isnull(row[node_id_col]):continue
                node_id_str=str(row[node_id_col])
                val=row[feature_col_name].tolist() if isinstance(row[feature_col_name],np.ndarray) else row[feature_col_name]
                
                # Find matching node and attach feature
                for n in graph_data["nodes"]:
                    if str(n["id"])==node_id_str:
                        "features" not in n and n.update({"features":{}})
                        n["features"][feature_col_name]=val
                        break

        logger.info("Feature embeddings attached to graph nodes.")
    else:logger.info("No advanced embeddings requested.")

    # Return results
    return {
        "graph":graph_data,
        "featureDataCsv":feature_data.to_csv(index=False) if feature_data is not None else None
    }

@app.post("/train-gnn")
async def train_gnn(request: TrainGNNRequest) -> StreamingResponse:
    """
    Trains a Graph Neural Network using provided graph data and model configuration.
    
    This endpoint:
    1. Builds a PyTorch Geometric Data object from the provided graph
    2. Instantiates the requested GNN model
    3. Sets up training components (optimizer, criterion, scheduler)
    4. Streams back training metrics for each epoch
    
    Args:
        request: TrainGNNRequest containing graph data and model configuration
        
    Returns:
        StreamingResponse that yields training metrics in JSON format
    
    Raises:
        HTTPException: If invalid model requested or missing labels in the data
    """
    try:
        # Build PyTorch Geometric Data object
        graph_builder = TorchGeometricGraphBuilder(request.graph)
        data = graph_builder.build_data()
        
        # Check if we have labels for training
        if data.y is None:
            raise HTTPException(
                status_code=400,
                detail="No labels found in the graph data. Node classification requires labeled nodes."
            )
        
        # Get the number of classes and features
        unique_labels = torch.unique(data.y)
        # Remove unlabeled nodes (marked with -1) when counting classes
        num_classes = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if num_classes < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient number of classes ({num_classes}) for classification. Need at least 2 classes."
            )
        
        # Split data into train/validation/test sets
        data = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        
        # Determine device (CPU or GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        
        # Extract configuration
        config = request.configuration
        model_name = config.model_name
        hidden_channels = config.hidden_channels
        in_channels = data.num_node_features
        dropout = config.dropout
        lr = config.lr
        epochs = config.epochs
        extra_params = config.extra_params or {}
        
        # Instantiate the requested model
        if model_name.upper() == "GCN":
            model = GCNModel(in_channels, hidden_channels, num_classes, dropout)
        elif model_name.upper() == "GRAPHSAGE" or model_name.upper() == "SAGE":
            model = GraphSageModel(in_channels, hidden_channels, num_classes, dropout)
        elif model_name.upper() == "GAT":
            heads = extra_params.get("heads", 8)
            model = GATModel(in_channels, hidden_channels // heads, num_classes, heads, dropout)
        elif model_name.upper() == "GIN":
            model = GINModel(in_channels, hidden_channels, num_classes, dropout)
        elif model_name.upper() == "CHEBCONV" or model_name.upper() == "CHEB":
            k = extra_params.get("K", 3)
            model = ChebConvModel(in_channels, hidden_channels, num_classes, k, dropout)
        elif model_name.upper() == "RESIDUALGCN" or model_name.upper() == "RESGCN":
            model = ResidualGCNModel(in_channels, hidden_channels, num_classes, dropout)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {model_name}. Available models: GCN, GraphSAGE, GAT, GIN, ChebConv, ResidualGCN"
            )
        
        # Move model to device
        model = model.to(device)
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Define the streaming function to yield training progress
        async def training_stream() -> Generator[str, None, None]:
            # Initial message
            yield json.dumps({
                "status": "started",
                "message": f"Training {model_name} model",
                "epoch": 0,
                "total_epochs": epochs
            }) + "\n"
            
            # Train for the specified number of epochs
            best_val_loss = float('inf')
            
            for epoch in range(1, epochs + 1):
                # Training step
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss = loss.item()
                
                # Validation step
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
                    
                    # Calculate validation accuracy
                    pred = out[data.val_mask].argmax(dim=1)
                    val_acc = (pred == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
                
                # Update learning rate scheduler
                scheduler.step(val_loss)
                
                # Check if this is the best model so far
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                # Yield the epoch results
                yield json.dumps({
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "is_best_model": is_best
                }) + "\n"
            
            # Final evaluation on test set
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                test_pred = out[data.test_mask].argmax(dim=1)
                test_acc = (test_pred == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            
            # Yield final results
            yield json.dumps({
                "status": "completed",
                "message": "Training completed",
                "test_accuracy": test_acc,
                "best_val_loss": best_val_loss
            }) + "\n"
        
        # Return streaming response
        return StreamingResponse(
            training_stream(),
            media_type="application/x-ndjson"
        )
        
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions
        raise http_ex
    except Exception as e:
        # Convert other exceptions to HTTP exceptions
        logger.error(f"Error training GNN: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error training GNN: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
import pandas as pd
import uvicorn
import logging
from networkx.readwrite import json_graph
import os
import numpy as np

from DataFrameToGraph import DataFrameToGraph
from FeatureSpaceCreator import FeatureSpaceCreator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessDataRequest(BaseModel):
    data: List[Dict[str, Any]]
    config: Dict[str, Any]

@app.post("/process-data")
def process_data(req: ProcessDataRequest):
    df = pd.DataFrame(req.data)
    config = req.config

    # Extract normal graph config
    nodes_config = config.get("nodes", [])
    relationships = config.get("relationships", [])
    graph_type = config.get("graph_type", "directed")

    # If advanced feature creation is requested
    use_feature_space = config.get("use_feature_space", False)
    feature_space_config = config.get("feature_space_config", {})
    # The user’s feature definitions (including node_id_column, column_name, etc.)
    user_features = config.get("features", [])

    # 1) Create a normal graph from the node/relationship config
    graph_config = {
        "nodes": nodes_config,
        "relationships": relationships,
        "graph_type": graph_type
    }
    df_to_graph = DataFrameToGraph(df, graph_config, graph_type=graph_type)
    graph = df_to_graph.get_graph()
    graph_data = json_graph.node_link_data(graph)

    # 2) If user wants embeddings, process them
    feature_data = None
    if use_feature_space and feature_space_config:
        # We pass the entire "feature_space_config" to FeatureSpaceCreator
        # The user_features array inside "feature_space_config.features" is used.
        logger.info("Generating embeddings with FeatureSpaceCreator.")
        fsc = FeatureSpaceCreator(config=feature_space_config, device="cuda")
        feature_data = fsc.process(df)  # includes columns like "text_embedding"
        
        # For each feature definition
        for feat in user_features:
            node_id_col = feat.get("node_id_column")
            col_name = feat.get("column_name")
            feat_type = feat.get("type", "text").lower()

            if not node_id_col or not col_name:
                logger.warning(f"Skipping feature {feat} because node_id_column or column_name is missing.")
                continue

            # If it's a text feature, the column is named <col>_embedding
            # If numeric, the column is named <col>_feature
            if feat_type == "numeric":
                feature_col_name = f"{col_name}_feature"
            else:
                feature_col_name = f"{col_name}_embedding"

            if feature_col_name not in feature_data.columns:
                logger.warning(f"Feature column '{feature_col_name}' not found in feature_data. Skipping.")
                continue

            # Ensure the node_id_column is in feature_data
            if node_id_col not in feature_data.columns:
                if node_id_col not in df.columns:
                    logger.error(f"Column '{node_id_col}' does not exist in CSV. Cannot attach features.")
                    continue
                feature_data[node_id_col] = df[node_id_col]

            # Attach the feature to the correct node in graph_data
            for idx, row in feature_data.iterrows():
                node_id_value = row[node_id_col]
                if pd.isnull(node_id_value):
                    continue
                node_id_str = str(node_id_value)

                val = row[feature_col_name]
                # If it's a numpy array, convert to list
                if isinstance(val, np.ndarray):
                    val = val.tolist()

                # find the node in graph_data
                for n in graph_data["nodes"]:
                    if str(n["id"]) == node_id_str:
                        if "features" not in n:
                            n["features"] = {}
                        n["features"][feature_col_name] = val
                        break



        logger.info("Feature embeddings attached to graph nodes.")
    else:
        logger.info("No advanced embeddings requested.")

    # Optionally, return the feature_data CSV
    feature_data_csv = None
    if feature_data is not None:
        feature_data_csv = feature_data.to_csv(index=False)

    return {
        "graph": graph_data,
        "featureDataCsv": feature_data_csv
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

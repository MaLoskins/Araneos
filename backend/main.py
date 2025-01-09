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

    nodes_config = config.get("nodes", [])
    relationships = config.get("relationships", [])
    graph_type = config.get("graph_type", "directed")
    label_column = config.get("label_column", "")

    use_feature_space = config.get("use_feature_space", False)
    feature_space_config = config.get("feature_space_config", {})
    user_features = config.get("features", [])

    # 1) Build the initial graph
    graph_config = {
        "nodes": nodes_config,
        "relationships": relationships,
        "graph_type": graph_type
    }
    df_to_graph = DataFrameToGraph(df, graph_config, graph_type=graph_type)
    graph = df_to_graph.get_graph()
    graph_data = json_graph.node_link_data(graph)

    # 2) Attach the user-chosen label if label_column is valid
    if label_column and label_column in df.columns:
        for node_info in graph_data["nodes"]:
            node_id = str(node_info["id"])
            for nc in nodes_config:
                node_id_col = nc["id"]
                matching_rows = df.loc[df[node_id_col].astype(str) == node_id]
                if not matching_rows.empty:
                    label_val = matching_rows[label_column].values[0]
                    if "features" not in node_info:
                        node_info["features"] = {}
                    node_info["features"]["label"] = str(label_val)
                    break

    # 3) If user wants embeddings
    feature_data = None
    if use_feature_space and feature_space_config:
        logger.info("Generating embeddings with FeatureSpaceCreator.")
        fsc = FeatureSpaceCreator(config=feature_space_config, device="cuda")
        feature_data = fsc.process(df)

        for feat in user_features:
            node_id_col = feat.get("node_id_column")
            col_name = feat.get("column_name")
            feat_type = feat.get("type", "text").lower()
            if not node_id_col or not col_name:
                logger.warning(f"Skipping feature {feat} because node_id_column or column_name is missing.")
                continue

            feature_col_name = f"{col_name}_embedding" if feat_type == "text" else f"{col_name}_feature"
            if feature_col_name not in feature_data.columns:
                logger.warning(f"Feature column '{feature_col_name}' not found in feature_data. Skipping.")
                continue

            if node_id_col not in feature_data.columns:
                if node_id_col not in df.columns:
                    logger.error(f"Column '{node_id_col}' does not exist in CSV. Cannot attach features.")
                    continue
                feature_data[node_id_col] = df[node_id_col]

            for idx, row_ in feature_data.iterrows():
                node_id_val = row_[node_id_col]
                if pd.isnull(node_id_val):
                    continue
                node_id_str = str(node_id_val)

                val = row_[feature_col_name]
                if isinstance(val, np.ndarray):
                    val = val.tolist()

                for n in graph_data["nodes"]:
                    if str(n["id"]) == node_id_str:
                        if "features" not in n:
                            n["features"] = {}
                        n["features"][feature_col_name] = val
                        break

        logger.info("Feature embeddings attached to graph nodes.")
    else:
        logger.info("No advanced embeddings requested.")

    feature_data_csv = None
    if feature_data is not None:
        feature_data_csv = feature_data.to_csv(index=False)

    return {
        "graph": graph_data,
        "featureDataCsv": feature_data_csv
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

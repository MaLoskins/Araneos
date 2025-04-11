# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any,Dict,List
import pandas as pd,uvicorn,logging,os,numpy as np
from networkx.readwrite import json_graph
from DataFrameToGraph import DataFrameToGraph
from FeatureSpaceCreator import FeatureSpaceCreator

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

class ProcessDataRequest(BaseModel):
    data:List[Dict[str,Any]]
    config:Dict[str,Any]

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

if __name__=="__main__":uvicorn.run(app,host="0.0.0.0",port=8000)

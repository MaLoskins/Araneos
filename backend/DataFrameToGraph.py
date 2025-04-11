# DataFrameToGraph.py
import pandas as pd,networkx as nx
from typing import Dict,Any
import logging as lg

lg.basicConfig(level=lg.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger=lg.getLogger(__name__)
class DataFrameToGraph:
    """Converts DataFrame and config into NetworkX graph"""
    def __init__(self,df:pd.DataFrame,config:Dict[str,Any],graph_type:str='directed'):
        self.df=df;self.config=config;self.graph_type=graph_type.lower()
        self.graph=self._initialize_graph();self.node_registry={}
        self._validate_config();self._parse_dataframe()

    def _initialize_graph(self)->nx.Graph:
        return nx.MultiDiGraph() if self.graph_type=='directed' else nx.MultiGraph() if self.graph_type=='undirected' else exec("raise ValueError(\"graph_type must be 'directed' or 'undirected'.\")")

    def _validate_config(self):
        [key not in self.config and exec(f"raise KeyError(\"Configuration missing required key: '{key}'\")") for key in ['nodes','relationships']]
        [('id' not in n and exec("raise KeyError(\"Each node configuration must have an 'id' key.\")") or 'type' not in n and logger.warning(f"Node configuration {n} missing 'type'. Defaulting to 'default'.")) for n in self.config['nodes']]
        [(('source' not in r or 'target' not in r) and exec("raise KeyError(\"Each relationship configuration must have 'source' and 'target' keys.\")") or 'type' not in r and logger.warning(f"Relationship configuration {r} missing 'type'. Defaulting to 'default'.")) for r in self.config['relationships']]

    def _parse_dataframe(self):
        """Create nodes/edges from DataFrame rows with embeddings"""
        for idx,r in self.df.iterrows():
            # Add nodes
            [self._process_node(idx,r,nc) for nc in self.config['nodes']]
            # Add edges
            [self._process_edge(idx,r,rc) for rc in self.config['relationships']]
    
    def _process_node(self,idx,r,nc):
        nid=r.get(nc['id'],None)
        if nid is None or pd.isnull(nid) or str(nid).strip()=="":
            logger.warning(f"Row {idx}: Missing or empty node ID for '{nc['id']}'. Skipping node.")
            return
        nid_str=str(nid);ntype=nc.get('type','default')
        # Fast feature collection with dict comprehension
        nfeats={c:r[c] for c in self.df.columns if c.endswith(("_embedding","_feature"))}
        self._add_node(nid_str,ntype,nfeats)

    def _process_edge(self,idx,r,rc):
        src_col,tgt_col=rc['source'],rc['target']
        rel_type=rc.get('type','default')
        src_id,tgt_id=r.get(src_col,None),r.get(tgt_col,None)
        
        # Skip if any ID is invalid
        if any(id is None or pd.isnull(id) or str(id).strip()=="" for id in [src_id,tgt_id]):
            logger.warning(f"Row {idx}: Missing or empty source/target for relationship '{rel_type}'. Skipping edge.")
            return
            
        src_str,tgt_str=str(src_id),str(tgt_id)
        
        # Skip if nodes not in registry
        if src_str not in self.node_registry:
            logger.warning(f"Row {idx}: Source node '{src_str}' not selected. Skipping edge.")
            return
        if tgt_str not in self.node_registry:
            logger.warning(f"Row {idx}: Target node '{tgt_str}' not selected. Skipping edge.")
            return
            
        self._add_edge(src_str,tgt_str,rel_type)

    def _add_node(self,node_id:str,node_type:str,node_features:dict):
        if node_id not in self.node_registry:
            self.node_registry[node_id]={'type':node_type,'features':node_features}
            self.graph.add_node(node_id,type=node_type,features=node_features)
            logger.info(f"Added node {node_id} of type '{node_type}' with features={list(node_features.keys())}.")
        else:logger.info(f"Node {node_id} already exists. Skipping duplicate.")

    def _add_edge(self,source_id:str,target_id:str,rel_type:str):
        if not self.graph.has_edge(source_id,target_id,key=rel_type):
            self.graph.add_edge(source_id,target_id,key=rel_type,type=rel_type)
            logger.info(f"Added edge from {source_id} to {target_id} of type '{rel_type}'.")
        else:logger.info(f"Edge from {source_id} to {target_id} of type '{rel_type}' already exists. Skipping.")

    def get_graph(self)->nx.Graph:return self.graph

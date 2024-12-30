# DataFrameToGraph.py
import pandas as pd
import networkx as nx
from typing import Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataFrameToGraph:
    """
    Converts a DataFrame and configuration into a NetworkX graph.
    
    Attributes:
        df (pd.DataFrame): The input DataFrame containing tabular data (including any embeddings).
        config (Dict[str, Any]): Configuration dictionary defining column roles for nodes/relationships,
                                 possibly including feature columns created by FeatureSpaceCreator.
        graph_type (str): 'directed' or 'undirected' graph type.
        graph (nx.Graph): The generated NetworkX graph.
        node_registry (dict): Registry of nodes added to the graph for quick checks.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        graph_type: str = 'directed'
    ):
        self.df = df
        self.config = config
        self.graph_type = graph_type.lower()
        self.graph = self._initialize_graph()
        self.node_registry = {}

        self._validate_config()
        self._parse_dataframe()

    def _initialize_graph(self) -> nx.Graph:
        if self.graph_type == 'directed':
            return nx.MultiDiGraph()
        elif self.graph_type == 'undirected':
            return nx.MultiGraph()
        else:
            raise ValueError("graph_type must be 'directed' or 'undirected'.")

    def _validate_config(self):
        required_keys = ['nodes', 'relationships']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Configuration missing required key: '{key}'")
        
        # Validate nodes
        for node_conf in self.config['nodes']:
            if 'id' not in node_conf:
                raise KeyError("Each node configuration must have an 'id' key.")
            if 'type' not in node_conf:
                logger.warning(f"Node configuration {node_conf} missing 'type'. Defaulting to 'default'.")

        # Validate relationships
        for rel_conf in self.config['relationships']:
            if 'source' not in rel_conf or 'target' not in rel_conf:
                raise KeyError("Each relationship configuration must have 'source' and 'target' keys.")
            if 'type' not in rel_conf:
                logger.warning(f"Relationship configuration {rel_conf} missing 'type'. Defaulting to 'default'.")

    def _parse_dataframe(self):
        """
        For each row, create nodes/edges. If the row has embedding columns (e.g., "text_embedding"),
        attach them as node features.
        """
        for index, row in self.df.iterrows():
            # Add nodes
            for node_conf in self.config['nodes']:
                node_id = row.get(node_conf['id'], None)

                # Treat empty strings the same as missing
                if node_id is None or pd.isnull(node_id) or str(node_id).strip() == "":
                    logger.warning(
                        f"Row {index}: Missing or empty node ID for '{node_conf['id']}'. Skipping node."
                    )
                    continue

                node_id_str = str(node_id)
                node_type = node_conf.get('type', 'default')

                # Collect precomputed features from columns that end in "_embedding" or "_feature"
                node_features = {}
                for col_name in self.df.columns:
                    if col_name.endswith("_embedding") or col_name.endswith("_feature"):
                        node_features[col_name] = row[col_name]

                self._add_node(node_id_str, node_type, node_features)

            # Add edges
            for rel_conf in self.config['relationships']:
                source_col = rel_conf['source']
                target_col = rel_conf['target']
                relationship_type = rel_conf.get('type', 'default')

                source_id = row.get(source_col, None)
                target_id = row.get(target_col, None)

                # Also skip if empty string
                if (
                    source_id is None or pd.isnull(source_id) or str(source_id).strip() == "" or
                    target_id is None or pd.isnull(target_id) or str(target_id).strip() == ""
                ):
                    logger.warning(
                        f"Row {index}: Missing or empty source/target for relationship '{relationship_type}'. Skipping edge."
                    )
                    continue

                source_id_str = str(source_id)
                target_id_str = str(target_id)

                if source_id_str not in self.node_registry:
                    logger.warning(f"Row {index}: Source node '{source_id_str}' not selected. Skipping edge.")
                    continue
                if target_id_str not in self.node_registry:
                    logger.warning(f"Row {index}: Target node '{target_id_str}' not selected. Skipping edge.")
                    continue

                self._add_edge(source_id_str, target_id_str, relationship_type)

    def _add_node(self, node_id: str, node_type: str, node_features: dict):
        if node_id not in self.node_registry:
            self.node_registry[node_id] = {'type': node_type, 'features': node_features}
            self.graph.add_node(node_id, type=node_type, features=node_features)
            logger.info(f"Added node {node_id} of type '{node_type}' with features={list(node_features.keys())}.")
        else:
            logger.info(f"Node {node_id} already exists. Skipping duplicate.")

    def _add_edge(self, source_id: str, target_id: str, rel_type: str):
        if not self.graph.has_edge(source_id, target_id, key=rel_type):
            self.graph.add_edge(source_id, target_id, key=rel_type, type=rel_type)
            logger.info(f"Added edge from {source_id} to {target_id} of type '{rel_type}'.")
        else:
            logger.info(f"Edge from {source_id} to {target_id} of type '{rel_type}' already exists. Skipping.")

    def get_graph(self) -> nx.Graph:
        return self.graph

"""
Backend package for geometric neural network feature space creation.

This package includes:
    - DataFrameToGraph: Converts DataFrame data into a NetworkX graph.
    - FeatureSpaceCreator: Processes raw text and numeric data into unified feature spaces.
    - TorchGeometricGraphBuilder: Builds and processes PyTorch Geometric graph data.
"""

from .DataFrameToGraph import DataFrameToGraph
from .FeatureSpaceCreator import FeatureSpaceCreator
from .TorchGeometricGraphBuilder import TorchGeometricGraphBuilder

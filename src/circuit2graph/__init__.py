"""
circuit2graph package
=====================

Utilities to represent cQED circuits as graphs and compress raw circuit
components into model-ready graph blocks.

Public modules:
- definitions.py: static subgraph types and physical attribute registry
- topology.py: CQEDNode and CQEDTopology data structures
- compression.py: graphlize() and graph-compression rules
"""

from circuit2graph.definitions import (
    ATTR_INDEX,
    LEGEND_ENTRIES,
    NODE_COLORS,
    SUBG_DEFS,
    SUBG_NODE,
    SubgDef,
    SubgType,
)
from circuit2graph.topology import CQEDNode, CQEDTopology
from circuit2graph.compression import branch_decomposition, find_root, graphlize
from circuit2graph.expansion import expand_graph, expand_macro_graph, expand_node_to_primitives, expand_topology

__all__ = [
    "ATTR_INDEX",
    "SubgType",
    "SubgDef",
    "SUBG_DEFS",
    "SUBG_NODE",
    "NODE_COLORS",
    "LEGEND_ENTRIES",
    "CQEDNode",
    "CQEDTopology",
    "branch_decomposition",
    "graphlize",
    "find_root",
    "expand_topology",
    "expand_graph",
    "expand_macro_graph",
    "expand_node_to_primitives",
]

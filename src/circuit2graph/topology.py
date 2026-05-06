"""
topology.py
===========

Core graph data structures for the cQED circuit representation.

This module contains:
- CQEDNode: one physical or compressed circuit block.
- CQEDTopology: a graph of CQEDNode objects with construction, inspection,
  drawing, symmetry and PyTorch-Geometric conversion utilities.

It intentionally does not contain graph-compression logic. Compression from
raw circuit elements into higher-level subgraph blocks is implemented in
compression.py.
"""

from __future__ import annotations

import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from circuit2graph.definitions import (
    SubgType, SUBG_DEFS, SUBG_NODE, NODE_COLORS, LEGEND_ENTRIES, ATTR_INDEX,
)

try:
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ---------------------------------------------------------------------------
# CQEDNode
# ---------------------------------------------------------------------------

class CQEDNode:
    """
    One node in a cQED circuit graph.

    Parameters
    ----------
    subg_type : SubgType
    attrs     : dict[str, float]   physical attribute values
    node_id   : int
    label     : str | None
    """

    def __init__(
        self,
        subg_type: SubgType,
        attrs:     dict,
        node_id:   int,
        label:     str | None = None,
    ):
        self.subg_type = subg_type
        self.attrs     = attrs
        self.node_id   = node_id
        self.label     = label

    # ------------------------------------------------------------------
    def feature_vector(self) -> list[float]:
        """Fixed-length attribute vector (len = ATTR_INDEX); unused slots = 0."""
        vec = [0.0] * len(ATTR_INDEX)
        for k, v in self.attrs.items():
            if k in ATTR_INDEX:
                vec[ATTR_INDEX[k]] = float(v)
        return vec

    # ------------------------------------------------------------------
    def inner_graph(self) -> tuple[list[dict], list[tuple]]:
        """
        Returns (nodes, edges) of this node's internal circuit sub-graph.
        Derived automatically from SUBG_DEFS — no manual branching needed.
        """
        defn  = SUBG_DEFS[self.subg_type]
        nodes = []
        for spec in defn.inner_nodes:
            val  = self.attrs.get(spec["val_attr"],  0.0) if spec["val_attr"]  else 0.0
            val2 = self.attrs.get(spec["val2_attr"], 0.0) if spec["val2_attr"] else 0.0
            nodes.append({"type_id": spec["type_id"], "val": val, "val2": val2})
        return nodes, list(defn.inner_edges)

    def __repr__(self) -> str:
        return (
            f"CQEDNode(id={self.node_id}, type={self.subg_type.name}, "
            f"label={self.label!r}, attrs={self.attrs})"
        )


# ---------------------------------------------------------------------------
# CQEDTopology
# ---------------------------------------------------------------------------

class CQEDTopology:
    """
    The topology of a complete cQED circuit as a graph of CQEDNodes.

    At encoding time the nodes are raw circuit elements (T, R, C, F).
    After graphlize() the nodes become compressed subgraph blocks.
    """

    def __init__(self, name: str):
        self.name      = name
        self._nodes:   list[CQEDNode]       = []
        self._edges:   list[tuple[int, int]] = []
        self._next_id: int                  = 0

    # ------------------------------------------------------------------
    # Node / edge construction
    # ------------------------------------------------------------------

    def _add(self, node: CQEDNode) -> CQEDNode:
        self._nodes.append(node)
        self._next_id += 1
        return node

    def add_node(
        self,
        subg_type: SubgType,
        attrs:     dict,
        label:     str | None = None,
    ) -> CQEDNode:
        return self._add(CQEDNode(
            subg_type = subg_type,
            node_id   = self._next_id,
            attrs     = attrs,
            label     = label,
        ))

    def add_edge(self, a: CQEDNode, b: CQEDNode) -> None:
        if a not in self._nodes or b not in self._nodes:
            raise ValueError("Both nodes must already belong to this CQEDTopology.")
        self._edges.append((a.node_id, b.node_id))

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _adj(self) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {n.node_id: [] for n in self._nodes}
        for a, b in self._edges:
            adj[a].append(b)
            adj[b].append(a)
        return adj

    # ------------------------------------------------------------------
    # Symmetry utilities
    # ------------------------------------------------------------------

    def symmetric_pairs(self) -> list[tuple[int, int]]:
        """
        Find pairs of nodes that are topologically symmetric — same SubgType
        and identical neighbour-type multisets in the outer graph.

        Returns a list of (node_id_a, node_id_b) pairs.
        """
        adj      = self._adj()
        node_map = {n.node_id: n for n in self._nodes}

        def _profile(nid):
            return tuple(sorted(int(node_map[nb].subg_type) for nb in adj[nid]))

        groups: dict = defaultdict(list)
        for n in self._nodes:
            key = (int(n.subg_type), _profile(n.node_id))
            groups[key].append(n.node_id)

        pairs = []
        for nids in groups.values():
            if len(nids) == 2:
                pairs.append((nids[0], nids[1]))
            elif len(nids) > 2:
                for i in range(0, len(nids) - 1, 2):
                    pairs.append((nids[i], nids[i + 1]))
        return pairs

    def is_symmetric(self) -> bool:
        """Return True if the topology has at least one symmetric pair."""
        return len(self.symmetric_pairs()) > 0

    # Attribute used to sort symmetric nodes, per SubgType.
    # Add entries here to enable canonical ordering for other types.
    _SORT_ATTR: dict[SubgType, str] = {
        SubgType.TRANSMON:  "L",      # Lmin first
        SubgType.RESONATOR: "length", # shorter first
    }

    def canonical_sort(self, param_dict: dict) -> dict:
        """
        Apply canonical ordering to symmetric node pairs.

        For each symmetric pair (a, b), ensures the node whose sort attribute
        is smaller comes first (lower node_id in the pair).

        Parameters
        ----------
        param_dict : dict[node_id → attrs dict]

        Returns
        -------
        sorted_param_dict : dict[node_id → attrs dict]
        """
        node_map = {n.node_id: n for n in self._nodes}
        result   = {nid: dict(attrs) for nid, attrs in param_dict.items()}

        for nid_a, nid_b in self.symmetric_pairs():
            st        = node_map[nid_a].subg_type
            sort_attr = self._SORT_ATTR.get(st)
            if sort_attr is None:
                continue
            val_a = result[nid_a].get(sort_attr, 0.0)
            val_b = result[nid_b].get(sort_attr, 0.0)
            if val_a > val_b:
                result[nid_a], result[nid_b] = result[nid_b], result[nid_a]

        return result

    # ------------------------------------------------------------------
    # Summary / drawing
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"CQEDTopology: '{self.name}'",
            f"  Nodes ({len(self._nodes)}):",
        ]
        for n in self._nodes:
            type_str = "+".join(SUBG_NODE[n.subg_type])
            lines.append(
                f"    [{n.node_id}] {str(n.label):22s} "
                f"type={type_str:16s} "
                f"attrs={n.attrs}"
            )
        lines.append("  Edges: " + ", ".join(f"({a}<->{b})" for a, b in self._edges))
        return "\n".join(lines)

    def draw(self, ax=None, with_labels: bool = True) -> None:
        """
        Draw the topology onto *ax* (matplotlib Axes).
        If ax is None a new figure is created and shown immediately.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 5))

        G = nx.Graph()
        for node in self._nodes:
            shown_label = node.label if node.label is not None else f"{node.subg_type}_{node.node_id}"
            G.add_node(node.node_id, label=shown_label, subg_type=node.subg_type)
        G.add_edges_from(self._edges)

        pos    = nx.spring_layout(G, seed=42)
        colors = [NODE_COLORS.get(G.nodes[n]["subg_type"], "#cccccc") for n in G.nodes]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1200)

        if with_labels:
            labels = {n.node_id: (n.label or f"id={n.node_id}") for n in self._nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

        ax.set_title(f"{self.name}", fontsize=10)
        ax.axis("off")

        if standalone:
            plt.tight_layout()
            plt.show()
            plt.close()

    # ------------------------------------------------------------------
    # PyG conversion
    # ------------------------------------------------------------------

    def to_pyg(self):
        """Convert to a PyTorch Geometric Data object."""
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for to_pyg().")

        N        = len(self._nodes)
        feat_dim = len(SUBG_NODE) + len(ATTR_INDEX)
        x        = torch.zeros((N, feat_dim), dtype=torch.float)

        for node in self._nodes:
            i = node.node_id
            x[i, int(node.subg_type)] = 1.0
            for j, v in enumerate(node.feature_vector()):
                x[i, len(SUBG_NODE) + j] = v

        fwd        = list(self._edges)
        bwd        = [(b, a) for a, b in fwd]
        all_e      = fwd + bwd
        edge_index = (
            torch.tensor(all_e, dtype=torch.long).t().contiguous()
            if all_e else torch.zeros((2, 0), dtype=torch.long)
        )
        return Data(
            x          = x,
            edge_index = edge_index,
            node_type  = torch.tensor([int(n.subg_type) for n in self._nodes], dtype=torch.long),
            num_nodes  = N,
        )

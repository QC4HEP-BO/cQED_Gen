"""
cqed.data.loader_vae
====================
Extension of the original data loader that adds VAE-specific fields to
each Data sample.

Fields on each Data object:
    obs_vals       [N_OBS_SLOTS]   scaled observable values
    obs_mask       [N_OBS_SLOTS]   binary mask
    topology_ids   [n_circuit]     SubgType int per macro-node
    block_param_lens [n_circuit]   number of attrs per macro-node
    y              [n_params]      scaled physical parameters (flat)
    dataset_name   str
    ei_outer       [2, E]          circuit↔circuit edge index
    g_true_ns      SimpleNamespace decoder ground-truth graph
    enc_inner_x        [total_inner_nodes, INNER_FEAT_DIM]
    enc_inner_ei       [2, total_inner_edges]
    enc_inner_n_nodes  list[int]
    enc_macro_subgtype [n_circuit]
    enc_macro_pos      [n_circuit, 1]
    enc_outer_ei       [2, E]
    enc_n_circuit      int

Public API
----------
    load_all_datasets_vae(train_frac, val_frac, seed, max_nodes)
        → (train_list, val_list, test_list, scalers)
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch_geometric.data import Data

from circuit2graph import CQEDTopology, SubgType, SUBG_DEFS
from circuit2graph import graphlize
from data_loader.schema import DATASETS, DatasetDef, OBS_PARSERS
from data_loader.schema import OBS_SLOTS, N_OBS_SLOTS, OBS_IDX
from data_loader.processing import ParamScaler, ObsScaler, DatasetScalers
from data_loader.schema import ROW_PARSERS, _is_header
from data_loader.processing import (
    _extract_params,
    _fill_attrs,
    _read_raw,
    N_SUBTYPES,
)
from vae_model.decoder import data_to_graph_ns
from vae_model.encoder import build_encoder_inner_feats, INNER_FEAT_DIM, N_SUBTYPES as _ENC_N_SUBTYPES


# ---------------------------------------------------------------------------
# Build a single VAE Data object
# ---------------------------------------------------------------------------

def _build_enc_tensors(
    compressed: CQEDTopology,
    y_scaled:   np.ndarray,
) -> dict:
    """
    Pre-computes all tensors needed by _prepare_batch() in GraphVAE for one sample.
    Called once at dataset-load time; stored in the Data object so that
    _prepare_batch() can just torch.cat instead of re-building per-batch.

    Returns a dict with:
        enc_inner_x        [total_inner_nodes, INNER_FEAT_DIM]
        enc_inner_ei       [2, total_inner_edges]  intra-sample offsets applied,
                           cross-sample offset deferred to _prepare_batch() in vae.py
        enc_inner_n_nodes  list[int]  number of inner nodes per macronode
        enc_macro_subgtype [n_circuit]  long
        enc_macro_pos      [n_circuit, 1]  float (rank / n_circuit)
        enc_outer_ei       [2, E]  (offsets NOT applied yet; only cc edges)
        enc_n_circuit      int
    """
    from circuit2graph.topology import CQEDNode as _CQEDNode

    circ_nodes = compressed._nodes
    n_circuit  = len(circ_nodes)

    # Build scaled-attrs dict per node from y_scaled (same order as SUBG_DEFS attrs)
    cursor = 0
    attrs_per_node: list[dict] = []
    for node in circ_nodes:
        attrs = SUBG_DEFS[node.subg_type].attrs
        d: dict = {}
        for attr_name in attrs:
            d[attr_name] = float(y_scaled[cursor])
            cursor += 1
        attrs_per_node.append(d)

    inner_x_list:   list[torch.Tensor] = []
    inner_ei_list:  list[torch.Tensor] = []
    inner_n_nodes:  list[int]          = []

    for node, attrs_sc in zip(circ_nodes, attrs_per_node):
        dummy = _CQEDNode(
            subg_type = node.subg_type,
            attrs     = dict(attrs_sc),
            node_id   = 0,
        )
        xi, ei = build_encoder_inner_feats(dummy, attrs_sc)
        n_in = xi.shape[0]
        inner_x_list.append(xi)
        # Store edge_index with LOCAL (zero-based) indices for this block only.
        # The intra-sample offset (block-to-block within one sample) and the
        # cross-sample offset (sample-to-sample within a batch) are both applied
        # in _prepare_batch() in vae.py, which holds the single source of truth
        # for index accumulation. Applying any offset here would cause a
        # double-shift when _prepare_batch adds inner_offset on top.
        inner_ei_list.append(ei)
        inner_n_nodes.append(n_in)

    # Assemble inner tensors: apply intra-sample (block-to-block) offsets so
    # enc_inner_ei is self-consistent within the sample. The cross-sample offset
    # is still deferred to _prepare_batch(), which adds its accumulated
    # inner_offset (= total inner nodes of all preceding samples) once.
    intra_offset = 0
    inner_ei_shifted: list[torch.Tensor] = []
    for ei, n_in in zip(inner_ei_list, inner_n_nodes):
        if ei.shape[1] > 0:
            inner_ei_shifted.append(ei + intra_offset)
        else:
            inner_ei_shifted.append(ei)
        intra_offset += n_in

    if inner_x_list:
        enc_inner_x = torch.cat(inner_x_list, dim=0)
    else:
        enc_inner_x = torch.zeros((0, INNER_FEAT_DIM), dtype=torch.float)

    if any(e.shape[1] > 0 for e in inner_ei_shifted):
        enc_inner_ei = torch.cat(inner_ei_shifted, dim=1)
    else:
        enc_inner_ei = torch.zeros((2, 0), dtype=torch.long)

    # Macro-node tensors
    enc_macro_subgtype = torch.tensor(
        [int(node.subg_type) for node in circ_nodes], dtype=torch.long
    )
    if n_circuit > 0:
        pos_vals = torch.arange(1, n_circuit + 1, dtype=torch.float) / max(n_circuit, 1)
    else:
        pos_vals = torch.zeros(0, dtype=torch.float)
    enc_macro_pos = pos_vals.unsqueeze(1)   # [n_circuit, 1]

    # Outer edge_index (circuit↔circuit only, no offset yet)
    adj       = compressed._adj()
    id_to_idx = {node.node_id: i for i, node in enumerate(circ_nodes)}
    oc_edges: list[tuple[int, int]] = []
    for u_id, v_id in compressed._edges:
        ui = id_to_idx.get(u_id)
        vi = id_to_idx.get(v_id)
        if ui is not None and vi is not None:
            oc_edges.append((ui, vi))
            oc_edges.append((vi, ui))
    if oc_edges:
        enc_outer_ei = torch.tensor(oc_edges, dtype=torch.long).t().contiguous()
    else:
        enc_outer_ei = torch.zeros((2, 0), dtype=torch.long)

    return {
        "enc_inner_x":        enc_inner_x,
        "enc_inner_ei":       enc_inner_ei,
        "enc_inner_n_nodes":  inner_n_nodes,
        "enc_macro_subgtype": enc_macro_subgtype,
        "enc_macro_pos":      enc_macro_pos,
        "enc_outer_ei":       enc_outer_ei,
        "enc_n_circuit":      n_circuit,
    }


def _topo_to_data_vae(
    compressed:      CQEDTopology,
    y_scaled:        np.ndarray,
    obs_vals_scaled: np.ndarray,
    obs_mask:        np.ndarray,
    ds_name:         str,
) -> Data:
    """
    Build a Data object with VAE fields.
    Encoder tensors are pre-computed and stored so _prepare_batch() is cheap.
    """
    # ── Inner graph topology ids ───────────────────────────────────────
    circ_nodes       = compressed._nodes
    topology_ids     = torch.tensor([int(n.subg_type) for n in circ_nodes], dtype=torch.long)
    block_param_lens = torch.tensor(
        [len(SUBG_DEFS[n.subg_type].attrs) for n in circ_nodes], dtype=torch.long
    )

    # ── Outer edge index (circuit↔circuit) ────────────────────────────
    id_to_idx = {n.node_id: i for i, n in enumerate(circ_nodes)}
    fwd = [(id_to_idx[u], id_to_idx[v]) for u, v in compressed._edges]
    bwd = [(v, u) for u, v in fwd]
    all_e = fwd + bwd
    ei_outer = (
        torch.tensor(all_e, dtype=torch.long).t().contiguous()
        if all_e else torch.zeros((2, 0), dtype=torch.long)
    )

    # ── Pre-computed encoder tensors ──────────────────────────────────
    enc = _build_enc_tensors(compressed, y_scaled)

    data                  = Data()
    data.ei_outer         = ei_outer
    data.obs_vals         = torch.tensor(obs_vals_scaled, dtype=torch.float)
    data.obs_mask         = torch.tensor(obs_mask,        dtype=torch.float)
    data.topology_ids     = topology_ids
    data.block_param_lens = block_param_lens
    data.y                = torch.tensor(y_scaled, dtype=torch.float)
    data.dataset_name     = ds_name
    # Pre-compute G_true_ns once at load time to avoid calling data_to_graph_ns()
    # at every forward pass.
    data.g_true_ns        = data_to_graph_ns(data, None)
    # Pre-computed encoder tensors
    data.enc_inner_x        = enc["enc_inner_x"]
    data.enc_inner_ei       = enc["enc_inner_ei"]
    data.enc_inner_n_nodes  = enc["enc_inner_n_nodes"]   # Python list[int]
    data.enc_macro_subgtype = enc["enc_macro_subgtype"]
    data.enc_macro_pos      = enc["enc_macro_pos"]
    data.enc_outer_ei       = enc["enc_outer_ei"]
    data.enc_n_circuit      = enc["enc_n_circuit"]       # Python int
    return data


# ---------------------------------------------------------------------------
# Build all samples for one dataset
# ---------------------------------------------------------------------------

def _build_samples_vae(
    ds_name:   str,
    defn:      DatasetDef,
    scalers,
    fit:       bool,
    rng:       random.Random,
    max_nodes: int,
) -> tuple[list[Data], DatasetScalers]:
    """
    Builds a list of VAE training samples from a raw dataset. This function 
    transforms raw entries (e.g. circuit parameters + observables) into structured 
    `Data` objects ready for the model.

    Workflow
    --------
    1. Load raw entries from disk:
    - Each entry consists of:
        attrs   : circuit parameters / attributes
        obs_kw  : raw observable values (keyword format)

    2. For each entry:
    - Instantiate a topology template and fill it with parameters (`_fill_attrs`)
    - Convert the topology into a compressed graph representation (`graphlize`)
    - Extract numerical parameter vector (Y_raw)
    - Parse observables into:
            obs_vals_raw : values
            obs_masks_raw: mask (1 = present, 0 = missing)

    3. Convert collected lists into numpy arrays:
    - Y_raw_np        : [N, param_dim]
    - obs_vals_np     : [N, N_OBS_SLOTS]
    - obs_mask_np     : [N, N_OBS_SLOTS]

    4. Fit scalers (only if `fit=True`, typically on training set):
    - ParamScaler: normalizes circuit parameters
    - ObsScaler  : normalizes observable values using masks
    - Store them inside DatasetScalers

    5. Apply scaling:
    - Parameters are scaled globally
    - Observables are scaled per-sample (respecting masks)

    6. Build final Data objects:
    - Combine:
            * graph structure (compressed topology)
            * scaled parameters
            * scaled observables
            * observable masks
    - Enforce max_nodes constraint if needed

    7. Return:
    - samples : list of Data objects (one per circuit)
    - scalers : fitted scalers (or reused ones if fit=False)
    """
    obs_parser  = OBS_PARSERS[ds_name]
    raw_entries = _read_raw(defn.path, ds_name, defn.n_samples, rng)
    if not raw_entries:
        raise RuntimeError(f"No rows loaded from {defn.path}")

    raw_template    = defn.topology_fn()
    compressed_list = []
    Y_raw           = []
    obs_vals_raw    = []
    obs_masks_raw   = []

    for attrs, obs_kw in raw_entries:
        topo       = _fill_attrs(raw_template, attrs)
        compressed = graphlize(topo)
        compressed_list.append(compressed)
        Y_raw.append(_extract_params(compressed))
        o_val, o_mask = obs_parser(**obs_kw)
        obs_vals_raw.append(o_val)
        obs_masks_raw.append(o_mask)

    Y_raw_np    = np.array(Y_raw, dtype=np.float64)
    obs_vals_np = np.array(obs_vals_raw, dtype=np.float64)
    obs_mask_np = np.array(obs_masks_raw, dtype=np.float64)

    if fit:
        ps = ParamScaler().fit(Y_raw_np)
        os_ = ObsScaler().fit(obs_vals_np, obs_mask_np)
        scalers = DatasetScalers(ps, os_)

    Y_scaled = scalers.param_scaler.transform(Y_raw_np)

    samples: list[Data] = []
    for i, compressed in enumerate(compressed_list):
        obs_scaled = scalers.obs_scaler.transform_row(obs_vals_np[i], obs_mask_np[i])
        data = _topo_to_data_vae(
            compressed, Y_scaled[i], obs_scaled, obs_mask_np[i],
            ds_name,
        )
        samples.append(data)

    return samples, scalers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_datasets_vae(
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
    max_nodes:  int   = 12,
) -> tuple[list, list, list, dict]:
    """
    Load all registered datasets with VAE-specific fields.

    Returns
    -------
    train_list, val_list, test_list : lists of Data objects
    scalers : dict[ds_name → DatasetScalers]
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    all_scalers: dict = {}
    all_samples: dict = {}

    print("Loading datasets (VAE mode)…")
    for ds_name, defn in DATASETS.items():
        print(f"\n[{ds_name}]")
        samples, ds_scalers = _build_samples_vae(
            ds_name, defn, scalers=None, fit=True, rng=rng,
            max_nodes=max_nodes,
        )
        all_scalers[ds_name] = ds_scalers
        all_samples[ds_name] = samples
        last    = samples[-1]
        n_nodes = int(last.enc_n_circuit)
        n_p     = int(last.y.shape[0])
        print(f"  → {len(samples)} samples | {n_nodes} outer nodes | {n_p} params")

    train_list: list = []
    val_list:   list = []
    test_list:  list = []

    for ds_name, samples in all_samples.items():
        rng.shuffle(samples)
        N = len(samples)
        n_test = max(1, int(round(N * (1 - train_frac - val_frac))))
        n_val = max(1, int(round(N * val_frac)))
        n_train = N - n_test - n_val
        train_list.extend(samples[:n_train])
        val_list.extend(samples[n_train: n_train + n_val])
        test_list.extend(samples[n_train + n_val:])

    rng.shuffle(train_list)
    rng.shuffle(val_list)

    print(f"\nFinal split: train={len(train_list)}  "
          f"val={len(val_list)}  test={len(test_list)}")
    return train_list, val_list, test_list, all_scalers
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

    load_inference_datasets_vae(seed, max_nodes)
        → (samples_dict, scalers_dict)
        Returns ALL datasets (including inference-only ones) as a dict
        keyed by dataset name. Scalers are fitted on each dataset independently.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch_geometric.data import Data

from circuit2graph import CQEDTopology, SubgType, SUBG_DEFS
from circuit2graph import graphlize
from data_loader.schema import (
    DATASETS, DatasetDef, OBS_PARSERS,
    OBS_SLOTS, N_OBS_SLOTS, OBS_IDX,
    ROW_PARSERS, _is_header,  # _is_header re-exported from datasets._base
    train_datasets,
)
from data_loader.processing import ParamScaler, ObsScaler, DatasetScalers
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
        inner_ei_list.append(ei)
        inner_n_nodes.append(n_in)

    # Apply intra-sample (block-to-block) offsets
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
    circ_nodes       = compressed._nodes
    topology_ids     = torch.tensor([int(n.subg_type) for n in circ_nodes], dtype=torch.long)
    block_param_lens = torch.tensor(
        [len(SUBG_DEFS[n.subg_type].attrs) for n in circ_nodes], dtype=torch.long
    )

    id_to_idx = {n.node_id: i for i, n in enumerate(circ_nodes)}
    fwd = [(id_to_idx[u], id_to_idx[v]) for u, v in compressed._edges]
    bwd = [(v, u) for u, v in fwd]
    all_e = fwd + bwd
    ei_outer = (
        torch.tensor(all_e, dtype=torch.long).t().contiguous()
        if all_e else torch.zeros((2, 0), dtype=torch.long)
    )

    enc = _build_enc_tensors(compressed, y_scaled)

    data                  = Data()
    data.ei_outer         = ei_outer
    data.obs_vals         = torch.tensor(obs_vals_scaled, dtype=torch.float)
    data.obs_mask         = torch.tensor(obs_mask,        dtype=torch.float)
    data.topology_ids     = topology_ids
    data.block_param_lens = block_param_lens
    data.y                = torch.tensor(y_scaled, dtype=torch.float)
    data.dataset_name     = ds_name
    data.g_true_ns        = data_to_graph_ns(data, None)
    data.enc_inner_x        = enc["enc_inner_x"]
    data.enc_inner_ei       = enc["enc_inner_ei"]
    data.enc_inner_n_nodes  = enc["enc_inner_n_nodes"]
    data.enc_macro_subgtype = enc["enc_macro_subgtype"]
    data.enc_macro_pos      = enc["enc_macro_pos"]
    data.enc_outer_ei       = enc["enc_outer_ei"]
    data.enc_n_circuit      = enc["enc_n_circuit"]
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
    Builds a list of VAE Data objects from one raw dataset file.

    Parameters
    ----------
    ds_name   : key in DATASETS / OBS_PARSERS
    defn      : DatasetDef descriptor
    scalers   : existing DatasetScalers (used when fit=False) or None
    fit       : if True, fit new scalers from this dataset's data
    rng       : seeded Random for reservoir sampling
    max_nodes : upper bound on compressed graph size (samples exceeding
                this are silently discarded)

    Returns
    -------
    samples   : list[Data]
    scalers   : DatasetScalers (fitted or passed through)
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
        if max_nodes and len(compressed._nodes) > max_nodes:
            continue
        compressed_list.append(compressed)
        Y_raw.append(_extract_params(compressed))
        o_val, o_mask = obs_parser(**obs_kw)
        obs_vals_raw.append(o_val)
        obs_masks_raw.append(o_mask)

    if not compressed_list:
        raise RuntimeError(
            f"All rows filtered out for {ds_name} (max_nodes={max_nodes}). "
            "Check block_params or topology builder."
        )

    Y_raw_np    = np.array(Y_raw, dtype=np.float64)
    obs_vals_np = np.array(obs_vals_raw, dtype=np.float64)
    obs_mask_np = np.array(obs_masks_raw, dtype=np.float64)

    if fit:
        ps  = ParamScaler().fit(Y_raw_np)
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
    Load datasets marked include_train=True and split into train/val/test.

    Datasets with include_train=False (e.g. Three_qubit_capacitive_line) are
    skipped here but remain available via load_inference_datasets_vae().
    To add a dataset to training, set include_train=True in its DatasetDef
    inside schema.py — no other change required.

    Returns
    -------
    train_list, val_list, test_list : lists of Data objects
    scalers : dict[ds_name → DatasetScalers]
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    train_defs = train_datasets()

    all_scalers: dict = {}
    all_samples: dict = {}

    print("Loading datasets (VAE mode, training split)…")
    print(f"  include_train=True  : {list(train_defs.keys())}")
    inference_only = [k for k, v in DATASETS.items() if not v.include_train]
    if inference_only:
        print(f"  include_train=False : {inference_only}  (inference only)")

    for ds_name, defn in train_defs.items():
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
        N       = len(samples)
        n_test  = max(1, int(round(N * (1 - train_frac - val_frac))))
        n_val   = max(1, int(round(N * val_frac)))
        n_train = N - n_test - n_val
        train_list.extend(samples[:n_train])
        val_list.extend(samples[n_train: n_train + n_val])
        test_list.extend(samples[n_train + n_val:])

    rng.shuffle(train_list)
    rng.shuffle(val_list)

    print(
        f"\nFinal split: train={len(train_list)}  "
        f"val={len(val_list)}  test={len(test_list)}"
    )
    return train_list, val_list, test_list, all_scalers


def load_inference_datasets_vae(
    seed:      int = 42,
    max_nodes: int = 12,
) -> tuple[dict[str, list[Data]], dict[str, DatasetScalers]]:
    """
    Load ALL datasets (including inference-only ones) without a train/val/test
    split.  Each dataset gets its own scaler fitted on the full set.

    Use this to evaluate generalisation on topologies not seen during training
    (e.g. Three_qubit_capacitive_line).

    Returns
    -------
    samples_dict : dict[ds_name → list[Data]]
    scalers_dict : dict[ds_name → DatasetScalers]
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    samples_dict: dict[str, list[Data]]          = {}
    scalers_dict: dict[str, DatasetScalers]      = {}

    print("Loading ALL datasets (inference mode)…")
    for ds_name, defn in DATASETS.items():
        tag = "" if defn.include_train else "  [inference-only]"
        print(f"\n[{ds_name}]{tag}")
        samples, ds_scalers = _build_samples_vae(
            ds_name, defn, scalers=None, fit=True, rng=rng,
            max_nodes=max_nodes,
        )
        samples_dict[ds_name] = samples
        scalers_dict[ds_name] = ds_scalers
        n_nodes = int(samples[-1].enc_n_circuit)
        n_p     = int(samples[-1].y.shape[0])
        print(f"  → {len(samples)} samples | {n_nodes} outer nodes | {n_p} params")

    return samples_dict, scalers_dict

"""
cqed.model.encoder
==================

Description
------------
For each compressed graph through graphlize

    1.  InnerEncoder  — for each SubgType block:
            inner graph 
                ↓  GIN message passing 
                ↓  global_add_pool
            h_v  ∈  R^hv_dim

    2.  OuterGIN  — on outer graph for each node, no pooling:
            [SubgType one-hot | h_v]  per ogni macronodo
                ↓  GIN message passing 
            A'  ∈  R^(N x outer_hidden) 

    3.  Embedding of x^t and x^p :
            x^t  = SubgType one-hot  → Linear → x'^t  ∈  R^(N x d)
            x^p  = normalized position of the node in the outer graph  → Linear → x'^p ∈ R^(N x d)
            A'   → Linear → A''  ∈  R^(N x d)  

    4.  Transformer Encoder:
            For each graph it takes:
                [µ_tok | Σ_tok | x'^t_1 … x'^t_N | (x'^t + x'^p + A'')_1 … _N ]

            Where the transformer's token is:
                tok_i  =  x'^t_i + x'^p_i + A''_i    ∈  R^d
            so it becomes:
                [µ_tok, Σ_tok, tok_1, …, tok_N]  ∈  R^((N+2) x d)

            First two outputs of the transformer are µ'^c e Σ'^c.

    5.  x^b projected using MLPs then concatenated:
            h_v  → MLP  →  hb  ∈  R^d
            µ'^c   → cat([µ'^c,  hb])  → Linear → µ''c   ∈  R^nz
            Σ'^c   → cat([Σ'^c,  hb])  → Linear → Σ''c   ∈  R^nz

    6.  Reparameterization:
            z = µ''c + ε · exp(0.5 · Σ''c),   ε ~ N(0, I)

Notes on inner-node one-hot encoding
------------------------------------
The type_id of an inner node is NOT the SubgType of the macro-node.
It is the index of the circuit element type INSIDE the subgraph
(e.g. for TCT: 0=T1_inductor, 1=T1_cap, 2=coupler, 3=T2_inductor, 4=T2_cap).
So the one-hot is based on the elements of the Subgraph

"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, Batch

from circuit2graph.definitions import SubgType, SUBG_DEFS, ATTR_INDEX


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SUBTYPES: int = len(SubgType)

N_INNER_TYPES: int = max(
    (
        nd["type_id"]
        for defn in SUBG_DEFS.values()
        for nd in defn.inner_nodes
    ),
    default=0,
) + 1

# [type_id one-hot (N_INNER_TYPES) | val_scaled (1)]
INNER_FEAT_DIM: int = N_INNER_TYPES + 1


# ---------------------------------------------------------------------------
#Build the inner features
# ---------------------------------------------------------------------------

def build_encoder_inner_feats(
    block:        "CQEDNode",
    attr_scaled:  dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds (x, edge_index) for the inner graph of a macro-node,
    including scaled physical values as features.

    Parameters
    ----------
    block        : compressed CQEDNode (after graphlize)
    attr_scaled  : dictionary {attr_name: scaled_value} for this block.
                   Obtained by applying ParamScaler to the node raw values.

    Features of each inner node:
        [type_id one-hot (N_INNER_TYPES) | scaled_value (1)]

    Returns
    -------
    x          : [n_inner_nodes, INNER_FEAT_DIM]
    edge_index : [2, n_inner_edges]  (bidirectional)
    """
    defn        = SUBG_DEFS[block.subg_type]
    inner_nodes = defn.inner_nodes
    inner_edges = defn.inner_edges
    n           = len(inner_nodes)

    # Node without internal structure
    # (e.g. FEEDLINE has 1 node and no edges)
    if n == 0:
        return (
            torch.zeros((0, INNER_FEAT_DIM), dtype=torch.float),
            torch.zeros((2, 0),              dtype=torch.long),
        )

    x = torch.zeros((n, INNER_FEAT_DIM), dtype=torch.float)

    for k, nd in enumerate(inner_nodes):
        # one-hot encoding of the circuit element type
        x[k, nd["type_id"]] = 1.0

        # scaled physical value (0 if absent or not scaled)
        if nd["val_attr"] is not None:
            x[k, N_INNER_TYPES] = float(
                attr_scaled.get(nd["val_attr"], 0.0)
            )

    # bidirectional edge_index
    if inner_edges:
        fwd = list(inner_edges)
        bwd = [(b, a) for a, b in fwd]
        ei  = torch.tensor(
            fwd + bwd,
            dtype=torch.long
        ).t().contiguous()
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)

    return x, ei


# ---------------------------------------------------------------------------
# InnerEncoder
# ---------------------------------------------------------------------------
class InnerEncoder(nn.Module):
    """
    GIN over the internal graph of a macro-node → h_v ∈ ℝ^hv_dim.

    A single shared network across all SubgTypes
    (the one-hot type_id already distinguishes the circuit element types).

    Node input:
        [type_id one-hot (N_INNER_TYPES) | scaled_value (1)]
        = INNER_FEAT_DIM dimensions
    """

    def __init__(
        self,
        hv_dim:  int   = 32,
        hidden:  int   = 32,
        layers:  int   = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hv_dim  = hv_dim
        self.dropout = dropout

        # initial projection: INNER_FEAT_DIM → hidden
        self.lin_in = nn.Linear(INNER_FEAT_DIM, hidden)

        # GIN stack
        self.convs = nn.ModuleList([
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden, hidden * 2),
                    nn.ReLU(),
                    nn.Linear(hidden * 2, hidden),
                ),
                train_eps=True,
            )
            for _ in range(layers)
        ])

        # final projection: hidden → hv_dim
        self.lin_out = nn.Linear(hidden, hv_dim)

    def forward(
        self,
        x:          torch.Tensor,   # [total_inner_nodes, INNER_FEAT_DIM]
        edge_index: torch.Tensor,   # [2, total_inner_edges]
        batch:      torch.Tensor,   # [total_inner_nodes]
    ) -> torch.Tensor:              # [n_blocks, hv_dim]
        """
        Processes all macro-nodes of a batch in parallel (PyG batching).

        Returns one embedding h_v for each macro-node.
        """
        x = F.relu(self.lin_in(x))

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # readout: sum all nodes belonging to the same block
        x = global_add_pool(x, batch)       # [n_blocks, hidden]

        return self.lin_out(x)              # [n_blocks, hv_dim]

# ---------------------------------------------------------------------------
# OuterGIN 
# ---------------------------------------------------------------------------
class OuterGIN(nn.Module):
    """
    GIN over the macro-node graph, returning PER-NODE embeddings (A').

    Unlike the previous OuterEncoder, it does NOT apply global_add_pool.
    The output A' ∈ ℝ^(total_macro_nodes × outer_hidden) is then
    projected and passed to the TransformerCircuitEncoder.

    Input node features:
        [SubgType one-hot (N_SUBTYPES=10) | h_v (hv_dim)]
    """

    def __init__(
        self,
        hv_dim:       int   = 32,
        outer_hidden: int   = 64,
        layers:       int   = 3,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.dropout    = dropout
        node_feat_dim   = N_SUBTYPES + hv_dim

        self.lin_in = nn.Linear(node_feat_dim, outer_hidden)

        self.convs = nn.ModuleList([
            GINConv(
                nn.Sequential(
                    nn.Linear(outer_hidden, outer_hidden * 2),
                    nn.ReLU(),
                    nn.Linear(outer_hidden * 2, outer_hidden),
                ),
                train_eps=True,
            )
            for _ in range(layers)
        ])

    def forward(
        self,
        x:          torch.Tensor,   # [total_macro_nodes, N_SUBTYPES + hv_dim]
        edge_index: torch.Tensor,   # [2, total_outer_edges]
    ) -> torch.Tensor:              # [total_macro_nodes, outer_hidden]
        """
        Returns A': one structural embedding for each macro-node.
        """
        x = F.relu(self.lin_in(x))

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x   # [total_macro_nodes, outer_hidden]

# ---------------------------------------------------------------------------
# TransformerCircuitEncoder
# ---------------------------------------------------------------------------
class TransformerCircuitEncoder(nn.Module):
    """
    Transformer Encoder that receives the macro-node sequence and produces
    µ'^c, Σ'^c through two learnable tokens prepended to the sequence.

    Pipeline (for each graph in the batch):
        sequence = [µ_tok | Σ_tok | tok_1 … tok_N]
        tok_i    = proj_t(x^t_i) + proj_p(x^p_i) + proj_A(A'_i)

    The two µ_tok and Σ_tok tokens are learnable, like CLS tokens in BERT.
    The outputs corresponding to tokens 0 and 1 are µ'^c and Σ'^c.

    Then:
        hb    = mlp_xb(h_v)                         projection of x^b
        µ''c  = fc_mu( cat([µ'^c,  hb]) )
        Σ''c  = fc_var( cat([Σ'^c, hb]) )

    Parameters
    ----------
    d            : common Transformer dimension (embedding dim)
    outer_hidden : output dimension of OuterGIN (for proj_A)
    hv_dim       : dimension of h_v / x^b (for mlp_xb)
    nz           : latent space dimension
    nhead        : number of Transformer heads
    tf_layers    : number of Transformer layers
    max_nodes    : maximum sequence length, excluding the 2 tokens, for positional encoding
    dropout      : dropout rate
    """

    def __init__(
        self,
        d:            int   = 64,
        outer_hidden: int   = 64,
        hv_dim:       int   = 32,
        nz:           int   = 32,
        nhead:        int   = 4,
        tf_layers:    int   = 2,
        max_nodes:    int   = 16,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.d  = d
        self.nz = nz

        # ── two learnable tokens (µ^c and Σ^c from CktGen) ───────────────
        self.mu_token  = nn.Parameter(torch.randn(1, 1, d))
        self.sig_token = nn.Parameter(torch.randn(1, 1, d))

        # ── projections for x^t, x^p, A' ─────────────────────────────────
        # x^t: SubgType one-hot → d
        self.proj_t = nn.Linear(N_SUBTYPES, d)

        # x^p: normalized scalar position → d
        self.proj_p = nn.Linear(1, d)

        # A': OuterGIN output → d
        self.proj_A = nn.Linear(outer_hidden, d)

        # ── fixed positional encoding, sinusoidal ────────────────────────
        # +2 for the two prepended tokens
        self._register_pos_enc(max_nodes + 2, d)

        # ── bidirectional Transformer Encoder ────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d,
            nhead           = nhead,
            dim_feedforward = d * 4,
            dropout         = dropout,
            batch_first     = True,   # (batch, seq, d)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)

        # ── x^b projection (h_v from inner physical parameters) ──────────
        self.mlp_xb = nn.Sequential(
            nn.Linear(hv_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # ── final FC layers: cat([token_out, hb]) → µ, logvar ────────────
        self.fc_mu  = nn.Linear(d + d, nz)
        self.fc_var = nn.Linear(d + d, nz)

    # ------------------------------------------------------------------
    def _register_pos_enc(self, max_len: int, d: int) -> None:
        """Creates and registers the fixed sinusoidal positional encoding."""
        pe   = torch.zeros(max_len, d)
        pos  = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div  = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float)
            * (-math.log(10000.0) / d)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pos_enc", pe.unsqueeze(0))   # [1, max_len, d]

    # ------------------------------------------------------------------
    def forward(
        self,
        x_t:         torch.Tensor,   # [total_macro, N_SUBTYPES]  SubgType one-hot
        x_p:         torch.Tensor,   # [total_macro, 1]            normalized position
        A_prime:     torch.Tensor,   # [total_macro, outer_hidden] OuterGIN output
        h_v:         torch.Tensor,   # [total_macro, hv_dim]       InnerEncoder output
        macro_batch: torch.Tensor,   # [total_macro]               sample index
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mu, logvar), shape [batch_size, nz].
        """
        device     = x_t.device
        batch_size = int(macro_batch.max().item()) + 1

        # ── 1. node tokens: x'^t + x'^p + A'' ────────────────────────────
        tok_nodes = self.proj_t(x_t) + self.proj_p(x_p) + self.proj_A(A_prime)
        # tok_nodes: [total_macro, d]

        # ── 2. x^b projection ─────────────────────────────────────────────
        hb = self.mlp_xb(h_v)   # [total_macro, d]

        # ── 3. build padded sequences for the Transformer ────────────────
        # Each graph has N_i nodes; we must pad to the maximum length.
        counts     = torch.bincount(macro_batch, minlength=batch_size)  # [B]
        max_N      = int(counts.max().item())
        seq_len    = max_N + 2   # +2 for the two tokens

        # sequence: (batch, seq_len, d)
        seq = torch.zeros(batch_size, seq_len, self.d, device=device)

        # key_padding_mask: True = position to ignore
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # prepend learnable tokens, same for all graphs
        mu_toks  = self.mu_token.expand(batch_size, -1, -1)    # [B, 1, d]
        sig_toks = self.sig_token.expand(batch_size, -1, -1)   # [B, 1, d]

        seq[:, 0, :] = mu_toks[:, 0, :]
        seq[:, 1, :] = sig_toks[:, 0, :]

        mask[:, 0] = False
        mask[:, 1] = False

        # fill node tokens for each graph in the batch
        offset = 0
        for b in range(batch_size):
            n = int(counts[b].item())

            seq[b, 2:2+n, :] = tok_nodes[offset:offset+n]
            mask[b, 2:2+n]   = False

            offset += n

        # positional encoding, truncated if seq_len exceeds max_nodes + 2
        seq = seq + self.pos_enc[:, :seq_len, :]

        # ── 4. Transformer ────────────────────────────────────────────────
        out = self.transformer(seq, src_key_padding_mask=mask)
        # out: [B, seq_len, d]

        mu_prime  = out[:, 0, :]   # output of µ^c token → µ'^c
        sig_prime = out[:, 1, :]   # output of Σ^c token → Σ'^c

        # ── 5. aggregate hb per graph (global_add_pool over node batch) ──
        # AMP-safe: hb may be float16/bfloat16 under autocast;
        # the scatter_add_ destination must use the same dtype.
        hb_graph = hb.new_zeros(batch_size, self.d)

        hb_graph.scatter_add_(
            0,
            macro_batch.unsqueeze(1).expand_as(hb),
            hb,
        )
        # hb_graph: [B, d]

        # ── 6. final FC layers ────────────────────────────────────────────
        mu = self.fc_mu(
            torch.cat([mu_prime, hb_graph], dim=-1)
        )   # [B, nz]

        logvar = self.fc_var(
            torch.cat([sig_prime, hb_graph], dim=-1)
        )   # [B, nz]

        return mu, logvar

# ---------------------------------------------------------------------------
# GraphVAEEncoder 
# ---------------------------------------------------------------------------
class GraphVAEEncoder(nn.Module):
    """
    Combines all previous components.
    """

    def __init__(
        self,
        hv_dim:       int   = 32,
        inner_hidden: int   = 32,
        inner_layers: int   = 2,
        outer_hidden: int   = 64,
        outer_layers: int   = 3,
        d:            int   = 64,
        nz:           int   = 32,
        nhead:        int   = 4,
        tf_layers:    int   = 2,
        max_nodes:    int   = 16,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.nz = nz

        self.inner_enc = InnerEncoder(
            hv_dim  = hv_dim,
            hidden  = inner_hidden,
            layers  = inner_layers,
            dropout = dropout,
        )

        self.outer_gin = OuterGIN(
            hv_dim       = hv_dim,
            outer_hidden = outer_hidden,
            layers       = outer_layers,
            dropout      = dropout,
        )

        self.tf_enc = TransformerCircuitEncoder(
            d            = d,
            outer_hidden = outer_hidden,
            hv_dim       = hv_dim,
            nz           = nz,
            nhead        = nhead,
            tf_layers    = tf_layers,
            max_nodes    = max_nodes,
            dropout      = dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, batch: "EncoderBatch") -> tuple[torch.Tensor, torch.Tensor]:
        """
        batch → (mu, logvar)

        batch must contain the fields built by _build_enc_tensors() / _prepare_batch():
            inner_x          [total_inner_nodes, INNER_FEAT_DIM]
            inner_ei         [2, total_inner_edges]
            inner_batch      [total_inner_nodes]   batch vector for inner GIN
            macro_subgtype   [total_macro_nodes]   SubgType int for each macro-node
            macro_pos        [total_macro_nodes, 1] normalized position
            macro_batch      [total_macro_nodes]   batch vector for outer GIN
            outer_ei         [2, total_outer_edges]  only circuit↔circuit edges
        """
        device = batch.inner_x.device

        # ── 1. Inner GIN: inner nodes → h_v for each macro-node ──────────
        h_v = self.inner_enc(
            batch.inner_x,
            batch.inner_ei,
            batch.inner_batch,
        )   # [total_macro_nodes, hv_dim]

        # ── 2. Build macro-node features: [one-hot | h_v] ────────────────
        one_hot = torch.zeros(
            h_v.shape[0], N_SUBTYPES, dtype=torch.float, device=device
        )

        one_hot.scatter_(1, batch.macro_subgtype.unsqueeze(1), 1.0)

        x_macro = torch.cat([one_hot, h_v], dim=-1)
        # [total_macro, N_SUBTYPES + hv_dim]

        # ── 3. Outer GIN node-level: x_macro → A' ────────────────────────
        A_prime = self.outer_gin(x_macro, batch.outer_ei)
        # A_prime: [total_macro_nodes, outer_hidden]

        # ── 4. Transformer Encoder: (x^t, x^p, A', h_v) → µ, logvar ─────
        mu, logvar = self.tf_enc(
            x_t         = one_hot,          # x^t: SubgType one-hot
            x_p         = batch.macro_pos,  # x^p: normalized position [total_macro, 1]
            A_prime     = A_prime,
            h_v         = h_v,
            macro_batch = batch.macro_batch,
        )

        return mu, logvar

    def reparameterize(
        self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        z = µ + ε·σ.
        During evaluation, directly returns µ.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        return mu

    def encode(
        self,
        batch: "EncoderBatch",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shortcut: forward + reparameterize.

        Returns (z, mu, logvar).
        """
        mu, logvar = self.forward(batch)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    @staticmethod
    def kl_loss(
        mu:     torch.Tensor,
        logvar: torch.Tensor,
        beta:   float = 1.0,
    ) -> torch.Tensor:
        """
        KL divergence KL(q(z|G) || N(0,I)), averaged over the batch.

        β controls the weight of the KL term (β-VAE).
        """
        kl = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return beta * kl



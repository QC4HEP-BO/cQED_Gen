"""
cqed.model.decoder
==================
Decoder GPT-like.

Architecture
------------_
The decoder is made up by three pieces:

1. TransformerTopologyDecoder  — topological decoder (GPT-like)
   z  →  x^t (node types) + x^p (positions) + x^e (edges)

   Workflow:
       z  ∈  R^nz
       ↓  fc_z  (Linear + tanh)
       z' ∈  R^d''

       Node generation (autogressive on x^t): # at each step the transformer sees z' + node generated and creates the next node; formally:
           x'^t_i  = EmbedLayer(x^t_i)                   ∈  R^d''                       
           x''^t   = [z'; x'^t_1; …; x'^t_i]             ∈  R^((i+1) x d'')
           g^t_i   = p(x^t_i | x''^t_{<i}, z)            via casual Transformer + Linear

       x^p analogous.

       Edge generation:
           A' ∈ R^(N x d'')                         #since we have already generate the graph nodes
           A''= cat([z'; A'])  ∈  R^((N+1) x d'')   #to this, we concatenate z'
           Transformer(A'') → x''^e ∈  R^((N+1) x d)
           y_{j→i} = [x''^e_i, x''^e_j]  ∈  R^2d
           for each i: P(edge j→i) = MLP_e(Y_i)  ∈  [0,1] #i.e. for each edge y_{j→i}, we compute its probability with a MLP

2. ParamDecoder  — parametrs decoder  
   (z, G)  →  θ̂' 

3.  Reco Loss:
       L_R = λ_t · L_t(x^t, x̂^t)          cross-entropy on node type
           + λ_p · L_p(x^p, x̂^p)          cross-entropy on positions
           + L_e(x^e, x̂^e)                 BCE on edges
           + λ_b · L_b(x^b, x̂^b)          MSE on phyiscal parameters


Inference
---------
    z^s  = vae.spec_encoder.encode(obs_vals, obs_mask)[0]
    G    = vae.decoder.decode(z^s)                
    θ̂'   = vae.param_decoder.predict(z^s, G)     


"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from circuit2graph.definitions import SubgType, SUBG_DEFS, ATTR_INDEX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SUBTYPES:   int = len(SubgType)           # 10
N_ATTRS:      int = len(ATTR_INDEX)         # 10
START_TYPE:   int = N_SUBTYPES              # 10 
END_TYPE:     int = N_SUBTYPES + 1          # 11 
N_NODE_TYPES: int = N_SUBTYPES + 2          # 12 
MAX_NODES:    int = 8                       
N_POSITIONS:  int = MAX_NODES + 1

# Mask
_ATTRS_MASK: dict[int, torch.Tensor] = {}
for _st in SubgType:
    _mask = torch.zeros(N_ATTRS)
    for _attr in SUBG_DEFS[_st].attrs:
        _mask[ATTR_INDEX[_attr]] = 1.0
    _ATTRS_MASK[int(_st)] = _mask


def get_attrs_mask(st_int: int, device: torch.device) -> torch.Tensor:
    return _ATTRS_MASK[st_int].to(device)


def build_type_class_weights(
    graphs: list[SimpleNamespace] | None = None,
    device: torch.device | None = None,
    end_weight: float = 1.0,
    min_count: float = 1.0,
) -> torch.Tensor:
    """
    Builds weight proportionally to 1/frequency for the loss_t.

    Layout vocaboulary:
        0..N_SUBTYPES-1 : real tyoes
        START_TYPE      : token START 
        END_TYPE        : token END

    Weights are defined for all types, but START is set to 0 since
    it should not appear in the loss.
    """
    counts = torch.full((N_NODE_TYPES,), float(min_count), dtype=torch.float)

    if graphs is not None:
        for g in graphs:
            for tp in g.node_types:
                if 0 <= int(tp) < N_SUBTYPES:
                    counts[int(tp)] += 1.0
            counts[END_TYPE] += 1.0

    counts[START_TYPE] = 0.0

    weights = torch.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / counts[nonzero]

    real_mask = torch.zeros_like(weights, dtype=torch.bool)
    real_mask[:N_SUBTYPES] = True
    real_mask[END_TYPE] = True
    if weights[real_mask].sum() > 0:
        weights[real_mask] = weights[real_mask] * (real_mask.sum() / weights[real_mask].sum())

    weights[END_TYPE] = weights[END_TYPE] * float(end_weight)

    if device is not None:
        weights = weights.to(device)
    return weights



class _CausalTransformer(nn.Module):
    """
    Casual Transformer encoder (GPT-like) with autoregressive masking.

    Uses nn.TransformerEncoder with causal mask.
    z' is passed as the first token and seerves as a global conditioning
    each subsequent position can see z through the casual mask .

    Input:  seq [B, seq_len, d] 
    Output: [B, seq_len, d]
    """

    def __init__(self, d: int, nhead: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d,
            nhead           = nhead,
            dim_feedforward = d * 4,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive: each position i con only see positions 0..i.
        """
        seq_len = x.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        return self.transformer(x, mask=causal_mask, is_causal=True)


class _BiTransformer(nn.Module):
    """
    Bidirectional Transformer.

    Used exclusively for the edges x^e.

    Input:  seq [B, seq_len, d]
    Output: [B, seq_len, d]
    """

    def __init__(self, d: int, nhead: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d,
            nhead           = nhead,
            dim_feedforward = d * 4,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class _GNNLayer(nn.Module):
    """GCN-style layer for generating edges (A → A')."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, N, d]
        adj : [B, N, N]  
        """
        return F.relu(self.lin(torch.bmm(adj, x)))


# ---------------------------------------------------------------------------
# TransformerTopologyDecoder
# ---------------------------------------------------------------------------

class TransformerTopologyDecoder(nn.Module):
    """
    Decoder GPT-like: z → nodes types + position + edges.

    Does NOT predict parameters (ParamDecoder does that).

    """

    def __init__(
        self,
        nz:        int   = 32,
        d:         int   = 512,
        nhead:     int   = 8,
        n_layers:  int   = 4,
        max_nodes: int   = MAX_NODES,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.nz        = nz
        self.d         = d
        self.max_nodes = max_nodes

        # ── z → z'  ────────────────────────────────────────────
        self.fc_z = nn.Sequential(
            nn.Linear(nz, d),
            nn.Tanh(),
        )

        # ── node types  ─────────────────────────────────────────
        self.embed_t   = nn.Embedding(N_NODE_TYPES, d)
        self.tf_t      = _CausalTransformer(d, nhead, n_layers, dropout)
        self.head_t    = nn.Linear(d, N_NODE_TYPES)   

        # ── positions  ──────────────────────────────
        self.embed_p   = nn.Embedding(N_POSITIONS, d)
        self.tf_p      = _CausalTransformer(d, nhead, n_layers, dropout)
        self.head_p    = nn.Linear(d, N_POSITIONS)   

        # ── edges  ───────────
        self.gnn_edge  = _GNNLayer(d)
        self.tf_e      = _BiTransformer(d, nhead, n_layers, dropout)
        self.proj_e    = nn.Linear(d, d)              # x'^e → x''^e [d]
        self.mlp_e     = nn.Sequential(               # [2d] → P(edge)
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self.register_buffer(
            "edge_pos_weight",
            torch.tensor([2.0]),
        )

        self._register_sinusoidal(max_nodes + 1, d)  

    # ------------------------------------------------------------------

    def _register_sinusoidal(self, max_len: int, d: int) -> None:
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))   # [1, max_len, d]

    def set_edge_pos_weight(self, w: float) -> None:
        """
        Sets the pos_weight of the BCE loss on edges.

        Recommended calibration: pos_weight = (# negative slots) / (# positive slots)
        on a sample of the dataset.  Quick example:
            edges_pos, edges_neg = 0, 0
            for g in train_graphs:
                n = len(g.node_types)
                tot = n * (n-1) // 2
                pos = len(g.edges)
                edges_pos += pos; edges_neg += tot - pos
            w = edges_neg / max(edges_pos, 1)
        Typical values for cQED: 2–5.
        """
        self.edge_pos_weight = torch.tensor([float(w)], device=self.edge_pos_weight.device)

    # ------------------------------------------------------------------
    # Branch x^t — node types
    # ------------------------------------------------------------------

    def _forward_types(
        self,
        z_prime:     torch.Tensor,   # [B, d]
        type_seq:    torch.Tensor,   # [B, seq_len]  integers (including START at pos 0)
    ) -> torch.Tensor:
        """
        Returns logits [B, seq_len, N_NODE_TYPES].

        Input sequence to the TransformerEncoder: [z'; embed(t_0); …; embed(t_{S-1})].
        z' is at position 0 with causal mask → every token always sees it.
        The output at position i (0-based in output, 1-based in seq) predicts t_{i}.
        """
        B, S = type_seq.shape
        emb = self.embed_t(type_seq)               # [B, S, d]
        # z' as first token: global conditioning visible to the entire sequence
        seq = torch.cat([z_prime.unsqueeze(1), emb], dim=1)  # [B, S+1, d]
        seq = seq + self.pos_enc[:, :S+1, :]
        out = self.tf_t(seq)                        # [B, S+1, d]
        # logits for positions 1..S (output[0] = z', output[1..S] predicts the types)
        return self.head_t(out[:, 1:, :])           # [B, S, N_NODE_TYPES]

    # ------------------------------------------------------------------
    # Branch x^p — positions
    # ------------------------------------------------------------------

    def _forward_positions(
        self,
        z_prime:  torch.Tensor,   # [B, d]
        pos_seq:  torch.Tensor,   # [B, seq_len]  interi 0..N_POSITIONS-1
    ) -> torch.Tensor:
        """Returns logits [B, seq_len, N_POSITIONS].
        z' at position 0 as global conditioning (causal mask).
        """
        B, S = pos_seq.shape
        emb = self.embed_p(pos_seq)
        seq = torch.cat([z_prime.unsqueeze(1), emb], dim=1)  # [B, S+1, d]
        seq = seq + self.pos_enc[:, :S+1, :]
        out = self.tf_p(seq)
        return self.head_p(out[:, 1:, :])           # [B, S, N_POSITIONS]

    # ------------------------------------------------------------------
    # Branch x^e — edges
    # ------------------------------------------------------------------

    def _forward_edges(
        self,
        z_prime:   torch.Tensor,   # [B, d]
        type_embs: torch.Tensor,   # [B, N, d]  embedding dei tipi reali
        adj:       torch.Tensor,   # [B, N, N]  adjacency (ALWAYS zero: no teacher forcing)
    ) -> torch.Tensor:
        """
        Returns edge logits [B, N*(N-1)/2, 1] in lower-triangular layout
        (edges j→i with j<i, scanned by increasing row i).

        Pipeline (§3.2.3 CktGen):
            A' = GNN(type_embs, adj)                ∈  ℝ^(N × d)
            A''= cat([z', A'])  con pos_enc         ∈  ℝ^((N+1) × d)
            x'^e = BiTransformer(A'')               ∈  ℝ^((N+1) × d)   ← BIDIREZIONALE
            x''^e= proj_e(x'^e)                     ∈  ℝ^((N+1) × d)
            for each i, j<i:
                y_{j→i} = [x''^e_i, x''^e_j]       ∈  ℝ^(2d)
            edge_logits = mlp_e(Y)                  ∈  ℝ^(N*(N-1)/2, 1)

        IMPLEMENTATION NOTES:
        - adj is always zero (both in training and inference): eliminates the
          mismatch caused by teacher forcing on the real topology.
          The GNN with adj=0 degenerates to a linear projection per node
          (bmm(0,x)=0, output = ReLU(lin(0)) = 0 → residual), but is kept
          for architectural compatibility and possible future use with partial structure.
        - tf_e is BIDIRECTIONAL: each node can attend to all others
          before predicting its own connectivity.
        """
        B, N, _ = type_embs.shape

        # GNN structurale
        A_prime = self.gnn_edge(type_embs, adj)     # [B, N, d]

        # concat z' come primo token
        seq = torch.cat([z_prime.unsqueeze(1), A_prime], dim=1)   # [B, N+1, d]
        seq = seq + self.pos_enc[:, :N+1, :]
        x_prime_e  = self.tf_e(seq)                  # [B, N+1, d]
        x_pp_e     = self.proj_e(x_prime_e)           # [B, N+1, d]

        # costruisci coppie (i, j) con j < i — versione VETTORIZZATA
        # tril_indices generates all (i, j) indices with j < i at once,
        # eliminating the O(N²) Python loop and N²/2 torch.cat calls.
        if N < 2:
            return torch.zeros(B, 0, 1, device=z_prime.device)

        # rows=i, cols=j  (1-based offset: +1 perché x_pp_e[:,0,:] è z')
        rows, cols = torch.tril_indices(N, N, offset=-1, device=z_prime.device)
        # x_pp_e: [B, N+1, d]; indicizziamo con rows+1 e cols+1 (0 è z')
        xi = x_pp_e[:, rows + 1, :]   # [B, n_pairs, d]
        xj = x_pp_e[:, cols + 1, :]   # [B, n_pairs, d]
        Y  = torch.cat([xi, xj], dim=-1)   # [B, n_pairs, 2d]
        return self.mlp_e(Y)               # [B, n_pairs, 1]

    # ------------------------------------------------------------------
    # tensorize_G_true — helper condiviso tra ramo z^c e ramo z^s
    # ------------------------------------------------------------------

    def tensorize_G_true(
        self,
        G_true: list[SimpleNamespace],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Converts G_true in tensors in order to compute the loss.

        It saves time since it can be computed tensorized and one time
        both for z_c and z_s.

        Returns
        -------
        type_seq : [B, M]      long
        pos_seq  : [B, M]      long
        adj_true : [B, M, M]   float
        seq_lens : list[int]
        """
        B = len(G_true)
        M = self.max_nodes
        type_seq = torch.full((B, M), START_TYPE, dtype=torch.long, device=device)
        pos_seq  = torch.zeros(B, M, dtype=torch.long, device=device)
        adj_true = torch.zeros(B, M, M, dtype=torch.float, device=device)
        seq_lens: list = []
        for b, g in enumerate(G_true):
            n = len(g.node_types)
            seq_lens.append(n)
            type_seq[b, 0] = START_TYPE
            for k, tp in enumerate(g.node_types):
                if k + 1 < M:
                    type_seq[b, k + 1] = tp
            end_pos = min(n + 1, M - 1)
            type_seq[b, end_pos] = END_TYPE #e.g. tensor([[START, 3, 7, 2, END]])
            for k in range(n):
                if k + 1 < M:
                    pos_seq[b, k + 1] = k + 1 #e.g. tensor([[0, 1, 2, 3, 0]])
            for u, v in g.edges:
                ui, vi = u + 1, v + 1
                if ui < M and vi < M:
                    adj_true[b, ui, vi] = 1.0
                    adj_true[b, vi, ui] = 1.0 #e.g. tensor([[0,0,0,0,0,0],# START [0,0,1,0,0,0],# nodo 3rd type [0,1,0,1,0,0],  # nodo 7th type, ...])
        return type_seq, pos_seq, adj_true, seq_lens

    # ------------------------------------------------------------------
    # loss — L_t + L_p + L_e  (senza L_b che è in ParamDecoder)
    # ------------------------------------------------------------------

    def loss(
        self,
        z:      torch.Tensor,                # [B, nz]
        G_true: list[SimpleNamespace],       # da data_to_graph_ns()
        mu:     torch.Tensor | None = None,
        logvar: torch.Tensor | None = None,
        beta:   float = 1.0,
        lambda_t: float = 0.5,
        lambda_p: float = 0.05,
        type_class_weights: torch.Tensor | None = None,
        # ── pre-built tensors (optional, fast path) ──────────────────────
        # If provided, skips the Python loop that tensorises G_true.
        # Call tensorize_G_true() once in vae.forward() and pass
        # the results to both loss() calls (z^c and z^s).
        _type_seq: "torch.Tensor | None" = None,
        _pos_seq:  "torch.Tensor | None" = None,
        _adj_true: "torch.Tensor | None" = None,
        _seq_lens: "list | None" = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Topological loss (Eq. 7, without L_b):
            L_topo = λ_t · L_t  +  λ_p · L_p  +  L_e  +  β · KL

        Teacher forcing on all branches.

        Parameters
        ----------
        lambda_t, lambda_p : relative weights (defaults from the paper)
        _type_seq, _pos_seq, _adj_true, _seq_lens : pre-built tensors
            from tensorize_G_true(). If provided, the tensorisation loop
            is skipped (saves ~18% time per epoch).

        Returns
        -------
        loss_topo  : scalar
        components : dict with individual terms for logging
        """
        device = z.device
        B      = z.shape[0]
        M      = self.max_nodes

        z_prime = self.fc_z(z)   # [B, d]

        # ── tensorise G_true (or use pre-built tensors) ──────────────────
        if _type_seq is not None:
            # fast path: tensors already built, no Python loop
            type_seq = _type_seq
            pos_seq  = _pos_seq
            adj_true = _adj_true
            seq_lens = _seq_lens
        else:
            # slow path: build on the fly (backward compat / slow-path fallback)
            type_seq, pos_seq, adj_true, seq_lens = self.tensorize_G_true(G_true, device)

        # ── L_t: cross-entropy on node types ────────────────────────────
        # Input:  type_seq[:, :-1]  (everything except the last)
        # Target: type_seq[:, 1:]   (everything except the first — shifted)
        # Sequence goes from 0 to M-1; we predict positions 1..M-1
        logits_t = self._forward_types(
            z_prime,
            type_seq[:, :-1],    # [B, M-1]
        )                         # [B, M-1, N_NODE_TYPES]

        target_t = type_seq[:, 1:].clone()   # [B, M-1]

        # START must never be emitted as decoder output.
        logits_t = logits_t.clone()
        logits_t[..., START_TYPE] = -1e9

        # direct CE target over the full vocabulary: preserves END_TYPE.
        target_t_ce = target_t.clone()

        # mask: do not compute loss on positions after END
        mask_t = torch.zeros(B, M - 1, dtype=torch.bool, device=device)
        for b, sl in enumerate(seq_lens):
            mask_t[b, :min(sl + 1, M - 1)] = True   # +1 per includere il token END

        ce_kwargs = {}
        if type_class_weights is not None:
            ce_kwargs["weight"] = type_class_weights.to(device)

        loss_t = F.cross_entropy(
            logits_t[mask_t].reshape(-1, N_NODE_TYPES),
            target_t_ce[mask_t].reshape(-1),
            **ce_kwargs,
        ) if mask_t.any() else torch.zeros(1, device=device)

        # ── L_p: cross-entropy on positions ──────────────────────────────
        logits_p = self._forward_positions(
            z_prime,
            pos_seq[:, :-1],     # [B, M-1]
        )                         # [B, M-1, N_POSITIONS]

        target_p = pos_seq[:, 1:].clone()   # [B, M-1]

        mask_p = torch.zeros(B, M - 1, dtype=torch.bool, device=device)
        for b, sl in enumerate(seq_lens):
            mask_p[b, :min(sl, M - 1)] = True   # real positions (no END)

        loss_p = F.cross_entropy(
            logits_p[mask_p].reshape(-1, N_POSITIONS),
            target_p[mask_p].reshape(-1),
        ) if mask_p.any() else torch.zeros(1, device=device)

        # ── L_e: BCE on edges ────────────────────────────────────────────
        # NOTE: we pass adj_zero (all zeros) to the GNN, NOT the real adjacency.
        # This eliminates the training/inference mismatch:
        #   - Training with teacher forcing on adj → the GNN learns to depend
        #     on the real adjacency, which is never available at inference.
        #   - Without teacher forcing → training and inference use the same input
        #     distribution; z carries all structural signal.
        # The GNN on the edge branch serves to propagate type information
        # between neighbouring nodes, not to "read" the real topology.
        # ── L_e: BCE on edges — BATCHED version ──────────────────────────
        # Invece di B forward pass separati, paddiamo tutti i grafi a
        # max_real_n nodi e facciamo una sola forward su [B_valid, max_n, d].
        # I grafi con n<2 sono esclusi (nessun arco possibile).
        # La maschera edge_mask annulla le coppie di padding.
        valid_bs = [b for b, g in enumerate(G_true) if len(g.node_types) >= 2]
        loss_e   = torch.zeros(1, device=device)

        if valid_bs:
            max_real_n = max(len(G_true[b].node_types) for b in valid_bs)
            Bv         = len(valid_bs)

            # Tensor of real node-type embeddings (padded to max_real_n)
            type_ids_pad = torch.zeros(Bv, max_real_n, dtype=torch.long, device=device)
            for bi, b in enumerate(valid_bs):
                g = G_true[b]
                n = len(g.node_types)
                type_ids_pad[bi, :n] = torch.tensor(
                    g.node_types, dtype=torch.long, device=device
                )

            type_embs_pad = self.embed_t(type_ids_pad)      # [Bv, max_real_n, d]
            adj_zero_pad  = torch.zeros(Bv, max_real_n, max_real_n, device=device)
            z_prime_valid = z_prime[[b for b in valid_bs]]  # [Bv, d]

            edge_logits_pad = self._forward_edges(
                z_prime_valid,
                type_embs_pad,
                adj_zero_pad,
            )   # [Bv, max_real_n*(max_real_n-1)//2, 1]

            # Build target and mask for every graph in the valid batch
            n_pairs   = max_real_n * (max_real_n - 1) // 2
            target_e  = torch.zeros(Bv, n_pairs,          device=device)
            edge_mask = torch.zeros(Bv, n_pairs, dtype=torch.bool, device=device)

            for bi, b in enumerate(valid_bs):
                g   = G_true[b]
                n   = len(g.node_types)
                idx = 0
                for i in range(max_real_n):
                    for j in range(i):
                        if i < n and j < n:
                            target_e[bi, idx]  = adj_true[b, i + 1, j + 1]
                            edge_mask[bi, idx] = True
                        idx += 1

            # BCE only on non-padding pairs
            if edge_mask.any():
                logits_valid  = edge_logits_pad[:, :, 0][edge_mask]
                targets_valid = target_e[edge_mask]
                loss_e = F.binary_cross_entropy_with_logits(
                    logits_valid,
                    targets_valid,
                    pos_weight=self.edge_pos_weight.to(device),
                    reduction="mean",
                )

        # ── KL ───────────────────────────────────────────────────────────
        kl_raw = torch.zeros(1, device=device)
        if mu is not None and logvar is not None:
            kl_raw = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # ── loss totale ───────────────────────────────────────────────────
        loss_topo = lambda_t * loss_t + lambda_p * loss_p + loss_e + beta * kl_raw

        components = {
            "loss_topo":   loss_topo.item(),
            "loss_t":      loss_t.item(),
            "loss_p":      loss_p.item(),
            "loss_e":      loss_e.item(),
            "loss_kl":     kl_raw.item(),
        }
        return loss_topo, components

    # ------------------------------------------------------------------
    # forward — shortcut for the training loop
    # ------------------------------------------------------------------

    def forward(
        self,
        z:        torch.Tensor,
        G_true:   list[SimpleNamespace],
        mu:       torch.Tensor | None = None,
        logvar:   torch.Tensor | None = None,
        beta:     float = 1.0,
        lambda_t: float = 0.5,
        lambda_p: float = 0.05,
        type_class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            z, G_true,
            mu=mu, logvar=logvar, beta=beta,
            lambda_t=lambda_t, lambda_p=lambda_p,
            type_class_weights=type_class_weights,
        )

    # ------------------------------------------------------------------
    # decode — genera topologia da z (inferenza autogressiva)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        z:          torch.Tensor,
        stochastic: bool = True,
    ) -> list[SimpleNamespace]:
        """
        Generates the topology for each z in the batch.

        Autoregressive flow (§3.2.3, inference stage):
            1. Generate node types one at a time until END or max_nodes
            2. Generate positions (one per node, conditioned on z)
            3. Generate edges with GNN + Transformer on the cumulative adjacency

        Returns a list of SimpleNamespace:
            node_types : list[int]        SubgType int for each macro-node
            edges      : list[(int,int)]  edges between macro-nodes (0-based)
            attrs      : list[dict]       empty dicts (to be filled by ParamDecoder)
        """
        device = z.device
        B      = z.shape[0]
        z_prime = self.fc_z(z)   # [B, d]

        results = []

        for b in range(B):
            zp = z_prime[b:b+1]   # [1, d]

            # ── 1. generate node types ───────────────────────────────────
            node_types: list[int] = []
            current_types = torch.tensor([START_TYPE], dtype=torch.long, device=device).unsqueeze(0)
            # current_types: [1, seq_so_far]

            for step in range(self.max_nodes):
                logits = self._forward_types(zp, current_types)   # [1, seq_so_far, N_NODE_TYPES]
                last_logit = logits[0, -1, :].clone()              # [N_NODE_TYPES]

                # START is not a valid output token during inference.
                last_logit[START_TYPE] = -1e9

                if stochastic:
                    probs    = F.softmax(last_logit, dim=-1)
                    new_type = int(torch.multinomial(probs, 1).item())
                else:
                    new_type = int(last_logit.argmax().item())

                if new_type == END_TYPE:
                    break

                # defensive fallback: ignore any unexpected special tokens
                if new_type >= N_SUBTYPES:
                    continue

                node_types.append(new_type)
                new_tok = torch.tensor([[new_type]], dtype=torch.long, device=device)
                current_types = torch.cat([current_types, new_tok], dim=1)

            if not node_types:
                r = SimpleNamespace()
                r.node_types = []
                r.edges      = []
                r.attrs      = []
                results.append(r)
                continue

            N = len(node_types)

            # ── 2. generate positions ────────────────────────────────────
            # cQED positions are essentially the rank along the path; we use
            # predicted positions only for consistency with loss L_p.
            # At inference they are used for completeness but do not change
            # the graph structure.
            current_pos = torch.zeros(1, 1, dtype=torch.long, device=device)
            positions: list[int] = []

            for step in range(N):
                logits_p = self._forward_positions(zp, current_pos)  # [1, *, N_POSITIONS]
                last_lp  = logits_p[0, -1, :]

                if stochastic:
                    probs_p = F.softmax(last_lp, dim=-1)
                    new_pos = int(torch.multinomial(probs_p, 1).item())
                else:
                    new_pos = int(last_lp.argmax().item())

                positions.append(new_pos)
                new_p = torch.tensor([[new_pos]], dtype=torch.long, device=device)
                current_pos = torch.cat([current_pos, new_p], dim=1)

            # ── 3. generate edges ────────────────────────────────────────
            edges: list[tuple[int, int]] = []

            if N >= 2:
                type_ids   = torch.tensor(node_types, dtype=torch.long, device=device)
                type_embs  = self.embed_t(type_ids).unsqueeze(0)   # [1, N, d]

                # cumulative adjacency (starts empty, updated after each edge)
                adj_running = torch.zeros(1, N, N, device=device)

                edge_logits = self._forward_edges(zp, type_embs, adj_running)
                # [1, N*(N-1)/2, 1]

                idx = 0
                for i in range(N):
                    for j in range(i):
                        if edge_logits.shape[1] > idx:
                            score = torch.sigmoid(edge_logits[0, idx, 0])
                            if stochastic:
                                if torch.rand(1, device=device).item() < score.item():
                                    edges.append((j, i))
                                    adj_running[0, i, j] = 1.0
                                    adj_running[0, j, i] = 1.0
                            else:
                                if score.item() > 0.5:
                                    edges.append((j, i))
                                    adj_running[0, i, j] = 1.0
                                    adj_running[0, j, i] = 1.0
                        idx += 1

            r            = SimpleNamespace()
            r.node_types = node_types
            r.edges      = edges
            r.attrs      = [{} for _ in node_types]
            results.append(r)

        return results


# ---------------------------------------------------------------------------
# ParamDecoder — predicts physical attributes given (z, G)
# (unchanged from the previous version — corresponds to x^b in the paper)
# ---------------------------------------------------------------------------

class ParamDecoder(nn.Module):
    """
    Physical parameter decoder: (z, G) -> theta_hat

    Pure z-first architecture (no structural residual):

        pred = base_from_z(z, x_struct)

    z is projected and used as the sole source for parameter prediction,
    in line with the CktGen approach (f_size directly from z).
    SAGE is kept to enrich the per-node representation with context
    information, but does not add a separate residual term.
    """

    def __init__(
        self,
        nz: int = 32,
        hidden: int = 64,
        sage_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nz          = nz
        self.hidden      = hidden
        self.sage_layers = sage_layers
        self.dropout_p   = dropout

        self.current_epoch = 0  # kept for checkpoint compatibility (set_epoch is a no-op)

        self.struct_dim = N_SUBTYPES + 2   # [one-hot tipo | deg_norm | pos_norm]

        self.z_proj = nn.Sequential(
            nn.Linear(nz, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.struct_in = nn.Linear(self.struct_dim, hidden)

        self.sage_self  = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(sage_layers)])
        self.sage_neigh = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(sage_layers)])
        self.sage_z     = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(sage_layers)])

        self.base_heads = nn.ModuleDict()

        for st in SubgType:
            n_out = len(SUBG_DEFS[st].attrs)
            if n_out == 0:
                continue
            key = str(int(st))
            self.base_heads[key] = nn.Sequential(
                nn.Linear(hidden + self.struct_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_out),
            )

    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """No-op kept for compatibility with train_vae.py."""
        self.current_epoch = int(epoch)

    def _build_struct_feats(
        self, node_types: list[int], edge_index: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        N = len(node_types)
        x = torch.zeros(N, self.struct_dim, device=device)
        deg = torch.zeros(N, device=device)
        if edge_index.shape[1] > 0:
            deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=device))
        max_deg  = max(float(deg.max().item()), 1.0) if N > 0 else 1.0
        pos_denom = max(N - 1, 1)
        for i, st_int in enumerate(node_types):
            x[i, st_int]         = 1.0
            x[i, N_SUBTYPES]     = deg[i] / max_deg
            x[i, N_SUBTYPES + 1] = i / pos_denom
        return x

    def _sage_forward(
        self, h: torch.Tensor, z_h: torch.Tensor, edge_index: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        N = h.shape[0]
        W_self, W_neigh, W_z = self.sage_self[layer_idx], self.sage_neigh[layer_idx], self.sage_z[layer_idx]
        z_term = W_z(z_h)
        if edge_index.shape[1] == 0:
            return F.relu(W_self(h) + z_term)
        src, dst = edge_index[0], edge_index[1]
        neigh_sum = torch.zeros_like(h)
        neigh_cnt = torch.zeros(N, 1, device=h.device)
        neigh_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(h[src]), h[src])
        neigh_cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.shape[0], 1, device=h.device))
        return F.relu(W_self(h) + W_neigh(neigh_sum / neigh_cnt.clamp(min=1.0)) + z_term)

    def _embed_nodes(
        self, z_single: torch.Tensor, node_types: list[int], edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device   = z_single.device
        N        = len(node_types)
        x_struct = self._build_struct_feats(node_types, edge_index, device)
        z_h      = self.z_proj(z_single).unsqueeze(0).expand(N, -1)
        h        = F.relu(self.struct_in(x_struct) + z_h)
        for li in range(self.sage_layers):
            h = self._sage_forward(h, z_h, edge_index, li)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        return x_struct, h

    def _predict_node(
        self, z_single: torch.Tensor, x_struct_i: torch.Tensor, h_i: torch.Tensor, st_int: int
    ) -> torch.Tensor:
        key    = str(st_int)
        z_base = self.z_proj(z_single)
        return self.base_heads[key](torch.cat([z_base, x_struct_i], dim=-1))

    # ------------------------------------------------------------------

    def _embed_nodes_batched(
        self,
        z:          torch.Tensor,         # [B, nz]
        G_true:     list[SimpleNamespace],
        device:     torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """
        Fully batched version of _embed_nodes.

        Builds a single "mega-batch" graph by concatenating all graphs in
        G_true, runs struct_in + SAGE layers with a global scatter_add_ (one
        kernel launch per layer instead of B separate launches),
        and returns the per-node vectors already split.

        Returns
        -------
        x_struct_all : [total_nodes, struct_dim]
        h_all        : [total_nodes, hidden]
        node_to_graph: [total_nodes]  int — graph index for each node
        node_types_flat: list[int]    — SubgType for each node (flat order)
        """
        B = len(G_true)

        # ── 1. Build flat x_struct and global edge_index ────────────────
        x_struct_list: list[torch.Tensor] = []
        ei_list:       list[torch.Tensor] = []
        node_to_graph: list[int]          = []
        node_types_flat: list[int]        = []
        node_offset = 0

        for b, g in enumerate(G_true):
            N  = len(g.node_types)
            ei = _build_edge_index_from_list(g.edges, device)

            # x_struct for this graph (CPU-side loop over small N ~3-4 nodes)
            xs = torch.zeros(N, self.struct_dim, device=device)
            deg = torch.zeros(N, device=device)
            if ei.shape[1] > 0:
                deg.scatter_add_(0, ei[1], torch.ones(ei.shape[1], device=device))
            max_deg   = max(float(deg.max().item()), 1.0)
            pos_denom = max(N - 1, 1)
            for i, st_int in enumerate(g.node_types):
                xs[i, st_int]         = 1.0
                xs[i, N_SUBTYPES]     = deg[i] / max_deg
                xs[i, N_SUBTYPES + 1] = i / pos_denom
            x_struct_list.append(xs)

            # global edge_index (index shift)
            if ei.shape[1] > 0:
                ei_list.append(ei + node_offset)
            node_to_graph.extend([b] * N)
            node_types_flat.extend(g.node_types)
            node_offset += N

        total_nodes = node_offset
        if total_nodes == 0:
            empty = torch.zeros(0, self.hidden, device=device)
            return (torch.zeros(0, self.struct_dim, device=device),
                    empty, [], [])

        x_struct_all = torch.cat(x_struct_list, dim=0)   # [total_nodes, struct_dim]

        # single global edge_index
        if ei_list:
            ei_global = torch.cat(ei_list, dim=1)         # [2, total_edges]
        else:
            ei_global = torch.zeros((2, 0), dtype=torch.long, device=device)

        # ── 2. z replicated per node (repeat_interleave = single kernel) ──
        n_per_graph = torch.tensor(
            [len(g.node_types) for g in G_true], dtype=torch.long, device=device
        )
        z_per_node = torch.repeat_interleave(z, n_per_graph, dim=0)  # [total_nodes, nz]

        # ── 3. SAGE layers on the mega-batch — one kernel launch per layer ─
        z_h = self.z_proj(z_per_node)                     # [total_nodes, hidden]
        h   = F.relu(self.struct_in(x_struct_all) + z_h)  # [total_nodes, hidden]

        for li in range(self.sage_layers):
            W_self  = self.sage_self[li]
            W_neigh = self.sage_neigh[li]
            W_z     = self.sage_z[li]
            z_term  = W_z(z_h)
            if ei_global.shape[1] == 0:
                h = F.relu(W_self(h) + z_term)
            else:
                src, dst = ei_global[0], ei_global[1]
                neigh_sum = torch.zeros_like(h)
                neigh_cnt = torch.zeros(total_nodes, 1, device=device)
                neigh_sum.scatter_add_(
                    0, dst.unsqueeze(1).expand_as(h[src]), h[src]
                )
                neigh_cnt.scatter_add_(
                    0, dst.unsqueeze(1),
                    torch.ones(src.shape[0], 1, device=device)
                )
                h = F.relu(
                    W_self(h)
                    + W_neigh(neigh_sum / neigh_cnt.clamp(min=1.0))
                    + z_term
                )
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        return x_struct_all, h, node_to_graph, node_types_flat

    def loss(
        self,
        z:      torch.Tensor,
        G_true: list[SimpleNamespace],
        attr_perm_indices: list[list[list[int]]] | None = None,
    ) -> torch.Tensor:
        """
        Physical parameter loss — fully batched.

        _embed_nodes_batched builds a single mega-graph with all nodes from
        all graphs in the batch and runs a single SAGE forward per layer
        (vs B separate forwards in the previous version).
        The per-SubgType heads still receive a [n_key, ...] tensor as before.
        """
        device = z.device

        # Fast compatibility path: if no permutations were requested, or every
        # sample only has identity, use exactly the old batched implementation.
        only_identity = True
        if attr_perm_indices is not None:
            for perms in attr_perm_indices:
                if len(perms) > 1:
                    only_identity = False
                    break
        if attr_perm_indices is None or only_identity:
            return self._loss_batched_no_permutation(z, G_true)

        pred_flat, target_flat = self._predict_and_target_flat_batched(z, G_true, device)
        if not pred_flat:
            return torch.zeros(1, device=device)

        losses: list[torch.Tensor] = []
        for b, (pred_b, target_b) in enumerate(zip(pred_flat, target_flat)):
            perms = attr_perm_indices[b] if attr_perm_indices is not None else [list(range(target_b.numel()))]
            if len(perms) <= 1:
                losses.append(F.mse_loss(pred_b, target_b))
                continue

            perm_losses: list[torch.Tensor] = []
            for perm in perms:
                if len(perm) != int(target_b.numel()):
                    # Defensive fallback: malformed saved permutation must not
                    # break training; use the unpermuted target for this entry.
                    perm_losses.append(F.mse_loss(pred_b, target_b))
                    continue
                idx = torch.tensor(perm, dtype=torch.long, device=device)
                perm_losses.append(F.mse_loss(pred_b, target_b[idx]))
            losses.append(torch.stack(perm_losses).min())

        return torch.stack(losses).mean()


    def _predict_and_target_flat_batched(
        self,
        z: torch.Tensor,
        G_true: list[SimpleNamespace],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return differentiable flat predictions and flat targets per graph."""
        x_struct_all, _h_all, _node_to_graph, node_types_flat = self._embed_nodes_batched(z, G_true, device)
        if len(node_types_flat) == 0:
            return [], []

        n_per_graph = torch.tensor([len(g.node_types) for g in G_true], dtype=torch.long, device=device)
        z_per_node = torch.repeat_interleave(z, n_per_graph, dim=0)
        z_base_all = self.z_proj(z_per_node)

        pred_by_node: list[torch.Tensor | None] = [None] * len(node_types_flat)
        type_to_indices: dict[str, list[int]] = {}
        for idx, st_int in enumerate(node_types_flat):
            if str(st_int) in self.base_heads:
                type_to_indices.setdefault(str(st_int), []).append(idx)

        for key, indices in type_to_indices.items():
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            base_in = torch.cat([z_base_all[idx_t], x_struct_all[idx_t]], dim=-1)
            preds = self.base_heads[key](base_in)
            for local_i, flat_i in enumerate(indices):
                pred_by_node[flat_i] = preds[local_i]

        pred_flat: list[torch.Tensor] = []
        target_flat: list[torch.Tensor] = []
        flat_idx = 0
        for g in G_true:
            p_parts: list[torch.Tensor] = []
            t_parts: list[torch.Tensor] = []
            for i, st_int in enumerate(g.node_types):
                attrs = SUBG_DEFS[SubgType(st_int)].attrs
                if not attrs or str(st_int) not in self.base_heads:
                    flat_idx += 1
                    continue
                p_parts.append(pred_by_node[flat_idx])
                t_parts.append(torch.tensor([g.attrs[i][a] for a in attrs], dtype=torch.float, device=device))
                flat_idx += 1
            if p_parts:
                pred_flat.append(torch.cat(p_parts, dim=0))
                target_flat.append(torch.cat(t_parts, dim=0))
            else:
                pred_flat.append(torch.zeros(0, device=device))
                target_flat.append(torch.zeros(0, device=device))
        return pred_flat, target_flat

    def _loss_batched_no_permutation(
        self,
        z: torch.Tensor,
        G_true: list[SimpleNamespace],
    ) -> torch.Tensor:
        """Original parameter loss, kept as the exact identity-only path."""
        device = z.device

        x_struct_all, h_all, node_to_graph, node_types_flat = self._embed_nodes_batched(z, G_true, device)

        if len(node_types_flat) == 0:
            return torch.zeros(1, device=device)

        n_per_graph = torch.tensor(
            [len(g.node_types) for g in G_true], dtype=torch.long, device=device
        )
        z_per_node   = torch.repeat_interleave(z, n_per_graph, dim=0)
        z_base_all   = self.z_proj(z_per_node)

        all_targets: list[torch.Tensor] = []
        all_st_int:  list[int]          = []
        for g in G_true:
            for i, st_int in enumerate(g.node_types):
                key = str(st_int)
                if key not in self.base_heads:
                    all_targets.append(None)
                    all_st_int.append(-1)
                    continue
                target_vals = [g.attrs[i][a] for a in SUBG_DEFS[SubgType(st_int)].attrs]
                all_targets.append(
                    torch.tensor(target_vals, dtype=torch.float, device=device)
                )
                all_st_int.append(st_int)

        type_to_indices: dict[str, list[int]] = {}
        for idx, st_int in enumerate(all_st_int):
            if st_int == -1:
                continue
            type_to_indices.setdefault(str(st_int), []).append(idx)

        total_loss  = torch.zeros(1, device=device)
        total_count = 0

        for key, indices in type_to_indices.items():
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            xstr  = x_struct_all[idx_t]
            zb    = z_base_all[idx_t]

            base_in = torch.cat([zb, xstr], dim=-1)
            preds   = self.base_heads[key](base_in)

            targets_key = torch.stack([all_targets[i] for i in indices])
            total_loss  = total_loss + F.mse_loss(preds, targets_key) * len(indices)
            total_count += len(indices)

        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def predict(
        self,
        z:      torch.Tensor,
        graphs: list[SimpleNamespace],
    ) -> list[list[dict]]:
        device  = z.device
        results = []

        for b, g in enumerate(graphs):
            z_b        = z[b]
            edge_index = _build_edge_index_from_list(g.edges, device)
            x_struct, h = self._embed_nodes(z_b, g.node_types, edge_index)

            attrs_list: list[dict] = []
            for i, st_int in enumerate(g.node_types):
                key = str(st_int)
                if key not in self.base_heads:
                    attrs_list.append({})
                    continue
                pred = self._predict_node(z_b, x_struct[i], h[i], st_int)
                attrs_list.append(
                    {a: float(pred[j].item()) for j, a in enumerate(SUBG_DEFS[SubgType(st_int)].attrs)}
                )
            results.append(attrs_list)

        return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _build_edge_index_from_list(
    edges: list[tuple[int, int]], device: torch.device
) -> torch.Tensor:
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    fwd   = edges
    bwd   = [(v, u) for u, v in fwd]
    all_e = fwd + bwd
    return torch.tensor(all_e, dtype=torch.long, device=device).t().contiguous()


def data_to_graph_ns(sample, scalers_param) -> SimpleNamespace:
    """
    Converts a Data sample into a SimpleNamespace for the decoder and ParamDecoder.

    Fields:
        node_types : list[int]    SubgType int for each macro-node
        edges      : list[(u,v)]  circuit↔circuit edges (0-based)
        attrs      : list[dict]   {attr_name: scaled_val} for each macro-node
    """
    topo_ids = sample.topology_ids.tolist()
    y_flat   = sample.y.tolist()

    cursor = 0
    scaled_attrs: list[dict] = []
    for st_int in topo_ids:
        st   = SubgType(st_int)
        d    = {}
        for attr_name in SUBG_DEFS[st].attrs:
            d[attr_name] = float(y_flat[cursor])
            cursor += 1
        scaled_attrs.append(d)

    n_circuit = len(topo_ids)
    ei        = sample.ei_outer
    edges: list[tuple[int, int]] = []
    if ei.shape[1] > 0:
        mask  = (ei[0] < n_circuit) & (ei[1] < n_circuit)
        ei_cc = ei[:, mask]
        seen: set[frozenset] = set()
        for u, v in ei_cc.t().tolist():
            k = frozenset({u, v})
            if k not in seen and u != v:
                seen.add(k)
                edges.append((int(u), int(v)))

    g            = SimpleNamespace()
    g.node_types = topo_ids
    g.edges      = edges
    g.attrs      = scaled_attrs
    return g
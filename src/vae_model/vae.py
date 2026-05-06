"""
cqed.model.vae
==============
GraphVAE — main module that combines the encoder and the decoders.

Architecture
------------
    GraphVAEEncoder   :  G  →  z^c, µ^c, log σ²^c
    SpecEncoder       :  obs → z^s, µ^s, log σ²^s
    GraphVAEDecoder   :  z  →  G'  
    ParamDecoder      :  (z, G)  →  θ̂' 
    ObsClassifier     :  z^c → s'

    L_R = loss_topo_c + attrs_scale * loss_attrs_c
        + spec_recon_scale * (loss_topo_s + attrs_scale * loss_attrs_s)

    loss_total = L_R
               + align_scale      * loss_align
               + nce_scale        * loss_nce
               + cg_scale         * loss_cg

β-annealing
-----------
    beta increases linearly from beta_start to beta_max during the first warmup_steps.
    It controls the weight of the KL term both in loss_topo_c (circuit branch)
    and in loss_align (spec/circuit alignment branch).
    It is not applied to loss_topo_s (the KL term of the spec branch is already included in loss_align).

Inference
---------
    z^s = vae.spec_encoder.encode(obs_vals, obs_mask)[0]
    G   = vae.decode(z^s)

    z^c = vae.encode(samples, scalers)[0]
    G   = vae.decode(z^c)

    # free generation
    G   = vae.sample(n)

"""

from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace

from circuit2graph.definitions import SubgType
from vae_model.encoder import (
    GraphVAEEncoder,
    N_SUBTYPES,
)
from vae_model.decoder import (
    TransformerTopologyDecoder,
    ParamDecoder,
    data_to_graph_ns,
    build_type_class_weights,
)

# Alias: vae.py usa GraphVAEDecoder, che è TransformerTopologyDecoder...DA AGGIUSTARE
GraphVAEDecoder = TransformerTopologyDecoder
from vae_model.obs_encoder import (
    SpecEncoder,
    ObsClassifier,
    alignment_loss,
    contrastive_loss,
)

# Dimension of outer feature for loader_vae.py:
# [SubgType one-hot (N_SUBTYPES) | deg_norm (1) | pos_norm (1)]
OUTER_FEAT_DIM_VAE: int = N_SUBTYPES + 2


# ---------------------------------------------------------------------------
# GraphVAE
# ---------------------------------------------------------------------------

class GraphVAE(nn.Module):
    """
    Full Graph VAE for cQED

    Parameters
    ----------
    nz               : latent space dimension
    hv_dim           : inner → macro-node embedding dimension (encoder)
    inner_hidden     : hidden dimension of InnerEncoder
    inner_layers     : number of InnerEncoder layers
    outer_hidden     : hidden dimension of OuterEncoder (dimension A' from OuterGIN)
    outer_layers     : number of OuterEncoder layers
    d                : common dimension of the Transformer encoder
    nhead            : number of attention heads in the Transformer encoder
    tf_layers        : number of Transformer encoder layers
    max_nodes        : maximum number of macro-nodes for encoder positional encoding
    spec_d           : embedding dimension for each slot in SpecEncoder
    spec_layers      : number of MLP layers in SpecEncoder
    cg_hidden        : hidden dimension of the ObsClassifier heads
    hs               : hidden state dimension of the decoder GGNN
    ggnn_rounds      : number of GGNN rounds in the topological decoder
    max_nodes_dec    : maximum number of decoder macro-nodes (including START/END)
    param_hidden     : hidden dimension of SAGEConv in the ParamDecoder
    param_layers     : number of SAGEConv layers in the ParamDecoder
    dropout          : shared dropout rate
    beta_start       : initial value of β
    beta_max         : final value of β after warmup
    warmup_steps     : number of steps to linearly reach beta_max
    attrs_scale      : weight of loss_attrs in the total loss (both branches)
    align_scale      : weight of loss_align (L_KL + L_C)
    nce_scale        : weight of loss_nce (contrastive InfoNCE)
    cg_scale         : weight of loss_cg (classifier guidance)
    spec_recon_scale : weight of the reconstruction branch from z^s (L^s_R).
                    1.0 = same weight as L^c_R (default full CktGen behavior).
                    0.0 = disables the spec branch (previous behavior).
    tau              : temperature parameter for InfoNCE
    class_weight_end : optional relative weight of the END class in loss_t
    """

    def __init__(
        self,
        nz:               int   = 32,
        hv_dim:           int   = 32,
        inner_hidden:     int   = 32,
        inner_layers:     int   = 2,
        outer_hidden:     int   = 64,
        outer_layers:     int   = 3,
        d:                int   = 64,
        nhead:            int   = 4,
        tf_layers:        int   = 2,
        max_nodes:        int   = 16,
        hs:               int   = 128,
        ggnn_rounds:      int   = 3,
        max_nodes_dec:    int   = 8,
        param_hidden:     int   = 64,
        param_layers:     int   = 2,
        spec_d:           int   = 64,
        spec_layers:      int   = 3,
        cg_hidden:        int   = 64,
        dropout:          float = 0.1,
        beta_start:       float = 0.0,
        beta_max:         float = 1.0,
        warmup_steps:     int   = 5000,
        attrs_scale:      float = 1.0,
        align_scale:      float = 1.0,
        nce_scale:        float = 1.0,
        cg_scale:         float = 1.0,
        spec_recon_scale: float = 1.0,
        tau:              float = 0.1,
        class_weight_end: float = 1.0,
    ):
        super().__init__()

        self.nz               = nz
        self.beta_start       = beta_start
        self.beta_max         = beta_max
        self.warmup_steps     = warmup_steps
        self.attrs_scale      = attrs_scale
        self.align_scale      = align_scale
        self.nce_scale        = nce_scale
        self.cg_scale         = cg_scale
        self.spec_recon_scale = spec_recon_scale
        self.tau              = tau
        self.class_weight_end = class_weight_end

        # to be saved in the checkpoint
        self._step: int = 0

        # ── Circuit Encoder ( GIN + Transformer) ────────────
        self.encoder = GraphVAEEncoder(
            hv_dim       = hv_dim,
            inner_hidden = inner_hidden,
            inner_layers = inner_layers,
            outer_hidden = outer_hidden,
            outer_layers = outer_layers,
            d            = d,
            nz           = nz,
            nhead        = nhead,
            tf_layers    = tf_layers,
            max_nodes    = max_nodes,
            dropout      = dropout,
        )

        # ── Topological decoder : z →  Ĝ  ─
        # TransformerTopologyDecoder uses `d` (hidden dim) and `n_layers`
        # instead of `hs` and `ggnn_rounds`... AGGIUSTA QUESTO MISMATCH
        self.decoder = GraphVAEDecoder(
            nz        = nz,
            d         = hs,          # hs  →  hidden dim Transformer
            n_layers  = ggnn_rounds, # ggnn_rounds → numero layer
            max_nodes = max_nodes_dec,
            dropout   = dropout,
        )

        # ── Parameters decoder: (z, G) → θ̂'───────────────────────────
        self.param_decoder = ParamDecoder(
            nz          = nz,
            hidden      = param_hidden,
            sage_layers = param_layers,
            dropout     = dropout,
        )

        # ── Spec Encoder: obs → (µ^s, Σ^s, z^s) ─────────────────────────
        self.spec_encoder = SpecEncoder(
            nz       = nz,
            d        = spec_d,
            n_layers = spec_layers,
            dropout  = dropout,
        )

        # ── Classifier guidance: z^c → ŝ (predicted observables) ──────────
        # Forces z^c to contain useful information about the Hamiltonian specs
        self.obs_classifier = ObsClassifier(
            nz     = nz,
            hidden = cg_hidden,
        )

    # ------------------------------------------------------------------
    # β-annealing
    # ------------------------------------------------------------------

    def current_beta(self) -> float:
        """
        Current β value:
            beta(t) = beta_start + (beta_max - beta_start) * min(1, t / warmup_steps)
        """
        if self.warmup_steps <= 0:
            return self.beta_max
        progress = min(1.0, self._step / self.warmup_steps)
        return self.beta_start + (self.beta_max - self.beta_start) * progress

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def _prepare_batch(
        self,
        samples: list,
        scalers: dict,
    ) -> tuple[SimpleNamespace, list[SimpleNamespace]]:
        """
        Convert a list of Data samples into the format expected by the encoder
        and decoder.

        If the samples already contain pre-computed enc_* fields from the loader 
        use them with simple torch.cat operations and index-offset
        corrections. (Used to fast things)

        Returns
        -------
        enc_batch : SimpleNamespace for GraphVAEEncoder.encode()
        G_true    : list[SimpleNamespace] for decoder.loss() and param_decoder.loss()
        """
        device = next(self.parameters()).device

        # ── Tensors pre-computed at load time ──────────────
        inner_x_parts:        list[torch.Tensor] = []
        inner_ei_parts:       list[torch.Tensor] = []
        inner_batch_parts:    list[torch.Tensor] = []
        macro_subgtype_parts: list[torch.Tensor] = []
        macro_pos_parts:      list[torch.Tensor] = []
        macro_batch_parts:    list[torch.Tensor] = []
        outer_ei_parts:       list[torch.Tensor] = []

        inner_offset = 0   # cumulative offset for inner node indices
        macro_offset = 0   # cumulative offset for macro node indices

        for sample_idx, s in enumerate(samples):
            n_circuit = s.enc_n_circuit
            n_nodes_list: list[int] = s.enc_inner_n_nodes  # per-block node counts

            inner_x_parts.append(s.enc_inner_x)

            # enc_inner_ei already has intra-sample/block offsets applied
            # at load time. Here we only add the cross-sample shift:
            # inner_offset = total number of inner nodes in all previous samples.
            ei_inner = s.enc_inner_ei
            if ei_inner.shape[1] > 0:
                inner_ei_parts.append(ei_inner + inner_offset)
            else:
                inner_ei_parts.append(ei_inner)

            # inner_batch: each inner node maps to its global macro-node index.
            # Vectorized version: torch.repeat_interleave removes the Python loop
            # over n_circuit blocks and avoids repeated torch.full + torch.cat calls.
            macro_global_start = macro_offset
            n_nodes_t = torch.tensor(n_nodes_list, dtype=torch.long)
            nonzero_mask = n_nodes_t > 0
            if nonzero_mask.any():
                macro_ids = torch.arange(
                    macro_global_start,
                    macro_global_start + n_circuit,
                    dtype=torch.long,
                )
                inner_batch_parts.append(
                    torch.repeat_interleave(
                        macro_ids[nonzero_mask],
                        n_nodes_t[nonzero_mask],
                    )
                )

            macro_subgtype_parts.append(s.enc_macro_subgtype)
            macro_pos_parts.append(s.enc_macro_pos)
            macro_batch_parts.append(
                torch.full((n_circuit,), sample_idx, dtype=torch.long)
            )

            # Outer edge_index: shift node indices by macro_offset.
            ei_outer = s.enc_outer_ei
            if ei_outer.shape[1] > 0:
                outer_ei_parts.append(ei_outer + macro_offset)
            else:
                outer_ei_parts.append(ei_outer)

            inner_offset += sum(n_nodes_list)
            macro_offset += n_circuit

            enc_batch = SimpleNamespace()
            enc_batch.inner_x = torch.cat(inner_x_parts, dim=0).to(device)

            if any(e.shape[1] > 0 for e in inner_ei_parts):
                enc_batch.inner_ei = torch.cat(inner_ei_parts, dim=1).to(device)
            else:
                enc_batch.inner_ei = torch.zeros((2, 0), dtype=torch.long, device=device)

            enc_batch.inner_batch = (
                torch.cat(inner_batch_parts, dim=0).to(device)
                if inner_batch_parts
                else torch.zeros(0, dtype=torch.long, device=device)
            )

            enc_batch.macro_subgtype = torch.cat(macro_subgtype_parts, dim=0).to(device)
            enc_batch.macro_pos      = torch.cat(macro_pos_parts,      dim=0).to(device)
            enc_batch.macro_batch    = torch.cat(macro_batch_parts,    dim=0).to(device)

            if any(e.shape[1] > 0 for e in outer_ei_parts):
                enc_batch.outer_ei = torch.cat(outer_ei_parts, dim=1).to(device)
            else:
                enc_batch.outer_ei = torch.zeros((2, 0), dtype=torch.long, device=device)

        # Use G_true pre-computed at load time.
        G_true = [s.g_true_ns for s in samples]

        return enc_batch, G_true

    # ------------------------------------------------------------------
    # Forward 
    # ------------------------------------------------------------------

    def forward(
        self,
        samples:  list,
        scalers:  dict,
        obs_vals: "torch.Tensor | None" = None,
        obs_mask: "torch.Tensor | None" = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Complete forward pass — bidirectional paradigm:

            L = λ_KL · L_KL  +  L^c_R  +  L^s_R  +  L_C  +  L_CG  +  L_NCE

        Flow:
            Circuit branch (L^c_R):
                G → GraphVAEEncoder → µ^c, Σ^c, z^c
                z^c → topological decoder → loss_topo_c
                z^c → param_decoder       → loss_attrs_c

            Specification branch (L^s_R, only if obs_vals and obs_mask are provided):
                obs → SpecEncoder → µ^s, Σ^s, z^s
                z^s → topological decoder → loss_topo_s   ← generative spec branch
                z^s → param_decoder       → loss_attrs_s  ← generative spec branch

            Auxiliary losses (only if observations are provided):
                alignment_loss(µ^c, Σ^c, z^c, µ^s, Σ^s, z^s) → loss_align
                contrastive_loss(z^s, z^c, obs_vals, obs_mask) → loss_nce
                obs_classifier.loss(z^c, obs_vals, obs_mask)   → loss_cg

        Parameters
        ----------
        samples  : list of Data objects (loader output)
        scalers  : dict {dataset_name: DatasetScalers}
        obs_vals : [B, N_OBS_SLOTS] scaled observables  (optional)
        obs_mask : [B, N_OBS_SLOTS] slot mask           (optional)

        Automatically updates the step counter and the current β.

        Returns
        -------
        loss_total : scalar torch.Tensor
        components : dict containing all loss terms for logging
        """
        beta = self.current_beta()

        enc_batch, G_true = self._prepare_batch(samples, scalers)
        _dev = next(self.parameters()).device

        # ── Pre-computes the type weights a single time per batch ──────────────────────

        type_class_weights = build_type_class_weights(
            G_true,
            device     = _dev,
            end_weight = self.class_weight_end,
        )
        # tensorize_G_true: tensorizes G_true to be faster
        _type_seq, _pos_seq, _adj_true, _seq_lens = self.decoder.tensorize_G_true(
            G_true, device=_dev
        )

        # ── Circuit VAE: z^c → L^c_R computes z, the mean and the variance for the circuit ─

        z_c, mu_c, logvar_c = self.encoder.encode(enc_batch)

        # Topological loss for the circuit
        loss_topo_c, comp_topo_c = self.decoder.loss(
            z_c, G_true,
            mu         = mu_c,
            logvar     = logvar_c,
            beta       = beta,
            type_class_weights = type_class_weights,
            _type_seq  = _type_seq,
            _pos_seq   = _pos_seq,
            _adj_true  = _adj_true,
            _seq_lens  = _seq_lens,
        )

        # Parametrical loss for the circuit
        loss_attrs_c = self.param_decoder.loss(z_c, G_true)

        # Total L^c_R
        loss_R_c = loss_topo_c + self.attrs_scale * loss_attrs_c

        # ── Specifics VAE + auxiliary losses ─────────────────────

        has_obs = (obs_vals is not None) and (obs_mask is not None)

        loss_R_s   = torch.zeros(1, device=z_c.device)
        loss_align = torch.zeros(1, device=z_c.device)
        loss_nce   = torch.zeros(1, device=z_c.device)
        loss_cg    = torch.zeros(1, device=z_c.device)
        comp_topo_s: dict = {"loss_topo_s": 0.0, "loss_t_s": 0.0,
                             "loss_p_s": 0.0,    "loss_e_s": 0.0}
        comp_align:  dict = {}
        comp_nce:    dict = {}
        comp_cg:     dict = {}
        loss_attrs_s_val = 0.0

        if has_obs:
            # ── Spec Encoder → z^s, mean and variance ─────────────────────────────────

            z_s, mu_s, logvar_s = self.spec_encoder.encode(obs_vals, obs_mask)

            # mu and sigma (logvar) are used in the alignment loss, not used here.
            # (mu and sigma are used to compute the KL loss, but since are used 
            # later, they will compute the KL later on, not now.)
            loss_topo_s, _comp_topo_s = self.decoder.loss(
                z_s, G_true,
                mu         = None,
                logvar     = None,
                beta       = 0.0,  
                type_class_weights = type_class_weights,
                _type_seq  = _type_seq,
                _pos_seq   = _pos_seq,
                _adj_true  = _adj_true,
                _seq_lens  = _seq_lens,
            )
            loss_attrs_s     = self.param_decoder.loss(z_s, G_true)
            loss_attrs_s_val = loss_attrs_s.item()

            loss_R_s = loss_topo_s + self.attrs_scale * loss_attrs_s

            comp_topo_s = {
                "loss_topo_s": _comp_topo_s["loss_topo"],
                "loss_t_s":    _comp_topo_s["loss_t"],
                "loss_p_s":    _comp_topo_s["loss_p"],
                "loss_e_s":    _comp_topo_s["loss_e"],
            }

            # ── L_KL + L_C (latent space alignment) ───────
            loss_align, comp_align = alignment_loss(
                mu_c     = mu_c,
                logvar_c = logvar_c,
                z_c      = z_c,
                mu_s     = mu_s,
                logvar_s = logvar_s,
                z_s      = z_s,
                beta     = beta,
            )

            # ── L_NCE (contrastive InfoNCE) ───────────────────────
            loss_nce, comp_nce = contrastive_loss(
                z_s      = z_s,
                z_c      = z_c,
                obs_vals = obs_vals,
                obs_mask = obs_mask,
                tau      = self.tau,
            )

            # ── L_CG (classifier guidance) ────────────────────────
            loss_cg, comp_cg = self.obs_classifier.loss(
                z_c      = z_c,
                obs_vals = obs_vals,
                obs_mask = obs_mask,
            )

        # ── Total loss ──────────────────────────────────────────────────
        loss_total = (
            loss_R_c
            + self.spec_recon_scale * loss_R_s
            + self.align_scale      * loss_align
            + self.nce_scale        * loss_nce
            + self.cg_scale         * loss_cg
        )

        components = {
            "loss_total": loss_total.item(),
            "loss_topo_c": comp_topo_c["loss_topo"],
            "loss_t_c": comp_topo_c["loss_t"],
            "loss_p_c": comp_topo_c["loss_p"],
            "loss_e_c": comp_topo_c["loss_e"],
            "loss_kl": comp_topo_c["loss_kl"],
            "loss_attrs_c":  loss_attrs_c.item(),
            **comp_topo_s,
            "loss_attrs_s":  loss_attrs_s_val,
            "loss_align":    loss_align.item(),
            "loss_nce":      loss_nce.item(),
            "loss_cg":       loss_cg.item(),
            "beta":          beta,
            **comp_align,
            **comp_nce,
            **comp_cg,
        }

        # The counter is increased only in training, not in validation
        if self.training:
            self._step += 1
        return loss_total, components

    # ------------------------------------------------------------------
    # encode 
    # ------------------------------------------------------------------

    def encode(
        self,
        samples: list,
        scalers: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Using the encoder (pre-trained), converts the batch data into the latent space.

        Retruns (z, mu, logvar)
        """
        enc_batch, _ = self._prepare_batch(samples, scalers)
        return self.encoder.encode(enc_batch)

    # ------------------------------------------------------------------
    # decode — genera topologia + parametri da z
    # ------------------------------------------------------------------

    def decode(
        self,
        z:          torch.Tensor,
        stochastic: bool = False,
    ) -> list[SimpleNamespace]:
        """
        Writes a batch in the latent space z into graphs (topology + parameters).

        Retruns
        -------
            node_types : list[int]        
            edges      : list[(int,int)]  
            attrs      : list[dict]      
        """
        graphs          = self.decoder.decode(z, stochastic=stochastic)
        attrs_per_graph = self.param_decoder.predict(z, graphs)
        for g, attrs in zip(graphs, attrs_per_graph):
            g.attrs = attrs
        return graphs

    # ------------------------------------------------------------------
    # sample — generates graphs from z ~ N(0,I)
    # ------------------------------------------------------------------

    def sample(
        self,
        n:          int,
        stochastic: bool = True,
        device:     torch.device | None = None,
    ) -> list[SimpleNamespace]:
        """
        Useful to generate graph by sampling the latent space
        """
        if device is None:
            device = next(self.decoder.parameters()).device
        z = torch.randn(n, self.nz, device=device)
        return self.decode(z, stochastic=stochastic)

    # ------------------------------------------------------------------
    # checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict_full(self) -> dict:
        """
        Extended state_dict that also includes the step counter.

        torch.compile replaces submodules with wrappers whose state_dict
        uses keys such as 'encoder._orig_mod.lin_in.weight' instead of
        'encoder.lin_in.weight'. Here we strip the '_orig_mod.' prefix
        so that the checkpoint can also be loaded by non-compiled models
        (e.g. inference scripts).
        """
        raw = self.state_dict()
        clean = {k.replace("._orig_mod.", "."): v for k, v in raw.items()}
        return {
            "model_state": clean,
            "step":        self._step,
        }

    def load_state_dict_full(self, checkpoint: dict) -> None:
        """
        Loads a checkpoint saved with state_dict_full().

        Normalizes the keys in case the checkpoint still contains
        residual '_orig_mod' prefixes (checkpoints saved before this fix).
        """
        raw   = checkpoint["model_state"]
        clean = {k.replace("._orig_mod.", "."): v for k, v in raw.items()}
        self.load_state_dict(clean, strict=True)
        self._step = checkpoint.get("step", 0)

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_params  = sum(p.numel() for p in self.parameters()               if p.requires_grad)
        n_enc     = sum(p.numel() for p in self.encoder.parameters()       if p.requires_grad)
        n_spec    = sum(p.numel() for p in self.spec_encoder.parameters()  if p.requires_grad)
        n_dec     = sum(p.numel() for p in self.decoder.parameters()       if p.requires_grad)
        n_pdec    = sum(p.numel() for p in self.param_decoder.parameters() if p.requires_grad)
        n_cg      = sum(p.numel() for p in self.obs_classifier.parameters() if p.requires_grad)

        return (
            f"GraphVAE("
            f"nz={self.nz}, "
            f"beta={self.current_beta():.3f}/{self.beta_max}, "
            f"spec_recon={self.spec_recon_scale}, "
            f"class_w_end={self.class_weight_end}, "
            f"step={self._step}, "
            f"total_params={n_params:,} | "
            f"circuit_enc={n_enc:,}, spec_enc={n_spec:,}, "
            f"topo_dec={n_dec:,}, param_dec={n_pdec:,}, "
            f"obs_clf={n_cg:,})"
        )
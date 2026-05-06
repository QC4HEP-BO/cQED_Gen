"""
cqed.model.obs_encoder
======================
SpecEncoder — encoder of the Hamiltonian specifics in the latent space.

Architecture
-----------------------------
Each slot is embedded individually into R^d,
then they are concatenated and passed to a simple MLP:

    obs_vals  ∈  R^(B x N_OBS_SLOTS)      
    obs_mask  ∈  R^(B x N_OBS_SLOTS)     

    input_masked  =  obs_vals * obs_mask               [B, N_OBS_SLOTS]

    For each slot i:
        s'_i  =  Embed_i(input_masked[:, i:i+1])       [B, d]

    s'  =  cat([s'_0, …, s'_6])                        [B, N_OBS_SLOTS * d]
    s'' =  MLP(s')                                      [B, d]

    µ^s    =  fc_mu(s'')                                [B, nz]
    Σ^s    =  fc_var(s'')                               [B, nz]
    z^s    ~  N(µ^s, exp(Σ^s))    

SpecEncoder produces µ^s e Σ^s.

Training congiunto (in GraphVAE.forward)
-----------------------------------------
VAE ha three branches for the loss:

    1. Circuit:    G   → GraphVAEEncoder → µ^c, Σ^c, z^c
    2. Observables: obs → SpecEncoder    → µ^s, Σ^s, z^s
    3. Auxiliaries:
           - Contrastive 
           - Classifier guidance

Alignment loss:

    L_KL = KL(a^c ‖ N(0,I))
         + KL(a^s ‖ N(0,I))
         + KL(a^c ‖ a^s)
         + KL(a^s ‖ a^c)

    L_C  = SmoothL1(z^c, z^s)

with a^c = N(µ^c, Σ^c),  a^s = N(µ^s, Σ^s).

Loss contrastive:
--------------------------------------------
In a M samples batch, (z^s_i, z^c_i) is a positive couple, since they have the same "i".
Each (z^s_i, z^c_j) with i!=j is a negative couple instead.
The samples that have the same specifics (Same H) but different circuit
are seen as "white pairs": neither positive nor negative, this to fullfill the
one-to-many relations.

    R_ij    = cos(z^s_i, z^c_j)                     [M, M]
    L_NCE   = -1/2M · Σ_i [ log(exp(R_ii/τ) / Σ_j w_ij·exp(R_ij/τ))   (spec→circ)
                           + log(exp(R_ii/τ) / Σ_j w_ij·exp(R_ji/τ)) ] (circ→spec)

where w_ij = 0 for the white pairs, 1 otherwise.

Classifier guidance:
-------------------------------------
To z^c are applied N_OBS_SLOTS independent MLP, one for each observable.
Each MLP predicts the slot value and is trained only in the data where
that slot is present (obs_mask[:,i] == 1).

    ŝ_i    =  head_i(z^c)                            [B, 1]
    L_CG   =  Σ_i MSE(ŝ_i[mask_i], obs_vals[mask_i] * obs_mask[mask_i])


Inference
------------------------------------------
    z^s  = vae.spec_encoder.encode(obs_vals, obs_mask)   # [B, nz]
    G    = vae.decode(z^s)                                # topology + θ̂'"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader.schema import N_OBS_SLOTS


# ---------------------------------------------------------------------------
# SpecEncoder
# ---------------------------------------------------------------------------

class SpecEncoder(nn.Module):
    """
    Encoder of Hamiltonian specifications → (µ^s, Σ^s, z^s).

    Parameters
    ----------
    nz         : latent space dimension (must match GraphVAE.nz)
    d          : per-slot embedding dimension and MLP hidden dimension
    n_layers   : number of Linear→LayerNorm→ReLU blocks in the MLP
    dropout    : dropout rate
    """

    def __init__(
        self,
        nz:       int   = 32,
        d:        int   = 64,
        n_layers: int   = 3,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.nz = nz
        self.d  = d

        # ── per-slot embedding ───────────────────────────────────────────
        # Each slot is a scalar → Linear(1, d)
        # N_OBS_SLOTS independent embedders, one for each slot
        self.slot_embeds = nn.ModuleList([
            nn.Linear(1, d) for _ in range(N_OBS_SLOTS)
        ])

        # ── MLP over the concatenated vector [N_OBS_SLOTS * d] → d ───────
        mlp: list[nn.Module] = []
        in_dim = N_OBS_SLOTS * d

        for _ in range(n_layers):
            mlp += [
                nn.Linear(in_dim, d),
                nn.LayerNorm(d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = d

        self.mlp = nn.Sequential(*mlp)

        # ── final FC layers → µ^s, Σ^s ───────────────────────────────────
        self.fc_mu  = nn.Linear(d, nz)
        self.fc_var = nn.Linear(d, nz)

    # ------------------------------------------------------------------
    def forward(
        self,
        obs_vals: torch.Tensor,   # [B, N_OBS_SLOTS] scaled
        obs_mask: torch.Tensor,   # [B, N_OBS_SLOTS] 0/1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (µ^s, logvar^s), each with shape [B, nz].
        """
        # mask: missing slots → 0 before embedding
        x = obs_vals * obs_mask                          # [B, N_OBS_SLOTS]

        # per-slot embedding: each column → [B, d], then concatenate
        slot_embs = [
            embed(x[:, i:i+1])                           # [B, d]
            for i, embed in enumerate(self.slot_embeds)
        ]

        s_prime = torch.cat(slot_embs, dim=-1)           # [B, N_OBS_SLOTS * d]

        # MLP → s''
        s_pp = self.mlp(s_prime)                         # [B, d]

        # final FC layers
        mu     = self.fc_mu(s_pp)                        # [B, nz]
        logvar = self.fc_var(s_pp)                       # [B, nz]

        return mu, logvar

    # ------------------------------------------------------------------
    def reparameterize(
        self,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        z^s = µ^s + ε · exp(0.5 · logvar^s).
        During evaluation, directly returns µ^s.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        return mu

    # ------------------------------------------------------------------
    def encode(
        self,
        obs_vals: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shortcut: forward + reparameterize.

        Returns (z^s, µ^s, logvar^s).
        """
        mu, logvar = self.forward(obs_vals, obs_mask)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"SpecEncoder("
            f"nz={self.nz}, d={self.d}, "
            f"in={N_OBS_SLOTS} slots, "
            f"params={n_params:,})"
        )


# ---------------------------------------------------------------------------
# ObsClassifier — classifier guidance (CktGen Eq. 6, adapted to regression)
# ---------------------------------------------------------------------------

class ObsClassifier(nn.Module):
    """
    N_OBS_SLOTS independent MLP heads over z^c → observable prediction.

    Each head is a small MLP(nz → hidden → 1) that predicts the scaled value
    of the corresponding slot. Training is performed only on samples for which
    that slot is active (obs_mask[:, i] == 1).

    In the original paper (CktGen, Eq. 6), the slots are categorical
    → cross-entropy. Here the observables are continuous scalars → MSE.

    The semantics are identical: z^c must contain predictive information
    about the Hamiltonian specifications.

    Parameters
    ----------
    nz     : latent space dimension
    hidden : hidden dimension of each head
    """

    def __init__(self, nz: int = 32, hidden: int = 64):
        super().__init__()
        self.nz = nz

        # one head for each observable slot
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nz, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(N_OBS_SLOTS)
        ])

    # ------------------------------------------------------------------
    def forward(
        self,
        z_c: torch.Tensor,   # [B, nz]
    ) -> torch.Tensor:
        """
        Returns the predictions for all slots: [B, N_OBS_SLOTS].
        """
        preds = torch.cat([head(z_c) for head in self.heads], dim=-1)

        return preds  # [B, N_OBS_SLOTS]

    # ------------------------------------------------------------------
    def loss(
        self,
        z_c:      torch.Tensor,   # [B, nz]
        obs_vals: torch.Tensor,   # [B, N_OBS_SLOTS] scaled target values
        obs_mask: torch.Tensor,   # [B, N_OBS_SLOTS] 1 = active slot
    ) -> tuple[torch.Tensor, dict]:
        """
        L_CG = average of the per-slot MSE values,
        computed only on active samples.

        If no sample in the batch has a given slot active, that term
        is skipped: it contributes 0, not NaN.

        Returns
        -------
        loss_cg    : scalar
        components : dict {f'cg_{slot_name}': float value}
        """
        from data_loader.schema import OBS_SLOTS

        preds      = self.forward(z_c)           # [B, N_OBS_SLOTS]
        total_loss = torch.zeros(1, device=z_c.device)
        active_slots = 0
        components: dict[str, float] = {}

        for i, slot_name in enumerate(OBS_SLOTS):
            mask_i = obs_mask[:, i].bool()       # [B]

            if not mask_i.any():
                components[f"cg_{slot_name}"] = 0.0
                continue

            pred_i   = preds[mask_i, i]          # [n_active]
            target_i = obs_vals[mask_i, i]       # [n_active]

            slot_loss = F.mse_loss(pred_i, target_i)

            total_loss = total_loss + slot_loss
            active_slots += 1

            components[f"cg_{slot_name}"] = slot_loss.item()

        if active_slots > 0:
            total_loss = total_loss / active_slots

        return total_loss, components

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"ObsClassifier("
            f"nz={self.nz}, slots={N_OBS_SLOTS}, "
            f"params={n_params:,})"
        )


# ---------------------------------------------------------------------------
# Loss helpers — called by GraphVAE.forward
# ---------------------------------------------------------------------------

def kl_to_normal(
    mu:     torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    KL( N(µ, Σ) ‖ N(0, I) ), averaged over the batch.

        = -½ · mean(1 + logvar - µ² - exp(logvar))
    """
    return -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )


def kl_between_gaussians(
    mu_p:     torch.Tensor,
    logvar_p: torch.Tensor,
    mu_q:     torch.Tensor,
    logvar_q: torch.Tensor,
) -> torch.Tensor:
    """
    KL( N(µ_p, Σ_p) ‖ N(µ_q, Σ_q) ), averaged over the batch.

    Closed-form formula for diagonal Gaussians:

        KL = ½ · Σ_i [
            logvar_q_i - logvar_p_i
            + (exp(logvar_p_i) + (µ_p_i - µ_q_i)²) / exp(logvar_q_i)
            - 1
        ]
    """
    var_p = logvar_p.exp()
    var_q = logvar_q.exp()

    kl = 0.5 * (
        logvar_q - logvar_p
        + (var_p + (mu_p - mu_q).pow(2)) / var_q
        - 1.0
    )

    return kl.mean()


def alignment_loss(
    mu_c:     torch.Tensor,
    logvar_c: torch.Tensor,
    z_c:      torch.Tensor,
    mu_s:     torch.Tensor,
    logvar_s: torch.Tensor,
    z_s:      torch.Tensor,
    beta:     float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Complete alignment loss between the two encoder branches
    (CktGen Eq. 3 + 4).

        L_KL = KL(α^c ‖ N(0,I))
             + KL(α^s ‖ N(0,I))
             + KL(α^c ‖ α^s)
             + KL(α^s ‖ α^c)

        L_C  = SmoothL1(z^c, z^s)

        L_align = beta * L_KL + L_C

    Parameters
    ----------
    mu_c, logvar_c : output of the GraphVAEEncoder, circuit branch
    z_c            : reparameterized sample from the circuit branch
    mu_s, logvar_s : output of the SpecEncoder, observable branch
    z_s            : reparameterized sample from the observable branch
    beta           : weight of the KL term, β-annealing, passed by the VAE

    Returns
    -------
    loss_align : scalar
    components : dict containing individual terms for logging
    """
    kl_c_normal = kl_to_normal(mu_c, logvar_c)
    kl_s_normal = kl_to_normal(mu_s, logvar_s)
    kl_c_s      = kl_between_gaussians(mu_c, logvar_c, mu_s, logvar_s)
    kl_s_c      = kl_between_gaussians(mu_s, logvar_s, mu_c, logvar_c)

    L_KL = kl_c_normal + kl_s_normal + kl_c_s + kl_s_c
    L_C  = F.smooth_l1_loss(z_c, z_s)

    loss_align = beta * L_KL + L_C

    components = {
        "kl_c_normal": kl_c_normal.item(),
        "kl_s_normal": kl_s_normal.item(),
        "kl_c_s":      kl_c_s.item(),
        "kl_s_c":      kl_s_c.item(),
        "L_KL":        L_KL.item(),
        "L_C":         L_C.item(),
        "loss_align":  loss_align.item(),
    }

    return loss_align, components


def contrastive_loss(
    z_s:      torch.Tensor,          # [B, nz] spec latents
    z_c:      torch.Tensor,          # [B, nz] circuit latents
    obs_vals: torch.Tensor,          # [B, N_OBS_SLOTS] scaled values
    obs_mask: torch.Tensor,          # [B, N_OBS_SLOTS] slot mask
    tau:      float = 0.1,
    eps:      float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    Bidirectional InfoNCE (CktGen Eq. 5), adapted to the cQED domain.

    Positive pairs: (z^s_i, z^c_i) — same circuit, same sample

    Negative pairs: (z^s_i, z^c_j) with i ≠ j

    White pairs: samples with identical specifications but different circuits
                 same dataset AND same active-slot pattern AND same observable
                 values discretized to the first decimal
                 → filtered out from the loss, weight 0

    Note on the adaptation to the cQED domain
    -----------------------------------------
    In CktGen, the specs are discrete categorical one-hot vectors,
    so equality is exact. Here the observables are continuous scaled values:
    we use a tolerance threshold (tol) on the active dimensions to decide
    whether two samples share "the same specifications".

    Only dimensions where obs_mask == 1 in BOTH samples are compared.
    If all common active values differ by less than tol, in normalized scale,
    the pair is considered white.

    Parameters
    ----------
    z_s, z_c  : spec and circuit latents, L2-normalized INTERNALLY
    obs_vals  : scaled observables for each sample
    obs_mask  : slot mask, 1 = active
    tau       : InfoNCE temperature
    eps       : epsilon for numerical stability in normalization

    Returns
    -------
    loss_nce   : scalar
    components : {"loss_nce": float}
    """
    B = z_s.shape[0]
    device = z_s.device

    # ── L2 normalization ────────────────────────────────────────────────
    z_s_n = F.normalize(z_s, p=2, dim=-1)   # [B, nz]
    z_c_n = F.normalize(z_c, p=2, dim=-1)   # [B, nz]

    # ── cosine similarity matrix [B, B] ─────────────────────────────────
    R = torch.mm(z_s_n, z_c_n.t())          # R[i,j] = cos(z^s_i, z^c_j)

    # ── build the white-pair mask ───────────────────────────────────────
    # white[i,j] = 1 ↔ samples i and j share the same specifications
    # but i ≠ j; the diagonal is always the positive pair
    tol = 0.05   # tolerance in normalized scale, roughly 5% of the typical range

    # slots active in common for each pair: [B, B, N_OBS_SLOTS]
    both_active = obs_mask.unsqueeze(1) * obs_mask.unsqueeze(0)
    # [B, B, N_OBS_SLOTS]

    # absolute value difference: [B, B, N_OBS_SLOTS]
    val_diff = (
        obs_vals.unsqueeze(1) - obs_vals.unsqueeze(0)
    ).abs()

    # two samples are "same spec" if:
    # 1. they have at least one active slot in common
    # 2. on ALL common active slots, the difference is < tol
    n_common  = both_active.sum(dim=-1)                      # [B, B]
    max_diff  = (val_diff * both_active).max(dim=-1).values  # [B, B]
    same_spec = (n_common > 0) & (max_diff < tol)            # [B, B]

    # white pairs: same spec AND i ≠ j
    eye = torch.eye(B, dtype=torch.bool, device=device)

    white_mask = same_spec & ~eye                            # [B, B]

    # weight matrix: 0 for white pairs, 1 otherwise
    # the diagonal always has weight 1 because it is the positive pair
    w = (~white_mask).float()                                # [B, B]

    # ── bidirectional InfoNCE ───────────────────────────────────────────
    logits = R / tau                                         # [B, B]

    # mask white pairs by subtracting a large value, equivalent to weight 0
    NEG_INF = -1e9
    logits_masked = logits + (1.0 - w) * NEG_INF             # [B, B]

    # spec → circuit:
    # for each z^s_i, the positive is z^c_i, column i
    log_softmax_sc = F.log_softmax(logits_masked, dim=1)     # [B, B]
    loss_sc        = -log_softmax_sc.diagonal().mean()

    # circuit → spec:
    # for each z^c_j, the positive is z^s_j, row j
    log_softmax_cs = F.log_softmax(logits_masked.t(), dim=1) # [B, B]
    loss_cs        = -log_softmax_cs.diagonal().mean()

    loss_nce = 0.5 * (loss_sc + loss_cs)

    components = {
        "loss_nce": loss_nce.item(),
        "white_pairs_frac": white_mask.float().mean().item(),
    }

    return loss_nce, components
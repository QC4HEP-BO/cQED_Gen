"""
train_vae.py
============
Training script for the GraphVAE (CktGen-style architecture) on cQED circuits.

Implements the full bidirectional CktGen paradigm (Eq. 8):

    L = λ_KL · L_KL  +  L^c_R  +  L^s_R  +  L_C  +  L_CG  +  L_NCE

where:
    L^c_R  = reconstruction from z^c (circuit branch)
    L^s_R  = reconstruction from z^s (observable branch, generative pathway)
    L_KL   = KL to N(0, I) + KL between the two latent distributions
    L_C    = SmoothL1(z^c, z^s)  (latent alignment)
    L_CG   = classifier guidance from z^c → observables
    L_NCE  = contrastive InfoNCE loss between (z^s, z^c)

Usage
-----
    # Full training with default config
    python train_vae.py

    # Training with overrides
    python train_vae.py --nz 64 --epochs 200 --batch_size 32

    # Evaluation only (load checkpoint and run test)
    python train_vae.py --eval_only --checkpoint best_vae.pt

    # Resume training from checkpoint (true resume)
    python train_vae.py --checkpoint best_vae.pt --epochs 400

    # Ablation mode: circuit branch only, no NCE/CG/spec
    python train_vae.py --spec_recon_scale 0 --nce_scale 0 --cg_scale 0


Checkpoint structure
-------------------
    model_state   : GraphVAE state_dict
    step          : global gradient step counter (for beta annealing)
    config        : configuration dict used
    scalers       : pickle(scalers) as bytes
    train_losses  : list[float] (mean loss per epoch)
    val_losses    : list[float]
    best_val_loss : float
    epoch         : int (last completed epoch)
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

# Fix for not having any trouble in the imports. Can be omitted if
# the import are correct and precise everywhere 
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT  = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader.loader_vae import load_all_datasets_vae
from vae_model.vae import GraphVAE


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
# Note on warmup_steps:
#   Let n_train be the total training samples, n_batch the batch size, and
#   n_warmup_epochs the number of warm-up epochs.
#
#       steps_per_epoch = n_train / n_batch
#       warmup_steps    = n_warmup_epochs * steps_per_epoch
#
#   Current sizing (9 train datasets × 10k samples × 0.70 train_frac = ~63 k):
#       steps_per_epoch ≈ 63_000 / 128 ≈ 492
#       20 warmup epochs → warmup_steps ≈ 9_840  (rounded to 10_000)
#
#   When adding/removing datasets, update warmup_steps proportionally or
#   use the helper printed at training start:
#       'beta-annealing: warmup_steps=X  ~Y epoche'
#   and adjust until Y ≈ 20.

DEFAULT_CONFIG: dict = {
    #architecture
    "nz":               128,
    "hv_dim":           32,
    "inner_hidden":     32,
    "inner_layers":     2,
    "outer_hidden":     64,
    "outer_layers":     3,
    "d":                64,
    "nhead":            4,
    "tf_layers":        4,
    "max_nodes":        12,    # DA AUMENTARE
    "hs":               512,
    "ggnn_rounds":      3,
    "max_nodes_dec":    8,
    "param_hidden":     128,
    "param_layers":     3,
    "spec_d":           128,
    "spec_layers":      4,
    "cg_hidden":        64,
    "dropout":          0.1,
    # beta-annealing
    "beta_start":       0.0,
    "beta_max":         1e-4,
    "warmup_steps":     10_000,  # ~20 epoche su ~63k samples / batch128 (9 train ds)
    # loss weights
    "attrs_scale":      0.01,
    "align_scale":      0.5,
    "nce_scale":        0.5,
    "cg_scale":         0.5,
    "spec_recon_scale": 0.3,
    "tau":              0.1,
    "class_weight_end": 1.0,
    "edge_pos_weight":  None,  # None = calibrazione automatica dal dataset
    # optimization
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "grad_clip":        1.0,   
    # training
    "batch_size":       128,
    "epochs":           300,
    "val_every":        5,     
    "patience":         40,
    "lr_patience":      10,
    "train_frac":       0.70,
    "val_frac":         0.15,
    "seed":             42,
    # I/O
    "save_path":        "best_vae.pt",
    "log_every":        2,
    "save_every":       20,    
    "plot_path":        "vae_curves.png",
    # performance
    "num_workers":      0,     # Da tenere a 0 per ora, se no da problemi
    "use_compile":      True,  # (PyTorch >= 2.0)
    "use_amp":          True,  # (only if CUDA is available)
}


# ---------------------------------------------------------------------------
# Collate function for GraphVAE
# ---------------------------------------------------------------------------

def collate_vae(samples: list) -> dict:
    """
    Collate function for the VAE DataLoader.

    Takes a list of sample objects (as returned by loader_vae) and builds
    a batch dictionary.

    Returns
    -------
    dict with keys:
        samples: original list of samples (passed directly to vae.forward) \\
        obs_vals: [B, N_OBS_SLOTS] float tensor of scaled observable values \\
        obs_mask: [B, N_OBS_SLOTS] float tensor indicating available observables
                    (1 = present, 0 = missing)
    """
    obs_vals = torch.stack([s.obs_vals for s in samples])
    obs_mask = torch.stack([s.obs_mask for s in samples])
    return {
        "samples":  samples,
        "obs_vals": obs_vals,
        "obs_mask": obs_mask,
    }


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    model:     GraphVAE,
    batch:     dict,
    scalers:   dict,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    grad_clip: float,
    scaler:    "amp.GradScaler | None" = None,
) -> tuple[float, dict]:
    """
    Forward + backward + optimizer step on a single batch.

    model.training needs to be True before the call.
    _step goes on in vae.forward() only if model.training is True,
    therefore the beta-annealing is locked to the gradient step.

    Returns (loss_scalar, components_dict).
    """
    optimizer.zero_grad()

    samples  = batch["samples"]
    obs_vals = batch["obs_vals"].to(device)
    obs_mask = batch["obs_mask"].to(device)

    use_autocast = (scaler is not None) and device.type == "cuda"

    with torch.amp.autocast("cuda", enabled=use_autocast):
        loss, components = model(
            samples  = samples,
            scalers  = scalers,
            obs_vals = obs_vals,
            obs_mask = obs_mask,
        )

    #backpropagation and weights update
    if scaler is not None and use_autocast:
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return loss.detach().item(), components


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loop(
    model:   GraphVAE,
    loader:  DataLoader,
    scalers: dict,
    device:  torch.device,
    use_amp: bool = False,
) -> tuple[float, dict]:
    """
    Compute the loss on a full loader in eval modality.

    _step does not go on during eval.

    Returns (mean_loss, mean_components).
    """
    model.eval()
    total_loss = 0.0
    comp_accum: dict = defaultdict(float)
    n_batches = 0

    use_autocast = use_amp and device.type == "cuda"

    for batch in loader:
        samples  = batch["samples"]
        obs_vals = batch["obs_vals"].to(device)
        obs_mask = batch["obs_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_autocast):
            loss, comp = model(
                samples  = samples,
                scalers  = scalers,
                obs_vals = obs_vals,
                obs_mask = obs_mask,
            )
        total_loss += float(loss)
        for k, v in comp.items():
            comp_accum[k] += float(v)
        n_batches += 1

    n = max(n_batches, 1)
    mean_comp = {k: v / n for k, v in comp_accum.items()}
    return total_loss / n, mean_comp


# ---------------------------------------------------------------------------
# Builder 
# ---------------------------------------------------------------------------

def _build_model(cfg: dict) -> GraphVAE:
    """Costruisce un GraphVAE dalla config."""
    return GraphVAE(
        nz               = cfg["nz"],
        hv_dim           = cfg["hv_dim"],
        inner_hidden     = cfg["inner_hidden"],
        inner_layers     = cfg["inner_layers"],
        outer_hidden     = cfg["outer_hidden"],
        outer_layers     = cfg["outer_layers"],
        d                = cfg["d"],
        nhead            = cfg["nhead"],
        tf_layers        = cfg["tf_layers"],
        max_nodes        = cfg["max_nodes"],
        hs               = cfg["hs"],
        ggnn_rounds      = cfg["ggnn_rounds"],
        max_nodes_dec    = cfg["max_nodes_dec"],
        param_hidden     = cfg["param_hidden"],
        param_layers     = cfg["param_layers"],
        spec_d           = cfg["spec_d"],
        spec_layers      = cfg["spec_layers"],
        cg_hidden        = cfg["cg_hidden"],
        dropout          = cfg["dropout"],
        beta_start       = cfg["beta_start"],
        beta_max         = cfg["beta_max"],
        warmup_steps     = cfg["warmup_steps"],
        attrs_scale      = cfg["attrs_scale"],
        align_scale      = cfg["align_scale"],
        nce_scale        = cfg["nce_scale"],
        cg_scale         = cfg["cg_scale"],
        spec_recon_scale = cfg["spec_recon_scale"],
        tau              = cfg["tau"],
        class_weight_end = cfg["class_weight_end"],
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:        GraphVAE,
    scalers:      dict,
    config:       dict,
    train_losses: list,
    val_losses:   list,
    best_val:     float,
    epoch:        int,
    path:         str,
) -> None:
    buf = io.BytesIO()
    pickle.dump(scalers, buf)
    ckpt = {
        **model.state_dict_full(),   # -> {"model_state": ..., "step": ...}
        "config":        config,
        "scalers":       buf.getvalue(),
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_val_loss": best_val,
        "epoch":         epoch,
    }
    torch.save(ckpt, path)
    print(f"  checkpoint salvato -> {path}  (epoch={epoch}, step={model._step})")


def load_checkpoint(
    path:         str,
    device:       torch.device,
    cfg_override: dict | None = None,
) -> tuple[GraphVAE, dict, dict, list, list, int]:
    """
    Load a checkpoint and reconstruct the model.

    Parameters
    ----------
    path         : path to the .pt checkpoint file
    device       : device on which to load the model
    cfg_override : optional dict to override keys in the saved config

    Returns
    -------
    model        : loaded GraphVAE (set to eval mode)
    scalers      : dict[ds_name -> DatasetScalers]
    config       : configuration dict (with overrides applied)
    train_losses : list[float] (previous training history)
    val_losses   : list[float]
    start_epoch  : int (epoch to resume from = saved_epoch + 1)
    """
    ckpt   = torch.load(path, map_location="cpu")
    config = {**ckpt["config"], **(cfg_override or {})}

    model = _build_model(config).to(device)
    model.load_state_dict_full({
        "model_state": ckpt["model_state"],
        "step":        ckpt.get("step", 0),
    })
    model.eval()

    # Apply torch.compile to submodules (same as in train())
    _apply_compile(model, config)

    scalers      = pickle.loads(ckpt["scalers"]) if "scalers" in ckpt else {}
    train_losses = ckpt.get("train_losses", [])
    val_losses   = ckpt.get("val_losses",   [])
    start_epoch  = ckpt.get("epoch", 0) + 1

    print(f"Checkpoint loaded from {path}")
    print(f"  best_val_loss = {ckpt.get('best_val_loss', float('nan')):.6f}")
    print(f"  last epoch    = {start_epoch - 1}  ->  resuming from epoch {start_epoch}")
    print(f"  global steps  = {model._step}  (beta={model.current_beta():.4f})")

    return model, scalers, config, train_losses, val_losses, start_epoch

# ---------------------------------------------------------------------------
# Plot learning curves...DA MODIFICARE SE val é NON SEMPRE
# ---------------------------------------------------------------------------

def plot_curves(
    train_losses: list[float],
    val_losses:   list[float],
    path:         str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train", color="#2196F3")
    ax.plot(epochs, val_losses,   label="val",   color="#FF5722")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss totale (media batch)")
    ax.set_title("GraphVAE — Learning curves (CktGen)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  curve salvate -> {path}")


# ---------------------------------------------------------------------------
# Core training loop 
# ---------------------------------------------------------------------------

def _run_training_loop(
    model:        GraphVAE,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    scalers:      dict,
    optimizer:    torch.optim.Optimizer,
    scheduler,
    cfg:          dict,
    device:       torch.device,
    train_losses: list,
    val_losses:   list,
    start_epoch:  int,
) -> GraphVAE:
    """
    Loop training/validation/early-stopping.

    It updates the Loss both for the training set and the validation test.

    It returns the model of the checkpoint with the lowest Loss after the training.
    """
    # useful for the resume train function
    best_val   = min(val_losses) if val_losses else float("inf")
    best_state = None
    bad_epochs = 0

    # AMP, if available
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    amp_scaler: "amp.GradScaler | None" = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print(f"  Mixed Precision (AMP) available")

    print("Training...")
    print("-" * 72)
    print(
        f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Best':>10}  "
        f"{'beta':>6}  {'LR':>8}  {'Elapsed':>8}"
    )

    t_start = time.time()

    for epoch in range(start_epoch, cfg["epochs"] + 1):

        # Update the current epoch (useful in the resume train function)
        model.param_decoder.current_epoch = epoch

        # train
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            loss_val, comp = train_step(
                model, batch, scalers, optimizer, device, cfg["grad_clip"],
                scaler=amp_scaler,
            )
            epoch_loss += loss_val
            n_batches  += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # validation every val_every epochs (default 5)
        val_every = int(cfg.get("val_every", 5))
        do_val = (epoch == start_epoch) or (epoch % val_every == 0)

        if do_val:
            val_loss, val_comp = eval_loop(model, val_loader, scalers, device, use_amp=use_amp)
            scheduler.step(val_loss)
            val_losses.append(val_loss)

            # early stopping
            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
        else:
            # if no validation in that epoch, the best one of the previous steps is considered
            val_loss  = val_losses[-1] if val_losses else float("nan")
            val_comp  = {}

        train_losses.append(train_loss)

        # logging
        if epoch == start_epoch or epoch % cfg["log_every"] == 0 or (do_val and bad_epochs == 0):
            elapsed = time.time() - t_start
            cur_lr  = optimizer.param_groups[0]["lr"]
            beta    = model.current_beta()
            val_tag = f"{val_loss:>10.5f}" if do_val else f"{'(skip)':>10}"
            print(
                f"{epoch:>6d}  {train_loss:>10.5f}  {val_tag}  "
                f"{best_val:>10.5f}  {beta:>6.4f}  {cur_lr:>8.2e}  "
                f"{elapsed:>7.1f}s"
            )
            if val_comp and epoch % cfg["log_every"] == 0:
                _print_components(val_comp)

        # checkpoint
        if cfg["save_every"] > 0 and epoch % cfg["save_every"] == 0:
            periodic_path = cfg["save_path"].replace(".pt", f"_ep{epoch}.pt")
            save_checkpoint(
                model, scalers, cfg,
                train_losses, val_losses,
                best_val, epoch, periodic_path,
            )

        # early stop
        if do_val and bad_epochs >= cfg["patience"]:
            print(
                f"\n  Early stopping all'epoch {epoch} "
                f"(patience={cfg['patience']} epoche senza miglioramento)."
            )
            break

    # resume best state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nModello riportato al miglior checkpoint (val={best_val:.6f})")

    save_checkpoint(
        model, scalers, cfg,
        train_losses, val_losses,
        best_val, epoch, cfg["save_path"],
    )

    # final test
    test_loss, test_comp = eval_loop(model, test_loader, scalers, device, use_amp=use_amp)
    print(f"\nTest loss: {test_loss:.6f}")
    _print_components(test_comp, prefix="  test ")

    plot_curves(train_losses, val_losses, cfg["plot_path"])

    elapsed_total = time.time() - t_start
    print(f"\nTraining completato in {elapsed_total:.1f}s  ({elapsed_total/60:.1f} min)")
    return model


# ---------------------------------------------------------------------------
# edge pos_weight calibration for edge BCE
# ---------------------------------------------------------------------------

def _calibrate_edge_pos_weight(
    model:      "GraphVAE",
    train_data: list,
    override:   float | None = None,
) -> None:
    """
    Compute the optimal pos_weight for edge BCE and set it in the model's
    topology decoder.

    Formula: pos_weight = n_negative / n_positive
    
    The idea is, if the weight of the edges loss is not given in input (in overrides), 
    it is computed by dividing the n_negative (n_tot - n_real_edges) by the n_positive 
    (n_real_edges). In this way if the edges are few, pos_weight is big, and a null 
    adjacency matrix is not favoured. Otherwise, when there are few edges and no pos_weight,
    the null adj. matrix is favoured since it gives a small loss.
    
    Parameters
    ----------
    model      : GraphVAE, already instantiated
    train_data : training dataset, list of Data objects with topology_ids and ei_outer
    override   : if provided, use this value instead of computing it
                 (e.g. from config)
    """
    if override is not None:
        model.decoder.set_edge_pos_weight(float(override))
        print(f"  edge_pos_weight (override): {override:.3f}")
        return

    from vae_model.decoder import data_to_graph_ns

    n_pos, n_neg = 0, 0

    # Sample at most 2000 examples for speed
    sample_size = min(len(train_data), 2000)

    import random as _random
    indices = _random.sample(range(len(train_data)), sample_size)

    for idx in indices:
        s = train_data[idx]
        n = len(s.topology_ids)
        if n < 2:
            continue
        tot = n * (n - 1) // 2

        # Count unique circuit-to-circuit edges in the sample
        ei = s.ei_outer
        if ei.shape[1] > 0:
            mask = (ei[0] < n) & (ei[1] < n)
            ei_cc = ei[:, mask]
            seen: set = set()
            for u, v in ei_cc.t().tolist():
                k = frozenset({int(u), int(v)})
                if k not in seen and u != v:
                    seen.add(k)
            n_real_edges = len(seen)
        else:
            n_real_edges = 0

        n_pos += n_real_edges
        n_neg += tot - n_real_edges

    if n_pos == 0:
        w = 2.0   # fallback
    else:
        w = n_neg / n_pos
    # Clamp to [1.0, 10.0] for numerical stability
    w = max(1.0, min(w, 10.0))

    model.decoder.set_edge_pos_weight(w)

    print(
        f"  calibrated edge_pos_weight: {w:.3f}  "
        f"(pos={n_pos}, neg={n_neg}, sample={sample_size})"
    )


# ---------------------------------------------------------------------------
# torch.compile on tensor-only submodules
# ---------------------------------------------------------------------------

def _apply_compile(model: "GraphVAE", cfg: dict) -> None:
    """
    Compile with torch.compile only the submodules whose forward() receives
    tensors exclusively, avoiding graph breaks caused by Python logic in
    _prepare_batch and G_true.

    Compiled submodules:
        encoder, decoder, param_decoder, spec_encoder, obs_classifier

    Not compiled: GraphVAE.forward() — contains arbitrary Python logic
    (loops over Data objects, SimpleNamespace, hasattr, dict lookups).

    With fullgraph=False, graph breaks are handled silently: if a pattern is
    unsupported, the compiler falls back to eager mode for that block without crashing.
    """
    if not bool(cfg.get("use_compile", True)):
        return
    if not hasattr(torch, "compile"):
        print("  torch.compile not available (PyTorch < 2.0) — skipping")
        return

    #these are the compatible modules of the VAE model
    submodules = [
        ("encoder", model.encoder),
        ("decoder", model.decoder),
        ("param_decoder", model.param_decoder),
        ("spec_encoder", model.spec_encoder),
        ("obs_classifier", model.obs_classifier),
    ]
    compiled = []
    for name, mod in submodules:
        try:
            setattr(model, name, torch.compile(mod, fullgraph=False))
            compiled.append(name)
        except Exception as e:
            print(f"  torch.compile({name}) failed: {e} — skipping")
    if compiled:
        print(f"  torch.compile enabled on: {', '.join(compiled)}")
        print("  (first epoch may be slower due to JIT compilation — expected)")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict | None = None) -> GraphVAE:
    """

    Parameters
    ----------
    config : dict | None
        Override of any DEFAULT_CONFIG's key.

    Returns
    -------
    model : GraphVAE trained at the best checkpoint with respect to val loss.
    """
    cfg    = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # data
    train_data, val_data, test_data, scalers = load_all_datasets_vae(
        train_frac = cfg["train_frac"],
        val_frac   = cfg["val_frac"],
        seed       = cfg["seed"],
        max_nodes  = cfg["max_nodes"],
    )
    print(f"\nSplit: train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    # beta info
    steps_per_epoch = max(1, len(train_data) // cfg["batch_size"])
    warmup_epochs   = cfg["warmup_steps"] / steps_per_epoch
    print(
        f"beta-annealing: warmup_steps={cfg['warmup_steps']}  "
        f"~{warmup_epochs:.1f} epoche  ({steps_per_epoch} step/epoca)"
    )

    train_loader = DataLoader(
        train_data, batch_size=cfg["batch_size"],
        shuffle=True,  collate_fn=collate_vae,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=False,  # collate_vae returns Data objects with Python fields; pin_memory is incompatible
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )
    val_loader = DataLoader(
        val_data,   batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_vae,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=False,  # collate_vae returns Data objects with Python fields; pin_memory is incompatible
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )
    test_loader = DataLoader(
        test_data,  batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_vae,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=False,  # collate_vae returns Data objects with Python fields; pin_memory is incompatible
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )

    # model
    model = _build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model}")
    print(f"Total parameters: {n_params:,}\n")

    # torch.compile on tensor-only submodules (the ones with no loops, Data object,...)
    _apply_compile(model, cfg)

    # computing the weight of the edges loss, if not given as input
    _calibrate_edge_pos_weight(model, train_data, cfg.get("edge_pos_weight", None))

    #optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["lr_patience"],
    )

    return _run_training_loop(
        model, train_loader, val_loader, test_loader,
        scalers, optimizer, scheduler, cfg, device,
        train_losses=[], val_losses=[], start_epoch=1,
    )


# ---------------------------------------------------------------------------
# Resume training da checkpoint...DA RIVEDERE
# ---------------------------------------------------------------------------

def resume_train(checkpoint_path: str, config_override: dict | None = None) -> GraphVAE:
    """
    Riprende il training da un checkpoint esistente.

    Parametri
    ----------
    checkpoint_path : path al .pt salvato da save_checkpoint()
    config_override : chiavi da sovrascrivere (es. {"epochs": 500, "lr": 5e-5})

    Ritorna
    -------
    model : GraphVAE al miglior checkpoint del run ripreso
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, scalers, cfg, train_losses, val_losses, start_epoch = load_checkpoint(
        checkpoint_path, device, cfg_override=config_override,
    )

    # Riallinea param_decoder all'epoca da cui si riprende
    model.param_decoder.current_epoch = start_epoch - 1

    # Ricrea i loader con la stessa config e seed
    train_data, val_data, test_data, _ = load_all_datasets_vae(
        train_frac = cfg["train_frac"],
        val_frac   = cfg["val_frac"],
        seed       = cfg["seed"],
        max_nodes  = cfg["max_nodes"],
    )
    print(f"\nSplit: train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    train_loader = DataLoader(
        train_data, batch_size=cfg["batch_size"],
        shuffle=True,  collate_fn=collate_vae,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=False,  # collate_vae returns Data objects with Python fields; pin_memory is incompatible
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )
    val_loader = DataLoader(
        val_data,   batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_vae,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=False,  # collate_vae returns Data objects with Python fields; pin_memory is incompatible
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )
    test_loader = DataLoader(
        test_data,  batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_vae,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=False,  # collate_vae returns Data objects with Python fields; pin_memory is incompatible
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["lr_patience"],
    )

    model.train()
    return _run_training_loop(
        model, train_loader, val_loader, test_loader,
        scalers, optimizer, scheduler, cfg, device,
        train_losses=train_losses,
        val_losses=val_losses,
        start_epoch=start_epoch,
    )


# ---------------------------------------------------------------------------
# Helper for printing components
# ---------------------------------------------------------------------------

def _print_components(comp: dict, prefix: str = "  ") -> None:
    """Print the loss components."""
    keys_main = [
        "loss_topo_c", "loss_attrs_c",
        "loss_topo_s", "loss_attrs_s",
        "loss_kl",     "loss_align",
        "loss_nce",    "loss_cg",
    ]
    parts = []
    for k in keys_main:
        if k in comp and comp[k] != 0.0:
            short = k.replace("loss_", "").replace("topo_", "top_")
            parts.append(f"{short}={comp[k]:.4f}")
    if "beta" in comp:
        parts.append(f"beta={comp['beta']:.4f}")
    print(f"{prefix}[{' | '.join(parts)}]")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Training GraphVAE for cQED circuits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--eval_only",  action="store_true",
                   help="Carica checkpoint ed esegue solo valutazione sul test set")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path checkpoint .pt (eval_only o resume training)")

    g = p.add_argument_group("architettura")
    g.add_argument("--nz",            type=int,   default=None)
    g.add_argument("--outer_hidden",  type=int,   default=None)
    g.add_argument("--outer_layers",  type=int,   default=None)
    g.add_argument("--inner_hidden",  type=int,   default=None)
    g.add_argument("--inner_layers",  type=int,   default=None)
    g.add_argument("--d",             type=int,   default=None)
    g.add_argument("--nhead",         type=int,   default=None)
    g.add_argument("--tf_layers",     type=int,   default=None)
    g.add_argument("--hs",            type=int,   default=None)
    g.add_argument("--spec_d",        type=int,   default=None)
    g.add_argument("--spec_layers",   type=int,   default=None)
    g.add_argument("--dropout",       type=float, default=None)
    g.add_argument("--max_nodes",     type=int,   default=None)

    g = p.add_argument_group("beta-annealing")
    g.add_argument("--beta_start",    type=float, default=None)
    g.add_argument("--beta_max",      type=float, default=None)
    g.add_argument("--warmup_steps",  type=int,   default=None)

    g = p.add_argument_group("pesi loss")
    g.add_argument("--attrs_scale",      type=float, default=None)
    g.add_argument("--align_scale",      type=float, default=None)
    g.add_argument("--nce_scale",        type=float, default=None)
    g.add_argument("--cg_scale",         type=float, default=None)
    g.add_argument("--spec_recon_scale", type=float, default=None,
                   help="0=solo ramo circuito, 1=CktGen completo")
    g.add_argument("--tau",              type=float, default=None)
    g.add_argument("--class_weight_end",   type=float, default=None,
                   help="Peso relativo della classe END nella loss_t (1.0 = neutro)")
    g.add_argument("--edge_pos_weight",    type=float, default=None,
                   help="pos_weight BCE archi (None=auto-calibrato, tipico 2-5)")

    g = p.add_argument_group("ottimizzatore")
    g.add_argument("--lr",           type=float, default=None)
    g.add_argument("--weight_decay", type=float, default=None)
    g.add_argument("--grad_clip",    type=float, default=None)

    g = p.add_argument_group("training")
    g.add_argument("--batch_size",   type=int,   default=None)
    g.add_argument("--epochs",       type=int,   default=None)
    g.add_argument("--val_every",    type=int,   default=None,
                   help="Esegue validation ogni N epoche (default 5; 1=ogni epoca)")
    g.add_argument("--patience",     type=int,   default=None)
    g.add_argument("--lr_patience",  type=int,   default=None)
    g.add_argument("--seed",         type=int,   default=None)

    g = p.add_argument_group("I/O")
    g.add_argument("--save_path",    type=str,   default=None)
    g.add_argument("--log_every",    type=int,   default=None)
    g.add_argument("--save_every",   type=int,   default=None)
    g.add_argument("--plot_path",    type=str,   default=None)

    g = p.add_argument_group("performance")
    g.add_argument("--num_workers",  type=int,   default=None,
                   help="DataLoader workers (default 4; 0 = single-thread)")
    g.add_argument("--no_compile",   action="store_true",
                   help="Disabilita torch.compile anche se disponibile")
    g.add_argument("--no_amp",       action="store_true",
                   help="Disabilita Mixed Precision anche se CUDA disponibile")

    return p.parse_args()


def _args_to_overrides(args: argparse.Namespace) -> dict:
    """Filters out what needs to be overridden"""
    skip = {"eval_only", "checkpoint", "no_amp", "no_compile"}
    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in skip}
    # --no_amp flag: if present, AMP is disabled
    if getattr(args, "no_amp", False):
        overrides["use_amp"] = False
    if getattr(args, "no_compile", False):
        overrides["use_compile"] = False
    return overrides


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args      = _parse_args()
    overrides = _args_to_overrides(args)

    if args.num_workers not in (0, None):
        raise ValueError("num_workers must be 0 for this run")

    if args.eval_only:
        if args.checkpoint is None:
            print("ERRORE: --eval_only richiede --checkpoint", file=sys.stderr)
            sys.exit(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, scalers, cfg, _, _, _ = load_checkpoint(
            args.checkpoint, device, cfg_override=overrides,
        )

        _, _, test_data, _ = load_all_datasets_vae(
            train_frac = cfg.get("train_frac", 0.70),
            val_frac   = cfg.get("val_frac",   0.15),
            seed       = cfg.get("seed",       42),
            max_nodes  = cfg.get("max_nodes",  12),
        )
        test_loader = DataLoader(
            test_data,
            batch_size  = cfg.get("batch_size", 32),
            shuffle     = False,
            collate_fn  = collate_vae,
            num_workers = 0,
        )
        test_loss, test_comp = eval_loop(model, test_loader, scalers, device)
        print(f"\nTest loss: {test_loss:.6f}")
        _print_components(test_comp)

    elif args.checkpoint:
        # resume from checkpoint
        resume_train(args.checkpoint, config_override=overrides)

    else:
        # new training 
        train(overrides)

# cQED-Fast

**Graph-based Variational Autoencoder for fast inverse design of superconducting quantum circuits.**

cQED-Fast learns a continuous latent representation of circuit topologies and physical parameters, enabling both *circuit reconstruction* and *observable-conditioned generation*: given a set of target Hamiltonian observables (qubit frequencies, decay rates, anharmonicities), the model can propose valid circuit designs that match them.

---

## Overview

Superconducting quantum circuits are described as **hierarchical graphs**: each macro-node represents a composite building block (transmon, resonator, coupler, …), which is in turn described by a small inner graph of circuit elements. cQED-Fast encodes this two-level structure into a compact latent vector and decodes it back into a valid graph with physical parameters.

The model follows the **bidirectional CktGen paradigm**, jointly training two branches:

| Branch | Input | Latent | Role |
|--------|-------|--------|------|
| Circuit branch | Graph `G` | `z^c` | Encodes topology + parameters |
| Observable branch | Hamiltonian observables | `z^s` | Encodes measured/simulated specs |

Both branches are aligned in latent space so that `z^c ≈ z^s` for the same circuit, making it possible to use observable queries at inference time.

---

## Architecture

```
               ┌──────────────────────────┐
Circuit G ───► │  GraphVAEEncoder         │ ──► μ^c, Σ^c, z^c ──► Decoder ──► G'
               │  (InnerGIN + OuterGIN    │                    └──► ParamDecoder ──► θ̂
               │   + Transformer)         │
               └──────────────────────────┘
                                                      ▲
               ┌──────────────────────────┐           │ alignment loss
Observables ──► │  SpecEncoder (MLP)       │ ──► μ^s, Σ^s, z^s
               └──────────────────────────┘
                          │
                          └──► ObsClassifier ──► ŝ (classifier guidance)
```

### Encoder (`src/vae_model/encoder.py`)
1. **InnerEncoder** — GIN message passing on each subgraph block → node embedding `h_v`
2. **OuterGIN** — GIN on the outer circuit graph using `[SubgType one-hot | h_v]` features
3. **Transformer Encoder** — cross-node attention with learnable `μ_tok` / `Σ_tok` tokens
4. Reparameterization → `z^c`

### Decoder (`src/vae_model/decoder.py`)
- **TransformerTopologyDecoder** — autoregressive generation of macro-nodes (GGNN-based)
- **ParamDecoder** — GraphSAGE network that predicts physical parameters `θ` given the generated topology

### Observable Encoder (`src/vae_model/obs_encoder.py`)
- **SpecEncoder** — per-slot MLP embedding of Hamiltonian observables with missing-value masking → `z^s`
- **ObsClassifier** — auxiliary heads for classifier guidance from `z^c`

### Circuit-to-Graph (`src/circuit2graph/`)
- `definitions.py` — canonical registry of subgraph types (`SubgType`) and their physical attributes (`ATTR_INDEX`)
- `topology.py` — constructs the two-level graph representation from raw circuit data
- `compression.py` — compresses repeated inner-graph patterns into macro-nodes

---

## Training objective

The full loss (Eq. 8 of the CktGen paper) is:

```
L = λ_KL · L_KL  +  L^c_R  +  L^s_R  +  L_C  +  L_CG  +  L_NCE
```

| Term | Description |
|------|-------------|
| `L^c_R` | Reconstruction from `z^c` (topology + parameters) |
| `L^s_R` | Reconstruction from `z^s` (generative pathway from observables) |
| `L_KL` | KL to `N(0,I)` + symmetric KL between the two branches |
| `L_C` | SmoothL1 latent alignment: `z^c ≈ z^s` |
| `L_CG` | Classifier guidance: `z^c → observables` |
| `L_NCE` | Contrastive InfoNCE loss between `(z^s, z^c)` pairs |

β-annealing is used to warm up the KL term over the first `warmup_steps` gradient steps.

---

## Supported circuit elements

| `SubgType` | Description |
|------------|-------------|
| `FEEDLINE` | Transmission-line feedline |
| `TRANSMON` | Single transmon qubit |
| `RESONATOR` | Microwave resonator |
| `C_COUPLER` | Capacitive coupler |
| `TC` | Transmon–capacitor block |
| `RC` | Resonator–capacitor block |
| `RCT` | Resonator–capacitor–transmon block |
| `TCT` | Transmon–coupler–transmon block |
| `I_COUPLER` | Inductive coupler |
| `RIND` | Inductive resonator |

Physical attributes tracked: inductance `L`, capacitance `C`, resonator length, coupling capacitances `Cc`, `Cc_qr`, `Cc_rf`, wire separation `D`, coupling length `l`.

Target Hamiltonian observables: qubit frequencies `f_1`, `f_2`, decay rates `κ_1`, `κ_2`, anharmonicities `χ_11`, `χ_22`, cross-Kerr `χ_12`.

---

## Project structure

```
cQED_Fast/
├── train_vae.py          # Training script (full CLI)
├── inference_vae.py      # Inference, sampling, and evaluation
└── src/
    ├── circuit2graph/
    │   ├── definitions.py    # SubgType enum, ATTR_INDEX, SubgDef registry
    │   ├── topology.py       # Two-level graph construction
    │   └── compression.py    # Macro-node compression
    ├── data_loader/
    │   ├── schema.py         # Dataset definitions and observable vocabulary
    │   ├── loader_vae.py     # DataLoader with train/val/test splits and scalers
    │   └── processing.py     # Feature normalization utilities
    └── vae_model/
        ├── vae.py            # GraphVAE — top-level module
        ├── encoder.py        # GraphVAEEncoder (InnerGIN + OuterGIN + Transformer)
        ├── decoder.py        # TransformerTopologyDecoder + ParamDecoder
        └── obs_encoder.py    # SpecEncoder + ObsClassifier + contrastive losses
```

---

## Installation

```bash
# Python 3.9+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install matplotlib numpy
```

> PyTorch ≥ 2.0 is recommended to take advantage of `torch.compile` acceleration.

---

## Usage

### Training

```bash
# Default training run
python train_vae.py

# Override hyperparameters
python train_vae.py --nz 64 --epochs 200 --batch_size 32

# Disable observable branch (circuit-only ablation)
python train_vae.py --spec_recon_scale 0 --nce_scale 0 --cg_scale 0
```

### Resume from checkpoint

```bash
python train_vae.py --checkpoint best_vae.pt --epochs 400
```

### Evaluation only

```bash
python train_vae.py --eval_only --checkpoint best_vae.pt
```

### Inference and generation

```bash
python inference_vae.py --checkpoint best_vae.pt
```

---

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nz` | 128 | Latent space dimension |
| `beta_max` | 1e-4 | Maximum KL weight |
| `warmup_steps` | 21 900 | Steps to reach `beta_max` (~20 epochs on 35 k samples, batch 32) |
| `spec_recon_scale` | 0.3 | Weight of the observable branch reconstruction (0 = circuit-only) |
| `nce_scale` | 0.5 | Contrastive loss weight |
| `cg_scale` | 0.5 | Classifier guidance loss weight |
| `align_scale` | 0.5 | Latent alignment loss weight |
| `max_nodes` | 12 | Maximum number of macro-nodes |
| `batch_size` | 128 | Training batch size |
| `epochs` | 300 | Maximum training epochs |
| `patience` | 40 | Early stopping patience (validation epochs) |

---

## Checkpoint format

Checkpoints saved by `train_vae.py` contain:

```python
{
    "model_state":   ...,   # GraphVAE state_dict
    "step":          int,   # global gradient steps (for β-annealing resume)
    "config":        dict,  # full configuration used
    "scalers":       bytes, # pickle(scalers) for feature normalization
    "train_losses":  list,  # per-epoch training loss history
    "val_losses":    list,
    "best_val_loss": float,
    "epoch":         int,
}
```

---

## Extending the model

To add a new circuit element type:

1. Add its physical attribute(s) to `ATTR_INDEX` in `src/circuit2graph/definitions.py` if not already present.
2. Add a new member to `SubgType`.
3. Add a `SubgDef` entry to `SUBG_DEFS` with `inner_nodes`, `inner_edges`, `attrs`, `color`, and `legend`.
4. Implement the corresponding merge rule in the compression module.

To add a new observable:

1. Append the new name to `OBS_SLOTS` in `src/data_loader/schema.py`.
2. Write a parser function and register it in `OBS_PARSERS` / `DATASETS`.

---

## References

- Flam-Shepherd et al., *"CktGen: Graph generative model for circuit design"* — the bidirectional VAE paradigm this model is based on.
- Kipf & Welling, *"Variational Graph Auto-Encoders"*, 2016.
- Li et al., *"Gated Graph Sequence Neural Networks"*, 2016.

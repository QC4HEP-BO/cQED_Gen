# cQED Gen

Generative framework for inverse design of superconducting quantum circuits.

**Goal:** given target Hamiltonian observables (qubit frequencies, coupling strengths, linewidths), automatically generate a circuit topology and its physical parameters.

---

## Model

A Graph VAE with two encoders and two decoders:

- **Circuit encoder** — GIN + Transformer, maps a compressed circuit graph → latent vector `z^c`
- **Spec encoder** — MLP, maps Hamiltonian observables → `z^s` in the same latent space
- **Topology decoder** — autoregressive GPT-like Transformer, `z` → macro-node sequence + edges
- **Parameter decoder** — GraphSAGE, `(z, topology)` → physical parameter values

Training combines topology reconstruction, parameter regression, KL regularisation, latent alignment (circuit ↔ observables), InfoNCE contrastive loss, and classifier guidance.

At inference, the spec encoder path `obs → z^s → decoder` generates candidate circuits from target quantum properties without ever seeing a labelled circuit.

---

## Repository layout

```
cQED_Gen/
├── train_vae.py              # training script
├── inference_vae.py          # evaluation, reconstruction, latent sampling
├── tests                     # tests
├── data/                     # raw dataset .txt files (one per topology)
└── src/
    ├── circuit2graph/        # circuit → compressed graph (graphlize)
    │   ├── definitions.py    # SubgType enum, SUBG_DEFS, ATTR_INDEX
    │   ├── topology.py       # CQEDTopology, CQEDNode
    │   └── compression.py    # graphlize() merge rules
    ├── data_loader/
    │   ├── datasets/         # one .py file per topology dataset  ← add new datasets here
    │   │   ├── _base.py      # DatasetBase interface + parsing helpers
    │   │   ├── qubit.py
    │   │   ├── resonator.py
    │   │   └── ...
    │   ├── schema.py         # auto-discovery registry + OBS_SLOTS
    │   ├── processing.py     # scalers, _fill_attrs, _extract_params
    │   └── loader_vae.py     # load_all_datasets_vae / load_inference_datasets_vae
    └── vae_model/
        ├── encoder.py        # GraphVAEEncoder (InnerGIN + OuterGIN + Transformer)
        ├── decoder.py        # TransformerTopologyDecoder + ParamDecoder
        ├── obs_encoder.py    # SpecEncoder + ObsClassifier + alignment/contrastive losses
        └── vae.py            # GraphVAE (full model, forward, encode, decode, sample)
```

---

## Adding new elements

### New dataset (new circuit topology)
Create one file in `src/data_loader/datasets/your_topology.py`, subclass `DatasetBase`, and implement:
- `build_topology()` — raw `CQEDTopology` template
- `parse_row(line)` — parse one data file row → `(attrs_dict, obs_kw_dict)`
- `parse_obs(...)` — fill the observable vector and mask
- Set `NAME`, `DATA_PATH`, `BLOCK_PARAMS`, `OBS_SLOTS_ACTIVE`, `N_SAMPLES`, `INCLUDE_TRAIN`

`schema.py` discovers it automatically on the next import. No other file needs to change.

> **Dataset generation** (running simulations to produce `.txt` files) will be integrated directly into the repo in a future release.

### New observable type (e.g. `f_4`, `chi_44`)
Append the new name to `OBS_SLOTS` in `src/data_loader/schema.py`. Always append at the end to keep existing checkpoint indices valid.

### New macro-node type (new compressed subgraph)
Edit `src/circuit2graph/definitions.py`:
1. Add the new value to the `SubgType` enum
2. Add the corresponding `ATTR_INDEX` entries for any new physical attribute
3. Add a `SubgDef` entry to `SUBG_DEFS` with `attrs`, `inner_nodes`, `inner_edges`
4. Add the merge rule in `src/circuit2graph/compression.py`

### New primitive circuit element (new inner node type inside a subgraph)
Add a new `type_id` in the relevant `SubgDef.inner_nodes` inside `definitions.py` and update `N_INNER_TYPES` accordingly (it is derived automatically as `max(type_id) + 1`).

---

## Global training size

To change the number of training samples for all datasets at once, set `N_SAMPLES_OVERRIDE` in `src/data_loader/schema.py`:

```python
N_SAMPLES_OVERRIDE: int | None = 5_000   # None → use each dataset's own N_SAMPLES
```

To control a single dataset, set `N_SAMPLES` in its file under `datasets/`.

---

## Installation

```bash
pip install torch torchvision torchaudio          # follow pytorch.org for your CUDA version
pip install torch-geometric
pip install numpy matplotlib networkx
pip install qultra                              # for validation in inference_vae
```

---

## Usage

```bash
# Training
python train_vae.py

# Evaluation on test split
python inference_vae.py --ckpt checkpoints/vae_best.pt

# Include inference-only topologies (never seen in training)
python inference_vae.py --ckpt checkpoints/vae_best.pt --inference-only \
  --plot-topo-errors --topo-errors-dataset Three_qubit_capacitive_line

# Sample from latent space
python inference_vae.py --ckpt checkpoints/vae_best.pt \
  --plot-latent-samples --latent-n 20 --stochastic
```

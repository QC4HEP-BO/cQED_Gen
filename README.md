# cQED-Gen

**Generative graph neural networks for the inverse design of superconducting circuit-QED (cQED) circuits.**

cQED-Gen addresses the inverse-design problem

```text
target Hamiltonian observables  ->  circuit topology + physical parameters
```

rather than the conventional forward workflow in which a circuit is specified first and its quantum response is computed afterwards.

The repository implements a conditional **Graph Variational Autoencoder (GraphVAE)** that maps superconducting circuit graphs and effective-Hamiltonian specifications into a shared latent space. Starting from a target set of observables, the model can generate both a circuit topology and its continuous physical parameters without fixing the topology in advance.

Generated candidates can then be:

- constrained during decoding by cQED connectivity rules;
- independently evaluated with **QuLTRA**;
- refined through **Cross-Entropy Method (CEM)** optimization in latent space;
- translated into an initial **Qiskit Metal layout-zero** through the `graph2metal` package.

This snapshot corresponds to the final code organization used for the thesis analyses.

---

## Table of contents

- [Thesis](#thesis)
- [Model overview](#model-overview)
  - [Circuit representation](#circuit-representation)
  - [Circuit encoder](#circuit-encoder)
  - [Hamiltonian encoder](#hamiltonian-encoder)
  - [Decoders](#decoders)
  - [Training objective](#training-objective)
- [Hamiltonian observables](#hamiltonian-observables)
- [Dataset configuration](#dataset-configuration)
- [Repository layout](#repository-layout)
  - [Core packages](#core-packages)
- [Main executable files](#main-executable-files)
- [`thesis_analysis/`](#thesis_analysis)
- [Checkpoint](#checkpoint)
- [Installation](#installation)
  - [QuLTRA](#qultra)
  - [Qiskit Metal](#qiskit-metal)
- [Quick start](#quick-start)
  - [Training](#training)
  - [Circuit/topology inference](#circuittopology-inference)
  - [QuLTRA Hamiltonian validation](#qultra-hamiltonian-validation)
  - [End-to-end Hamiltonian-conditioned optimization](#end-to-end-hamiltonian-conditioned-optimization)
  - [Prior sampling](#prior-sampling)
- [Reproducing the thesis analyses](#reproducing-the-thesis-analyses)
- [Extending cQED-Gen](#extending-cqed-gen)
  - [Adding a new dataset / circuit topology](#adding-a-new-dataset--circuit-topology)
  - [Adding a new observable](#adding-a-new-observable)
  - [Adding a new primitive circuit element](#adding-a-new-primitive-circuit-element)
  - [Adding a new macro-node / subgraph basis element](#adding-a-new-macro-node--subgraph-basis-element)
- [Next steps](#next-steps)
- [Validation of this snapshot](#validation-of-this-snapshot)
- [Scope of the physical-layout extension](#scope-of-the-physical-layout-extension)

---

## Thesis

The physical motivation, graph representation, model architecture, training objective, inverse-design formulation, latent-space optimization, validation studies, ablations, and graph-to-layout extension are described in the thesis.

**Thesis:** [LINK TO BE ADDED]

---

## Model overview

cQED-Gen uses two complementary encoders that map the same physical device into a common latent space:

```text
Circuit graph G --------> Circuit encoder --------> z_c
                                                   |
                                                   | shared latent space
                                                   |
Hamiltonian H ----------> Spec encoder -----------> z_s
                                                   |
                                                   v
                                      Topology + parameter decoders
                                                   |
                                                   v
                                         Generated cQED circuit
```

For inverse design, the relevant path is

```text
target Hamiltonian
       |
       v
Specification encoder
       |
      z_s
       |
       v
Topology decoder + parameter decoder
       |
       v
generated macro-graph
       |
       v
primitive circuit
       |
       +------------> QuLTRA physical validation
       |
       `------------> graph2metal / Qiskit Metal layout-zero
```

### Circuit representation

Primitive cQED networks are compressed with `graphlize()` into a macro-node representation. The current macro-node vocabulary is defined in `src/circuit2graph/definitions.py`:

```text
FEEDLINE
TRANSMON
RESONATOR
C_COUPLER
TC
RC
RCT
TCT
I_COUPLER
RI
```

The compression/expansion machinery is implemented in `src/circuit2graph/`, together with symmetry handling and decoder-side physical compatibility constraints.

### Circuit encoder

`src/vae_model/encoder.py` implements the graph branch using hierarchical GIN processing followed by a Transformer representation that produces the circuit posterior parameters `mu_c` and `logvar_c`.

### Hamiltonian encoder

`src/vae_model/obs_encoder.py` implements the specification branch. Hamiltonian slots are embedded with an explicit mask for missing observables, allowing inference from **partially specified Hamiltonians**. The branch produces `mu_s` and `logvar_s` in the same latent space as the circuit encoder.

### Decoders

`src/vae_model/decoder.py` contains:

- an autoregressive Transformer topology decoder;
- connectivity prediction with bidirectional attention;
- decoder-side cQED compatibility constraints;
- a GraphSAGE-style continuous parameter decoder.

The topology and physical parameters are therefore generated jointly from the latent representation.

### Training objective

`train_vae.py` jointly trains the circuit and specification branches using topology reconstruction, continuous-parameter reconstruction, KL regularization, latent alignment, InfoNCE contrastive learning, auxiliary Hamiltonian/classifier guidance, and reconstruction from both latent branches.

---

## Hamiltonian observables

The global observable vocabulary is defined in `src/data_loader/schema.py`:

```text
f_1, f_2,
kappa_1, kappa_2,
chi_11, chi_22, chi_12,
f_3,
chi_33, chi_13, chi_23
```

Each dataset activates only the observables available for that circuit family. Missing entries remain masked and do not contribute as observed targets.

---

## Dataset configuration

Dataset definitions live in `src/data_loader/datasets/` and are automatically discovered by `src/data_loader/schema.py`.

The final training registry contains eight circuit families:

```text
Qubit
Resonator
Qubit_resonator_feedline_capacitive
Qubit_resonator_feedline_inductive_nognd
Qubit_resonator_resonator
Resonator_qubit_resonator
Two_qubit_with_capacitive_coupling
Three_qubit_capacitive_star
```

`Three_qubit_capacitive_line` is configured with `INCLUDE_TRAIN = False` and is used as a held-out topology for unseen-circuit evaluation.

Each registered dataset currently requests up to `10_000` samples. With the default `70% / 15% / 15%` split, the eight training families correspond approximately to `56 000 / 12 000 / 12 000` train/validation/test configurations before loader filtering.

The final loader fits normalization statistics only on the training partition and uses a **global attribute-aware parameter scaler** together with a global observable scaler. The final thesis checkpoint must therefore contain `scalers["__global__"]`.

---

## Repository layout

```text
cQED_Gen/
|
|-- train_vae.py
|-- inference_circuit_elements.py
|-- inference_hamiltonian_qultra.py
|-- inference_end2end.py
|
|-- data/
|
|-- src/
|   |-- circuit2graph/
|   |-- data_loader/
|   |-- vae_model/
|   |-- graph2qultra/
|   `-- graph2metal/
|
|-- test/
|   |-- test_graphlize.py
|   |-- test_expansion.py
|   `-- test_symmetries.py
|
|-- thesis_analysis/
|   |-- analysis_4_2_1_training_validation_loss.py
|   |-- analysis_4_2_3_latent_space.py
|   |-- analysis_4_3_1_seen_circuits.py
|   |-- analyze_unseen_threequbit_and_stress_4T_errorbars.py
|   |-- analysis_4_3_3_ablation_study_revised.py
|   |-- inference_threequbit_constraint.py
|   |-- sampling.py
|   `-- run_graph2metal_test.py
|
|-- LICENSE
`-- README.md
```

### Core packages

| Path | Purpose |
| --- | --- |
| `src/circuit2graph/` | Primitive/macro graph definitions, `graphlize()` compression, expansion, symmetry handling, and physical compatibility constraints. |
| `src/data_loader/` | Dataset discovery, parsing, train/validation/test construction, global scaling, and conversion to model inputs. |
| `src/vae_model/` | Circuit encoder, specification encoder, topology decoder, parameter decoder, losses, and complete `GraphVAE`. |
| `src/graph2qultra/` | Conversion from generated circuits to QuLTRA-compatible networks and forward physical evaluation helpers. |
| `src/graph2metal/` | Conversion from a primitive circuit graph to an initial planar layout plan and optional Qiskit Metal rendering. |

---

## Main executable files

| File | Purpose |
| --- | --- |
| `train_vae.py` | Trains the complete dual-branch GraphVAE and stores weights, configuration, loss history, and scaler state in the checkpoint. |
| `inference_circuit_elements.py` | Evaluates topology reconstruction and continuous circuit-parameter prediction from both the circuit and Hamiltonian branches, including symmetry-aware parameter alignment. |
| `inference_hamiltonian_qultra.py` | Final QuLTRA-enabled inference implementation. Generated circuits are expanded, simulated, mode-aligned, and compared with target frequencies, Kerr terms, and linewidths. |
| `inference_end2end.py` | End-to-end Hamiltonian-conditioned inference followed by CEM latent-space optimization with QuLTRA in the loop. Supports partial targets, primitive-count guidance, and optional graph-to-layout export. |


---

## `thesis_analysis/`

The final thesis-analysis scripts are grouped in `thesis_analysis/`.

| Script | Purpose |
| --- | --- |
| `analysis_4_2_1_training_validation_loss.py` | Reconstructs the training/validation history and produces the training-loss and loss-component analyses used in Secs. 4.2.1-4.2.2. |
| `analysis_4_2_3_latent_space.py` | Performs t-SNE visualization, original-dimensional latent geometry, circuit/specification alignment, cross-modal retrieval, and repeated prior sampling with validity/novelty statistics. |
| `analysis_4_3_1_seen_circuits.py` | Evaluates seen circuit families on the test split: typed topology reconstruction, symmetry-aware physical-parameter errors, repeated subsampling statistics, and QuLTRA Hamiltonian validation. |
| `analyze_unseen_threequbit_and_stress_4T_errorbars.py` | Evaluates the held-out `Three_qubit_capacitive_line` topology and plausible Hamiltonian stress tests up to four-transmon structural requirements, comparing direct inference and CEM refinement with uncertainty estimates. |
| `analysis_4_3_3_ablation_study_revised.py` | Runs the final 100-epoch objective ablations, decoder-constraint ablations, and optional latent-dimension sensitivity study. |
| `inference_threequbit_constraint.py` | Diagnostic constraint-aware inference on the held-out three-qubit line and latent sampling under primitive-count requirements. |
| `sampling.py` | Samples `z ~ N(0,I)`, decodes/expands generated circuits, and evaluates structural/physical validity with optional QuLTRA evaluation. |
| `run_graph2metal_test.py` | Tests graph-to-layout planning, automatic chip fitting, and optional Qiskit Metal rendering. |

---

## Checkpoint

A trained `.pt` checkpoint is **not bundled in this repository snapshot**.

The thesis-analysis scripts are designed around the final global-scaler checkpoint, typically named:

```text
best_vae_global.pt
```

Place the checkpoint in the repository root, or pass its path explicitly through `--ckpt`, `--checkpoint`, or `--reference` depending on the script.

For unseen-topology inference, end-to-end optimization, and the final thesis analyses, use a checkpoint containing the global scaler (`scalers["__global__"]`).

---

## Installation

Python **3.10+** is recommended.

Install PyTorch first using the build appropriate for your CPU/CUDA environment, then install the core dependencies:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy matplotlib networkx scipy scikit-learn shapely
```

For development/testing:

```bash
pip install pytest
```

### QuLTRA

QuLTRA is required for the physical-validation and CEM-in-the-loop workflows. It must be installed separately in the active environment and importable by the repository wrapper as `qultra` or `qu`.

It is used by, among others:

```text
inference_hamiltonian_qultra.py
inference_end2end.py
thesis_analysis/analysis_4_3_1_seen_circuits.py
thesis_analysis/analyze_unseen_threequbit_and_stress_4T_errorbars.py
thesis_analysis/analysis_4_3_3_ablation_study_revised.py
thesis_analysis/sampling.py        # only with --run-qultra
```

### Qiskit Metal

Qiskit Metal is optional and is required only for native layout rendering. The graph2metal planner can still be exercised without Qiskit Metal by using `--plan-only`.

---

## Quick start

Run commands from the repository root.

### Training

```bash
python train_vae.py --save_path best_vae.pt
```

Example with explicit overrides:

```bash
python train_vae.py --epochs 300 --batch_size 128 --nz 128
```

Resume training:

```bash
python train_vae.py --checkpoint best_vae.pt --epochs 400
```

### Circuit/topology inference

```bash
python inference_circuit_elements.py \
  --ckpt best_vae_global.pt \
  --split test \
  --out-dir results/circuit_inference
```

### QuLTRA Hamiltonian validation

```bash
python inference_hamiltonian_qultra.py \
  --ckpt best_vae_global.pt \
  --split test \
  --out-dir results/qultra_inference
```

### End-to-end Hamiltonian-conditioned optimization

Only observables explicitly provided on the command line are treated as fixed targets.

```bash
python inference_end2end.py \
  --ckpt best_vae_global.pt \
  --f_1 5.0 \
  --f_2 5.35 \
  --chi_11 0.040 \
  --required-transmon 2 \
  --population 64 \
  --elite 8 \
  --iters 20 \
  --print-best \
  --plot-best
```

To additionally export the best candidate through the graph-to-layout pipeline:

```bash
python inference_end2end.py \
  --ckpt best_vae_global.pt \
  --f_1 5.0 \
  --required-transmon 1 \
  --quantum-metal
```

### Prior sampling

```bash
python thesis_analysis/sampling.py \
  --checkpoint best_vae_global.pt \
  --n-samples 10000 \
  --out-dir analysis_results/prior_sampling
```

Add `--run-qultra` to include QuLTRA analyzability.

---

## Reproducing the thesis analyses

### Training / validation losses

```bash
python thesis_analysis/analysis_4_2_1_training_validation_loss.py \
  --checkpoint best_vae_global.pt \
  --output-dir analysis_results/4_2_training_losses
```

### Latent-space analysis

```bash
python thesis_analysis/analysis_4_2_3_latent_space.py \
  --checkpoint best_vae_global.pt
```

### Seen-circuit inference

```bash
python thesis_analysis/analysis_4_3_1_seen_circuits.py \
  --ckpt best_vae_global.pt
```

Fast diagnostic without QuLTRA:

```bash
python thesis_analysis/analysis_4_3_1_seen_circuits.py \
  --ckpt best_vae_global.pt \
  --max-per-family 100 \
  --no-qultra
```

### Held-out topology and stress tests

```bash
python thesis_analysis/analyze_unseen_threequbit_and_stress_4T_errorbars.py \
  --ckpt best_vae_global.pt \
  --n-unseen 20 \
  --n-stress-per-case 5
```

The final stress-test success definition is based on a maximum requested-frequency relative error below `10%` by default. Kerr terms can still participate in conditioning and in the CEM objective.

### Ablation study

```bash
python thesis_analysis/analysis_4_3_3_ablation_study_revised.py --plan core
```

Available plans are `core`, `constraints`, `latent_dim`, and `all`. Trainable ablation variants use a 100-epoch budget by default.

### Constraint-aware three-qubit diagnostic

```bash
python thesis_analysis/inference_threequbit_constraint.py \
  --checkpoint best_vae_global.pt \
  --out-dir outputs/threequbit_constraint \
  --n-pred 12 \
  --guidance-strength 1.5 \
  --stochastic
```

### Graph-to-layout test

Planner-only test, which does not require Qiskit Metal:

```bash
python thesis_analysis/run_graph2metal_test.py \
  --case four_bus \
  --plan-only \
  -o runs/graph2metal_test_four_bus
```

Available representative cases are:

```text
single
qbus
four_bus
four_full
```

Without `--plan-only`, the script also attempts native Qiskit Metal rendering.

---

## Extending cQED-Gen

The repository is designed so that the circuit vocabulary and training data can be extended without rewriting the complete VAE pipeline. There are three main extension points: **new datasets/topologies**, **new primitive circuit elements**, and **new macro-nodes in the compressed subgraph basis**. Dataset discovery and most dimensional bookkeeping are derived from the repository definitions, so changes should be made at the representation level and then propagated through the existing automatic loaders.

### Adding a new dataset / circuit topology

The easiest route is to copy:

```text
src/data_loader/datasets/_template_new_dataset.py
```

into a new file under `src/data_loader/datasets/` and subclass `DatasetBase`.

A dataset definition provides:

```text
NAME
DATA_PATH
OBS_SLOTS_ACTIVE
build_topology()
parse_row()
parse_obs()
```

Optional fields include:

```text
N_SAMPLES
INCLUDE_TRAIN
```

`BLOCK_PARAMS` should **not** be declared manually. The parameter ordering is derived automatically from `build_topology()`, `graphlize()`, and `SUBG_DEFS`.

`schema.py` automatically discovers new `DatasetBase` subclasses on import.

### Adding a new observable

Append the new slot to the **end** of `OBS_SLOTS` in:

```text
src/data_loader/schema.py
```

Do not reorder existing slots because their indices are part of the checkpoint interface.

### Adding a new primitive circuit element

The source of truth for circuit elements and subgraphs is:

```text
src/circuit2graph/definitions.py
```

If a genuinely new primitive element or physical attribute is introduced:

1. add the physical attribute to `ATTR_INDEX` when necessary;
2. add/update the appropriate `SubgType` / `SubgDef` entry;
3. define the primitive components and exported attributes;
4. update connectivity semantics in `src/circuit2graph/constraints.py` when the new element introduces new physical connection rules.

Derived quantities such as the primitive vocabulary and inner-type count are generated from `SUBG_DEFS` and should not normally be edited manually.

### Adding a new macro-node / subgraph basis element

To introduce a new compressed motif:

1. add a new `SubgType` in `src/circuit2graph/definitions.py`;
2. define its `SubgDef`, including exported attributes, primitive inner nodes, inner edges, and orientation information when applicable;
3. add the corresponding recognition/merge rule in `src/circuit2graph/compression.py`;
4. ensure `src/circuit2graph/expansion.py` reconstructs the expected primitive structure;
5. add or update tests in `test/` for compression, expansion round-trips, and symmetries.

Because the decoder constraints derive compatibility from the same circuit definitions, extending the subgraph basis in this way keeps the representation, generation, and validation layers aligned.

---

## Next steps

The natural continuation of cQED-Gen is to extend both the scale of the training distribution and the physical richness of the design space. This includes larger multi-qubit circuits, more diverse global connectivity patterns, additional primitive components and a richer subgraph basis, together with higher-capacity or more hierarchical generative architectures as circuit complexity grows. Future work should also progressively connect network-level validation to more accurate physical objectives, combining QuLTRA with layout-aware and electromagnetic optimization and extending latent-space refinement to multi-objective design.

On the data-engineering side, a useful next step is to move beyond the current flat `.txt` datasets toward a structured data layer based on **SQL and Pandas**. Storing circuit parameters, topology metadata, Hamiltonian observables, simulation provenance, and train/validation/test information in queryable tables would make larger dataset collections easier to filter, merge, version, validate, and analyse, while preserving Pandas-based interfaces for model preprocessing and exploratory analysis.

---

## Validation of this snapshot

The final checked snapshot has been validated with:

```bash
python -m compileall -q -f .
PYTHONPATH=./src:. pytest -q test
```

The current test suite contains **69 tests**, all passing in the checked snapshot.

The graph2metal planner was also smoke-tested in `--plan-only` mode for all four bundled examples (`single`, `qbus`, `four_bus`, and `four_full`), with valid plans and no reported layout violations.

Full QuLTRA and native Qiskit Metal execution additionally require those external environments and are therefore separate runtime dependencies.

---

## Scope of the physical-layout extension

cQED-Gen operates primarily at the **circuit-network level**. QuLTRA provides an independent forward evaluation of the generated circuit, while `graph2metal` provides an initial geometric realization.

The resulting design should therefore be interpreted as a **layout-zero**: a physically meaningful starting geometry for subsequent electromagnetic/device optimization, not a fabrication-ready final layout.

# 5G UPF Energy Profiling — Digital Twin Foundation

Reproducible ML pipeline for profiling, analysing, and predicting the energy
consumption of 5G User Plane Function (UPF) deployments.
Measurements come from **Scaphandre** (process-level power monitoring) and
**Keysight LoadCore** (traffic generation / emulation).

The trained models are designed as the **predictive core of a data-driven
digital twin**: given an offered traffic load, the twin outputs power
consumption, CPU utilisation, delivered throughput, packet loss, and
end-to-end delay — without requiring a live UPF.

---

## UPF Variants Under Test

| Variant | Short name | I/O model | Implemented by |
|---|---|---|---|
| SD-Core DPDK | **DPDK** | Poll-mode (DPDK driver) | Open Networking Foundation SD-Core |
| OpenAirInterface UPF | **USR** | Interrupt-driven (Linux sockets / eBPF) | OpenAirInterface (user-space application) |

Key findings from the profiling campaign (see `reports/chapter_upf_profiling.pdf`):

- **DPDK**: near-constant power (0.82 ± 0.005 W) across 0–5 Gbps; zero packet
  loss; deterministically low latency. The polling loop consumes a fixed CPU
  share regardless of load.
- **USR**: power scales 0–4.5 W proportionally with load; saturates at
  ~0.17 Gbps (loss onset) and ~0.57 Gbps (near-total loss); safe operating
  boundary ≈ 0.10 Gbps.
- **SEC paradox**: the `SEC_net` metric misleadingly reports ~0 for DPDK
  because idle power ≈ load power. `SEC_total` is the honest cross-variant
  metric.

---

## Two-Layer Digital Twin Architecture

The predictive stack is organised in two layers that reflect the causal
structure of the system.

```
 ┌─────────────────────────────────────────────────────────┐
 │              INPUTS  (offered load — what you control)  │
 │   DL throughput,  UL throughput,  packet size,          │
 │   L2/L3 overhead ratio,  packet counts (TX side)        │
 └──────────────────────────┬──────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼                ▼
  [L1: Throughput]   [L1: CPU]        [L1: Loss]        [L1: Delay]
   (delivered Gbps)  (cpu_pct %)   (pkts lost/interval)  (DL µs)
          │                 │                  │                │
          └─────────────────┴──────────────────┴────────────────┘
                                    │
                                    ▼
                         [L2: Power model]
                          → power_watts (W)
```

### Why two layers?

- **Layer 1** models predict UPF *responses* from *offered load*.
  These are independently useful (e.g. to predict QoS degradation).
- **Layer 2** predicts power from offered load **plus** L1 outputs.
  Using L1 outputs as L2 features captures the true causal pathway
  (load → CPU → power) without circular dependency at inference time.
- At training time, Layer 2 trains on **out-of-fold predictions** from
  Layer 1 (stacking), so it never sees perfect ground-truth L1 values.
  This prevents distribution shift at inference time.

### Feature boundary

| Feature group | Layer 1 input? | Layer 2 input? | Notes |
|---|---|---|---|
| TX-side throughput (DL + UL) | ✅ | ✅ | Offered load — controlled |
| TX packet counts | ✅ | ✅ | Offered load — controlled |
| Avg packet size, L2/L3 ratio | ✅ | ✅ | Deployment config constants |
| Delivered throughput (`throughput_gbps`) | ❌ target | ✅ (L1 prediction) | UPF response |
| CPU utilisation (`cpu_pct`) | ❌ target | ✅ (L1 prediction) | UPF response |
| Packet loss (`packets_lost_delta`) | ❌ target | ✅ (L1 prediction) | UPF response |
| DL delay (`weighted_mean_delay_us`) | ❌ target | ✅ (L1 prediction) | UPF response |
| Rolling power features | ❌ leakage | ❌ leakage | Derived from power target |
| `cpu_per_watt`, `sec_*`, `net_power_watts` | ❌ leakage | ❌ leakage | Derived from power target |

---

## NetMob 2023 Compatibility

The digital twin can be driven by real-world traffic traces from the
**NetMob 2023** dataset (https://arxiv.org/abs/2305.06933), which provides
normalised UL/DL traffic loads per cell but no packet-level detail.

### Input adapter

A thin adapter layer translates NetMob's normalised loads into the model's
input space:

```
netmob_dl_norm  ×  C_max_dl  →  throughput_dl_gbps   (absolute load)
netmob_ul_norm  ×  C_max_ul  →  throughput_ul_gbps

Missing features (packet size, overhead ratio):
  → imputed with deployment constants from params.yaml
  → packet counts derived as:
      throughput × 1e9/8 / avg_packet_size_bytes × interval_sec
```

`C_max` (rated UPF link capacity) is a deployment parameter in `params.yaml`
(default: 10 Gbps; override per deployment scenario).

### Full vs Lite models

Two model variants are trained side by side:

| Model | Input features | Use case |
|---|---|---|
| **Full** (primary) | All offered-load features (TX throughput, packet counts, size, overhead) | Lab experiments; high-fidelity simulation |
| **Lite** (companion) | UL + DL throughput only | NetMob-driven simulation; any scenario where only load volume is known |

The lite model quantifies how much accuracy is lost when packet-level
features are unavailable, and serves as a lower bound / sanity check.

---

## Training Strategy

### Training variants

Three model variants are trained to capture different operating regimes:

| Variant | Data | Purpose |
|---|---|---|
| `dpdk` | All 4 388 DPDK samples | Evidence + baseline; power nearly constant (52 mW range) |
| `usr_full` | All 4 393 USR samples | Full range including saturated regime (>0.5 Gbps) |
| `usr_safe` | USR samples with throughput < 0.5 Gbps | Clean unsaturated regime; practical operating range |

### Model types (compared per slot)

| Model | Hyperparameter search |
|---|---|
| Ridge regression | Grid search: `alpha` ∈ {0.01, 0.1, 1, 10, 100, 1000} |
| Random Forest | RandomizedSearch 20 iter: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| Gradient Boosted Trees | RandomizedSearch 20 iter: `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

All tuning uses 5-fold cross-validation on the training split.
Best model per slot (by test R²) is saved to `models/`.

### MLflow experiment structure

One experiment: `upf-energy-profiling`.
Each run tagged with: `variant`, `layer`, `target`, `model_type`, `model_variant` (full/lite), `best`.

Total runs: 3 variants × (4 L1 targets + 1 L2 target) × 3 model types × 2 (full/lite) = **90 MLflow runs**.

---

## Pipeline Architecture

```
raw data ──→ ingest ──→ merge ──→ featurize ──→ train ──→ evaluate
  (DVC)       (DVC)     (DVC)      (DVC)      (MLflow)   (MLflow)
```

| Stage | Script | Key output |
|---|---|---|
| `ingest` | `src/ingest.py` | `data/interim/loadcore/`, `data/interim/scaphandre/` |
| `merge` | `src/merge.py` | `data/interim/merged.csv` — LoadCore + Scaphandre aligned by timestamp |
| `featurize` | `src/features.py` | `data/processed/features.csv` — engineered feature matrix |
| `train` | `src/train.py` | `models/` — 30 serialised models + `manifest.json`; `reports/metrics.json` |
| `evaluate` | `src/evaluate.py` | `reports/figures/feature_importance.png`; extended metrics |

---

## Model Outputs

```
models/
  layer1/
    {variant}__{target}.pkl          # best full L1 model per (variant, target)
    {variant}__{target}__lite.pkl    # lite version (throughput-only inputs)
  layer2/
    {variant}__power_watts.pkl       # best full L2 power model
    {variant}__power_watts__lite.pkl # lite version
  manifest.json                      # maps (variant, layer, target) → file path,
                                     #   best model type, test R², MAE, RMSE
reports/
  metrics.json          # DVC-tracked: L2 power model metrics per variant
  metrics_full.json     # all 90 runs summary
  chapter_upf_profiling.pdf    # thesis chapter (compile from .tex)
  figures/              # publication-quality figures (fig1–fig7, PDF + PNG)
```

---

## Quick Start

```bash
git clone https://github.com/youruser/upf-energy-profiling.git
cd upf-energy-profiling
make setup
source .venv/bin/activate      # Windows: .venv\Scripts\activate
dvc pull                       # download data from remote
dvc repro                      # run full pipeline
mlflow ui --port 5000          # view experiment runs
```

### Run training only

```bash
dvc repro train
```

### Drive the digital twin from NetMob data

```python
from src.twin import DigitalTwin

twin = DigitalTwin.load("models/manifest.json", variant="usr_safe", mode="lite")

# netmob_dl/ul are normalised loads (0–1); c_max in Gbps
result = twin.predict(netmob_dl=0.42, netmob_ul=0.18, c_max_dl=10.0, c_max_ul=10.0)
# → {"power_watts": 1.23, "cpu_pct": 2.1, "throughput_gbps": 0.41,
#    "loss_delta": 0.0, "delay_us": 320.4}
```

---

## Project Structure

```
├── params.yaml              # Single source of truth for all parameters
├── dvc.yaml                 # Pipeline DAG definition
├── data/
│   ├── raw/                 # Untouched source files (DVC-tracked)
│   │   ├── loadcore/        # Keysight LoadCore zip files (220 runs)
│   │   └── scaphandre/      # Scaphandre CSVs (dpdk/ and usr/)
│   ├── interim/             # Extracted, merged, time-aligned
│   └── processed/           # Final feature matrix (features.csv)
├── src/
│   ├── ingest.py            # Extract CSVs from zips via manifest
│   ├── merge.py             # Align LoadCore + Scaphandre by timestamp
│   ├── features.py          # Feature engineering (SEC, rolling, deltas)
│   ├── train.py             # Two-layer training with MLflow + CV tuning
│   ├── evaluate.py          # Evaluation, feature importance, residual plots
│   └── twin.py              # DigitalTwin inference class + NetMob adapter
├── configs/
│   └── file_manifest.json   # Which CSVs to extract from which zips
├── notebooks/
│   └── eda.ipynb            # Exploratory analysis: variant profiles,
│                            #   saturation analysis, SEC paradox
├── models/                  # Serialised models (DVC-tracked)
├── reports/
│   ├── figures/             # Publication figures (fig1–fig7, PDF + PNG)
│   ├── chapter_upf_profiling.tex   # Thesis chapter LaTeX source
│   └── metrics.json         # DVC pipeline metrics
├── scripts/
│   └── generate_thesis_figures.py  # Standalone figure generation
└── tests/                   # Unit tests
```

---

## Key Parameters (`params.yaml`)

```yaml
train:
  target: power_watts           # L2 target; L1 targets defined in script
  test_split: 0.2
  random_state: 42
  usr_safe_threshold_gbps: 0.5  # USR safe operating region boundary
  models: [ridge, random_forest, gradient_boosting]
  netmob_compat_features:       # Lite model input set
    - gtpu_kbitss_dn__kbits_tx_s
    - gtpu_kbitss_ngran__gtpu_kbits_tx_s
  adapter:
    c_max_dl_gbps: 10.0         # Rated DL capacity — override per deployment
    c_max_ul_gbps: 10.0         # Rated UL capacity — override per deployment
    avg_packet_size_bytes: 1250 # Default imputation for lite/NetMob mode
```

---

## Tech Stack

| Concern | Tool |
|---|---|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Pipeline orchestration | DVC pipelines |
| Modelling | scikit-learn (Ridge, RF, GBM) |
| CI/CD | GitHub Actions |
| Reporting | LaTeX (`reports/chapter_upf_profiling.tex`) |

---

## Thesis Chapter

A full academic write-up is available at `reports/chapter_upf_profiling.tex`
(compile with `pdflatex` twice from the `reports/` directory).
Covers: measurement methodology, variant profiles, USR saturation analysis,
SEC paradox, feature correlations, and implications for energy modelling.

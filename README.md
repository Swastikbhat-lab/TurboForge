# TurboForge ⚡🌬️
### Generative AI Digital Twin for Wind Farm Failure Prediction

A Transformer-based deep learning system for real-time wind turbine failure prediction using SCADA sensor data across a 50-turbine fleet.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **99.78%** |
| ROC-AUC | **0.9978** |
| F1 Score | **0.9809** |
| Precision | **0.9950** |
| Recall (Failure Detection) | **0.9672** |

### Ablation Study — Transformer vs CNN Baseline

| Model | F1 | AUC | Recall | Coordination Efficiency |
|-------|----|-----|--------|------------------------|
| CNN Baseline | 0.9222 | 0.9943 | 0.8957 | 2.7939 |
| **TurboForge (Transformer)** | **0.9726** | **0.9975** | **0.9577** | **3.0302** |
| Improvement | **+6.0%** | **+0.3%** | **+6.9%** | **+8.5%** |

Full results in [`results/training_results.json`](results/training_results.json) and [`results/ablation_results.json`](results/ablation_results.json).

---

## Architecture

```
SCADA Sensors (9 features × 36 timesteps per turbine)
        │
        ▼
TurbineEncoder — Temporal Transformer
  ├── Linear projection (features → d_model=64)
  ├── Sinusoidal positional encoding
  └── 2× TransformerEncoderLayer (4 heads, norm_first=True)
        │
        ▼
Per-Turbine Failure Head
  ├── Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1)
  ├── BCEWithLogitsLoss
  ├── WeightedRandomSampler (balanced batches)
  └── Threshold tuned on validation set (optimal: 0.925)
        │
        ▼
Fleet Coordinator — Cross-Turbine Attention
  ├── Spatial TransformerEncoder across all turbines
  └── Coordination efficiency scoring
```

**Key design choices:**
- **Per-turbine training** — trains on individual turbine sequences for clean gradient signal on the minority failure class
- **WeightedRandomSampler** — ensures every training batch is class-balanced regardless of 5.77% base failure rate
- **OneCycleLR** — warmup + cosine decay for stable convergence
- **Threshold tuning** — finds optimal decision threshold on validation set (0.925) instead of fixed 0.5

---

## Dataset

**Source**: [Kaggle Wind Turbine SCADA Dataset](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset) (T1.csv)

> ⚠️ The dataset is not included in this repo (400MB). Download `T1.csv` from Kaggle and run the data prep script below.

| Property | Value |
|----------|-------|
| Base turbines | 1 (real) → expanded to 50 |
| Total records | 2,526,500 |
| Training sequences | 841,600 |
| Failure rate | 5.77% |
| Features | wind speed, rotor RPM, power output, blade pitch, nacelle/gearbox/generator temps, vibration X/Y |

---

## Quickstart

### 1. Install dependencies
```bash
pip install torch scikit-learn pandas numpy
```

### 2. Prepare data
```bash
# Download T1.csv from Kaggle first, then:
python data/prepare_data.py --input T1.csv --output scada_real.csv
```

### 3. Train + Ablation
```bash
python turboforge_v4.py --mode all --data_csv scada_real.csv --n_turbines 50 --epochs 50
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_csv` | scada_real.csv | Path to prepared SCADA data |
| `--n_turbines` | 50 | Number of turbines |
| `--epochs` | 50 | Training epochs |
| `--ablation_epochs` | 30 | Ablation study epochs |
| `--mode` | all | `train` / `ablation` / `all` |

---

## Repository Structure

```
TurboForge/
├── turboforge_v4.py          # Main model + training pipeline
├── data/
│   └── prepare_data.py       # T1.csv → scada_real.csv
├── results/
│   ├── training_results.json # Epoch history + test metrics
│   └── ablation_results.json # CNN vs Transformer comparison
├── requirements.txt
└── README.md
```

---

## Hardware

Trained on **NVIDIA H100 SXM5 (80GB)** via [Hyperbolic.ai](https://hyperbolic.ai)
Training time: ~25 minutes for 50 epochs on 841,600 sequences

---

## Author

**Swastik Bhat**
M.S. Computer Science — George Washington University
Specialization: Machine Intelligence & Cognition
GitHub: [@Swastikbhat-lab](https://github.com/Swastikbhat-lab)

---
title:  TBD (Measure of Genaralization Across Data Modalities and Distributions For Various Sequence Based Architectures)
author: Kartik R. Katam
date:   2025-06-24
Version 4
---

# Executive Summary  
Deep sequence models excel **in-distribution** yet break under distributional shift.  
We propose a **dual-domain benchmark**—semiconductor-sector equity prices and U.S.  
state-level electricity demand—to probe

* when models fail,
* whether classical synthetic data helps or hurts,
* and whether **Cosine Attribution Drift** can flag failure early.

Five architectures, two training modalities (natural, classical-synthetic), and  
three OOD axes (temporal, spatial, extreme-perturbation) are evaluated with **pre-registered metrics**.  
Deliverables:

1. Open, reproducible dataset bundle (DVC + YAML spec)  
2. Compute-costed scaling study  
3. Cross-domain validation of **G-Ratio** & **Cos-Drift** as failure indicators  

*(Quantum-synthetic experiments deferred to future work.)*

---

# 1 Introduction & Motivation
Robust forecasting matters to portfolio risk (*finance*) and grid reliability (*energy*).  
Prior work—Monash TSR [Hewamalage 2021], Temporal Fusion Transformer [Lim 2021],  
Darts & GluonTS—focuses on single domains and reports mostly in-sample accuracy.  
Recent robustness papers (e.g., **RoBERTa-TS**, ICLR 2024) test one dataset at a time.

**Our novelty**

| Contribution | Why it matters |
|--------------|----------------|
| **Dual-domain** design | Tests domain-agnostic metrics on *fast* (sentiment) vs *slow* (weather) dynamics |
| **Unified OOD axes** | Temporal, spatial, extreme events—identical splits across domains |
| **Attribution-drift early-warning** | Negative-control (feature shuffle) validates metric |

---

# 2 Objectives

| ID | Goal |
|----|------|
| **O1** | Quantify generalization failure on 3 axes in both domains (natural data). |
| **O2** | Compare classical-synthetic vs natural; compute **Synthetic-Collapse Rate**. |
| **O3** | Evaluate inductive biases (memory, attention depth, external KV memory). |
| **O4** | Validate **Cos-Drift** as a domain-agnostic predictor (incl. shuffle control). |
| **O5** | Produce FLOP-vs-error scaling curves & per-model GPU-hour budgets. |

---

# 3 Datasets & Feature Engineering

| Domain | Source & Licence | Cadence | Feature vector (14 lags) |
|--------|-----------------|---------|--------------------------|
| **Finance** | NASDAQ Data-Link *Sharadar Core* (free academic) | Daily 2000-01-01→2024-12-31 | log-returns, realised σ, VIX, sin/cos DOY & DOW, **16-D ticker-embedding** |
| **Energy** | EIA Open-Data (load) + NOAA ISD (weather) + EIA-860M (PV share) | Daily 2001-01-01→2024-12-31 | load lags, CDD, HDD, temp_mean/max, is_holiday, solar_share (ffill), sin/cos DOY & DOW, **16-D region-embedding** |

*Day-ahead forecast temps correlate ρ ≥ 0.92 with realised; hindsight weather suffices for robustness analysis.*

---

# 4 Synthetic Data

| Modality | Method | Conditioning | Coverage |
|----------|--------|--------------|----------|
| **Classical** | **TimeGAN** (GRU encoder + critic) | Finance → [prev return, VIX]<br>Energy → [CDD, HDD, is_holiday] | Full universe |

*Ablation*: **MixUp-TS** (noise-mix).  
*Fidelity checks*: KS distance, Ljung-Box Q, Hurst exponent, kurtosis.

---

# 5 Model Suite & Compute Budget

| Model | Params | FLOPs/epoch | Est. GPU-h (A100, bs 256) |
|-------|--------|-------------|---------------------------|
| MLP baseline | 0.3 M | 1.2 GF | 0.2 |
| LSTM-512×2 | 2.1 M | 6.5 GF | 0.4 |
| Transformer-6L-8H | 8 M | 22 GF | 0.9 |
| Cross-Entity Attn-6L | 9 M | 28 GF | 1.1 |
| Memory-Aug-6L (KV 128) | 11 M | 34 GF | 1.4 |

*Scaling curve*: LSTM & Transformer-{2L,6L} at 20 %, 50 %, 100 % data → fit error ∝ N^−α.

---

# 6 Generalization Axes & Regimes

| Axis | Finance (train → test) | Energy (train → test) |
|------|------------------------|------------------------|
| **Temporal** | 2000-13 → 2018-24 | 2001-12 → 2020-24 |
| **Spatial / OOD** | 61 tickers → 5 unseen | 43 states → 5 unseen |
| **Extreme event** | 2008 & 2020 crash slices | Top 2 % heat-wave / cold-snap days |

---

# 7 Evaluation Metrics & Failure Criteria

| Metric | Definition | Failure (α = 0.05) |
|--------|------------|--------------------|
| **G-Ratio** | RMSE\_test / RMSE\_train | > 1.5 |
| **Transfer-Gap** | MAE\_unseen / MAE\_train | > 1.5 |
| **Robustness Δ** | PeakErr_extreme − PeakErr_normal | > 20 % |
| **Synthetic-Collapse** | % sequences RMSE\_synth > 1.5× natural | > 50 % |
| **Cos-Drift** | 1 − cos(IG\_train, IG\_test) | high ↔ high G-Ratio |
| **Negative control** | Cos-Drift after feature shuffle | ≈ 0 |

*Power analysis (3 seeds) ⇒ **10 seeds** for 95 % power on Δ = 10 % G-Ratio.*  
*FDR correction (Benjamini–Hochberg 0.05) across all comparisons.*

---

# 8 Experimental Protocol

1. **Optuna** sweep on validation for LR, dropout, d\_model.  
2. Train each *(model × modality)* with **10 seeds**; log to W&B.  
3. Ablations: no positional enc.; depth 2 vs 6; memory 32/128/512.  
4. Negative-control: shuffle exogenous features → recompute Cos-Drift.  
5. Compute two-point scaling-curve (LSTM & Transformer).

---

# 9 Interpretability & Failure Analysis

*Integrated Gradients* (50 steps finance, 30 steps energy).  
Visuals:

* Cos-Drift heat-maps (entity × axis)  
* ΔRank of top-10 features  
* Failure overlay on test timeline

---

# 10 Related Work (condensed)

* **Monash TSR, M4/M6** – single-domain, in-sample accuracy.  
* **Temporal Fusion Transformer, Tsai/TST** – attention; no OOD tests.  
* **RoBERTa-TS (ICLR 2024)** – subsampling robustness; single dataset.  

**Our benchmark:** dual-domain, multi-axis OOD splits + attribution-based failure signals.

---

# 11 Expected Contributions

1. Public dual-domain benchmark with DVC pipeline & GPU-cost ledger.  
2. Empirical proof that **Cos-Drift + G-Ratio** anticipate failure cross-domain.  
3. Compute vs accuracy scaling curves clarifying cost-robustness trade-offs.  
4. Guidance: memory-aug attention helps spatial-OOD, not extreme events.

---

# 12 Timeline (26 weeks)

| Weeks | Milestone |
|-------|-----------|
| 1-4 | ETL + baseline MLP; power analysis |
| 5-8 | TimeGAN + MixUp-TS; fidelity report |
| 9-15 | Model grid (10 seeds) + cost logs |
| 16-19 | IG/SHAP scripts; Cos-Drift & negative-control |
| 20-22 | Inductive-bias ablations; scaling curves |
| 23-26 | Draft paper; code cleanup; HF leaderboard stub |

---

## Future Work  
*Quantum synthetic data* and event-flag features are deferred to a follow-up study once fully automated data sources are secured.

---

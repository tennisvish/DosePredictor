

## Overview

Conventional biodosimetry requires manual foci scoring, which is low-throughput and observer-dependent. This model automates dose estimation directly from raw fluorescence images, bypassing explicit foci counting. It is trained on 4-hour post-exposure images exposed to both X-ray and Iron across five dose levels (0, 0.10, 0.30, 0.82, 1.0 Gy).

**Key architectural contributions:**

- **Convolutional patch embedding stem** — replaces the standard single linear projection with a 3-layer conv stem, giving the model local spatial inductive bias suited to foci detection
- **Depthwise conv MLP** — augments each transformer block's feedforward network with a spatial 3×3 depthwise convolution, mixing neighbouring patch tokens within the MLP pass
- **Cell-cycle conditioning** — nuclear area (R² = 0.876 as an auxiliary prediction) is used as a proxy for cell cycle phase (G1/S/G2M). A differentiable soft quantisation layer converts the model's own nuclear area prediction into a learned 32-dim phase embedding that conditions the dose head, encoding the known ~2–3× variation in γH2AX response across cell cycle phases
- **Multi-task auxiliary learning** — joint prediction of foci count and nuclear area regularises the shared encoder and provides interpretable auxiliary outputs

---

## Results

| Method | MAE (Gy) | R² |
|---|---|---|
| Standard inference | 0.1880 | 0.5556 |
| Test-time augmentation (8×) | 0.1865 | 0.5602 |

Auxiliary task performance: foci count R² = 0.903, nuclear area R² = 0.876.

---

## Architecture
```
Input (224 × 224 × 1, single-channel fluorescence)
    │
    ▼
ConvPatchEncoder
  Conv(3×3, dim//4) → BN → GELU        # local edges / texture
  Conv(3×3, dim//2) → BN → GELU        # mid-level features
  Conv(P×P, dim, stride=P) → BN → GELU # tokenisation (P=14)
  + CLS token + positional embedding
  → 257 tokens × 256 dim
    │
    ▼
6 × TransformerBlock
  LayerNorm → MultiHeadAttention (8 heads) → residual
  LayerNorm → Dense(512) → DepthwiseConv(3×3) → Dense(256) → residual
    │
    ▼
LayerNorm → CLS token (256 dim)
    │
    ▼
Shared head: Dense(512) → Dense(256)
    ├── Nuclear area head → Dense(128) → scalar
    │       │
    │       ▼
    │   NucAreaConditioner (soft G1/S/G2M quantisation → 32-dim embedding)
    │       │
    ├── Dose head: concat(shared[256], CC[32]) → Dense(128) → scalar  ← primary
    ├── Foci count head → Dense(128) → scalar                          ← auxiliary
    └── Spot mean head  → Dense(128) → scalar                          ← disabled (weight=0)
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Patch size | 14 |
| Embedding dim | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Dropout | 0.15 |
| Optimizer | AdamW (weight_decay=0.02, clipnorm=1.0) |
| Learning rate | 1e-4 with 5-epoch linear warmup + cosine decay restarts |
| Batch size | 32 |
| Augmentation | Random flips, 90° rotations, ±20% brightness + Mixup λ∈[0.3,0.7] |
| Loss (dose) | MSE |
| Loss (aux) | MAE |
| Aux loss weights | foci=0.5, nuc_area=0.3, spot_mean=0.0 |

---

## Data

The model expects single-channel `.tif` fluorescence microscopy images of individual cell nuclei, with a corresponding CSV containing:

| Column | Description |
|---|---|
| `filename` | Image filename (`.tif`) |
| `dose_Gy` | Radiation dose label (Gy) |
| `nfoci` | γH2AX foci count per nucleus |
| `nuc_area` | Nuclear area (µm²) |
| `spot_mean` | Mean foci intensity (optional) |
| `hr_post_exposure` | Hours post-irradiation (model filters for `== 4`) |

Update `CSV_PATH` and `IMAGES_DIR` in the script to point to your data. The dataset is not included in this repository.

**Train/val/test split:** 70% / 20% / 10%, stratified by dose level (`random_state=42`).

---

## Requirements
```
tensorflow >= 2.12
scikit-learn
opencv-python
pandas
numpy
matplotlib
scipy
```

Install with:
```bash
pip install tensorflow scikit-learn opencv-python pandas numpy matplotlib scipy
```

---

## Usage
```bash
python vit_convpatch_cc.py
```

Outputs saved to the working directory:

| File | Contents |
|---|---|
| `vit_convpatch_cc.keras` | Trained model weights |
| `results.csv` | Per-cell predictions (standard + TTA) and auxiliary outputs |
| `results.png` | Scatter plot, prediction distributions, training loss curves |

---

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{vit_gammaH2AX_dosimetry,
  title   = {Radiation Dose Estimation via Vision Transformer},
  author  = {[Author]},
  year    = {2025},
  url     = {[repository URL]}
}
```

---

## License

[Add license here]

# VascuLens - A Multi-Backbone Ensemble Pipeline for Classification of Vascular Ulcer Images

Binary classification of clinical ulcer images using an ensemble of three
pre-trained backbones: **BiomedCLIP**, **UNI**, and **EfficientNet-B0**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Phase                           │
│   (Sequential — only ONE backbone in VRAM at a time)        │
│                                                             │
│  EfficientNet-B0  ─── train/fold ─── cache logits/embeds   │
│  BiomedCLIP ViT-B ─── train/fold ─── cache logits/embeds   │
│  UNI ViT-L        ─── train/fold ─── cache logits/embeds   │
└─────────────────────────────────────────────────────────────┘
                          │ cached .pt files (no backbone in VRAM)
┌─────────────────────────▼───────────────────────────────────┐
│                    Fusion Phase                             │
│  Late Fusion:    weighted avg of cached logits  (<0.1 GB)   │
│  Feature Fusion: MLP on concat embeddings       (<0.5 GB)   │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Explainability                             │
│  EfficientNet → GradCAM++ (last conv block)                 │
│  BiomedCLIP   → Attention Rollout (all ViT layers)          │
│  UNI          → Attention Rollout (all ViT layers)          │
│  (each backbone loaded and released independently)          │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Test Evaluation                            │
│  Load fold checkpoint → infer on held-out test set          │
│  Late fusion (equal weights) → metrics + CI                 │
│  Heatmaps: EfficientNet │ BiomedCLIP │ fused overlay        │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Windows
1. Clone the repository
2. Double-click `setup_env.bat` or run it from Command Prompt
3. Place training images in `data/VASCULAR/` and `data/NON-VASCULAR/`
4. Run: `venv\Scripts\activate.bat && python main.py --config config.yaml --mode all`

### Linux / macOS
1. Clone the repository
2. Run: `chmod +x setup_env.sh && ./setup_env.sh`
3. Place training images in `data/VASCULAR/` and `data/NON-VASCULAR/`
4. Run: `source venv/bin/activate && python main.py --config config.yaml --mode all`

### HuggingFace Login (required for UNI model)
After activating the venv:
```bash
python -c "from huggingface_hub import login; login()"
```
Accept the UNI model license at https://huggingface.co/MahmoodLab/uni

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA with 8 GB VRAM | RTX 3080 (10 GB) |
| CUDA | 11.7+ | 12.4 |
| Python | 3.10 | 3.11 |
| System RAM | 8 GB | 16 GB |
| Disk | 10 GB | 20 GB |

### VRAM Usage Per Phase

| Phase | Estimated VRAM |
|---|---|
| Train EfficientNet-B0 (FP16) | ~1.5 GB |
| Train BiomedCLIP (FP16) | ~3.5 GB |
| Train UNI (FP16 + grad ckpt) | ~5–6 GB |
| Late Fusion (cached logits) | < 0.1 GB |
| Feature Fusion (cached embeddings) | < 0.5 GB |
| GradCAM++ on EfficientNet | ~1 GB |
| Attention Rollout on BiomedCLIP | ~2.5 GB |
| Attention Rollout on UNI | ~4 GB |
| Test inference (one backbone at a time) | same as training phases |

---

## Dataset Preparation

### Training / validation data

```
data/
├── VASCULAR/
│   ├── patient_001.jpg
│   ├── patient_002.png
│   └── ...
└── NON-VASCULAR/
    ├── patient_050.jpg
    └── ...
```

Configured via `data.root_dir` in `config.yaml` (default: `data/`).

### Test set

The test set uses the **same folder structure** in a separate directory:

```
test_data/
├── VASCULAR/
│   └── ...
└── NON-VASCULAR/
    └── ...
```

Configured via `test.test_dir` in `config.yaml` (default: `test_data/`).
The test set is **never used during training or cross-validation**.

---

## Memory Management

The pipeline is designed specifically for the RTX 3080's 10 GB VRAM budget:

1. **Sequential backbone training**: Backbones are trained one at a time
   (EfficientNet → BiomedCLIP → UNI). Each backbone is deleted from VRAM
   before the next is loaded.

2. **Embedding caching**: After each backbone is trained on a fold, its
   embeddings and logits are extracted and saved to disk as `.pt` files.
   Fusion training reads these cached tensors — no backbone in VRAM.

3. **Mixed precision (FP16)**: All training and inference uses
   `torch.cuda.amp.autocast`. Halves activation memory on Tensor Cores.

4. **Gradient checkpointing (UNI)**: UNI's ViT-L activations exceed 4 GB
   in FP32 at batch_size=16. Gradient checkpointing is mandatory and
   enabled automatically.

5. **OOM recovery**: If a CUDA out-of-memory error occurs, the pipeline
   automatically halves the batch size and retries (configurable).

---

## Configuration

All parameters are in `config.yaml`. Key sections:

| Section | Purpose |
|---|---|
| `general` | Seed, device, logging level |
| `hardware` | VRAM budget, mixed precision, OOM recovery |
| `data` | Training image paths, class names, augmentation |
| `backbones` | Per-backbone learning rates, layer freezing |
| `training` | Epochs, optimizer, scheduler, early stopping |
| `fusion` | Late fusion weights, feature fusion MLP |
| `evaluation` | Metrics, bootstrap CI, plot saving |
| `explainability` | GradCAM++, Attention Rollout settings |
| `test` | Test set path, fold selection, heatmap options |

### Test section reference

```yaml
test:
  test_dir: "test_data/"      # Root with VASCULAR/ and NON-VASCULAR/ subfolders
  fold: 0                     # Checkpoint fold to load (0 .. n_splits-1)
  output_dir: "outputs/test"
  num_heatmap_samples: 20     # Max images for heatmaps (0 = skip heatmaps)
  heatmap_selection: "all"    # all | random | best | worst
  colormap: "jet"
  alpha: 0.5
  # Optional: explicit fusion weights (must sum to 1, one per enabled backbone)
  # fusion_weights: [0.4, 0.3, 0.3]
```

`heatmap_selection` values:

| Value | Behaviour |
|---|---|
| `all` | First N images in dataset order |
| `random` | N random images |
| `best` | N images with highest-confidence **correct** predictions |
| `worst` | N images with highest-confidence **wrong** predictions |

---

## Outputs

### Training + cross-validation (`outputs/`)

```
outputs/
├── checkpoints/
│   ├── fold_0/
│   │   ├── biomedclip_best.pt
│   │   ├── uni_best.pt
│   │   └── efficientnet_best.pt
│   └── fold_1/ ... fold_4/
├── embedding_cache/
│   ├── efficientnet_fold0_embeddings.pt
│   ├── efficientnet_fold0_logits.pt
│   ├── biomedclip_fold0_embeddings.pt
│   ├── fold0_labels.pt
│   └── ...
├── metrics/
│   ├── aggregated_results.json
│   ├── confusion_matrix_fold_0.png
│   ├── roc_curve_late_fusion.png
│   └── ablation_comparison.json
├── heatmaps/
│   └── sample_001/
│       ├── original.png
│       ├── efficientnet_gradcam.png
│       ├── biomedclip_attention.png
│       ├── uni_attention.png
│       └── combined.png
└── training.log
```

### Test evaluation (`outputs/test/`)

```
outputs/test/
├── metrics.json                        # All metrics + 95% CI per model
├── per_sample_results.csv              # Per-image predictions and confidence
├── confusion_matrix_late_fusion.png
├── confusion_matrix_efficientnet.png
├── confusion_matrix_biomedclip.png
├── roc_curve_test.png
└── heatmaps/
    └── <image_stem>/
        ├── original.png
        ├── efficientnet_gradcam.png        # GradCAM++ panel (orig│heatmap│overlay)
        ├── biomedclip_attention.png        # Attention Rollout panel
        └── late_fusion_aggregated.png      # Pixel-wise avg of the two heatmaps
```

---

## CLI Reference

```bash
# Full pipeline: train + evaluate + explain
python main.py --config config.yaml --mode all

# Training only (5-fold CV + fusion ablation)
python main.py --config config.yaml --mode train

# Evaluate from saved checkpoints (uses cached embeddings)
python main.py --config config.yaml --mode evaluate

# Generate explainability heatmaps (requires saved checkpoints)
python main.py --config config.yaml --mode explain

# Test on held-out test set with late fusion + heatmaps
python main.py --config config.yaml --mode test
```

### Test mode workflow

```
1. Edit config.yaml:
     test.test_dir  ← path to your test folder
     test.fold      ← which checkpoint fold to use (default: 0)

2. python main.py --config config.yaml --mode test

3. Results in outputs/test/:
     metrics.json            ← AUC, F1, accuracy, sensitivity, specificity + CI
     per_sample_results.csv  ← per-image breakdown
     confusion_matrix_*.png  ← one per backbone + late fusion
     roc_curve_test.png
     heatmaps/<image>/       ← EfficientNet, BiomedCLIP, fused
```

---

## Troubleshooting

### CUDA not found
```
RuntimeError: No CUDA GPUs are available
```
- Verify NVIDIA drivers: `nvidia-smi`
- Re-run `setup_env.bat`/`setup_env.sh` — PyTorch with CUDA is installed
  with `--index-url`. Installing from `requirements.txt` alone gives the
  CPU-only wheel.
- Set `device: "cpu"` in `config.yaml` to run without GPU (slow).

### CUDA Out of Memory
- Reduce `training.batch_size` in `config.yaml`.
- Enable `hardware.auto_reduce_batch_size: true` for automatic recovery.
- Ensure `hardware.mixed_precision: true` and `backbones.uni.gradient_checkpointing: true`.

### No images found in test_dir
```
RuntimeError: No images found in test_dir='test_data/'
```
- Check that `test.test_dir` in `config.yaml` points to the correct folder.
- The folder must contain `VASCULAR/` and `NON-VASCULAR/` subfolders
  matching `data.class_names` exactly (case-sensitive).

### Test mode — checkpoint not found
```
WARNING: Checkpoint not found: outputs/checkpoints/fold_0/efficientnet_best.pt
```
- Run training first (`--mode train`) before running `--mode test`.
- Or change `test.fold` to a fold that has been trained.

### PowerShell activation error (Windows)
If `activate.bat` fails in PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### python3-venv missing (Ubuntu/Debian)
```bash
sudo apt install python3-venv python3-dev
```

### HuggingFace authentication (UNI)
```
requests.exceptions.HTTPError: 401 Unauthorized
```
Run the following and accept the model license at
https://huggingface.co/MahmoodLab/uni:
```bash
python -c "from huggingface_hub import login; login()"
```

---

## Acknowledgements

- **BiomedCLIP**: Zhang et al., "A Large-Scale Expert-Level Biomedical
  Vision-Language Model", Microsoft Research.
  [HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

- **UNI**: Chen et al., "A General-Purpose Self-Supervised Model for
  Computational Pathology", MahmoodLab.
  [HuggingFace](https://huggingface.co/MahmoodLab/uni)

- **pytorch-grad-cam**: Jacobgilpython grad-cam library.
  [GitHub](https://github.com/jacobgil/pytorch-grad-cam)

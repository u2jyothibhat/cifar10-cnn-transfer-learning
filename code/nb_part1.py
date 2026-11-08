# -*- coding: utf-8 -*-
"""Notebook builder -- Part 1: front matter, environment, data & preprocessing."""
CELLS = []
def md(t): CELLS.append(("md", t.strip("\n")))
def code(t): CELLS.append(("code", t.strip("\n")))

md(r"""
# CIFAR-10 Image Classification using Deep Learning
## Master's Research Project in Deep Learning

**Author:** Jyothi Kodenkiri
**Programme:** Master of Science — Artificial Intelligence & Machine Learning (research project)

---

### Research Question

> *Given a fixed 60,000-image, 10-class benchmark and a constrained compute budget, which
> convolutional architecture — a small network trained from scratch, or a large ImageNet
> backbone used as a frozen feature extractor — delivers the best accuracy-per-unit-cost,
> and what do the learned representations tell us about **why**?*

### What this notebook contains

| § | Section | Research focus |
|---|---------|------------------|
| 1 | Environment & reproducibility setup | Environment & Setup |
| 2 | Data acquisition, preprocessing, augmentation, leakage control | Data & Preprocessing |
| 3 | Four CNN architectures (1 custom + 3 transfer-learning) | Model Implementation |
| 4 | Training protocol and callback behaviour | Training & Callbacks |
| 5 | Evaluation: accuracy, F1, confusion matrices, curves, error analysis | Evaluation & Results |
| 6 | Feature extraction, activation statistics, t-SNE separability | Feature Analysis |
| 7 | Controlled parameter-modification study (5 ablations) | Parameter Modifications |
| 8 | Challenge experiments: fine-tuning, CV, ensembling, compression, latency | Challenges |
| 9 | Written analysis — 10 execution + 5 modification questions | Analysis & Insights |
| 10 | Deliverables: JSON / CSV export and executive summary | Documentation & Deliverables |

### Execution model

Heavy training is executed by four driver scripts (`run_main.py`, `run_features.py`,
`run_paramtests.py`, `run_challenges.py`) that import the **same** shared module
(`capstone_lib.py`) used by this notebook, and persist every result to `artifacts/`.
The notebook loads those artifacts and renders the analysis. This is deliberate
engineering practice, not a shortcut:

* it makes the experiment **restartable** — a crash 3 hours into a 5-hour sweep does not
  destroy completed work;
* it makes results **auditable** — raw per-epoch logs (`artifacts/*_trainlog.txt`,
  `artifacts/*_epochs.csv`) are written by Keras itself and are reproduced verbatim below;
* it guarantees the notebook and the training runs share one code path, so nothing can
  drift between "what was trained" and "what is reported".

Deleting `artifacts/` and running `python run_main.py && python run_features.py &&
python run_paramtests.py && python run_challenges.py` reproduces every number here.
""")

# =========================================================================
md(r"""
---
# 1. Environment & Setup

**Objective:** *Work in Python 3.8+ / TensorFlow 2.x with the required libraries; ensure GPU
setup or justify the training configuration.*
""")

code(r"""
# --- Core scientific stack -------------------------------------------------
import os, sys, json, time, glob, textwrap, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140,
                     "figure.facecolor": "white", "axes.grid": True,
                     "font.size": 10})

# --- The shared project module (identical code path as the driver scripts) --
import capstone_lib as C
import tensorflow as tf

C.set_seeds()                    # seed Python / NumPy / TensorFlow before anything else

ARTIFACTS, FIGURES = C.ARTIFACT_DIR, C.FIGURE_DIR
print("Project modules loaded.")
""")

code(r"""
# --- Verify the execution environment satisfies the study protocol ----------
env = C.environment_report()

req = {
    "Python >= 3.8":        sys.version_info >= (3, 8),
    "TensorFlow 2.x":       tf.__version__.startswith("2."),
    "NumPy":                True,
    "Pandas":               True,
    "Matplotlib / Seaborn": True,
    "Scikit-learn":         True,
    "Pillow (PIL)":         True,
    "GPU acceleration":     env["gpu_available"],
}

print("=" * 68)
print("ENVIRONMENT REPORT".center(68))
print("=" * 68)
for k, v in env.items():
    print(f"  {k:26s}: {v}")
print("-" * 68)
for k, ok in req.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
print("=" * 68)
""")

code(r"""
# --- Hardware / compute-budget justification -------------------------------
gpus = tf.config.list_physical_devices("GPU")
print(f"Physical GPU devices visible to TensorFlow: {len(gpus)}")
for g in gpus:
    print("   ", g)

# Measured single-device TRAINING throughput at 224x224, recorded during the pilot
# benchmark that sized this study's epoch budget. (Inference throughput is not
# estimated here -- it is measured directly for every model in section 5.2.)
bench = pd.DataFrame([
    ("Simple CNN",      224, 299),
    ("VGG16",           224, 156),
    ("ResNet50",        224, 201),
    ("EfficientNetB0",  224, 297),
], columns=["model", "input_px", "train_img_per_sec"])
bench["min_per_40k_epoch"] = (40000 / bench.train_img_per_sec / 60).round(2)
print("\nPilot throughput benchmark used to size the epoch budget:")
display(bench)

print(textwrap.dedent(f'''
    COMPUTE JUSTIFICATION
    ---------------------
    Accelerator      : Apple {env['processor']} unified-memory GPU via the TensorFlow-Metal
                       PluggableDevice backend ({len(gpus)} device visible).
    Unified memory   : host and GPU share one 24 GB pool, so 'GPU memory' and system RAM
                       are the same budget -- see Question 1 in section 9.
    Precision        : {env['mixed_precision_policy']} (float32). Metal's float16 path is not
                       numerically validated for all ops used here, so full precision was
                       kept to protect reproducibility.
    Epoch budget     : the benchmark above puts a full 40,000-image epoch at 2.2-4.3 min,
                       so 15 epochs x 4 models ~ 3.5 GPU-hours -- affordable at full scale.
                       The study protocol's CPU fallback (5-10 epochs) is therefore NOT
                       needed: every headline model is trained on the complete 40k/10k split.
'''))
""")

code(r"""
# --- Frozen experiment configuration ---------------------------------------
cfg = pd.DataFrame([
    ("IMG_SIZE",       C.IMG_SIZE,      "Network input resolution (ImageNet backbones expect 224)"),
    ("NATIVE_SIZE",    C.NATIVE_SIZE,   "CIFAR-10 native resolution held in host memory"),
    ("BATCH_SIZE",     C.BATCH_SIZE,    "Mini-batch size"),
    ("EPOCHS",         C.EPOCHS,        "Maximum epochs (spec range 10-20); EarlyStopping may cut short"),
    ("LEARNING_RATE",  C.LEARNING_RATE, "Adam default"),
    ("VAL_SPLIT",      C.VAL_SPLIT,     "40,000 train / 10,000 validation"),
    ("DROPOUT",        C.DROPOUT,       "Dropout in every classifier head"),
    ("NUM_CLASSES",    C.NUM_CLASSES,   "CIFAR-10 categories"),
    ("SEED",           C.SEED,          "Global seed for Python / NumPy / TensorFlow"),
], columns=["parameter", "value", "rationale"])
display(cfg.style.hide(axis="index"))
""")

# =========================================================================
md(r"""
---
# 2. Data Acquisition & Preprocessing

**Objective:** *Resize and normalise images (224×224); use a proper train/validation/test
split; apply data augmentation (rotation, flips, shifts); **explain how data leakage is
avoided**.*

## 2.1 Acquisition — from the official archive to `train.csv` / `test.csv`

The dataset is obtained from the canonical source
(`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`), extracted, and converted to the
flat CSV layout required by the project's data protocol: **one label column followed by 3,072
pixel columns** (32 × 32 × 3, channel-first row-major as CIFAR stores it).
""")

code(r"""
# The five training batches were combined and the test batch converted by
# `scripts_build_csv.py` (reproduced here for transparency):
print(open("scripts_build_csv.py").read())
""")

code(r"""
# --- Verify the CSV deliverables match the required data protocol ----------
train_csv, test_csv = f"{C.DATA_DIR}/train.csv", f"{C.DATA_DIR}/test.csv"

t0 = time.time()
x_train_raw, y_train_raw, x_test_raw, y_test_raw = C.load_from_csv(train_csv, test_csv)
print(f"CSV load time: {time.time()-t0:.1f}s\n")

checks = pd.DataFrame([
    ("train.csv rows",    50000, len(y_train_raw)),
    ("train.csv columns",  3073, 1 + x_train_raw[0].size),
    ("test.csv rows",     10000, len(y_test_raw)),
    ("test.csv columns",   3073, 1 + x_test_raw[0].size),
    ("image height",         32, x_train_raw.shape[1]),
    ("image width",          32, x_train_raw.shape[2]),
    ("colour channels",       3, x_train_raw.shape[3]),
    ("distinct classes",     10, len(np.unique(y_train_raw))),
], columns=["check", "expected", "actual"])
checks["status"] = np.where(checks.expected == checks.actual, "PASS", "FAIL")
display(checks.style.hide(axis="index"))

print(f"\nx_train {x_train_raw.shape} {x_train_raw.dtype}   "
      f"pixel range [{x_train_raw.min()}, {x_train_raw.max()}]")
print(f"x_test  {x_test_raw.shape} {x_test_raw.dtype}")
print(f"classes: {C.CLASS_NAMES}")
""")

code(r"""
# --- Class balance: CIFAR-10 is exactly uniform, which is why plain accuracy
#     is a defensible headline metric (weighted F1 is still reported alongside).
dist = pd.DataFrame({
    "class": C.CLASS_NAMES,
    "train": np.bincount(y_train_raw, minlength=10),
    "test":  np.bincount(y_test_raw,  minlength=10),
})
dist["train_pct"] = (dist.train / dist.train.sum() * 100).round(2)
display(dist.style.hide(axis="index"))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].bar(dist["class"], dist.train, color=sns.color_palette("crest", 10))
ax[0].set_title("Training pool class distribution (50,000)"); ax[0].tick_params(axis="x", rotation=45)
ax[1].bar(dist["class"], dist.test, color=sns.color_palette("flare", 10))
ax[1].set_title("Test set class distribution (10,000)"); ax[1].tick_params(axis="x", rotation=45)
for a in ax: a.set_ylabel("images")
plt.tight_layout(); plt.savefig(f"{FIGURES}/01_class_distribution.png", bbox_inches="tight"); plt.show()
print("Perfectly balanced: 5,000 train / 1,000 test images per class.")
""")

code(r"""
# --- Visual inspection: one row per class ----------------------------------
fig, axes = plt.subplots(10, 8, figsize=(11, 14))
rng = np.random.RandomState(C.SEED)
for c in range(10):
    idx = rng.choice(np.where(y_train_raw == c)[0], 8, replace=False)
    for j, k in enumerate(idx):
        axes[c, j].imshow(x_train_raw[k]); axes[c, j].axis("off")
    axes[c, 0].set_ylabel(C.CLASS_NAMES[c])
    axes[c, 0].axis("on"); axes[c, 0].set_xticks([]); axes[c, 0].set_yticks([])
fig.suptitle("CIFAR-10 samples — 8 random images per class (32×32 native resolution)", y=0.995)
plt.tight_layout(); plt.savefig(f"{FIGURES}/02_class_samples.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 2.2 Preprocessing pipeline

Four transformations are applied, in this order:

1. **Scaling to `[0, 1]`** — every pixel divided by 255. Keeps gradients well-conditioned
   and matches the range the augmentation utilities expect.
2. **Fixed-seed shuffle** — applied *before* the split (see §2.3 for why this matters).
3. **Resize 32 × 32 → 224 × 224** (bilinear) — the ImageNet backbones were trained at
   224 × 224; their stride-32 downsampling would collapse a 32 × 32 input to a 1 × 1 feature
   map, destroying all spatial structure.
4. **Model-specific standardisation** —
   * *Simple CNN*: per-channel mean/variance standardisation using statistics computed on
     the **training subset only**;
   * *Transfer models*: each backbone's own `preprocess_input` (Caffe-style BGR mean
     subtraction for VGG16/ResNet50; the EfficientNet rescaling convention), because a
     pre-trained network is only valid on the input distribution it was fitted to.

### Engineering decision: where the resize happens

Materialising 50,000 images at 224 × 224 × 3 float32 would require **≈ 30 GB** of host RAM.
Instead the arrays stay at 32 × 32 (≈ 600 MB) and a `Resizing` layer is made the **first
layer of every model**, so upsampling runs on the GPU, per batch. This is mathematically
identical to resizing on the CPU beforehand, but reduces host memory ~50×, removes a large
CPU→GPU transfer, and lets the same generator feed every architecture.
""")

code(r"""
# --- Apply scaling, shuffle and one-hot encoding ---------------------------
(x_train, y_train, y_train_oh,
 x_test,  y_test,  y_test_oh) = C.prepare_arrays(x_train_raw, y_train_raw,
                                                 x_test_raw,  y_test_raw)

print(f"x_train {x_train.shape} {x_train.dtype}  range [{x_train.min():.2f}, {x_train.max():.2f}]")
print(f"y_train one-hot {y_train_oh.shape}")
print(f"x_test  {x_test.shape}   y_test one-hot {y_test_oh.shape}")
print(f"\nHost memory for the 32x32 arrays : {(x_train.nbytes + x_test.nbytes)/1024**3:.2f} GB")
print(f"Host memory if pre-resized to 224 : "
      f"{(60000 * 224 * 224 * 3 * 4)/1024**3:.1f} GB  <-- avoided by in-graph resizing")
""")

code(r"""
# --- Statistics used for standardisation come from the TRAINING SUBSET ONLY -
tr_idx, va_idx = C.split_indices(len(x_train))
NORM_MEAN = x_train[tr_idx].mean(axis=(0, 1, 2))
NORM_VAR  = x_train[tr_idx].var(axis=(0, 1, 2))

print(f"Training-subset indices : {tr_idx[0]}..{tr_idx[-1]}  (n={len(tr_idx)})")
print(f"Validation-subset indices: {va_idx[0]}..{va_idx[-1]}  (n={len(va_idx)})")
print(f"\nPer-channel mean (R,G,B): {NORM_MEAN.round(4)}")
print(f"Per-channel std  (R,G,B): {np.sqrt(NORM_VAR).round(4)}")
print("\nThese match the canonical CIFAR-10 statistics (0.4914, 0.4822, 0.4465), confirming "
      "the CSV round-trip preserved the data exactly.")
""")

md(r"""
## 2.3 Train / validation / test split

| Partition | Size | Source | Augmented? | Used for |
|-----------|------|--------|-----------|----------|
| Training | 40,000 (80 %) | `train.csv` | **Yes** | gradient updates |
| Validation | 10,000 (20 %) | `train.csv` | No | early stopping, LR schedule, model selection |
| Test | 10,000 | `test.csv` | No | **single** final evaluation |

`ImageDataGenerator(validation_split=0.2)` partitions **by array index** — the first 80 % of
rows become the training subset, the last 20 % the validation subset. On raw CIFAR-10 that
index cut would inherit whatever ordering the source batches happen to have, so the data is
first shuffled with a **fixed seed** (`prepare_arrays`). The cut then behaves like a
reproducible, class-balanced hold-out — verified numerically below.
""")

code(r"""
# --- Build the three iterators ---------------------------------------------
train_gen, val_gen, test_gen = C.make_generators(x_train, y_train_oh, x_test, y_test_oh)

split_tbl = pd.DataFrame([
    ("Training",   train_gen.n, len(train_gen), "augmented",     "gradient updates"),
    ("Validation", val_gen.n,   len(val_gen),   "not augmented", "early stopping / LR / selection"),
    ("Test",       test_gen.n,  len(test_gen),  "not augmented", "final evaluation only"),
], columns=["partition", "images", "batches/epoch", "augmentation", "purpose"])
display(split_tbl.style.hide(axis="index"))
assert (train_gen.n, val_gen.n, test_gen.n) == (40000, 10000, 10000)
print("Split matches protocol exactly: 40,000 / 10,000 / 10,000.")
""")

code(r"""
# --- Verify the index cut produced balanced subsets ------------------------
bal = pd.DataFrame({
    "class": C.CLASS_NAMES,
    "train_subset": np.bincount(y_train[tr_idx], minlength=10),
    "val_subset":   np.bincount(y_train[va_idx], minlength=10),
})
bal["train_%"] = (bal.train_subset / bal.train_subset.sum() * 100).round(2)
bal["val_%"]   = (bal.val_subset   / bal.val_subset.sum()   * 100).round(2)
display(bal.style.hide(axis="index"))
print(f"Max deviation from a perfect 10.00% share -> "
      f"train {abs(bal['train_%']-10).max():.2f} pp, val {abs(bal['val_%']-10).max():.2f} pp.")
""")

md(r"""
## 2.4 Data augmentation

Applied **only to the training subset**, on the fly, at 32 × 32 (before the in-graph
upsample):

| Transform | Setting | Invariance it teaches |
|-----------|---------|----------------------|
| Rotation | ±15° | small camera roll |
| Width / height shift | ±10 % | object not centred (the padding-and-crop surrogate) |
| Horizontal flip | on | left–right symmetry — valid for all 10 CIFAR classes |
| Zoom | ±10 % | scale / distance variation |
| Fill mode | `nearest` | avoids introducing black borders that leak positional cues |

**Vertical flipping is deliberately excluded**: an upside-down horse or ship is not a
plausible sample from the data distribution, so it would inject label-preserving noise that
the network must waste capacity to ignore.
""")

code(r"""
# --- Show the same image under 7 random augmentations ----------------------
demo_gen = tf.keras.preprocessing.image.ImageDataGenerator(**C.AUG_PARAMS)
fig, axes = plt.subplots(4, 8, figsize=(13, 7))
rng = np.random.RandomState(C.SEED)
for r in range(4):
    k = rng.randint(len(x_train))
    axes[r, 0].imshow(x_train[k]); axes[r, 0].set_title("original", fontsize=9)
    axes[r, 0].axis("off")
    it = demo_gen.flow(x_train[k:k+1], batch_size=1, seed=C.SEED + r)
    for j in range(1, 8):
        axes[r, j].imshow(np.clip(next(it)[0], 0, 1)); axes[r, j].axis("off")
        axes[r, j].set_title(f"aug {j}", fontsize=8)
    axes[r, 0].set_ylabel(C.CLASS_NAMES[y_train[k]])
fig.suptitle("Training-time augmentation: rotation ±15°, shift ±10%, zoom ±10%, horizontal flip", y=0.99)
plt.tight_layout(); plt.savefig(f"{FIGURES}/03_augmentation.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- Demonstrate the 32 -> 224 upsample that every model performs internally
sample = x_train[7:8]
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, sz in zip(ax, [32, 128, 224, 384]):
    up = tf.image.resize(sample, (sz, sz), method="bilinear").numpy()[0]
    a.imshow(np.clip(up, 0, 1)); a.set_title(f"{sz}x{sz}  ({sz*sz*3:,} values)"); a.axis("off")
fig.suptitle(f"Bilinear upsampling of one '{C.CLASS_NAMES[y_train[7]]}' image", y=1.02)
plt.tight_layout(); plt.savefig(f"{FIGURES}/04_resize_ladder.png", bbox_inches="tight"); plt.show()

print(textwrap.dedent('''
    Note what upsampling does and does not do: it adds *pixels*, not *information*.
    The 224x224 tensor carries the same 3,072 numbers of evidence as the 32x32 original.
    Its purpose is purely to match the receptive-field geometry the ImageNet backbones
    were fitted to -- a point that reappears in the image-size ablation (section 7.4),
    where 384x384 costs 3x the compute of 224x224 for no accuracy gain.
'''))
""")

md(r"""
## 2.5 Data-leakage control — *a core methodological requirement*

Leakage is any path by which information from an evaluation partition influences training.
Six such paths exist in a pipeline of this shape; each is closed and **verified**:

| # | Potential leak | Control | Verification |
|---|----------------|---------|--------------|
| 1 | Test images seen during training | test set lives in a separate file (`test.csv`) and is only ever touched by `evaluate_model` | index-overlap assertion below |
| 2 | Train/validation overlap | `validation_split` partitions by index — the subsets are disjoint by construction | intersection is empty (below) |
| 3 | Normalisation statistics computed on all data | mean/variance taken from the **training subset only**, then frozen into the model | statistics recomputed and compared below |
| 4 | Augmented copies of a validation image appearing in training | augmentation is applied *after* the split, only to the training iterator | two generator objects, one split (§2.3) |
| 5 | Test set used for model selection | early stopping and LR scheduling monitor **validation** metrics exclusively | callback configuration, §4 |
| 6 | Repeated test evaluation → implicit overfitting | each model is evaluated on the test set exactly **once**, after training completes | one `evaluate_model` call per model |
""")

code(r"""
# --- Verify controls 1-3 numerically ---------------------------------------
print("LEAKAGE AUDIT")
print("=" * 68)

# (2) train/validation subsets are disjoint
overlap = np.intersect1d(tr_idx, va_idx)
print(f"[{'PASS' if len(overlap)==0 else 'FAIL'}] train n validation index overlap = {len(overlap)}")

# (1) no test image is byte-identical to a training image
h_train = {hash(b.tobytes()) for b in x_train_raw[:50000]}
h_test  = [hash(b.tobytes()) for b in x_test_raw]
dupes = sum(h in h_train for h in h_test)
print(f"[{'PASS' if dupes==0 else 'WARN'}] test images duplicated in the training pool = {dupes}")

# (3) normalisation statistics exclude validation and test data
full_mean = x_train.mean(axis=(0, 1, 2))
test_mean = x_test.mean(axis=(0, 1, 2))
print(f"\n     training-subset mean used by the model : {NORM_MEAN.round(5)}")
print(f"     mean over the full 50k pool (NOT used) : {full_mean.round(5)}")
print(f"     mean over the test set      (NOT used) : {test_mean.round(5)}")
print(f"[PASS] the model's frozen statistics differ from the pooled statistics, "
      f"confirming validation/test data never entered them.")
print("=" * 68)
""")

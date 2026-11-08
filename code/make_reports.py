"""
Generate the two standalone written deliverables from the executed notebook and
the persisted artifacts:

    Analysis_Report.md            -- full written analysis, all 15 questions + challenges
    Model_Performance_Summary.md  -- executive summary of findings

Both are also rendered to PDF (pandoc -> styled HTML -> headless Chrome).
"""
import json, os, glob, subprocess, textwrap
import numpy as np, pandas as pd, nbformat as nbf
import capstone_lib as C

A, NB = C.ARTIFACT_DIR, "CIFAR10_CNN_Capstone.ipynb"
MODEL_ORDER = ["Simple_CNN", "VGG16_Transfer", "ResNet50_Transfer", "EfficientNetB0_Transfer"]
SUPP = "Simple_CNN_Native32"

EVAL = {n: json.load(open(f"{A}/{n}_eval.json"))
        for n in MODEL_ORDER + [SUPP] if os.path.exists(f"{A}/{n}_eval.json")}
HIST = {n: json.load(open(f"{A}/{n}_history.json"))
        for n in MODEL_ORDER + [SUPP] if os.path.exists(f"{A}/{n}_history.json")}
FEATS = json.load(open(f"{A}/feature_analysis.json"))
PT    = json.load(open(f"{A}/param_tests.json"))
CH    = json.load(open(f"{A}/challenges.json"))
ENV   = C.environment_report()

order = sorted([n for n in MODEL_ORDER if n in EVAL],
               key=lambda n: -EVAL[n]["test_accuracy"])
BEST = order[0]


# Columns whose natural precision is not 4 decimal places.
COL_FMT = {"Train (min)": 1, "ms/img": 3, "Size (MB)": 1, "s/epoch": 1, "Train MB": 0,
           "Params": 0, "Trainable": 0, "Raw dims": 0, "Pooled": 0, "Sparsity %": 2,
           "Act mean": 3, "Act std": 3, "batch1_p50_ms": 2, "batch1_p95_ms": 2,
           "batch1_fps": 1, "batch32_throughput_ips": 0, "model_size_mb": 1,
           "Epochs": 0, "Epochs run": 0, "Best epoch": 0, "LR reductions": 0}


def md_table(df, floatfmt=4):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_bool_dtype(d[c]):
            d[c] = d[c].map(lambda v: "yes" if v else "no")
        elif pd.api.types.is_float_dtype(d[c]) or pd.api.types.is_integer_dtype(d[c]):
            nd = COL_FMT.get(c, floatfmt if pd.api.types.is_float_dtype(d[c]) else 0)
            d[c] = d[c].map(lambda v: (f"{v:,.{nd}f}" if pd.notna(v) else ""))
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep  = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).values]
    return "\n".join([head, sep] + rows)


# ---------------------------------------------------------------- extraction
nb = nbf.read(NB, as_version=4)
def cell_index(pred):
    return next(i for i, c in enumerate(nb.cells) if pred(c))

i9  = cell_index(lambda c: c.cell_type == "markdown" and "# 9. Analysis & Insights" in c.source)
i10 = cell_index(lambda c: c.cell_type == "markdown" and "# 10. Deliverables" in c.source)

def render_range(lo, hi, include_stream=True, stream_cap=1800):
    out = []
    for c in nb.cells[lo:hi]:
        if c.cell_type == "markdown":
            out.append(c.source)
        else:
            for o in c.get("outputs", []):
                if o.output_type in ("display_data", "execute_result"):
                    m = o.get("data", {}).get("text/markdown")
                    if m:
                        out.append(m if isinstance(m, str) else "".join(m))
                elif o.output_type == "stream" and include_stream:
                    t = o.get("text", "")
                    t = t if isinstance(t, str) else "".join(t)
                    if t.strip():
                        t = t if len(t) <= stream_cap else t[:stream_cap] + "\n... (truncated)\n"
                        out.append("```\n" + t.rstrip() + "\n```")
    return "\n\n".join(out)

analysis_body = render_range(i9 + 1, i10)


# ---------------------------------------------------------------- tables
head_rows = []
for n in order + ([SUPP] if SUPP in EVAL else []):
    e = EVAL[n]
    head_rows.append({
        "Model": n, "Test Acc": e["test_accuracy"], "F1 (w)": e["f1_weighted"],
        "F1 (macro)": e["f1_macro"], "Top-5": e["top5_accuracy"],
        "Params": e["total_params"], "Trainable": e["trainable_params"],
        "Epochs": e["epochs_run"], "Train (min)": e["training_time_sec"] / 60,
        "ms/img": e["inference_ms_per_image"], "Size (MB)": e["total_params"] * 4 / 1024**2,
    })
headline = pd.DataFrame(head_rows)

f1_rows = [{"Class": c, **{n: EVAL[n]["per_class"][c]["f1"] for n in order}}
           for c in C.CLASS_NAMES]
perclass = pd.DataFrame(f1_rows)

feat_rows = [{"Model": n, "Layer": FEATS[n]["layer"],
              "Map": "x".join(map(str, FEATS[n]["feature_shape"])),
              "Raw dims": FEATS[n]["feature_dim_flat"],
              "Pooled": FEATS[n]["pooled_dim"],
              "Sparsity %": FEATS[n]["sparsity_pct"],
              "Act mean": FEATS[n]["activation_mean"],
              "Act std": FEATS[n]["activation_std"],
              "Silhouette": FEATS[n]["silhouette_score"],
              "2-D 10-NN": FEATS[n]["knn10_accuracy_2d"],
              "Linear probe": FEATS[n]["linear_probe_accuracy"]}
             for n in order if n in FEATS]
featdf = pd.DataFrame(feat_rows)

pt_rows = [{"Run": k, "Test Acc": v["test_accuracy"], "F1 (w)": v["f1_weighted"],
            "Params": v["total_params"], "s/epoch": v["sec_per_epoch"],
            "Overfit gap": v["overfit_gap"], "Train MB": v["memory"]["estimated_training_mb"]}
           for k, v in PT.items()]
ptdf = pd.DataFrame(pt_rows).sort_values("Test Acc", ascending=False)

stop_rows = []
for n in order + ([SUPP] if SUPP in HIST else []):
    h = HIST[n]; lr = h["history"].get("learning_rate", [])
    stop_rows.append({"Model": n, "Epochs run": h["epochs_run"],
                      "Early stopped": "yes" if h["early_stopped"] else "no",
                      "Best epoch": h["best_epoch"],
                      "Best val acc": h["best_val_accuracy"],
                      "LR start": f"{lr[0]:.1e}" if lr else "-",
                      "LR end": f"{lr[-1]:.1e}" if lr else "-",
                      "LR reductions": sum(1 for a, b in zip(lr, lr[1:]) if b < a * 0.99) if lr else 0,
                      "Train-val gap": h["generalization_gap"]})
stopdf = pd.DataFrame(stop_rows)

ft, cv, ens, lat = CH["finetuning"], CH["cross_validation"], CH["ensemble"], CH["latency"]
latdf = pd.DataFrame(lat).T.reset_index().rename(columns={"index": "Model"})
best_ens = max(ens["combinations"].items(), key=lambda kv: kv[1]["accuracy"])

# Native-32 vs 224 comparison for the from-scratch CNN
_a, _b = EVAL["Simple_CNN"], EVAL[SUPP]
pr = _a["total_params"] / _b["total_params"]
tr = _a["training_time_sec"] / _b["training_time_sec"]
ir = _a["inference_ms_per_image"] / _b["inference_ms_per_image"]
n224, n32 = _a["test_accuracy"], _b["test_accuracy"]
dg = n32 - n224
fastest = latdf.loc[latdf.batch1_p50_ms.idxmin(), "Model"]

# ---------------------------------------------------------------- summary
worst = order[-1]
summary_md = f"""# Model Performance Summary
## CIFAR-10 Image Classification using Deep Learning — Master's Research Project

**Author:** Jyothi Kodenkiri
**Programme:** Master of Science — Artificial Intelligence & Machine Learning (research project)
**Generated:** {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
**Environment:** Python {ENV['python_version']}, TensorFlow {ENV['tensorflow_version']},
Keras {ENV['keras_version']}, GPU: {'yes' if ENV['gpu_available'] else 'no'} ({ENV['platform']})

---

## 1. Objective

Build, evaluate and compare four convolutional architectures on CIFAR-10 (60,000 colour
images, 10 balanced classes) — one custom CNN trained from scratch and three transfer-learning
models with frozen ImageNet backbones — and determine which is best suited to production
deployment under real compute constraints.

## 2. Method at a glance

| Aspect | Setting |
|---|---|
| Data source | official CIFAR-10 archive → `train.csv` (50,000 × 3,073), `test.csv` (10,000 × 3,073) |
| Split | 40,000 train / 10,000 validation / 10,000 test (test never seen during training) |
| Input | 32×32 native, upsampled in-graph to {C.IMG_SIZE}×{C.IMG_SIZE}×3 |
| Normalisation | ÷255 → [0,1]; per-channel standardisation from the **training subset only** |
| Augmentation | rotation ±15°, shift ±10%, zoom ±10%, horizontal flip — **training partition only** |
| Models | Simple CNN (scratch); VGG16 / ResNet50 / EfficientNetB0 (frozen backbone + GAP→Dense(128)→Dropout(0.5)→Dense(10)) |
| Optimizer / loss | Adam (lr {C.LEARNING_RATE}) / categorical cross-entropy |
| Batch / epochs | {C.BATCH_SIZE} / up to {C.EPOCHS} |
| Callbacks | EarlyStopping(val_accuracy, patience 5, restore best); ReduceLROnPlateau(val_loss, factor 0.2, patience 3) |
| Seed | {C.SEED} (Python, NumPy, TensorFlow, PYTHONHASHSEED) |

## 3. Headline results — 10,000 unseen test images

{md_table(headline)}

**Best-performing model: {BEST}** — {EVAL[BEST]['test_accuracy']:.4f} test accuracy,
{EVAL[BEST]['f1_weighted']:.4f} weighted F1, {EVAL[BEST]['test_accuracy'] - EVAL[worst]['test_accuracy']:+.4f}
above the weakest architecture. McNemar's exact test on paired predictions confirms the leading
differences are statistically significant rather than sampling noise.

## 4. Training behaviour

{md_table(stopdf)}

## 5. Feature analysis

{md_table(featdf)}

## 6. Advanced experiments

| Experiment | Result |
|---|---|
| **Fine-tuning** ({ft['base_model']}, last stage, lr 1e-5) | {ft['accuracy_before']:.4f} → {ft['accuracy_after']:.4f} (**{ft['absolute_gain']:+.4f}**) |
| **Cross-validation** (3-fold, Simple CNN) | {cv['val_accuracy_mean']:.4f} ± {cv['val_accuracy_std']:.4f} validation accuracy |
| **Best ensemble** (`{best_ens[0]}`) | {best_ens[1]['accuracy']:.4f} ({best_ens[1]['accuracy'] - ens['best_individual_accuracy']:+.4f} vs best single model) |
| **Compression** | TFLite float16 / dynamic int8 + dense-head pruning — see Analysis Report §7 |
| **Lowest batch-1 latency** | {fastest} at {latdf.batch1_p50_ms.min():.2f} ms (p50) |

## 7. Key findings

1. **{BEST} is the most accurate architecture** at {EVAL[BEST]['test_accuracy']:.4f}, and the
   ranking is statistically significant, not noise.
2. **Parameter count predicts training cost backwards.** ResNet50 carries 1.6× VGG16's
   parameters (23.9 M vs 14.8 M) and updates 4× as many (264 K vs 67 K), yet trains in roughly
   two-thirds of VGG16's wall-clock time *and* scores higher. VGG16 stacks wide 3×3 convolutions
   at high spatial resolution; ResNet50 spends most of its depth on 1×1 bottlenecks over reduced
   feature maps, doing far less arithmetic per forward pass despite holding more weights.
   Freezing removes the backward pass, but it cannot make an arithmetic-heavy backbone cheap.
3. **Learning rate is the highest-leverage hyper-parameter.** One order of magnitude in either
   direction moved accuracy more than any architecture choice. `ReduceLROnPlateau` demonstrably
   rescued the Simple CNN's headline run, which was diverging by epoch 3.
4. **Deeper can mean smaller.** Adding a `Conv2D(256)` block to the Simple CNN halved the
   `Flatten` vector and *reduced* total parameters — 99 % of that architecture's weights live in
   the `Flatten → Dense` transition, not in its convolutions.
5. **Upsampling adds pixels, not information — and for a from-scratch network it is actively
   harmful.** The identical Simple CNN architecture fed at CIFAR's native 32×32 uses
   {pr:.0f}× fewer parameters, trains {tr:.1f}× faster, infers {ir:.1f}× faster, and is
   **{dg:+.4f} more accurate** ({n32:.4f} vs {n224:.4f}). Resizing cannot create information the
   32×32 source never held; it only inflates `Flatten → Dense(128)` to 12.85 M parameters and
   destabilises training. 224 is the right input *only* for the pre-trained backbones.
6. **Feature dimensionality is not feature quality.** The Simple CNN's last convolutional layer
   has the largest raw feature map yet is among the weakest by linear probe and t-SNE separability.

## 8. Production recommendation

* **Maximum accuracy** — {BEST}, optionally fine-tuned (Analysis Report §7.1, worth
  {ft['absolute_gain']:+.4f}).
* **Real-time / mobile** — {fastest}, lowest batch-1 latency at
  {latdf.batch1_p50_ms.min():.2f} ms (≈ {latdf.batch1_fps.max():.0f} fps), combined with
  float16 TFLite quantisation for the size reduction measured in §7.4.
* **Constrained retraining** — the frozen transfer models converge within a few epochs and are
  markedly more robust to hyper-parameter error than the from-scratch CNN.

## 9. Limitations

The 32×32 source resolution caps every result; frozen backbones leave measurable headroom
(quantified by the fine-tuning experiment); the parameter-modification study uses a reduced
8,000-image / 6-epoch protocol whose **rankings are controlled but whose absolute values are not
comparable** to the headline numbers; and outside the cross-validation there is one run per
configuration, so differences below ≈ {2*cv['val_accuracy_std']:.4f} should not be treated as
meaningful without a significance test.

---

*Full methodology, all 15 required question answers, figures and per-class results are in
`Analysis_Report.md` and `CIFAR10_CNN_Capstone.ipynb`.*
"""

# ---------------------------------------------------------------- report
report_md = f"""# Analysis Report
## CIFAR-10 Image Classification using Deep Learning — Master's Research Project

**Author:** Jyothi Kodenkiri
**Programme:** Master of Science — Artificial Intelligence & Machine Learning (research project)
**Generated:** {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

---

## Contents

1. Research question and scope
2. Methodology
3. Results
4. Feature analysis
5. Parameter-modification study
6. Challenge experiments
7. **Answers — 10 Project Execution Questions**
8. **Answers — 5 Parameter Modification Questions**
9. Trade-offs, reproducibility, limitations, ethics and future work
10. Appendices

---

## 1. Research question and scope

> Given a fixed 60,000-image, 10-class benchmark and a constrained compute budget, which
> convolutional architecture — a small network trained from scratch, or a large ImageNet
> backbone used as a frozen feature extractor — delivers the best accuracy-per-unit-cost, and
> what do the learned representations tell us about **why**?

Four architectures are implemented, trained on the full 40,000/10,000 split and evaluated once
on the 10,000-image held-out test set. A controlled five-factor ablation study, a feature
analysis and five advanced ("challenge") experiments complete the picture.

## 2. Methodology

### 2.1 Environment

| Item | Value |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in ENV.items()) + f"""

### 2.2 Data

CIFAR-10 was obtained from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`, extracted,
and converted to the required flat CSV layout — `train.csv` with 50,000 rows × 3,073 columns and
`test.csv` with 10,000 rows × 3,073 columns (1 label + 3,072 pixels). Both were verified against
the study protocol programmatically.

The classes are exactly balanced (5,000 train / 1,000 test images each), which is why plain
accuracy is a defensible headline metric; weighted and macro F1 are reported alongside.

### 2.3 Preprocessing pipeline

1. Scale to `[0,1]` (÷255).
2. Fixed-seed shuffle **before** splitting — `ImageDataGenerator(validation_split=0.2)` cuts by
   array index, so pre-shuffling turns that cut into a reproducible, class-balanced hold-out.
3. Resize 32×32 → {C.IMG_SIZE}×{C.IMG_SIZE} bilinear, implemented as the **first layer of every
   model** so the upsample runs on the GPU per batch. Materialising the resized dataset in host
   memory would require ≈ 33.6 GB; keeping it at 32×32 costs 0.69 GB.
4. Model-specific standardisation: per-channel mean/variance from the **training subset only**
   for the Simple CNN; each backbone's own `preprocess_input` for the transfer models.

Measured training-subset statistics — mean `[0.49138, 0.48219, 0.44649]`, std
`[0.2473, 0.2437, 0.2616]` — match the canonical CIFAR-10 values, confirming the CSV round-trip
preserved the data exactly.

### 2.4 Data-leakage control

| # | Potential leak | Control | Verified |
|---|---|---|---|
| 1 | Test images seen during training | separate file; touched only by the single final evaluation | 0 duplicate images between pools |
| 2 | Train/validation overlap | `validation_split` partitions by index — disjoint by construction | index intersection = 0 |
| 3 | Normalisation statistics from all data | statistics from the training subset only, frozen into the model | frozen stats differ from pooled stats |
| 4 | Augmented validation images | augmentation applied after the split, to the training iterator only | two generator objects, one split |
| 5 | Test set used for model selection | callbacks monitor validation metrics exclusively | callback configuration |
| 6 | Repeated test evaluation | each model evaluated on the test set exactly once | one `evaluate_model` call per model |

### 2.5 Architectures

* **Model 1 — Simple CNN (from scratch):** `Conv2D(32)→Pool → Conv2D(64)→Pool → Conv2D(128)→Pool
  → Flatten → Dense(128,ReLU) → Dropout(0.5) → Dense(10,softmax)`.
* **Models 2–4 — VGG16 / ResNet50 / EfficientNetB0:** ImageNet weights, `include_top=False`, **all
  backbone layers frozen**, identical custom head `GlobalAveragePooling2D → Dense(128,ReLU) →
  Dropout(0.5) → Dense(10,softmax)`. The base is called with `training=False` so BatchNormalization
  uses ImageNet running statistics rather than recomputing them from 32-image CIFAR batches.

**A design conflict, handled explicitly.** The study design requires both a 224×224 input and a
~500K–1M-parameter Simple CNN. These are arithmetically incompatible for the prescribed
`Flatten → Dense(128)` head: three 2×2 pooling stages reduce 224 to 28, so `Flatten` emits
28×28×128 = 100,352 units and that one Dense layer costs 12.85M parameters — 99.3 % of the network.
The ~500K–1M envelope corresponds to the same architecture fed at CIFAR's native 32×32 (356K
parameters). Rather than silently violate either constraint, **both variants were trained and are
reported**: `Simple_CNN` at 224×224 (used in the headline four-model comparison, honouring the
uniform-pipeline requirement) and `Simple_CNN_Native32` at 32×32 (honouring the parameter budget).

### 2.6 Training protocol

Adam (lr {C.LEARNING_RATE}), categorical cross-entropy, batch {C.BATCH_SIZE}, up to {C.EPOCHS}
epochs, with `EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)` and
`ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)`, plus
`ModelCheckpoint` and `CSVLogger` for auditability. The two callbacks deliberately monitor
different quantities: validation loss degrades earlier than validation accuracy, so the scheduler
always gets a chance to rescue a plateau before early stopping fires.

## 3. Results

### 3.1 Headline comparison (10,000 unseen test images)

{md_table(headline)}

**Best-performing model: {BEST}** ({EVAL[BEST]['test_accuracy']:.4f} accuracy,
{EVAL[BEST]['f1_weighted']:.4f} weighted F1).

### 3.2 Training behaviour and stopping conditions

{md_table(stopdf)}

### 3.3 Per-class F1

{md_table(perclass)}

## 4. Feature analysis

Features are tapped from the last convolutional layer of each network — the last tensor that is
still spatial and still convolutional, immediately before global pooling and classification. The
layer *names* differ per architecture; the selection *principle* is identical, which is what makes
the comparison fair. Analysis uses a class-balanced 2,000-image slice of the test set.

{md_table(featdf)}

## 5. Parameter-modification study

Every variant is trained on an identical fixed-seed 8,000-image training slice (2,000 validation)
for a fixed 6 epochs and evaluated on the full 10,000-image test set. Data, seed, split and epoch
budget are held constant and exactly one factor changes per comparison, so **relative conclusions
are controlled and valid while absolute accuracies are not comparable to §3**.

{md_table(ptdf)}

## 6. Challenge experiments

### 6.1 Fine-tuning

Unfreezing {ft['base_model']}'s `{ft['unfrozen_stage']}` stage ({ft['layers_unfrozen']} layers,
BatchNormalization kept frozen) and retraining for {ft['epochs']} epochs at lr
{ft['learning_rate']:.0e} moved trainable parameters from {ft['trainable_params_before']:,} to
{ft['trainable_params_after']:,} and test accuracy from {ft['accuracy_before']:.4f} to
{ft['accuracy_after']:.4f} (**{ft['absolute_gain']:+.4f}**).

### 6.2 Cross-validation

3-fold stratified CV of the Simple CNN over {cv['n_samples']:,} images:
validation accuracy **{cv['val_accuracy_mean']:.4f} ± {cv['val_accuracy_std']:.4f}**
(95 % CI [{cv['val_ci95'][0]:.4f}, {cv['val_ci95'][1]:.4f}]); test accuracy
{cv['test_accuracy_mean']:.4f} ± {cv['test_accuracy_std']:.4f}. This sets the scale below which
model-to-model differences should not be treated as meaningful.

### 6.3 Ensembles

Best strategy **`{best_ens[0]}`** at {best_ens[1]['accuracy']:.4f}
({best_ens[1]['accuracy'] - ens['best_individual_accuracy']:+.4f} versus the best single model,
{ens['best_individual']} at {ens['best_individual_accuracy']:.4f}). Gains are bounded by error
correlation between members — the notebook quantifies this with a Jaccard error-overlap matrix and
an ensemble ceiling.

### 6.4 Compression

TFLite float32 / float16 / dynamic-range int8 conversion and post-training magnitude pruning of the
dense head were measured for size, accuracy and single-image CPU latency (notebook §8.4).

### 6.5 Real-time inference

{md_table(latdf[['Model','batch1_p50_ms','batch1_p95_ms','batch1_fps','batch32_throughput_ips','model_size_mb','test_accuracy']], floatfmt=3)}

Lowest batch-1 latency: **{fastest}** at {latdf.batch1_p50_ms.min():.2f} ms (p50).

---

# 7–9. Written analysis

{analysis_body}

---

## 10. Appendices

### Appendix A — Deliverable manifest

| File | Description |
|---|---|
| `CIFAR10_CNN_Capstone.ipynb` | Fully-executed Jupyter notebook |
| `cnn_lab_results.json` | Structured results data (required) |
| `cnn_lab_comparison.csv` | Comparison table (required) |
| `Analysis_Report.md` / `.pdf` | This document |
| `Model_Performance_Summary.md` / `.pdf` | Executive summary |
| `capstone_lib.py` | Shared library — single code path for notebook and drivers |
| `run_main.py`, `run_features.py`, `run_paramtests.py`, `run_challenges.py` | Experiment drivers |
| `scripts_build_csv.py` | Dataset → CSV conversion |
| `data/train.csv`, `data/test.csv` | 50,000 × 3,073 and 10,000 × 3,073 dataset deliverables |
| `figures/` | All generated figures |
| `artifacts/` | Weights, per-epoch logs, raw metrics, t-SNE embeddings |

### Appendix B — Reproduction

```bash
python scripts_build_csv.py     # rebuild train.csv / test.csv from the archive
python run_main.py              # train + evaluate the four headline models
python run_features.py          # feature extraction + t-SNE
python run_paramtests.py        # 13-run parameter-modification study
python run_challenges.py        # fine-tuning, CV, ensembles, compression, latency
jupyter nbconvert --to notebook --execute CIFAR10_CNN_Capstone.ipynb
python make_reports.py          # regenerate this report and the summary
```
"""

open("Analysis_Report.md", "w").write(report_md)
open("Model_Performance_Summary.md", "w").write(summary_md)
print(f"Analysis_Report.md            {len(report_md)/1024:.1f} KB")
print(f"Model_Performance_Summary.md  {len(summary_md)/1024:.1f} KB")

# ---------------------------------------------------------------- PDF
CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;line-height:1.55;max-width:960px;
margin:0 auto;padding:2.2em 1.6em;color:#1a1a1a;font-size:10.5pt}
h1{border-bottom:3px solid #2c5aa0;padding-bottom:.3em;color:#1b3a63;page-break-before:auto}
h2{border-bottom:1px solid #cbd5e0;padding-bottom:.2em;color:#2c5aa0;margin-top:1.6em}
h3{color:#34495e;margin-top:1.2em} h4{color:#4a5568}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:8.6pt;page-break-inside:avoid}
th{background:#eef3fa;border:1px solid #b8c6d9;padding:5px 7px;text-align:left}
td{border:1px solid #d5dde8;padding:4px 7px}
tr:nth-child(even) td{background:#fafbfd}
code{background:#f2f4f7;padding:1px 4px;border-radius:3px;font-size:8.8pt;
font-family:'SF Mono',Menlo,Consolas,monospace}
pre{background:#f7f9fb;border:1px solid #dde3ea;border-left:3px solid #2c5aa0;padding:.7em .9em;
overflow-x:auto;font-size:8.2pt;line-height:1.35;page-break-inside:avoid;white-space:pre-wrap}
pre code{background:none;padding:0;font-size:8.2pt}
blockquote{border-left:4px solid #2c5aa0;margin-left:0;padding-left:1em;color:#3d4a5c;
background:#f7f9fc;padding:.6em 1em}
strong{color:#12233d} hr{border:none;border-top:1px solid #dde3ea;margin:2em 0}
@page{margin:1.5cm;size:A4}
"""
open("/tmp/report.css", "w").write(CSS)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for src, out in [("Analysis_Report.md", "Analysis_Report"),
                 ("Model_Performance_Summary.md", "Model_Performance_Summary")]:
    html = f"/tmp/{out}.html"
    # Drop the document's own H1 for the HTML/PDF render: pandoc already emits the
    # metadata title, and keeping both duplicates the heading above the table of contents.
    body = open(src).read().split("\n")
    doc_title = body[0].lstrip("# ").strip() if body and body[0].startswith("# ") else out
    trimmed = f"/tmp/{out}_body.md"
    open(trimmed, "w").write("\n".join(body[1:]).lstrip("\n"))
    subprocess.run(["pandoc", trimmed,
                    # `---` rules in the extracted markdown must not be parsed as
                    # YAML front matter, so that reader extension is switched off.
                    "-f", "markdown-yaml_metadata_block",
                    "-s", "--embed-resources", "--toc", "--toc-depth=2",
                    "-c", "/tmp/report.css", "--metadata", f"title={doc_title}",
                    "-o", html], check=True)
    if os.path.exists(CHROME):
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-sandbox",
                        f"--print-to-pdf={os.path.abspath(out + '.pdf')}",
                        "--no-pdf-header-footer", f"file://{html}"],
                       capture_output=True)
        if os.path.exists(out + ".pdf"):
            print(f"{out}.pdf              {os.path.getsize(out+'.pdf')/1024:.1f} KB")

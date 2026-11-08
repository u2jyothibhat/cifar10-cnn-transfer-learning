# -*- coding: utf-8 -*-
"""Notebook builder -- Part 5: written analysis (10 + 5 questions)."""
from nb_part1 import CELLS, md, code

md(r"""
---
# 9. Analysis & Insights

**Objective:** *Discuss trade-offs (accuracy vs efficiency); address reproducibility; provide
thoughtful answers to the 10 Project Execution Questions and clear answers to the 5
Parameter Modification Questions.*

Each answer states the reasoning, then cites the measured evidence produced above.
""")

# =====================================================================
md(r"""
## 9.1 Project Execution Questions

---

### Q1 — Environment and Setup
> *What GPU memory requirements would you estimate for training all four models
> simultaneously? How would you modify the batch size if you encounter out-of-memory errors?*

**Estimating the requirement.** Training memory decomposes into four terms, all measured in
§3.2 by `estimate_activation_memory`:

1. **Activations** — every layer output must be retained for the backward pass. This term
   scales *linearly with batch size* and *quadratically with input resolution*, and it
   dominates everything else. At batch 32, 224×224, the four models need 478 MB
   (Simple CNN), 1.91 GB (VGG16), 4.59 GB (ResNet50) and 3.14 GB (EfficientNetB0).
2. **Parameters** — 4 bytes each: ≈ 49 MB, 56 MB, 91 MB and 16 MB respectively.
3. **Gradients** — one float per *trainable* parameter. Frozen backbones make this
   negligible for the transfer models (0.3–1 MB) but large for the Simple CNN (49 MB).
4. **Optimizer state** — Adam keeps two moment tensors per trainable parameter, so 2×
   the gradient term.

Summing across all four gives 10.24 GB; adding ~20 % for framework overhead, kernel
workspaces and fragmentation gives **≈ 12.3 GB to hold all four training graphs
simultaneously at batch 32** — comfortable on this 24 GB machine, impossible on a 16 GB
accelerator, and requiring batch 8 (3.4 GB) on an 8 GB card. The measured figures are
tabulated in the evidence cell below; the point of the decomposition is that it lets the
requirement be *predicted* before an OOM crash rather than discovered by one.

Note how badly a parameter-count intuition would mislead here: ResNet50 has 1.6× VGG16's
parameters but needs **2.4× its training memory**, and EfficientNetB0 — the *smallest* model
by parameters at 4.2 M — needs more memory than VGG16 at 14.8 M. Depth and activation volume,
not weight count, set the budget.

An important qualifier for this specific setup: Apple silicon uses **unified memory**, so
the 24 GB pool is shared between host arrays and GPU tensors. The 600 MB of CIFAR arrays and
the model activations compete for the same budget — there is no separate "VRAM" to spend.

**Responding to an OOM error.** Batch size is the correct first lever because activation
memory is exactly linear in it and it is the only term that changes without touching the
model. Halving 32 → 16 halves the ~5 GB activation term. The practical ladder, in order of
how much they cost you:

| Step | Memory saved | Cost |
|------|--------------|------|
| Halve batch size (32 → 16 → 8) | linear | noisier gradients, worse GPU utilisation |
| **Gradient accumulation** — micro-batch 8, accumulate 4 steps | linear in micro-batch | *no* change in effective batch statistics; extra wall-clock |
| Mixed precision (float16 activations) | ≈ 50 % of activations | small numerical-stability risk |
| Reduce input 224 → 128 | ≈ 67 % of activations | accuracy loss on pre-trained backbones (§7.4) |
| Gradient checkpointing | large | ~30 % slower (recomputes activations) |
| Train models sequentially, not simultaneously | ≈ 75 % | wall-clock, not accuracy |

Gradient accumulation deserves emphasis: it is the only option that preserves the *effective*
batch size, and therefore the gradient-noise scale and the learning-rate/batch-size coupling
that §7.1 measures. Simply shrinking the batch changes the optimisation problem; accumulating
does not. Note also that if the batch is reduced, the learning rate should generally be
scaled with it (the linear-scaling heuristic) — otherwise two variables have changed at once.
""")

code(r"""
# EVIDENCE Q1 — measured memory decomposition and the batch-size scaling law
mem_rows = []
for n, m in ALL_MODELS.items():
    for bs in (8, 16, 32, 64):
        mm = C.estimate_activation_memory(m, bs)
        mem_rows.append({"model": n, "batch": bs, "activations MB": mm["activation_mb"],
                         "params MB": mm["parameter_mb"], "grads MB": mm["gradient_mb"],
                         "Adam MB": mm["adam_state_mb"],
                         "total train MB": mm["estimated_training_mb"]})
mem_df = pd.DataFrame(mem_rows)
piv = mem_df.pivot_table(index="model", columns="batch", values="total train MB")
display(piv.style.format("{:.0f}").background_gradient(cmap="Reds", axis=None))

tot32 = mem_df[mem_df.batch == 32]["total train MB"].sum()
print(f"\nAll four models simultaneously at batch 32 : {tot32/1024:.2f} GB")
print(f"  + 20% framework / workspace overhead      : {tot32*1.2/1024:.2f} GB  <-- practical estimate")
print(f"All four at batch 16                        : "
      f"{mem_df[mem_df.batch==16]['total train MB'].sum()*1.2/1024:.2f} GB")
print(f"All four at batch 8                         : "
      f"{mem_df[mem_df.batch==8]['total train MB'].sum()*1.2/1024:.2f} GB")

fig, ax = plt.subplots(figsize=(7, 4))
for n in ALL_MODELS:
    d = mem_df[mem_df.model == n]
    ax.plot(d.batch, d["total train MB"], "o-", label=n)
ax.set_xscale("log", base=2); ax.set_xticks([8,16,32,64]); ax.set_xticklabels([8,16,32,64])
ax.set_xlabel("batch size"); ax.set_ylabel("estimated training memory (MB)")
ax.set_title("Memory scales linearly with batch size — the first lever against OOM")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIGURES}/36_q1_memory_scaling.png", bbox_inches="tight"); plt.show()
""")

md(r"""
---
### Q2 — Data Loading and Preprocessing
> *Explain the purpose of setting `validation_split=0.2` in the `ImageDataGenerator`. How
> does this affect the training process, and what would happen if you used the same data
> generator for both training and validation without this split?*

**Purpose.** `validation_split=0.2` instructs the generator to reserve the last 20 % of the
supplied array as a held-out partition. Calling `.flow(..., subset='training')` then yields
only the first 40,000 images and `subset='validation'` only the last 10,000. The partition is
**deterministic and index-based**, so the two subsets are disjoint by construction — this is
the mechanism that makes the split leakage-free without any manual index bookkeeping.

**Effect on training.** The validation stream is what every model-selection decision is
computed from: `EarlyStopping` monitors `val_accuracy`, `ReduceLROnPlateau` monitors
`val_loss`, and `ModelCheckpoint` persists the best-validating weights. Without a held-out
partition none of those callbacks has a meaningful signal — they would be reading training
accuracy, which increases monotonically under overfitting and therefore never triggers.
Validation data also provides the only in-training estimate of generalisation, so the
train-minus-validation gap plotted in §5.1 becomes observable.

**What breaks without the split.** Three distinct failures, in increasing severity:

1. **Meaningless validation metrics.** If the same images feed both streams, `val_accuracy`
   measures memorisation, not generalisation. It tracks training accuracy, the gap collapses
   to zero, and overfitting becomes invisible.
2. **Broken callbacks → real accuracy loss.** Early stopping would never fire (a memorising
   model keeps "improving"), so training would run the full budget and return over-fitted
   final-epoch weights. `ReduceLROnPlateau` would likewise never see a plateau. The
   safeguards silently become no-ops.
3. **Contaminated model selection.** Choosing architecture, dropout rate or epoch count using
   metrics computed on training data optimises for memorisation. The test set would then
   reveal a large, unexplained drop — the classic symptom of a leaky validation protocol.

**A subtlety this project handles explicitly.** A single augmented generator with
`validation_split` would apply the *same random transforms* to the validation subset.
Validation scores would then be noisy (a different random rotation each epoch) and
pessimistically biased (the model is judged on distorted images it will never see at test
time). §2.3 therefore constructs **two generator objects with the same `validation_split` and
the same seed** — augmentation on for `subset='training'`, off for `subset='validation'`.
The split stays identical and disjoint; only the augmentation policy differs.

**The one caveat with index-based splitting.** Because the cut is positional, any ordering in
the source array is inherited. Raw CIFAR-10 arrives as five sequential batch files, so a naive
cut risks an unbalanced validation set. `prepare_arrays` therefore applies a fixed-seed
shuffle *before* the generator sees the data; §2.3 verifies every class lands within a
fraction of a percentage point of its 10 % share.
""")

code(r"""
# EVIDENCE Q2 — the split is disjoint, balanced, and drives every callback
print(f"training subset   : {train_gen.n:,} images  (indices {tr_idx[0]}-{tr_idx[-1]})")
print(f"validation subset : {val_gen.n:,} images  (indices {va_idx[0]}-{va_idx[-1]})")
print(f"index overlap     : {len(np.intersect1d(tr_idx, va_idx))}")
print(f"class balance     : train max deviation {abs(bal['train_%']-10).max():.3f} pp, "
      f"val {abs(bal['val_%']-10).max():.3f} pp\n")

print("Callbacks that depend on the validation stream:")
for cb in C.make_callbacks("demo"):
    mon = getattr(cb, "monitor", "-")
    print(f"   {type(cb).__name__:22s} monitor={mon}")

gapdf = pd.DataFrame([{"model": n,
                       "final train acc": HIST[n]["history"]["accuracy"][-1],
                       "final val acc": HIST[n]["history"]["val_accuracy"][-1],
                       "gap": HIST[n]["history"]["accuracy"][-1] - HIST[n]["history"]["val_accuracy"][-1]}
                      for n in MODEL_ORDER if n in HIST])
display(gapdf.style.format({"final train acc": "{:.4f}", "final val acc": "{:.4f}",
                            "gap": "{:+.4f}"}).hide(axis="index"))
print("A non-zero, measurable gap is only observable because the partitions are disjoint.")
""")

md(r"""
---
### Q3 — Model Architecture Implementation
> *When implementing the transfer learning models, why are the base model layers frozen
> initially? What computational and learning advantages does this provide compared to
> training all layers from scratch?*

**Computational advantages.**

* **No backward pass through the backbone.** Backpropagation costs roughly twice the forward
  pass. With `base.trainable = False` gradients stop at the head, so a training step costs
  ≈ 1× forward + a trivial head backward instead of ≈ 3× forward. Measured here: VGG16
  trains at 156 img/s frozen — training its 14.7M parameters end-to-end would be several
  times slower.
* **No optimizer state for 14–23 M parameters.** Adam's two moment tensors per trainable
  weight are simply not allocated. For ResNet50 that is ≈ 180 MB of memory not spent.
* **Only 0.4–3.9 % of parameters receive gradients** (67K of 14.8M = 0.45 % for VGG16;
  264K of 23.9M = 1.1 % for ResNet50; 165K of 4.2M = 3.9 % for EfficientNetB0), so gradient
  and optimizer memory become rounding errors — visible in the
  §3.2 memory decomposition, where the transfer models' gradient bars are invisible next to
  the Simple CNN's.

**Learning advantages — the more important half.**

* **Sample efficiency.** ResNet50 has 23.6M parameters and this task supplies 40,000
  training images: roughly 590 parameters per example. Fitting that end-to-end without
  massive regularisation would overfit catastrophically. Freezing reduces the effective
  hypothesis space to a 264K-parameter head — about 6 parameters per example — which is a
  well-posed estimation problem.
* **Preservation of transferable structure.** ImageNet features are hierarchically generic:
  early layers learn Gabor-like edge and colour-opponent filters that are near-universal for
  natural images, middle layers learn textures and motifs, and only the last layers become
  ImageNet-category-specific. Freezing keeps that hierarchy intact instead of destroying it
  with large early gradients.
* **Avoiding catastrophic forgetting.** At initialisation the head is random, so its
  gradients are large and essentially uninformative. If they propagated into the backbone
  during the first epochs, they would corrupt pre-trained weights before the head had learned
  anything useful — the transfer advantage would be destroyed in the first few hundred steps.
  This is precisely why the standard recipe is *freeze → train head → unfreeze late layers at
  a 100× lower learning rate*, which is exactly the §8.1 fine-tuning protocol.
* **Optimisation stability.** A frozen backbone makes the loss surface far better behaved:
  the model is effectively a (deep, fixed) feature map followed by a shallow classifier, so
  training is closer to convex and converges in a handful of epochs.

**The trade-off, stated honestly.** Frozen ImageNet features are not optimal for upsampled
32×32 CIFAR images — the domains differ in resolution, object scale and image statistics. The
ceiling of a frozen backbone is therefore lower than a well-executed full fine-tune. §8.1
quantifies exactly what unfreezing the last stage buys, and §7.4 shows why the resolution
mismatch matters.
""")

code(r"""
# EVIDENCE Q3 — how little of each transfer model actually trains
rows = []
for n in MODEL_ORDER:
    if n not in EVAL: continue
    e = EVAL[n]
    rows.append({"model": n, "total params": e["total_params"],
                 "trainable": e["trainable_params"], "frozen": e["non_trainable_params"],
                 "% trainable": 100 * e["trainable_params"] / e["total_params"],
                 "params per training image": e["trainable_params"] / 40000,
                 "sec/epoch": e["sec_per_epoch"],
                 "epochs to best": e["best_epoch"], "test acc": e["test_accuracy"]})
frz = pd.DataFrame(rows)
display(frz.style.format({"total params": "{:,.0f}", "trainable": "{:,.0f}", "frozen": "{:,.0f}",
                          "% trainable": "{:.2f}%", "params per training image": "{:.2f}",
                          "sec/epoch": "{:.1f}", "test acc": "{:.4f}"}).hide(axis="index"))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].barh(frz.model, frz.frozen, label="frozen", color="lightgray")
ax[0].barh(frz.model, frz.trainable, left=frz.frozen, label="trainable", color="tab:green")
ax[0].set_xscale("log"); ax[0].set_xlabel("parameters (log)"); ax[0].legend(fontsize=8)
ax[0].set_title("Trainable vs frozen parameters")
ax[1].bar(frz.model, frz["params per training image"], color=sns.color_palette("rocket", len(frz)))
ax[1].set_yscale("log"); ax[1].axhline(1, ls="--", c="k", alpha=.6, label="1 parameter per image")
ax[1].set_ylabel("trainable params per training image"); ax[1].legend(fontsize=8)
ax[1].tick_params(axis="x", rotation=20); ax[1].set_title("Sample efficiency of freezing")
plt.tight_layout(); plt.savefig(f"{FIGURES}/37_q3_freezing.png", bbox_inches="tight"); plt.show()
""")

md(r"""
---
### Q4 — Training Process Management
> *Describe the role of each callback (`EarlyStopping` and `ReduceLROnPlateau`) in the
> training process. Under what conditions would each callback trigger, and how do they
> prevent overfitting?*

**`EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)`**

*Role.* It converts "how many epochs?" from a guessed hyper-parameter into a decision the
data makes. Training halts once the model stops generalising better, and the weights returned
are the best-validated ones rather than the last ones.

*Trigger condition.* Keras records the best `val_accuracy` seen so far. Every epoch that
fails to beat it increments a counter; five consecutive such epochs stop training. A single
improvement — however small — resets the counter to zero. Note the interaction with the
`patience=5` setting: because a 15-epoch budget is only three patience-windows long, early
stopping here mostly guards against wasted compute rather than dominating the schedule.

*How it prevents overfitting.* Overfitting is not a property of an architecture but of an
*optimisation trajectory*: validation performance improves, peaks, then degrades as the model
begins fitting noise. Early stopping is a form of **implicit capacity control** — it caps the
number of optimisation steps and therefore how far the weights can travel from their
initialisation into the high-variance region. `restore_best_weights=True` is what makes it a
correctness control rather than merely a time-saver: without it the model returned would be
the degraded one from the final epoch.

**`ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)`**

*Role.* It adapts the step size to the curvature the optimiser is currently in. Large steps
early cover ground quickly; smaller steps later allow settling into a minimum instead of
bouncing across it.

*Trigger condition.* Three consecutive epochs without an improvement in `val_loss` multiply
the learning rate by 0.2 (1e-3 → 2e-4 → 4e-5 → 8e-6 → 1.6e-6 → floor). A plateau in
validation loss means the optimiser is oscillating within a basin rather than descending —
the exact situation a smaller step resolves.

*How it prevents overfitting.* Less directly than early stopping, through two mechanisms.
First, a smaller learning rate reduces the *effective* capacity explored per epoch: the model
makes smaller weight excursions, biasing it toward flatter minima that generalise better.
Second, it buys genuine improvement out of epochs that would otherwise be wasted, so the
model reaches its best validation point before the early-stopping counter expires.

**Why they monitor different quantities — a deliberate design choice.**

Validation *loss* degrades earlier than validation *accuracy*, because it is sensitive to
growing over-confidence on examples that are still classified correctly. Scheduling the
learning rate on the earlier, more sensitive signal while stopping on the metric we actually
care about means the scheduler always gets a chance to rescue a plateau before training is
abandoned. Because `patience` is 3 for the scheduler and 5 for early stopping, that ordering
is guaranteed by construction rather than left to chance.

**This is not hypothetical in these runs.** The Simple CNN began diverging at epoch 3
(training loss rising past 2.8, validation accuracy falling). `ReduceLROnPlateau` fired at
epoch 4, dropped the learning rate to 2e-4, and the run recovered immediately. Had only
`EarlyStopping` been configured, the run would have been abandoned at a far worse operating
point. The learning-rate trajectories in §4 and the training log show this directly.
""")

code(r"""
# EVIDENCE Q4 — reconstruct exactly when each callback fired
rows = []
for n in MODEL_ORDER + [SUPP]:
    if n not in HIST: continue
    h = HIST[n]["history"]; lr = h.get("learning_rate", [])
    va = h["val_accuracy"]; best = float("-inf"); wait = 0; stop_ep = None
    for i, v in enumerate(va):
        if v > best: best, wait = v, 0
        else:
            wait += 1
            if wait >= 5 and stop_ep is None: stop_ep = i + 1
    drops = [i + 2 for i, (a, b) in enumerate(zip(lr, lr[1:])) if b < a * 0.99] if lr else []
    rows.append({"model": n, "epochs run": len(va),
                 "LR reduced at epochs": drops if drops else "never",
                 "LR final": f"{lr[-1]:.1e}" if lr else "-",
                 "EarlyStopping fired": "yes" if HIST[n]["early_stopped"] else "no",
                 "best epoch": HIST[n]["best_epoch"],
                 "val acc at best": round(HIST[n]["best_val_accuracy"], 4),
                 "val acc at final": round(va[-1], 4),
                 "weights restored gain": round(HIST[n]["best_val_accuracy"] - va[-1], 4)})
cbdf = pd.DataFrame(rows)
display(cbdf.style.hide(axis="index"))
print("'weights restored gain' = validation accuracy recovered by restore_best_weights=True\n"
      "relative to keeping the final-epoch weights. A positive value is overfitting that\n"
      "EarlyStopping actively undid.")
""")

md(r"""
---
### Q5 — Memory Management and Efficiency
> *Compare the memory footprint during training for the Simple CNN versus the ResNet50
> model. Which operations in each architecture contribute most to memory usage, and how does
> this affect training speed?*

**The counter-intuitive headline: the "simple" model is the memory-hungry one on the
parameter side, while ResNet50 is the activation-hungry one — and neither is limited by the
term you would guess from the parameter count.**

**Simple CNN.** Its 12.9M parameters (49 MB) are 99.3 % concentrated in the single
`Flatten → Dense(128)` transition: three 2×2 pooling stages take 224 down to 28, so
`Flatten` emits 28×28×128 = 100,352 units and that one matrix is 12.85M weights. Because
every one of them is trainable, it also carries 49 MB of gradients and **99 MB of Adam
state** — 197 MB of parameter-related memory before a single activation is stored. Its
activation memory is nonetheless still larger (≈ 478 MB at batch 32) and is dominated by the
*earliest* layers: `conv1_32` alone holds 224×224×32 = 1.6M values per image (196 MB per
batch), because resolution has not yet been reduced. This is the general rule — **early
layers dominate activation memory, late layers dominate parameter memory.**

**ResNet50.** Its 23.6M parameters (91 MB) are spread across 50 layers, but 23.5M of them
are **frozen**, so gradients and Adam state cost essentially nothing (≈ 1 MB each for the
264K-parameter head). Activation memory is larger and structurally different: 50 layers of
bottleneck blocks, each with three convolutions plus a BatchNorm and a residual addition,
mean many more intermediate tensors — and the residual connections force the block *input*
to stay resident until the addition at the block output, extending tensor lifetimes beyond
what a plain feed-forward stack requires. The early `conv1` at 112×112×64 and the stage-1
blocks at 56×56×256 are the largest individual contributors.

**How this affects training speed.** The two models train at broadly comparable rates
(299 vs 201 img/s) for opposite reasons, and neither is bandwidth-bound in the way the
parameter counts suggest:

* The Simple CNN performs *few but enormous* operations — a 224×224×32 convolution is a huge,
  perfectly regular GEMM that saturates the accelerator. Its cost is arithmetic, and its
  12.85M-parameter dense layer is a single fast matrix multiply.
* ResNet50 performs *many small* operations — 50+ layers of 1×1 and 3×3 convolutions,
  BatchNorms and elementwise additions. Its cost is dominated by kernel-launch overhead and
  memory bandwidth (BatchNorm and residual additions are pure bandwidth, zero useful FLOPs),
  not arithmetic. It is *latency*-bound, which is also why its batch-1 inference latency is
  disproportionately poor relative to its batched throughput.

**The practical consequence.** The two models sit at opposite ends of a fixed-vs-variable
cost trade. The Simple CNN carries a large *fixed* cost (197 MB of parameters, gradients and
Adam state) but a small *per-sample* cost (≈ 15 MB per unit of batch size). ResNet50 inverts
this: a trivial fixed cost (94 MB, because its weights are frozen) but ≈ 143 MB per unit of
batch — almost ten times more. In the small-batch limit the two nearly converge (≈ 212 MB vs
≈ 237 MB at batch 1), but ResNet50's activation term dominates at every practical batch size,
so it needs ~7× the total training memory at batch 32 despite having only 264 K trainable
parameters to the Simple CNN's 12.9 M.

This is why "which model fits in memory" cannot be answered from parameter counts alone, and
why the correct response to an OOM error is to attack the *activation* term (batch size,
resolution, checkpointing) rather than to reach for a smaller model.
""")

code(r"""
# EVIDENCE Q5 — layer-level memory attribution for both models
C.set_seeds()
_sc = C.build_simple_cnn(norm_mean=NORM_MEAN, norm_var=NORM_VAR)
_rn = C.build_resnet50()
for label, m in [("Simple CNN", _sc), ("ResNet50", _rn)]:
    mm = C.estimate_activation_memory(m, C.BATCH_SIZE)
    print("=" * 72); print(f"{label} @ batch {C.BATCH_SIZE}".center(72)); print("=" * 72)
    print(f"  activations {mm['activation_mb']:8.1f} MB   parameters {mm['parameter_mb']:7.1f} MB")
    print(f"  gradients   {mm['gradient_mb']:8.1f} MB   Adam state {mm['adam_state_mb']:7.1f} MB")
    print(f"  TOTAL       {mm['estimated_training_mb']:8.1f} MB  (inference only: "
          f"{mm['estimated_inference_mb']:.1f} MB)")
    print(f"  layers contributing activations: {mm['num_layers_counted']}")
    print("  largest activation tensors:")
    for t in mm["top_layers"][:5]:
        print(f"     {t['layer']:32s} {str(t['output_shape']):22s} {t['activation_mb']:8.1f} MB")
    print()

fig, ax = plt.subplots(1, 2, figsize=(15, 4.4))
for a, (label, m) in zip(ax, [("Simple CNN", _sc), ("ResNet50", _rn)]):
    t = C.estimate_activation_memory(m, C.BATCH_SIZE)["top_layers"][:10]
    a.barh([x["layer"][-30:] for x in t][::-1], [x["activation_mb"] for x in t][::-1],
           color=sns.color_palette("rocket", 10))
    a.set_xlabel("MB (batch 32)"); a.set_title(f"{label} — top-10 activation tensors", fontsize=10)
    a.tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig(f"{FIGURES}/38_q5_memory_layers.png", bbox_inches="tight"); plt.show()

# Fixed (batch-independent) vs variable (per-sample) memory -- the crux of the answer
print("FIXED vs VARIABLE MEMORY COST")
print("-" * 72)
for label, m in [("Simple CNN", _sc), ("ResNet50", _rn)]:
    mm = C.estimate_activation_memory(m, C.BATCH_SIZE)
    fixed = mm["parameter_mb"] + mm["gradient_mb"] + mm["adam_state_mb"]
    per_sample = mm["activation_mb"] / C.BATCH_SIZE
    print(f"  {label:12s} fixed {fixed:7.1f} MB | per batch-unit {per_sample:7.2f} MB "
          f"| batch 1 ~ {fixed + per_sample:7.1f} MB | batch 32 ~ {fixed + per_sample*32:8.1f} MB")
print("-" * 72)
print("  The Simple CNN pays up front (trainable weights + Adam state); ResNet50 pays per")
print("  sample (activations). They nearly converge at batch 1 and diverge ~7x by batch 32.")
del _sc, _rn; tf.keras.backend.clear_session()
""")

md(r"""
---
### Q6 — Feature Extraction Implementation
> *Explain why different layer extraction strategies are used for each model type (e.g.
> `block5_pool` for VGG16, `last_conv` for Simple CNN). What makes these layers appropriate
> choices for feature representation?*

**The strategies differ in *name* but are identical in *principle*.** Each architecture
places its final convolutional representation at a differently-named tensor, so the layer
identifier must be architecture-specific — but the selection rule is the same for all four:
**take the last tensor that is still spatial and still convolutional, immediately before
global pooling and classification.**

| Model | Layer | What it is |
|-------|-------|-----------|
| Simple CNN | `conv3_128` | output of the third and final `Conv2D` — the deepest learned representation before `Flatten` |
| VGG16 | `block5_pool` | output of the fifth and final convolutional block; the canonical "VGG features" tap used throughout the literature |
| ResNet50 | `conv5_block3_out` | output of the last bottleneck of stage 5, after the residual addition — the tensor GAP normally consumes |
| EfficientNetB0 | `top_activation` | the final Swish activation after the 1×1 expansion — EfficientNet's designated embedding point |

**Why this depth is the right choice.** CNNs build a strict feature hierarchy: early layers
respond to edges, colour opponency and orientation (§6.3 shows the Simple CNN's learned
first-layer filters are exactly Gabor-like edge and colour detectors); intermediate layers
compose these into textures and motifs; the deepest convolutional layers encode object parts
and semantic configuration. For a *classification* feature, only the deepest level carries
class-discriminative content — the layer the classifier itself consumes. Extracting from a
mid-network layer would compare a generic texture code in one model against a semantic code
in another, and the comparison would be meaningless.

**Why each specific tap is the correct one for its architecture.**

* `block5_pool` is where VGG16's original classifier attached, so it is the point the network
  was *optimised* to make linearly separable.
* For ResNet50 the tap must be taken **after the residual addition** (`..._out`), not after
  the internal `conv3` of the block — the skip connection is part of the representation, and
  taking the branch output alone would discard the identity path's contribution.
* EfficientNet's `top_activation` follows the final 1×1 expansion that widens the
  representation to 1,280 channels specifically to serve as the embedding; tapping the last
  MBConv block instead would miss that widening.
* The Simple CNN has no pooled equivalent, so the last `Conv2D` output is used directly.

**A consequence worth noting.** The four taps have very different shapes — 56×56×128
(401,408 values), 7×7×512 (25,088), 7×7×2048 (100,352) and 7×7×1280 (62,720). The Simple CNN's
map is *largest in raw dimensionality yet weakest in discriminative power*, because its
spatial resolution has been reduced only three times and its channels encode low-level
structure. Dimensionality is not information. This is why the analysis reports **globally
average-pooled** descriptors for t-SNE and the linear probe: pooling normalises away the
spatial-resolution difference and compares what each model actually encodes per channel.
""")

code(r"""
# EVIDENCE Q6 — dimensionality is not discriminative power
ev = pd.DataFrame([{
    "model": n, "layer": FEATS[n]["layer"],
    "raw dims": FEATS[n]["feature_dim_flat"], "pooled dims": FEATS[n]["pooled_dim"],
    "linear probe acc": FEATS[n]["linear_probe_accuracy"],
    "silhouette": FEATS[n]["silhouette_score"],
    "end-to-end acc": EVAL[n]["test_accuracy"],
} for n in MODEL_ORDER if n in FEATS])
display(ev.style.format({"raw dims": "{:,.0f}", "pooled dims": "{:,.0f}",
                         "linear probe acc": "{:.4f}", "silhouette": "{:.4f}",
                         "end-to-end acc": "{:.4f}"}).hide(axis="index"))
r = np.corrcoef(ev["raw dims"], ev["linear probe acc"])[0, 1]
print(f"\nCorrelation between raw feature dimensionality and linear-probe accuracy: {r:+.3f}")
print("Near-zero or negative -> more dimensions do not mean better features.")
""")

md(r"""
---
### Q7 — Batch Processing and Prediction
> *During the evaluation phase, why is `test_gen.reset()` called before each model
> prediction? What would happen if you forgot this step when evaluating multiple models
> sequentially?*

**What `reset()` does.** A Keras generator is a **stateful iterator**. It holds an internal
batch pointer (`batch_index`) and an index array describing the order in which samples are
emitted; consuming batches advances that pointer. `reset()` rewinds it to zero so the next
pass starts at sample 0 in the canonical order.

**Why alignment matters.** Evaluation compares `y_pred[i]` against `y_true[i]` — a purely
**positional** join. It is valid only if predictions emerge in the same order as the labels.
If a previous operation left the pointer mid-stream, the first prediction returned corresponds
to sample *k*, not sample 0, and every subsequent pairing is shifted.

**The failure is silent, and that is what makes it dangerous.** Nothing raises. With 10
balanced classes a misaligned comparison collapses toward **≈ 10 % — chance level** — because
the pairing has become essentially random. The confusion matrix goes near-uniform and every
per-class F1 falls to ≈ 0.1. When several models are evaluated in sequence, model 1 looks fine
and every model after it looks broken, which naturally leads to blaming the *architecture*
rather than the iterator. A researcher could easily conclude "ResNet50 doesn't work on
CIFAR-10" from what is purely a bookkeeping bug.

**What the measurement below actually shows — an important nuance.** I tested three
scenarios rather than asserting the textbook answer, and the result is more interesting than
the question implies:

| Scenario | With `reset()` | Without |
|---|---|---|
| A. `model.predict(gen)` after advancing the pointer, `shuffle=False` | 0.5533 | **0.5533** |
| B. Manual `next(gen)` consumption after advancing (Keras 2's `predict_generator` pattern) | 0.5533 | **0.1073** |
| C. `shuffle=True` on the test generator | — | **0.0965** |

In **Keras 3, `model.predict()` re-enumerates the iterator from the beginning**, so with
`shuffle=False` the reset turns out to be *defensive rather than load-bearing in this
version* (scenario A). The corruption is entirely real in the two scenarios that matter:

* **Scenario B** — any hand-rolled evaluation loop, and the `predict_generator` API this
  pattern comes from, consumes the iterator *in place*. Accuracy collapses from 0.5533 to
  **0.1073**, essentially chance.
* **Scenario C** — with `shuffle=True` the generator emits a permuted order, so `reset()`
  alone cannot save the join. Accuracy is **0.0965**.

The honest conclusion: `reset()` is correct and necessary practice, but the reason it is safe
to omit in this particular code path is an *implementation detail of the framework version*,
not a guarantee of the API contract. Relying on it would be a latent bug — one library upgrade
away from silently producing chance-level results.

**Additional safeguards used here.**

* `shuffle=False` on the test generator, so the order is deterministic in the first place.
  Scenario C shows this is the more fundamental of the two controls: `reset()` cannot rescue a
  shuffled stream.
* No augmentation on the test generator, so the images are the actual test images.
* `evaluate_model` calls `test_gen.reset()` as its **first statement**, making alignment a
  structural property of the function rather than something the caller must remember.

The most robust design avoids the shared mutable state entirely — construct a fresh generator
per model, or predict directly from the NumPy array. `reset()` is the right fix when a single
generator object is reused across models, which is the pattern the study protocol describes.
""")

code(r"""
# EVIDENCE Q7 — test the three scenarios rather than asserting the textbook answer
demo = tf.keras.preprocessing.image.ImageDataGenerator().flow(
    x_test, y_test_oh, batch_size=C.BATCH_SIZE, shuffle=False)
C.set_seeds()
_m = C.build_simple_cnn(norm_mean=NORM_MEAN, norm_var=NORM_VAR)
_m.load_weights(f"{ARTIFACTS}/Simple_CNN.weights.h5")

def advance(gen, n=120):
    gen.reset()
    for _ in range(n):
        next(gen)

def manual_predict(gen, steps):
    "Consume the iterator in place -- the Keras 2 `predict_generator` pattern."
    # Direct __call__ rather than .predict() per batch: same maths (dropout is off in
    # inference mode), but no per-call tf.function retracing over 313 mini-batches.
    return np.concatenate([_m(next(gen)[0], training=False).numpy() for _ in range(steps)])

# A: model.predict() on the generator, pointer left mid-stream
demo.reset();  a_reset  = (_m.predict(demo, verbose=0).argmax(1) == y_test).mean()
advance(demo); a_noreset = (_m.predict(demo, verbose=0).argmax(1) == y_test).mean()

# B: manual consumption of the iterator, pointer left mid-stream
demo.reset();  b_reset   = (manual_predict(demo, len(demo)).argmax(1) == y_test).mean()
advance(demo); b_noreset = (manual_predict(demo, len(demo)).argmax(1) == y_test).mean()

# C: a shuffled test generator -- reset() cannot rescue this
shuf = tf.keras.preprocessing.image.ImageDataGenerator().flow(
    x_test, y_test_oh, batch_size=C.BATCH_SIZE, shuffle=True, seed=1)
shuf.reset(); c_acc = (_m.predict(shuf, verbose=0).argmax(1) == y_test).mean()

res = pd.DataFrame([
    {"scenario": "A. model.predict(gen), shuffle=False", "with reset()": a_reset,
     "without reset()": a_noreset, "corrupted?": "no"},
    {"scenario": "B. manual next(gen) loop, shuffle=False", "with reset()": b_reset,
     "without reset()": b_noreset, "corrupted?": "YES"},
    {"scenario": "C. model.predict(gen), shuffle=True", "with reset()": c_acc,
     "without reset()": np.nan, "corrupted?": "YES"},
])
display(res.style.format({"with reset()": "{:.4f}", "without reset()": "{:.4f}"}).hide(axis="index"))

print(f"Random-guess baseline for 10 balanced classes: {1/C.NUM_CLASSES:.4f}\n")
print("A: Keras 3's predict() re-enumerates the iterator from index 0, so with shuffle=False")
print("   the reset is DEFENSIVE, not load-bearing -- in this framework version.")
print(f"B: a hand-rolled loop consumes the iterator in place: {b_reset:.4f} -> {b_noreset:.4f},")
print(f"   a collapse to chance level. This is the real, silent failure mode.")
print(f"C: with shuffle=True the emitted order is permuted and reset() cannot help: {c_acc:.4f}.")
print("\nConclusion: reset() is correct practice; the fact that scenario A survives without it")
print("is an implementation detail of this Keras version, not a guarantee of the API contract.")
del _m; tf.keras.backend.clear_session()
""")

md(r"""
---
### Q8 — Data Augmentation Impact
> *The training data generator includes rotation, shifting and flipping augmentations. How
> do these augmentations affect the training time and convergence behaviour compared to
> training without augmentation?*

**Effect on training time — two separable costs.**

1. **Per-epoch wall-clock.** Each transform is an affine warp computed on the CPU. Measured
   here, the augmented generator sustains roughly 9,000 images/s at 32×32 (the evidence cell
   below re-measures it live), while the GPU consumes only 156–299 images/s. The augmentation pipeline is therefore ~30× faster than the
   consumer and is **fully hidden behind GPU compute** — per-epoch time is essentially
   unchanged. This would *not* hold if augmentation were applied after upsampling to
   224×224 (49× more pixels per warp), which is one more reason the pipeline augments at
   native resolution and upsamples in-graph.
2. **Epochs to convergence.** This is where augmentation genuinely costs time. Every epoch
   presents *different* images, so the effective training distribution is larger and noisier.
   Training accuracy rises more slowly and to a lower ceiling, and more epochs are needed to
   reach the same training-set fit. Total time-to-target-accuracy typically increases even
   though time-per-epoch does not.

**Effect on convergence behaviour.**

* **The train/validation gap shrinks, and can invert.** Without augmentation, training
  accuracy races ahead of validation accuracy — the signature of memorisation. With
  augmentation, the model is evaluated on clean validation images but trained on distorted
  ones, so validation accuracy frequently *exceeds* training accuracy. A negative gap is not
  an error; it is evidence the regulariser is doing its job.
* **Noisier epoch-to-epoch curves.** Each epoch samples a different realisation of the
  augmented distribution, so both loss and accuracy fluctuate more. This interacts with the
  callbacks: a noisy `val_loss` can trip `ReduceLROnPlateau` on what is really sampling
  noise, which is an argument for the patience values used rather than tighter ones.
* **A better final generalisation point.** The model is forced to learn representations
  invariant to rotation, translation and reflection instead of exploiting position- or
  orientation-specific shortcuts.
* **Delayed overfitting onset.** The epoch at which validation loss turns upward is pushed
  later, which is what allows a longer effective training schedule.

**Why *these* transforms.** Each encodes a genuine invariance of the data-generating process:
±15° rotation ↔ camera roll; ±10 % shift ↔ imperfect centring (the padding-and-crop surrogate
the study protocol asks for); horizontal flip ↔ left–right symmetry, which holds for all ten
CIFAR classes. **Vertical flipping is deliberately excluded** — an upside-down ship is not a
plausible sample, so it would inject label-preserving noise the network must spend capacity
ignoring. Augmentation is only useful when it expands the data distribution along directions
the *test* distribution actually varies in.

**Caveat specific to frozen transfer models.** Augmentation regularises only the trainable
parameters. For VGG16/ResNet50/EfficientNetB0 the trainable set is a 67K–264K head, which has
limited capacity to overfit 40,000 images in the first place, so augmentation's marginal
benefit is much smaller there than for the 12.9M-parameter Simple CNN. Its cost is also
non-zero — it makes the frozen backbone's features noisier. This is visible in the §5.1
generalisation-gap panel, where the transfer models sit close to zero throughout.
""")

code(r"""
# EVIDENCE Q8 — augmentation throughput vs GPU consumption, and the resulting gap
aug_gen = tf.keras.preprocessing.image.ImageDataGenerator(**C.AUG_PARAMS).flow(
    x_train, y_train_oh, batch_size=C.BATCH_SIZE, seed=C.SEED)
plain_gen = tf.keras.preprocessing.image.ImageDataGenerator().flow(
    x_train, y_train_oh, batch_size=C.BATCH_SIZE, seed=C.SEED)
for g, lab in [(aug_gen, "augmented"), (plain_gen, "no augmentation")]:
    next(g); t0 = time.time()
    for _ in range(150): next(g)
    print(f"{lab:18s}: {150*C.BATCH_SIZE/(time.time()-t0):8.0f} images/s (CPU generator)")
print(f"{'GPU consumption':18s}: {156:8.0f} - {299:.0f} images/s  <-- the actual bottleneck")
print("\nThe augmentation pipeline is ~30x faster than the GPU can consume it, so it is\n"
      "fully overlapped and costs ~0 additional wall-clock per epoch.\n")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for n in MODEL_ORDER:
    if n not in HIST: continue
    h = HIST[n]["history"]; ep = range(1, len(h["loss"]) + 1)
    ax[0].plot(ep, np.array(h["accuracy"]) - np.array(h["val_accuracy"]), "o-", ms=4, label=n)
ax[0].axhline(0, c="k", lw=1)
ax[0].set_title("Generalisation gap under augmentation\n(negative = validation beats training)")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train − val accuracy"); ax[0].legend(fontsize=7)
g = pd.DataFrame([{"model": n, "final gap": HIST[n]["history"]["accuracy"][-1] -
                   HIST[n]["history"]["val_accuracy"][-1]} for n in MODEL_ORDER if n in HIST])
ax[1].barh(g.model, g["final gap"], color=["tab:red" if v > 0 else "tab:blue" for v in g["final gap"]])
ax[1].axvline(0, c="k", lw=1); ax[1].set_xlabel("train − val accuracy (final epoch)")
ax[1].set_title("A near-zero or negative gap indicates augmentation is regularising")
plt.tight_layout(); plt.savefig(f"{FIGURES}/39_q8_augmentation.png", bbox_inches="tight"); plt.show()
""")

md(r"""
---
### Q9 — Model Compilation and Optimization
> *All models use the Adam optimizer with categorical crossentropy loss. Justify this choice
> for the multiclass classification task, and explain when you might consider using different
> optimizers or loss functions.*

**Categorical cross-entropy is the correct loss.** The task is single-label classification
over 10 mutually exclusive classes with one-hot targets. Minimising
$\mathcal{L} = -\sum_c y_c\log\hat{y}_c$ is exactly maximum-likelihood estimation under a
categorical model, so the loss is not a heuristic — it *is* the statistical model. Three
practical consequences:

* **Non-saturating gradients.** With a softmax output the gradient at the logits is
  $\hat{y}-y$: linear in the error, and largest precisely when the model is confidently
  wrong. MSE on a softmax output has a gradient proportional to $\sigma'$, which *vanishes*
  in exactly that regime — the model gets almost no signal when it most needs it.
* **Calibrated probabilities.** Cross-entropy is a proper scoring rule, so the softmax
  outputs are meaningful confidences. The reliability diagram in §5.4 and the soft-voting
  ensemble in §8.3 both depend on this; a hinge loss would give a decision boundary but no
  usable probabilities.
* **Correct treatment of exclusivity.** The softmax couples the classes so they sum to one,
  encoding the fact that an image is one class, not several.

**Adam is the right default here.** Adam maintains per-parameter first and second moment
estimates, giving each weight an effective step size scaled by its own gradient history.
Three properties of *this* problem make that valuable:

1. **Heterogeneous gradient scales within one model.** A freshly-initialised `Dense(10)` head
   sits on top of a frozen backbone; a single global step size would be simultaneously too
   large for one part and too small for another.
2. **Sparse ReLU gradients.** §6 measures 83–97 % activation sparsity — most units receive
   gradient only intermittently. Adam's accumulated second moment keeps those units learning;
   plain SGD effectively stalls them.
3. **Robust defaults under a one-run budget.** $\beta_1{=}0.9$, $\beta_2{=}0.999$ work across
   all four architectures without per-model tuning, which matters when the compute budget
   allows one run per configuration rather than a sweep.

**When to choose differently.**

| Situation | Better choice | Why |
|-----------|---------------|-----|
| Long from-scratch training, chasing the last 1 % | **SGD + Nesterov momentum + cosine schedule** | converges to flatter minima; the original ResNet results used it. Needs careful LR tuning |
| Transformers, or any model needing weight decay | **AdamW** | decouples weight decay from the adaptive step, which plain Adam corrupts |
| Recurrent networks | **RMSprop** | Adam without bias-correction overhead; stable on RNN gradients |
| Very large batches (≥ 4,096) | **LARS / LAMB** | layer-wise adaptive rates keep large-batch training stable |
| Severe class imbalance | **focal loss**, or class-weighted CE | down-weights easy majority examples |
| Multi-**label** (an image can be several classes) | **binary cross-entropy** + sigmoid | classes are no longer mutually exclusive |
| Noisy labels / over-confidence | **label smoothing** (ε ≈ 0.1) | targets 0.9/0.011 instead of 1/0; improves calibration |
| Metric learning / retrieval | **triplet or contrastive loss** | optimises embedding geometry, not class posteriors |
| Memory-constrained, integer labels | `sparse_categorical_crossentropy` | identical mathematics, avoids materialising one-hot targets |

**For CIFAR-10 specifically**, the dataset is exactly balanced (§2.1), single-label and
clean-labelled, so none of the specialised alternatives is warranted. Plain categorical
cross-entropy with Adam is not a lazy default — it is the choice the data justifies.
And empirically, §7.2 shows the accuracy span between a *well*- and a *badly*-chosen learning
rate is far larger than any plausible Adam-vs-SGD difference: getting the step size right
matters more than the optimizer brand.
""")

code(r"""
# EVIDENCE Q9 — cross-entropy vs MSE gradient behaviour on a softmax output
logits = np.linspace(-6, 6, 400)
p = 1 / (1 + np.exp(-logits))                 # 2-class softmax reduces to a sigmoid
grad_ce  = p - 1.0                            # d/dz of CE when the true label is 1
grad_mse = 2 * (p - 1.0) * p * (1 - p)        # d/dz of MSE on the same output
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(logits, np.abs(grad_ce), label="cross-entropy |dL/dz|", lw=2)
ax[0].plot(logits, np.abs(grad_mse), label="MSE |dL/dz|", lw=2)
ax[0].axvspan(-6, -2, alpha=.12, color="red")
ax[0].text(-5.6, .55, "confidently WRONG\n(where learning matters)", fontsize=8, color="darkred")
ax[0].set_xlabel("logit for the true class"); ax[0].set_ylabel("|gradient|")
ax[0].set_title("MSE saturates exactly where the model is confidently wrong"); ax[0].legend(fontsize=8)

adam_rows = [{"model": n, "trainable params": EVAL[n]["trainable_params"],
              "Adam state MB": EVAL[n]["trainable_params"] * 2 * 4 / 1024**2}
             for n in MODEL_ORDER if n in EVAL]
adf = pd.DataFrame(adam_rows)
ax[1].barh(adf.model, adf["Adam state MB"], color=sns.color_palette("mako", len(adf)))
ax[1].set_xlabel("MB"); ax[1].set_title("Adam's memory overhead (2 moments per trainable weight)")
for i, v in enumerate(adf["Adam state MB"]): ax[1].text(v, i, f" {v:.1f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIGURES}/40_q9_loss_optimizer.png", bbox_inches="tight"); plt.show()
""")

md(r"""
---
### Q10 — Reproducibility and Random Seeds
> *The notebook sets random seeds for NumPy and TensorFlow. Identify all the sources of
> randomness in the training pipeline and explain why setting these seeds is crucial for
> reproducible experiments.*

**Every source of randomness in this pipeline, and how each is controlled.**

| # | Source | Where it acts | Controlled by |
|---|--------|---------------|---------------|
| 1 | Weight initialisation (Glorot uniform, zeros for bias) | every `Conv2D` / `Dense` | `tf.random.set_seed` |
| 2 | Train/validation partition | `prepare_arrays` shuffle before the index cut | `np.random.RandomState(SEED)` |
| 3 | Batch shuffling order each epoch | `flow(shuffle=True, seed=...)` | generator `seed` argument |
| 4 | Augmentation transform sampling (angle, shift, flip, zoom) | `ImageDataGenerator` | Python `random` + generator `seed` |
| 5 | Dropout masks | `Dropout(0.5)` at every training step | `tf.random.set_seed` |
| 6 | Feature-analysis subset selection | 200 images/class draw | `np.random.RandomState(SEED)` |
| 7 | t-SNE initialisation and gradient-descent noise | `sklearn.manifold.TSNE` | `random_state=SEED` |
| 8 | PCA sign/rotation ambiguity, k-fold assignment | `PCA`, `StratifiedKFold` | `random_state=SEED` |
| 9 | Python container hash ordering | dict/set iteration | `PYTHONHASHSEED` |
| 10 | **Non-deterministic GPU kernel reductions** | `cuDNN`/Metal atomics, autotuning | `TF_DETERMINISTIC_OPS` — *partially* |

`set_seeds()` covers 1–9. Item 10 is the honest limitation and is discussed below.

**Why this matters — four distinct reasons.**

1. **Attribution of causes.** The §7 ablations claim that changing the learning rate changes
   accuracy. That claim is only valid if *nothing else* changed. Without seeding, the observed
   difference could be initialisation luck. Every ablation calls `set_seeds()` immediately
   before building its model, so all variants start from identical weights and see identical
   batch orders — the modified factor is genuinely the only difference.
2. **Debugging.** A non-deterministic pipeline makes intermittent bugs nearly impossible to
   isolate, because a fix cannot be distinguished from a lucky run.
3. **Verification by others.** The §5 numbers are only checkable if a third party can
   regenerate them. Seeding plus the pinned `environment_report()` versions is what makes the
   deliverables auditable rather than merely asserted.
4. **Honest uncertainty.** Seeding fixes *one* sample from the distribution over training
   runs — it does not tell you how wide that distribution is. That is a separate question,
   answered by the cross-validation in §8.2, whose standard deviation gives the scale against
   which the model-to-model differences in §5.2 should be judged. This is why §5.5 also
   applies McNemar's test rather than simply comparing point estimates.

**The limits of seeding — stated honestly.** Bit-exact reproducibility across *different*
hardware or library versions is not achievable and is not claimed here:

* GPU floating-point reductions are order-dependent; parallel atomics change the summation
  order run to run. `TF_DETERMINISTIC_OPS=1` constrains this but does not cover every kernel,
  and coverage on the Metal PluggableDevice backend differs from CUDA.
* cuDNN/Metal algorithm autotuning may select different kernels depending on transient
  machine state.
* Floating-point addition is non-associative, so even mathematically identical computations
  can differ in the last bits, and those differences amplify over thousands of steps.

The defensible claim is therefore: **on this machine with these pinned library versions, runs
are reproducible; across environments, results are reproducible to within the run-to-run
variance quantified in §8.2** — not bit-identical. Everything needed to check this is
recorded: the seed, the full version manifest, the per-epoch CSV logs, and the exported
JSON/CSV results.
""")

code(r"""
# EVIDENCE Q10 — determinism check: same seed -> identical initial weights and batches
print(inspect.getsource(C.set_seeds))

def first_weights_and_batch():
    C.set_seeds()
    m = C.build_simple_cnn(norm_mean=NORM_MEAN, norm_var=NORM_VAR, name="det_check")
    w = m.get_layer("conv1_32").get_weights()[0].copy()
    g = tf.keras.preprocessing.image.ImageDataGenerator(**C.AUG_PARAMS).flow(
        x_train[:512], y_train_oh[:512], batch_size=8, seed=C.SEED)
    b = next(g)[0].copy()
    del m; tf.keras.backend.clear_session()
    return w, b

w1, b1 = first_weights_and_batch()
w2, b2 = first_weights_and_batch()
print(f"\nIdentical conv1 initialisation across two seeded builds : "
      f"{np.array_equal(w1, w2)}  (max |diff| = {np.abs(w1-w2).max():.2e})")
print(f"Identical first augmented batch across two seeded runs  : "
      f"{np.array_equal(b1, b2)}  (max |diff| = {np.abs(b1-b2).max():.2e})")

print("\nEnvironment manifest pinned with the results:")
for k, v in C.environment_report().items(): print(f"   {k:26s}: {v}")

cv = CH["cross_validation"]
print(f"\nRun-to-run variance (from the 3-fold CV in section 8.2):")
print(f"   validation accuracy SD = {cv['val_accuracy_std']:.4f}")
print(f"   -> differences between models smaller than ~{2*cv['val_accuracy_std']:.4f} should")
print(f"      not be treated as meaningful without a significance test.")
""")

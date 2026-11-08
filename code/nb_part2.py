# -*- coding: utf-8 -*-
"""Notebook builder -- Part 2: model implementation and training protocol."""
from nb_part1 import CELLS, md, code

md(r"""
---
# 3. Model Implementation

**Objective:** *Build a simple CNN; implement VGG16, ResNet50 and EfficientNetB0 with transfer
learning; justify the choice of optimizer and loss function.*

Every model shares an identical interface — it accepts a `32×32×3` tensor in `[0,1]` and
performs resizing plus model-specific preprocessing as its own first layers. That single
convention is what makes the comparison fair: one generator, one preprocessing contract,
one training loop, four architectures.
""")

code(r"""
# Reproduce the architecture definitions from capstone_lib.py so the notebook is
# self-documenting.
import inspect
for fn in (C.build_simple_cnn, C._transfer_model, C.build_vgg16,
           C.build_resnet50, C.build_efficientnet):
    print(inspect.getsource(fn))
    print("-" * 100)
""")

md(r"""
## 3.1 Model 1 — Simple Custom CNN (trained from scratch)

```
Input(32×32×3) → Resizing(224) → Normalization(train mean/std)
  → Conv2D(32,3×3,ReLU,same)  → MaxPooling2D(2×2)
  → Conv2D(64,3×3,ReLU,same)  → MaxPooling2D(2×2)
  → Conv2D(128,3×3,ReLU,same) → MaxPooling2D(2×2)
  → Flatten → Dense(128,ReLU) → Dropout(0.5) → Dense(10,softmax)
```

Three convolutional blocks with a doubling filter schedule (32 → 64 → 128), each halving the
spatial resolution — the classical "trade space for depth" pattern that lets receptive fields
grow while activation volume stays bounded.
""")

code(r"""
C.set_seeds()
simple_cnn = C.build_simple_cnn(norm_mean=NORM_MEAN, norm_var=NORM_VAR)
simple_cnn.summary(line_length=104)
""")

code(r"""
# --- Reconciling the parameter count with the study protocol ----------------
pc = C.param_counts(simple_cnn)
print(f"Simple CNN trainable parameters at 224x224 input : {pc['trainable_params']:,}")
print(f"Design target                                    : ~500,000 - 1,000,000\n")

rows = []
for sz in (32, 48, 64, 128, 224):
    m = C.build_simple_cnn(img_size=sz, norm_mean=NORM_MEAN, norm_var=NORM_VAR,
                           name=f"tmp{sz}")
    side = sz // 8                                    # three 2x2 pooling stages
    p = C.param_counts(m)
    rows.append({"input_px": sz, "feature_map_before_flatten": f"{side}x{side}x128",
                 "flatten_units": side * side * 128,
                 "dense_128_params": side * side * 128 * 128 + 128,
                 "total_params": p["total_params"],
                 "in_spec_500k_1M": 500_000 <= p["total_params"] <= 1_000_000})
    del m; tf.keras.backend.clear_session()
display(pd.DataFrame(rows).style.hide(axis="index"))

print(textwrap.dedent('''
    WHY THE COUNT EXCEEDS THE SPEC, AND WHAT WE DO ABOUT IT
    ------------------------------------------------------
    The study protocol simultaneously requires (a) a 224x224 model input and (b) a
    ~500K-1M parameter Simple CNN. Those two constraints are arithmetically
    incompatible for the prescribed 'Flatten -> Dense(128)' head: three 2x2 pooling
    stages reduce 224 to 28, so Flatten emits 28*28*128 = 100,352 units and the single
    Dense(128) layer alone costs 12.85M parameters -- 99.3% of the network.
    The ~500K-1M envelope corresponds to the SAME architecture fed at CIFAR's native
    32x32 (356K parameters), as the table shows.

    Rather than silently violate either constraint, both are reported:
      * Simple_CNN          -- 224x224 input, honouring the uniform-pipeline requirement
                               and used in the headline four-model comparison;
      * Simple_CNN_Native32 -- identical layer stack at 32x32, honouring the parameter
                               budget, trained and evaluated as a supplementary baseline.
    The comparison between them is itself informative (section 5.5) and is revisited
    quantitatively in the image-size ablation (section 7.4).
'''))
""")

md(r"""
## 3.2 Models 2–4 — Transfer learning with frozen ImageNet backbones

All three share the specified custom head:

```
frozen backbone → GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(10, softmax)
```

**Why `GlobalAveragePooling2D` rather than `Flatten`?** GAP collapses each `7×7×C` feature
map to a single `C`-vector, cutting the head from millions of parameters to tens of
thousands, and imposes a translation-invariance prior that suits a frozen extractor. It is also
why total parameter count is a poor predictor of training cost here: VGG16 carries 14.8 M
weights but trains only 67 K of them (0.45 %), so its per-step cost is set by forward-pass
arithmetic through the frozen backbone, not by its weight count.

**Why freeze?** With `base.trainable = False` the backbone becomes a fixed, deterministic
function; only ~67K–264K head parameters receive gradients. This (i) removes the backward
pass through 16–50 layers, (ii) makes 40K training images sufficient — fitting 23M ResNet50
parameters on 40K images would overfit catastrophically — and (iii) preserves the ImageNet
features, which is the entire point of transfer learning.

`base(x, training=False)` additionally pins BatchNormalization to inference mode, so
ResNet50 and EfficientNetB0 use their ImageNet running statistics instead of recomputing
them from 32-image CIFAR batches. Omitting this is the single most common transfer-learning
bug and silently degrades frozen-backbone accuracy.
""")

code(r"""
C.set_seeds()
vgg16_model = C.build_vgg16()
vgg16_model.summary(line_length=104)
print("\nFrozen VGG16 backbone — last 6 layers:")
base = next(l for l in vgg16_model.layers if isinstance(l, tf.keras.Model))
for l in base.layers[-6:]:
    print(f"   {l.name:24s} trainable={l.trainable}  output={tuple(l.output.shape[1:])}")
""")

code(r"""
C.set_seeds()
resnet_model = C.build_resnet50()
print(f"ResNet50: {len(next(l for l in resnet_model.layers if isinstance(l, tf.keras.Model)).layers)} "
      f"backbone layers")
resnet_model.summary(line_length=104)
""")

code(r"""
C.set_seeds()
effnet_model = C.build_efficientnet()
print(f"EfficientNetB0: {len(next(l for l in effnet_model.layers if isinstance(l, tf.keras.Model)).layers)} "
      f"backbone layers")
effnet_model.summary(line_length=104)
""")

code(r"""
# --- Side-by-side architectural comparison ---------------------------------
ALL_MODELS = {"Simple_CNN": simple_cnn, "VGG16_Transfer": vgg16_model,
              "ResNet50_Transfer": resnet_model, "EfficientNetB0_Transfer": effnet_model}

DESIGN = {
    "Simple_CNN":              ("scratch",  "plain 3-block conv stack",    "Flatten"),
    "VGG16_Transfer":          ("ImageNet", "3x3 conv stacks, no BN",      "GlobalAvgPool"),
    "ResNet50_Transfer":       ("ImageNet", "bottleneck residual blocks",  "GlobalAvgPool"),
    "EfficientNetB0_Transfer": ("ImageNet", "MBConv + squeeze-excitation", "GlobalAvgPool"),
}
rows = []
for n, m in ALL_MODELS.items():
    p = C.param_counts(m)
    mem = C.estimate_activation_memory(m, C.BATCH_SIZE)
    src, block, head = DESIGN[n]
    rows.append({
        "model": n, "weights": src, "core block": block, "pooling head": head,
        "total params": f"{p['total_params']:,}",
        "trainable": f"{p['trainable_params']:,}",
        "trainable %": f"{100*p['trainable_params']/p['total_params']:.1f}%",
        "feature layer": C.FEATURE_LAYERS[n],
        "activation MB (bs=32)": mem["activation_mb"],
        "est. train MB": mem["estimated_training_mb"],
    })
arch_tbl = pd.DataFrame(rows)
display(arch_tbl.style.hide(axis="index"))
arch_tbl.to_csv(f"{ARTIFACTS}/architecture_comparison.csv", index=False)
""")

code(r"""
# --- Where does training memory actually go? (feeds Question 5) ------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
mem_rows = []
for n, m in ALL_MODELS.items():
    mm = C.estimate_activation_memory(m, C.BATCH_SIZE)
    mem_rows.append({"model": n, "activations": mm["activation_mb"],
                     "parameters": mm["parameter_mb"], "gradients": mm["gradient_mb"],
                     "Adam state": mm["adam_state_mb"]})
mdf = pd.DataFrame(mem_rows).set_index("model")
mdf.plot(kind="barh", stacked=True, ax=axes[0], colormap="viridis")
axes[0].set_xlabel("MB (batch size 32, float32)"); axes[0].set_title("Estimated training memory decomposition")
axes[0].legend(fontsize=8)

top = C.estimate_activation_memory(simple_cnn, C.BATCH_SIZE)["top_layers"][:6]
axes[1].barh([t["layer"] for t in top][::-1], [t["activation_mb"] for t in top][::-1],
             color=sns.color_palette("rocket", 6))
axes[1].set_xlabel("MB"); axes[1].set_title("Simple CNN — largest activation tensors")
plt.tight_layout(); plt.savefig(f"{FIGURES}/05_memory_decomposition.png", bbox_inches="tight"); plt.show()
display(mdf.round(1))
""")

md(r"""
## 3.3 Optimizer and loss justification

### Loss — categorical cross-entropy

The task is single-label classification over 10 mutually exclusive classes with one-hot
targets, so the natural likelihood is categorical: minimising
$\mathcal{L} = -\sum_c y_c \log \hat{y}_c$ is exactly maximum-likelihood estimation under a
categorical model. Paired with a softmax output it has a decisive practical property: the
gradient at the logits reduces to $\hat{y} - y$. That is linear in the error and **cannot
saturate**, unlike MSE on a softmax output, whose gradient vanishes precisely when the model
is confidently wrong — the regime where learning matters most. Cross-entropy also produces
calibrated probabilities, which the soft-voting ensemble in §8.3 depends on.

*Alternatives and when they would apply:* `sparse_categorical_crossentropy` (identical
mathematics, integer labels, saves the one-hot memory); `binary_crossentropy` for
multi-**label** problems; focal loss under severe class imbalance; label smoothing to reduce
over-confidence. CIFAR-10 is exactly balanced and single-label, so plain categorical
cross-entropy is the correct default.

### Optimizer — Adam (lr = 0.001)

Adam maintains per-parameter first and second moment estimates, giving each weight an
effective learning rate scaled by its own gradient history. Three properties matter here:

1. **Heterogeneous gradient scales.** A single model spans a freshly-initialised
   `Dense(10)` head and a frozen backbone; a single global step size (plain SGD) would be
   simultaneously too large for one and too small for the other.
2. **Sparse ReLU gradients.** Section 6 measures 83–97 % activation sparsity — most units
   receive gradient only intermittently. Adam's accumulated second moment keeps those units
   learning; SGD effectively stalls them.
3. **Minimal tuning.** Adam's defaults ($\beta_1{=}0.9$, $\beta_2{=}0.999$) are robust,
   which matters when the compute budget allows one run per configuration, not a sweep.

*When to prefer something else:* SGD + Nesterov momentum with a cosine schedule generally
reaches a slightly better final optimum for long from-scratch training runs (it is what the
original ResNet paper used) and generalises marginally better; AdamW decouples weight decay
and is standard for transformers; RMSprop suits recurrent models. The cost is that SGD needs
a carefully tuned learning-rate schedule — the §7.2 ablation shows the accuracy span between
a good and a bad learning rate is far larger than the Adam-vs-SGD gap.

Adam's memory price is quantified above: two extra float32 tensors per trainable weight.
For the frozen transfer models that is negligible (≈ 0.5–2 MB); for the 12.9M-parameter
Simple CNN it is ≈ 99 MB — one more consequence of the `Flatten` head.
""")

# =========================================================================
md(r"""
---
# 4. Training Protocol & Callbacks

**Objective:** *Apply training callbacks (EarlyStopping, ReduceLROnPlateau); document training
epochs and stopping conditions.*
""")

code(r"""
print(inspect.getsource(C.make_callbacks))
print(inspect.getsource(C.train_model))
""")

md(r"""
### Callback configuration and interaction

| Callback | Monitor | Trigger | Effect |
|----------|---------|---------|--------|
| `EarlyStopping` | `val_accuracy` (max) | no improvement for **5** epochs | stop; **restore best weights** |
| `ReduceLROnPlateau` | `val_loss` (min) | no improvement for **3** epochs | `lr ← 0.2 × lr`, floor `1e-6` |
| `ModelCheckpoint` | `val_accuracy` (max) | every improvement | persist best weights to disk |
| `CSVLogger` | — | every epoch | immutable per-epoch audit record |

Two deliberate choices deserve defending.

**The two callbacks monitor different quantities.** `ReduceLROnPlateau` watches `val_loss`
and `EarlyStopping` watches `val_accuracy`. Validation loss degrades *earlier* than
validation accuracy once a model begins to over-fit — it is sensitive to growing
over-confidence on examples that are still classified correctly. Reducing the learning rate
on the earlier signal, while stopping on the metric we actually care about, means the
scheduler always gets at least one chance to rescue a plateau before early stopping fires.
Because `patience` is 3 versus 5, that ordering is guaranteed rather than incidental.

**`restore_best_weights=True` is what makes early stopping a *correctness* control**, not
just a time saver. Without it the reported model would be the one from the final,
already-degrading epoch. With it, the weights evaluated on the test set are provably the
best-validated ones — and since selection uses validation data only, no test information
enters the choice.

Both callbacks combat over-fitting through different mechanisms: early stopping caps the
*number* of optimisation steps (implicit capacity control), while LR reduction shrinks the
*size* of each step, letting the optimiser settle into a flatter minimum instead of
oscillating across it.
""")

code(r"""
# --- Verbatim Keras training logs, as written during the runs --------------
MODEL_ORDER = ["Simple_CNN", "VGG16_Transfer", "ResNet50_Transfer", "EfficientNetB0_Transfer"]
SUPP = "Simple_CNN_Native32"

HIST, EVAL = {}, {}
for n in MODEL_ORDER + [SUPP]:
    hp, ep = f"{ARTIFACTS}/{n}_history.json", f"{ARTIFACTS}/{n}_eval.json"
    if os.path.exists(hp): HIST[n] = json.load(open(hp))
    if os.path.exists(ep): EVAL[n] = json.load(open(ep))
print(f"Loaded histories for: {list(HIST)}")

for n in MODEL_ORDER:
    p = f"{ARTIFACTS}/{n}_trainlog.txt"
    if not os.path.exists(p): continue
    print("\n" + "=" * 96)
    print(f"TRAINING LOG — {n}".center(96))
    print("=" * 96)
    print(open(p).read().strip())
""")

code(r"""
# --- Stopping-condition summary --------------------------------------------
rows = []
for n in MODEL_ORDER + [SUPP]:
    if n not in HIST: continue
    h = HIST[n]
    lr = h["history"].get("learning_rate", [])
    rows.append({
        "model": n,
        "epochs configured": h["epochs_configured"],
        "epochs run": h["epochs_run"],
        "stopped early": "yes" if h["early_stopped"] else "no (budget reached)",
        "best epoch": h["best_epoch"],
        "best val acc": round(h["best_val_accuracy"], 4),
        "final train acc": round(h["final_train_accuracy"], 4),
        "train-val gap": round(h["generalization_gap"], 4),
        "LR start": f"{lr[0]:.1e}" if lr else "-",
        "LR end": f"{lr[-1]:.1e}" if lr else "-",
        "LR reductions": int(sum(1 for a, b in zip(lr, lr[1:]) if b < a * 0.99)) if lr else 0,
        "min/epoch": round(h["sec_per_epoch"] / 60, 2),
    })
stop_tbl = pd.DataFrame(rows)
display(stop_tbl.style.hide(axis="index"))
stop_tbl.to_csv(f"{ARTIFACTS}/stopping_conditions.csv", index=False)
""")

code(r"""
# --- Learning-rate trajectories: did ReduceLROnPlateau actually fire? ------
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for n in MODEL_ORDER:
    if n not in HIST: continue
    lr = HIST[n]["history"].get("learning_rate")
    if lr: ax[0].plot(range(1, len(lr) + 1), lr, marker="o", ms=4, label=n)
    ax[1].plot(range(1, len(HIST[n]["history"]["val_loss"]) + 1),
               HIST[n]["history"]["val_loss"], marker="o", ms=4, label=n)
ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("learning rate")
ax[0].set_title("ReduceLROnPlateau trajectory (factor 0.2, patience 3)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation loss")
ax[1].set_title("Validation loss — the quantity the scheduler monitors")
for a in ax: a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIGURES}/06_lr_schedule.png", bbox_inches="tight"); plt.show()
""")

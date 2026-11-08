# -*- coding: utf-8 -*-
"""Notebook builder -- Part 4: parameter-modification study and challenge experiments."""
from nb_part1 import CELLS, md, code

md(r"""
---
# 7. Parameter-Modification Study

**Objective:** *Experiment with different batch sizes, learning rates, dropout, image sizes and
model depth; report the effect of these changes on performance.*

### Experimental design and its limitations — stated up front

Re-running the full 40k/10k protocol for all 13 variants below would cost roughly
20 GPU-hours. Instead every variant is trained on an **identical, fixed-seed 10,000-image
slice** of the training pool (8,000 train / 2,000 validation) for a **fixed 6-epoch budget
with no early stopping**, and evaluated on the **full, untouched 10,000-image test set**.

* Because data, seed, split and epoch budget are held constant and exactly **one factor
  changes per comparison**, the *relative* conclusions within each ablation are controlled
  and valid.
* Because the training set is 5× smaller and the schedule shorter, the *absolute*
  accuracies are systematically lower than the §5 headline results and must not be compared
  across the two studies. This is the honest cost of the compute budget, and it is why the
  conclusions below are phrased as rankings and trends rather than as absolute numbers.
* Fixing the epoch budget (rather than training to convergence) is what makes the timing
  comparisons interpretable: every variant performs the same number of passes over the same
  data, so differences in wall-clock time are attributable to the modified factor alone.
""")

code(r"""
PT = json.load(open(f"{ARTIFACTS}/param_tests.json"))
print(f"{len(PT)} parameter-modification runs loaded:")
for k in PT: print("   ", k)

def pt_frame(prefix, label_fn):
    rows = []
    for k, v in PT.items():
        if not k.startswith(prefix): continue
        rows.append({
            label_fn[0]: label_fn[1](k, v),
            "Test Accuracy": v["test_accuracy"], "F1 (weighted)": v["f1_weighted"],
            "Best Val Acc": v["best_val_accuracy"], "Final Train Acc": v["final_train_accuracy"],
            "Final Val Acc": v["final_val_accuracy"], "Overfit Gap": v["overfit_gap"],
            "Sec/Epoch": v["sec_per_epoch"], "Total Time (s)": v["training_time_sec"],
            "Steps/Epoch": v["steps_per_epoch"], "Params": v["total_params"],
            "Activation MB": v["memory"]["activation_mb"],
            "Est. Train MB": v["memory"]["estimated_training_mb"],
            "Peak RSS MB": v["peak_host_rss_mb"],
            "ms/img": v["inference_ms_per_image"],
        })
    return pd.DataFrame(rows)

PTFMT = {"Test Accuracy": "{:.4f}", "F1 (weighted)": "{:.4f}", "Best Val Acc": "{:.4f}",
         "Final Train Acc": "{:.4f}", "Final Val Acc": "{:.4f}", "Overfit Gap": "{:+.4f}",
         "Sec/Epoch": "{:.1f}", "Total Time (s)": "{:.1f}", "Params": "{:,.0f}",
         "Activation MB": "{:.1f}", "Est. Train MB": "{:.1f}", "Peak RSS MB": "{:.0f}",
         "ms/img": "{:.3f}"}

def curves(prefix, title, labeller, fname):
    ks = [k for k in PT if k.startswith(prefix)]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    for k in ks:
        h = PT[k]["history"]; ep = range(1, len(h["loss"]) + 1); lb = labeller(k, PT[k])
        ax[0].plot(ep, h["accuracy"], "o--", ms=3.5, alpha=.65, label=f"{lb} train")
        ax[0].plot(ep, h["val_accuracy"], "s-", ms=4, label=f"{lb} val")
        ax[1].plot(ep, h["val_loss"], "s-", ms=4, label=lb)
        ax[2].plot(ep, np.array(h["accuracy"]) - np.array(h["val_accuracy"]), "s-", ms=4, label=lb)
    ax[0].set_title(f"{title} — accuracy"); ax[1].set_title("validation loss")
    ax[2].set_title("generalisation gap (train − val)"); ax[2].axhline(0, c="k", lw=.8)
    for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIGURES}/{fname}", bbox_inches="tight"); plt.show()
""")

md(r"""
## 7.1 Batch size — 16 vs 32 vs 64

*Everything else fixed: Simple CNN, 224×224 input, Adam lr = 1e-3, 6 epochs.*
""")

code(r"""
lab = ("Batch Size", lambda k, v: v["batch_size"])
b_tbl = pt_frame("q1_batch", lab).sort_values("Batch Size")
display(b_tbl.style.format(PTFMT).hide(axis="index")
        .background_gradient(subset=["Test Accuracy"], cmap="Greens"))
curves("q1_batch", "Batch size", lambda k, v: f"bs={v['batch_size']}", "22_ablation_batch.png")

fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
ax[0].plot(b_tbl["Batch Size"], b_tbl["Test Accuracy"], "o-", ms=8); ax[0].set_title("accuracy vs batch size")
ax[1].plot(b_tbl["Batch Size"], b_tbl["Sec/Epoch"], "o-", ms=8, c="tab:red"); ax[1].set_title("seconds per epoch")
ax[2].plot(b_tbl["Batch Size"], b_tbl["Activation MB"], "o-", ms=8, c="tab:purple"); ax[2].set_title("activation memory (MB)")
for a in ax: a.set_xlabel("batch size"); a.set_xscale("log", base=2); a.set_xticks(b_tbl["Batch Size"]); a.set_xticklabels(b_tbl["Batch Size"])
plt.tight_layout(); plt.savefig(f"{FIGURES}/23_batch_tradeoff.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 7.2 Learning rate — 1e-4 vs 1e-3 vs 1e-2

*Everything else fixed: Simple CNN, 224×224, batch 32, 6 epochs.*
""")

code(r"""
lab = ("Learning Rate", lambda k, v: float(k.split("_")[-1]))
lr_tbl = pt_frame("q2_lr", lab).sort_values("Learning Rate")
display(lr_tbl.style.format({**PTFMT, "Learning Rate": "{:.0e}"}).hide(axis="index")
        .background_gradient(subset=["Test Accuracy"], cmap="Greens"))
curves("q2_lr", "Learning rate", lambda k, v: f"lr={k.split('_')[-1]}", "24_ablation_lr.png")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for k in [x for x in PT if x.startswith("q2_lr")]:
    h = PT[k]["history"]
    ax[0].plot(range(1, len(h["loss"]) + 1), h["loss"], "o-", ms=4, label=f"lr={k.split('_')[-1]}")
ax[0].set_yscale("log"); ax[0].set_title("Training loss (log scale) — divergence is visible here first")
ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8)
ax[1].semilogx(lr_tbl["Learning Rate"], lr_tbl["Test Accuracy"], "o-", ms=9)
ax[1].set_xlabel("learning rate (log)"); ax[1].set_ylabel("test accuracy")
ax[1].set_title("The classic inverted-U learning-rate response")
plt.tight_layout(); plt.savefig(f"{FIGURES}/25_lr_response.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 7.3 Dropout rate — 0.3 vs 0.5 vs 0.8

*Applied to the transfer-learning head (EfficientNetB0, frozen backbone), everything else fixed.*
EfficientNetB0 is used because dropout in these models regularises **only** the 128-unit head —
the frozen backbone cannot overfit — which isolates the effect cleanly.
""")

code(r"""
lab = ("Dropout", lambda k, v: float(k.split("_")[-1]))
dr_tbl = pt_frame("q3_dropout", lab).sort_values("Dropout")
display(dr_tbl.style.format(PTFMT).hide(axis="index")
        .background_gradient(subset=["Test Accuracy"], cmap="Greens"))
curves("q3_dropout", "Dropout rate", lambda k, v: f"p={k.split('_')[-1]}", "26_ablation_dropout.png")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(dr_tbl["Dropout"], dr_tbl["Test Accuracy"], "o-", ms=9, label="test accuracy")
ax[0].plot(dr_tbl["Dropout"], dr_tbl["Final Val Acc"], "s--", ms=7, label="final val accuracy")
ax[0].set_xlabel("dropout rate"); ax[0].legend(); ax[0].set_title("Accuracy vs dropout")
ax[1].bar(dr_tbl["Dropout"].astype(str), dr_tbl["Overfit Gap"],
          color=["tab:red" if g > 0 else "tab:blue" for g in dr_tbl["Overfit Gap"]])
ax[1].axhline(0, c="k", lw=.8); ax[1].set_xlabel("dropout rate")
ax[1].set_ylabel("train − val accuracy"); ax[1].set_title("Overfitting gap (positive = overfitting)")
plt.tight_layout(); plt.savefig(f"{FIGURES}/27_dropout_tradeoff.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 7.4 Image size — 128 vs 224 vs 384

*Simple CNN at all three sizes, plus VGG16 at 128 and 224 to separate the effect on a model
that was **pre-trained** at 224 from one that was not.*
""")

code(r"""
lab = ("Image Size", lambda k, v: int(k.split("_")[-1]))
sz_tbl = pt_frame("q4_imgsize", lab).sort_values("Image Size")
display(sz_tbl.style.format(PTFMT).hide(axis="index")
        .background_gradient(subset=["Test Accuracy"], cmap="Greens"))
vgg_tbl = pt_frame("q4_vgg_imgsize", lab).sort_values("Image Size")
print("\nVGG16 transfer model (pre-trained at 224x224):")
display(vgg_tbl.style.format(PTFMT).hide(axis="index"))

fig, ax = plt.subplots(1, 4, figsize=(19, 3.9))
ax[0].plot(sz_tbl["Image Size"], sz_tbl["Test Accuracy"], "o-", ms=9, label="Simple CNN")
if len(vgg_tbl): ax[0].plot(vgg_tbl["Image Size"], vgg_tbl["Test Accuracy"], "s-", ms=9, label="VGG16")
ax[0].set_title("accuracy vs input size"); ax[0].legend(fontsize=8)
ax[1].plot(sz_tbl["Image Size"], sz_tbl["Sec/Epoch"], "o-", ms=9, c="tab:red", label="Simple CNN")
if len(vgg_tbl): ax[1].plot(vgg_tbl["Image Size"], vgg_tbl["Sec/Epoch"], "s-", ms=9, c="tab:orange", label="VGG16")
ax[1].set_title("seconds per epoch"); ax[1].legend(fontsize=8)
ax[2].plot(sz_tbl["Image Size"], sz_tbl["Params"], "o-", ms=9, c="tab:green")
ax[2].set_yscale("log"); ax[2].set_title("parameters (Flatten head scales with area)")
ax[3].plot(sz_tbl["Image Size"], sz_tbl["Est. Train MB"], "o-", ms=9, c="tab:purple")
ax[3].set_title("estimated training memory (MB)")
for a in ax: a.set_xlabel("input size (px)"); a.set_xticks(sz_tbl["Image Size"])
plt.tight_layout(); plt.savefig(f"{FIGURES}/28_ablation_imgsize.png", bbox_inches="tight"); plt.show()

base = sz_tbl[sz_tbl["Image Size"] == 224].iloc[0]
rel = sz_tbl.assign(**{
    "relative compute (px²)": (sz_tbl["Image Size"] ** 2 / 224 ** 2).round(2),
    "relative time": (sz_tbl["Sec/Epoch"] / base["Sec/Epoch"]).round(2),
    "accuracy delta vs 224": (sz_tbl["Test Accuracy"] - base["Test Accuracy"]).round(4)})
display(rel[["Image Size", "relative compute (px²)", "relative time",
             "Test Accuracy", "accuracy delta vs 224"]].style.hide(axis="index"))
""")

md(r"""
## 7.5 Architecture depth — adding a `Conv2D(256) → MaxPooling2D` block

*Simple CNN with three blocks vs four, everything else fixed. t-SNE of the last
convolutional layer is compared for both.*
""")

code(r"""
d_tbl = pt_frame("q5_depth", ("Variant", lambda k, v: "4 blocks (+Conv2D 256)"
                              if "plus4" in k else "3 blocks (baseline)"))
display(d_tbl.style.format(PTFMT).hide(axis="index"))

if "q5_depth_base" in PT and "q5_depth_plus4" in PT:
    b, p = PT["q5_depth_base"], PT["q5_depth_plus4"]
    delta = pd.DataFrame([
        ("Total parameters", f"{b['total_params']:,}", f"{p['total_params']:,}",
         f"{(p['total_params']-b['total_params'])/b['total_params']*100:+.1f}%"),
        ("Flatten input units", f"{b['feature_stats']['feature_dim_flat']:,}",
         f"{p['feature_stats']['feature_dim_flat']:,}", ""),
        ("Test accuracy", f"{b['test_accuracy']:.4f}", f"{p['test_accuracy']:.4f}",
         f"{p['test_accuracy']-b['test_accuracy']:+.4f}"),
        ("Weighted F1", f"{b['f1_weighted']:.4f}", f"{p['f1_weighted']:.4f}",
         f"{p['f1_weighted']-b['f1_weighted']:+.4f}"),
        ("Seconds / epoch", f"{b['sec_per_epoch']:.1f}", f"{p['sec_per_epoch']:.1f}",
         f"{(p['sec_per_epoch']-b['sec_per_epoch'])/b['sec_per_epoch']*100:+.1f}%"),
        ("Inference ms / image", f"{b['inference_ms_per_image']:.3f}",
         f"{p['inference_ms_per_image']:.3f}", ""),
        ("Feature sparsity %", f"{b['feature_stats']['sparsity_pct']:.1f}",
         f"{p['feature_stats']['sparsity_pct']:.1f}", ""),
        ("Overfit gap", f"{b['overfit_gap']:+.4f}", f"{p['overfit_gap']:+.4f}", ""),
    ], columns=["metric", "3 blocks", "4 blocks", "change"])
    display(delta.style.hide(axis="index"))
""")

code(r"""
# --- t-SNE comparison of the two depths ------------------------------------
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
sub_lab = y_test[:2000]
fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
depth_sep = []
for a, (key, title) in zip(ax, [("q5_depth_base", "3 blocks — conv3_128 (baseline)"),
                                ("q5_depth_plus4", "4 blocks — conv4_256 (+Conv2D 256)")]):
    p = f"{ARTIFACTS}/pt_{key}_pooled.npy"
    if not os.path.exists(p): a.axis("off"); continue
    P = np.load(p)
    emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
               random_state=C.SEED, max_iter=1000).fit_transform(
                   PCA(n_components=min(50, P.shape[1]), random_state=C.SEED).fit_transform(P))
    for c in range(10):
        m = sub_lab == c
        a.scatter(emb[m, 0], emb[m, 1], s=7, alpha=.65, color=pal10[c], label=C.CLASS_NAMES[c])
    s = C.separability(emb, sub_lab)
    depth_sep.append({"variant": title, **s})
    a.set_title(f"{title}\nsilhouette {s['silhouette_score']:.3f} | 10-NN {s['knn10_accuracy_2d']:.3f}",
                fontsize=10)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
ax[0].legend(markerscale=2.5, fontsize=7, ncol=2)
fig.suptitle("Effect of depth on feature separability (2,000 test images)", y=1.0)
plt.tight_layout(); plt.savefig(f"{FIGURES}/29_depth_tsne.png", bbox_inches="tight"); plt.show()
display(pd.DataFrame(depth_sep).style.format({"silhouette_score": "{:.4f}",
                                              "knn10_accuracy_2d": "{:.4f}"}).hide(axis="index"))
""")

code(r"""
# --- All 13 ablation runs in one view --------------------------------------
allpt = pd.DataFrame([{
    "run": k, "test acc": v["test_accuracy"], "F1": v["f1_weighted"],
    "params": v["total_params"], "s/epoch": v["sec_per_epoch"],
    "overfit gap": v["overfit_gap"], "est. train MB": v["memory"]["estimated_training_mb"],
} for k, v in PT.items()]).sort_values("test acc", ascending=False)
display(allpt.style.format({"test acc": "{:.4f}", "F1": "{:.4f}", "params": "{:,.0f}",
                            "s/epoch": "{:.1f}", "overfit gap": "{:+.4f}",
                            "est. train MB": "{:.0f}"}).hide(axis="index")
        .background_gradient(subset=["test acc"], cmap="RdYlGn"))
allpt.to_csv(f"{ARTIFACTS}/parameter_modifications.csv", index=False)
""")

# =========================================================================
md(r"""
---
# 8. Challenge Experiments

**Objective:** *Explore fine-tuning; attempt cross-validation; create ensembles; investigate
model compression or quantisation; measure inference speed for real-time deployment.*
""")

code(r"""
CH = json.load(open(f"{ARTIFACTS}/challenges.json"))
print("Challenge experiments completed:", list(CH))
""")

md(r"""
## 8.1 Fine-tuning the best transfer model

The best frozen transfer model's **final convolutional stage** is unfrozen and retrained for
5 epochs at `lr = 1e-5` — 100× below the head-training rate. Two deliberate constraints:

* **Only the last stage.** Early layers encode edges and textures that transfer unchanged;
  the late layers encode ImageNet-specific object parts, and those are what need to adapt to
  upsampled 32×32 CIFAR imagery.
* **BatchNormalization layers stay frozen.** Recomputing BN statistics from 32-image batches
  of a distribution the backbone was not fitted to is the standard way to destroy a working
  fine-tune. The base is also still called with `training=False`, so BN uses its ImageNet
  running statistics throughout.

A drastically lower learning rate is essential: at `1e-3` the large gradients from the
randomly-initialised head would overwrite the pre-trained weights in the first few steps —
"catastrophic forgetting" — and the transfer advantage would be lost entirely.
""")

code(r"""
ft = CH["finetuning"]
display(pd.DataFrame([
    ("Base model",              ft["base_model"]),
    ("Unfrozen stage",          ft["unfrozen_stage"]),
    ("Layers unfrozen",         ft["layers_unfrozen"]),
    ("Fine-tune learning rate", f"{ft['learning_rate']:.0e}"),
    ("Epochs",                  ft["epochs"]),
    ("Trainable params before", f"{ft['trainable_params_before']:,}"),
    ("Trainable params after",  f"{ft['trainable_params_after']:,}"),
    ("Test accuracy before",    f"{ft['accuracy_before']:.4f}"),
    ("Test accuracy after",     f"{ft['accuracy_after']:.4f}"),
    ("Absolute gain",           f"{ft['absolute_gain']:+.4f}"),
    ("Relative error reduction",f"{(ft['accuracy_after']-ft['accuracy_before'])/(1-ft['accuracy_before'])*100:+.1f}%"),
    ("Weighted F1 before/after",f"{ft['f1_before']:.4f} -> {ft['f1_after']:.4f}"),
    ("Fine-tune wall-clock",    f"{ft['finetune_time_sec']/60:.1f} min"),
], columns=["", "value"]).style.hide(axis="index"))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
h = ft["history"]; ep = range(1, len(h["loss"]) + 1)
ax[0].plot(ep, h["accuracy"], "o-", label="train"); ax[0].plot(ep, h["val_accuracy"], "s-", label="validation")
ax[0].set_title("Fine-tuning accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(ep, h["loss"], "o-", label="train"); ax[1].plot(ep, h["val_loss"], "s-", label="validation")
ax[1].set_title("Fine-tuning loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
ax[2].bar(["frozen", "fine-tuned"], [ft["accuracy_before"], ft["accuracy_after"]],
          color=["tab:gray", "tab:green"])
for i, v in enumerate([ft["accuracy_before"], ft["accuracy_after"]]):
    ax[2].text(i, v, f"{v:.4f}", ha="center", va="bottom")
ax[2].set_ylabel("test accuracy"); ax[2].set_title(f"{ft['base_model']} — effect of fine-tuning")
plt.tight_layout(); plt.savefig(f"{FIGURES}/30_finetuning.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 8.2 K-fold cross-validation

A single train/validation split gives a point estimate with no uncertainty attached.
3-fold stratified cross-validation of the Simple CNN over a 12,000-image subset yields a
**distribution** of accuracies, so the reported number can be quoted with a confidence
interval — and so that differences between models can be judged against run-to-run variance
rather than assumed to be real.
""")

code(r"""
cv = CH["cross_validation"]
display(pd.DataFrame(cv["folds"]).style.format({
    "val_accuracy": "{:.4f}", "test_accuracy": "{:.4f}",
    "test_f1_weighted": "{:.4f}", "time_sec": "{:.1f}"}).hide(axis="index"))
print(f"\nValidation accuracy : {cv['val_accuracy_mean']:.4f} +/- {cv['val_accuracy_std']:.4f} (SD)")
print(f"95% CI of the mean  : [{cv['val_ci95'][0]:.4f}, {cv['val_ci95'][1]:.4f}]")
print(f"Test accuracy       : {cv['test_accuracy_mean']:.4f} +/- {cv['test_accuracy_std']:.4f} (SD)")
print(f"Protocol            : k={cv['k']}, n={cv['n_samples']:,}, {cv['epochs_per_fold']} epochs/fold")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
f = pd.DataFrame(cv["folds"])
ax[0].bar(f.fold - .18, f.val_accuracy, .36, label="validation")
ax[0].bar(f.fold + .18, f.test_accuracy, .36, label="test")
ax[0].axhline(cv["val_accuracy_mean"], ls="--", c="k", alpha=.6, label="mean val")
ax[0].set_xlabel("fold"); ax[0].set_xticks(f.fold); ax[0].legend(fontsize=8)
ax[0].set_title("Per-fold accuracy")
ax[1].errorbar([0], [cv["val_accuracy_mean"]], yerr=[cv["val_accuracy_std"]], fmt="o", capsize=8, ms=10)
ax[1].errorbar([1], [cv["test_accuracy_mean"]], yerr=[cv["test_accuracy_std"]], fmt="s", capsize=8, ms=10)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["validation", "test"]); ax[1].set_xlim(-.5, 1.5)
ax[1].set_ylabel("accuracy"); ax[1].set_title("Mean ± 1 SD across folds")
plt.tight_layout(); plt.savefig(f"{FIGURES}/31_cross_validation.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 8.3 Ensembles

Four combination rules are compared. Soft voting averages the softmax vectors and therefore
uses each model's *confidence*; hard voting uses only the arg-max and discards it. Ensembling
helps exactly to the extent that members make **decorrelated** errors, so the error-overlap
analysis below is the diagnostic that explains the result.
""")

code(r"""
ens = CH["ensemble"]
rows = [{"strategy": "best individual (" + ens["best_individual"] + ")",
         "members": 1, "accuracy": ens["best_individual_accuracy"], "F1 (weighted)": np.nan}]
for k, v in ens["combinations"].items():
    rows.append({"strategy": k, "members": len(v["members"]),
                 "accuracy": v["accuracy"], "F1 (weighted)": v["f1_weighted"]})
edf = pd.DataFrame(rows)
edf["gain vs best individual"] = edf.accuracy - ens["best_individual_accuracy"]
display(edf.style.format({"accuracy": "{:.4f}", "F1 (weighted)": "{:.4f}",
                          "gain vs best individual": "{:+.4f}"}).hide(axis="index")
        .background_gradient(subset=["accuracy"], cmap="Greens"))

# Error-correlation: do the models fail on the same images?
names_e = [n for n in MODEL_ORDER if os.path.exists(f"{ARTIFACTS}/{n}_probs.npy")]
err = {n: (np.load(f"{ARTIFACTS}/{n}_probs.npy").argmax(1) != y_test) for n in names_e}
M = np.zeros((len(names_e), len(names_e)))
for i, a in enumerate(names_e):
    for j, b in enumerate(names_e):
        M[i, j] = (err[a] & err[b]).sum() / max((err[a] | err[b]).sum(), 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
sns.heatmap(M, annot=True, fmt=".3f", cmap="Oranges", ax=ax[0],
            xticklabels=[n.replace("_Transfer", "") for n in names_e],
            yticklabels=[n.replace("_Transfer", "") for n in names_e])
ax[0].set_title("Error overlap (Jaccard) — lower off-diagonal = more complementary")
ax[0].tick_params(axis="x", rotation=30)
ax[1].bar(range(len(edf)), edf.accuracy, color=sns.color_palette("crest", len(edf)))
ax[1].axhline(ens["best_individual_accuracy"], ls="--", c="r", label="best individual")
ax[1].set_xticks(range(len(edf))); ax[1].set_xticklabels(edf.strategy, rotation=35, ha="right", fontsize=7)
ax[1].set_ylabel("test accuracy"); ax[1].legend(fontsize=8); ax[1].set_title("Ensemble strategies")
plt.tight_layout(); plt.savefig(f"{FIGURES}/32_ensemble.png", bbox_inches="tight"); plt.show()

n_all_wrong = int(np.all(np.stack([err[n] for n in names_e]), axis=0).sum())
n_any_wrong = int(np.any(np.stack([err[n] for n in names_e]), axis=0).sum())
print(f"Images every model gets wrong : {n_all_wrong:,}  -> irreducible ensemble error "
      f"{n_all_wrong/len(y_test)*100:.2f}%  (ensemble ceiling {(1-n_all_wrong/len(y_test))*100:.2f}%)")
print(f"Images at least one gets wrong: {n_any_wrong:,}")
""")

md(r"""
## 8.4 Model compression and quantisation

Two independent compression axes are measured on the best model: **TFLite numeric
quantisation** (float32 → float16 → dynamic-range int8) and **post-training magnitude
pruning** of the dense head.
""")

code(r"""
cp = CH["compression"]
print(f"Target model: {cp['target_model']}")
print(f"Keras float32 weight footprint: {cp['keras_fp32_size_mb']:.1f} MB")
print(f"Baseline accuracy on the {cp['tflite'].get('float32', {}).get('n_eval_images', 1000)}-image "
      f"evaluation slice: {cp['baseline_accuracy_subset']:.4f}\n")

if "error" not in cp["tflite"]:
    q = pd.DataFrame([v for v in cp["tflite"].values()])
    q["size vs fp32"] = (q.size_mb / q.size_mb.iloc[0]).round(3)
    q["accuracy delta"] = (q.accuracy - q.accuracy.iloc[0]).round(4)
    display(q.style.format({"size_mb": "{:.2f}", "accuracy": "{:.4f}",
                            "latency_ms_per_image": "{:.2f}"}).hide(axis="index"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    ax[0].bar(q.variant, q.size_mb, color=sns.color_palette("mako", len(q))); ax[0].set_ylabel("MB")
    ax[0].set_title("TFLite model size")
    ax[1].bar(q.variant, q.accuracy, color=sns.color_palette("crest", len(q)))
    ax[1].set_ylim(0, 1); ax[1].set_title("Accuracy after quantisation")
    ax[2].bar(q.variant, q.latency_ms_per_image, color=sns.color_palette("flare", len(q)))
    ax[2].set_ylabel("ms/image"); ax[2].set_title("Single-image CPU latency")
    plt.tight_layout(); plt.savefig(f"{FIGURES}/33_quantisation.png", bbox_inches="tight"); plt.show()
else:
    print("TFLite conversion note:", cp["tflite"]["error"])

pr = pd.DataFrame(cp["dense_head_pruning"])
display(pr.style.format({"sparsity_target": "{:.0%}", "accuracy": "{:.4f}",
                         "nonzero_dense_weights": "{:,.0f}"}).hide(axis="index"))
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(pr.sparsity_target * 100, pr.accuracy, "o-", ms=9)
ax.set_xlabel("% of dense weights zeroed"); ax.set_ylabel("accuracy")
ax.set_title("Post-training magnitude pruning of the classifier head")
plt.tight_layout(); plt.savefig(f"{FIGURES}/34_pruning.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 8.5 Real-time inference and deployment suitability

Batch-1 latency is the metric that matters for interactive deployment — a camera frame
arrives alone, not in a batch of 32. Both are reported because the gap between them is
itself a deployment finding.
""")

code(r"""
lat = pd.DataFrame(CH["latency"]).T.reset_index().rename(columns={"index": "Model"})
# The nested dict yields object-dtype columns; cast the numeric ones so plotting works.
_num = [c for c in lat.columns if c not in ("Model", "realtime_30fps_capable")]
lat[_num] = lat[_num].apply(pd.to_numeric, errors="coerce")
lat["accuracy per MB"] = lat.test_accuracy / lat.model_size_mb
display(lat[["Model", "batch1_p50_ms", "batch1_p95_ms", "batch1_fps",
             "batch32_throughput_ips", "model_size_mb", "test_accuracy",
             "accuracy per MB", "realtime_30fps_capable"]].style.format({
    "batch1_p50_ms": "{:.2f}", "batch1_p95_ms": "{:.2f}", "batch1_fps": "{:.1f}",
    "batch32_throughput_ips": "{:.0f}", "model_size_mb": "{:.1f}",
    "test_accuracy": "{:.4f}", "accuracy per MB": "{:.4f}"}).hide(axis="index"))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
xs = range(len(lat))
ax[0].bar([i - .18 for i in xs], lat.batch1_p50_ms, .36, label="p50")
ax[0].bar([i + .18 for i in xs], lat.batch1_p95_ms, .36, label="p95")
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels([m.replace("_", "\n") for m in lat.Model], fontsize=7.5)
ax[0].set_ylabel("ms"); ax[0].set_title("Batch-1 latency"); ax[0].legend(fontsize=8)
ax[1].axhline(30, ls="--", c="r", label="30 fps real-time threshold")
ax[1].bar(xs, lat.batch1_fps, color=sns.color_palette("crest", len(lat)))
ax[1].set_xticks(list(xs)); ax[1].set_xticklabels([m.replace("_", "\n") for m in lat.Model], fontsize=7.5)
ax[1].set_ylabel("frames per second"); ax[1].set_title("Achievable frame rate"); ax[1].legend(fontsize=8)
ax[2].scatter(lat.model_size_mb, lat.test_accuracy, s=lat.batch1_fps * 3 + 40, alpha=.75,
              c=range(len(lat)), cmap="viridis")
for _, r in lat.iterrows():
    ax[2].annotate(r.Model.replace("_Transfer", ""), (r.model_size_mb, r.test_accuracy),
                   textcoords="offset points", xytext=(8, 4), fontsize=8)
ax[2].set_xlabel("model size (MB)"); ax[2].set_ylabel("test accuracy")
ax[2].set_title("Deployment envelope (marker size ∝ fps)")
plt.tight_layout(); plt.savefig(f"{FIGURES}/35_latency.png", bbox_inches="tight"); plt.show()

mobile = lat.sort_values("accuracy per MB", ascending=False).iloc[0]
print(f"\nBest accuracy-per-megabyte (mobile-deployment proxy): {mobile.Model} "
      f"— {mobile.test_accuracy:.4f} accuracy in {mobile.model_size_mb:.1f} MB "
      f"at {mobile.batch1_p50_ms:.1f} ms/frame.")
""")

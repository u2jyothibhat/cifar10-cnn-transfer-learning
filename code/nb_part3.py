# -*- coding: utf-8 -*-
"""Notebook builder -- Part 3: evaluation, results and feature analysis."""
from nb_part1 import CELLS, md, code

md(r"""
---
# 5. Evaluation & Results

**Objective:** *Report accuracy and F1 for all models; provide confusion matrices; plot
training curves; compare parameters, training time and inference time; identify and justify
the best-performing model.*

Every model is evaluated **once**, on the full untouched 10,000-image test set, with
`test_gen.reset()` called first so predictions align with the labels.
""")

code(r"""
print(inspect.getsource(C.evaluate_model))
""")

md("## 5.1 Training curves")

code(r"""
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for j, n in enumerate(MODEL_ORDER):
    if n not in HIST:
        for r in (0, 1): axes[r, j].axis("off")
        continue
    h = HIST[n]["history"]; ep = range(1, len(h["loss"]) + 1); best = HIST[n]["best_epoch"]
    axes[0, j].plot(ep, h["accuracy"], "o-", ms=4, label="train")
    axes[0, j].plot(ep, h["val_accuracy"], "s-", ms=4, label="validation")
    axes[0, j].axvline(best, ls="--", c="green", alpha=.7, label=f"best (ep {best})")
    axes[0, j].set_title(f"{n}\naccuracy", fontsize=10); axes[0, j].set_ylim(0, 1)
    axes[0, j].set_xlabel("epoch"); axes[0, j].legend(fontsize=7)
    axes[1, j].plot(ep, h["loss"], "o-", ms=4, label="train")
    axes[1, j].plot(ep, h["val_loss"], "s-", ms=4, label="validation")
    axes[1, j].axvline(best, ls="--", c="green", alpha=.7)
    axes[1, j].set_title("loss", fontsize=10); axes[1, j].set_xlabel("epoch")
    axes[1, j].legend(fontsize=7)
fig.suptitle("Training dynamics — accuracy (top) and loss (bottom); dashed line = best validation epoch", y=1.0)
plt.tight_layout(); plt.savefig(f"{FIGURES}/07_training_curves.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- Overlaid validation curves make the ranking and its stability obvious --
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
for n in MODEL_ORDER + [SUPP]:
    if n not in HIST: continue
    h = HIST[n]["history"]; ep = range(1, len(h["loss"]) + 1)
    ls = "--" if n == SUPP else "-"
    ax[0].plot(ep, h["val_accuracy"], ls, marker="o", ms=3.5, label=n)
    ax[1].plot(ep, h["val_loss"], ls, marker="o", ms=3.5, label=n)
    ax[2].plot(ep, np.array(h["accuracy"]) - np.array(h["val_accuracy"]), ls, marker="o", ms=3.5, label=n)
ax[0].set_title("Validation accuracy"); ax[1].set_title("Validation loss")
ax[2].set_title("Generalisation gap (train − val accuracy)"); ax[2].axhline(0, c="k", lw=.8)
for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIGURES}/08_validation_comparison.png", bbox_inches="tight"); plt.show()
""")

md("## 5.2 Headline comparison table")

code(r"""
rows = []
for n in MODEL_ORDER + [SUPP]:
    if n not in EVAL: continue
    e, h = EVAL[n], HIST[n]
    rows.append({
        "Model": n,
        "Test Accuracy": e["test_accuracy"],
        "F1 (weighted)": e["f1_weighted"],
        "F1 (macro)": e["f1_macro"],
        "Top-5 Accuracy": e["top5_accuracy"],
        "Test Loss": e["test_loss"],
        "Best Val Accuracy": e["best_val_accuracy"],
        "Total Params": e["total_params"],
        "Trainable Params": e["trainable_params"],
        "Epochs Run": e["epochs_run"],
        "Training Time (s)": e["training_time_sec"],
        "Training Time (min)": e["training_time_sec"] / 60,
        "Sec / Epoch": e["sec_per_epoch"],
        "Inference Time (s)": e["inference_time_sec"],
        "Inference (ms/img)": e["inference_ms_per_image"],
        "Throughput (img/s)": e["inference_throughput_ips"],
        "Model Size (MB)": e["total_params"] * 4 / 1024**2,
        "Generalisation Gap": e["generalization_gap"],
    })
comparison = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

fmt = {"Test Accuracy": "{:.4f}", "F1 (weighted)": "{:.4f}", "F1 (macro)": "{:.4f}",
       "Top-5 Accuracy": "{:.4f}", "Test Loss": "{:.4f}", "Best Val Accuracy": "{:.4f}",
       "Total Params": "{:,.0f}", "Trainable Params": "{:,.0f}",
       "Training Time (s)": "{:.1f}", "Training Time (min)": "{:.1f}", "Sec / Epoch": "{:.1f}",
       "Inference Time (s)": "{:.2f}", "Inference (ms/img)": "{:.3f}",
       "Throughput (img/s)": "{:.0f}", "Model Size (MB)": "{:.1f}",
       "Generalisation Gap": "{:+.4f}"}
display(comparison.style.format(fmt).hide(axis="index")
        .background_gradient(subset=["Test Accuracy", "F1 (weighted)"], cmap="Greens")
        .background_gradient(subset=["Training Time (min)", "Inference (ms/img)"], cmap="Reds"))
""")

code(r"""
# --- Multi-panel visual comparison -----------------------------------------
sub = comparison[comparison.Model.isin(MODEL_ORDER)]
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
pal = sns.color_palette("Set2", len(sub))
def bar(a, col, title, fmtstr="{:.3f}", log=False):
    b = a.bar(range(len(sub)), sub[col], color=pal)
    a.set_xticks(range(len(sub))); a.set_xticklabels([m.replace("_", "\n") for m in sub.Model], fontsize=8)
    a.set_title(title, fontsize=10)
    if log: a.set_yscale("log")
    for r, v in zip(b, sub[col]): a.text(r.get_x()+r.get_width()/2, r.get_height(),
                                         fmtstr.format(v), ha="center", va="bottom", fontsize=8)
bar(ax[0,0], "Test Accuracy", "Test accuracy")
bar(ax[0,1], "F1 (weighted)", "Weighted F1-score")
bar(ax[0,2], "Training Time (min)", "Training wall-clock (min)", "{:.1f}")
bar(ax[1,0], "Total Params", "Total parameters", "{:,.0f}", log=True)
bar(ax[1,1], "Inference (ms/img)", "Inference latency (ms/image)", "{:.2f}")
bar(ax[1,2], "Model Size (MB)", "Model size (MB, float32)", "{:.0f}")
plt.tight_layout(); plt.savefig(f"{FIGURES}/09_model_comparison.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- Efficiency frontier: accuracy against the three costs that matter ------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for i, (xcol, xlab, logx) in enumerate([
        ("Training Time (min)", "training wall-clock (min)", False),
        ("Inference (ms/img)",  "inference latency (ms/image)", False),
        ("Total Params",        "total parameters", True)]):
    for _, r in comparison.iterrows():
        mk = "D" if r.Model == SUPP else "o"
        ax[i].scatter(r[xcol], r["Test Accuracy"], s=150, marker=mk, alpha=.85,
                      label=r.Model if i == 0 else None)
        ax[i].annotate(r.Model.replace("_Transfer", "").replace("_", " "),
                       (r[xcol], r["Test Accuracy"]), textcoords="offset points",
                       xytext=(7, 5), fontsize=8)
    ax[i].set_xlabel(xlab); ax[i].set_ylabel("test accuracy")
    if logx: ax[i].set_xscale("log")
    ax[i].set_title(f"Accuracy vs {xlab}", fontsize=10)
ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Efficiency frontier — up and to the left is better", y=1.02)
plt.tight_layout(); plt.savefig(f"{FIGURES}/10_efficiency_frontier.png", bbox_inches="tight"); plt.show()
""")

md("## 5.3 Confusion matrices")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
for a, n in zip(axes.ravel(), MODEL_ORDER):
    if n not in EVAL: a.axis("off"); continue
    cm = np.array(EVAL[n]["confusion_matrix"])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=a,
                annot_kws={"size": 7}, linewidths=.4, linecolor="white")
    a.set_title(f"{n}\naccuracy {EVAL[n]['test_accuracy']:.4f} | weighted F1 {EVAL[n]['f1_weighted']:.4f}",
                fontsize=10)
    a.set_xlabel("predicted"); a.set_ylabel("true")
    a.tick_params(axis="x", rotation=45); a.tick_params(axis="y", rotation=0)
fig.suptitle("Confusion matrices — cell colour = row-normalised recall, cell text = image count", y=1.0)
plt.tight_layout(); plt.savefig(f"{FIGURES}/11_confusion_matrices.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- Full per-class classification reports ---------------------------------
for n in MODEL_ORDER:
    if n not in EVAL: continue
    print("=" * 74); print(f"{n}".center(74)); print("=" * 74)
    print(EVAL[n]["classification_report"])
""")

code(r"""
# --- Per-class F1 across models --------------------------------------------
f1_tbl = pd.DataFrame({n: {c: EVAL[n]["per_class"][c]["f1"] for c in C.CLASS_NAMES}
                       for n in MODEL_ORDER if n in EVAL})
fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
sns.heatmap(f1_tbl, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax[0],
            cbar_kws={"label": "F1"}, linewidths=.4)
ax[0].set_title("Per-class F1-score by model")
f1_tbl.plot(kind="bar", ax=ax[1], width=.8, colormap="Set2")
ax[1].set_ylabel("F1"); ax[1].set_title("Per-class F1 — class difficulty is consistent across architectures")
ax[1].legend(fontsize=7); ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(f"{FIGURES}/12_per_class_f1.png", bbox_inches="tight"); plt.show()

rank = f1_tbl.mean(axis=1).sort_values()
print("Mean F1 per class across all four models (hardest first):")
for c, v in rank.items(): print(f"   {c:12s} {v:.4f}")
""")

md("## 5.4 Error analysis")

code(r"""
# --- Which class pairs does the best model actually confuse? ---------------
BEST = comparison[comparison.Model.isin(MODEL_ORDER)].iloc[0].Model
print(f"Best-performing required model: {BEST}\n")

cm = np.array(EVAL[BEST]["confusion_matrix"])
pairs = []
for i in range(10):
    for j in range(10):
        if i != j: pairs.append((C.CLASS_NAMES[i], C.CLASS_NAMES[j], int(cm[i, j]),
                                 cm[i, j] / cm[i].sum() * 100))
top_conf = pd.DataFrame(sorted(pairs, key=lambda p: -p[2])[:12],
                        columns=["true class", "predicted as", "count", "% of true class"])
top_conf["% of true class"] = top_conf["% of true class"].round(2)
display(top_conf.style.hide(axis="index"))

# symmetric confusion strength -> which pairs are mutually ambiguous?
sym = cm + cm.T; np.fill_diagonal(sym, 0)
fig, ax = plt.subplots(figsize=(7, 5.6))
sns.heatmap(sym, annot=True, fmt="d", cmap="Reds", xticklabels=C.CLASS_NAMES,
            yticklabels=C.CLASS_NAMES, ax=ax, annot_kws={"size": 7}, linewidths=.4)
ax.set_title(f"{BEST} — symmetric confusion (errors in both directions)")
plt.tight_layout(); plt.savefig(f"{FIGURES}/13_symmetric_confusion.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- Look at actual misclassified images -----------------------------------
probs_best = np.load(f"{ARTIFACTS}/{BEST}_probs.npy")
pred_best = probs_best.argmax(1); conf_best = probs_best.max(1)
wrong = np.where(pred_best != y_test)[0]
print(f"{BEST}: {len(wrong):,} / {len(y_test):,} misclassified ({len(wrong)/len(y_test)*100:.2f}%)")
print(f"Mean confidence — correct {conf_best[pred_best==y_test].mean():.3f}, "
      f"incorrect {conf_best[wrong].mean():.3f}")

order = wrong[np.argsort(-conf_best[wrong])][:16]     # most confidently wrong
fig, axes = plt.subplots(2, 8, figsize=(17, 5))
for a, k in zip(axes.ravel(), order):
    a.imshow(x_test[k]); a.axis("off")
    a.set_title(f"true: {C.CLASS_NAMES[y_test[k]]}\npred: {C.CLASS_NAMES[pred_best[k]]}"
                f" ({conf_best[k]:.2f})", fontsize=7.5)
fig.suptitle(f"{BEST} — the 16 most confidently misclassified test images", y=1.02)
plt.tight_layout(); plt.savefig(f"{FIGURES}/14_misclassified.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- Confidence calibration: is the model's certainty meaningful? ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(conf_best[pred_best == y_test], bins=40, alpha=.7, label="correct", density=True)
ax[0].hist(conf_best[wrong], bins=40, alpha=.7, label="incorrect", density=True)
ax[0].set_xlabel("softmax confidence"); ax[0].set_ylabel("density")
ax[0].set_title(f"{BEST} — confidence distribution"); ax[0].legend()

bins = np.linspace(0, 1, 11); mid, acc_b, cnt = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (conf_best >= lo) & (conf_best < hi)
    if m.sum() > 10:
        mid.append((lo + hi) / 2); acc_b.append((pred_best[m] == y_test[m]).mean()); cnt.append(m.sum())
ax[1].plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax[1].plot(mid, acc_b, "o-", label="observed")
ax[1].set_xlabel("predicted confidence"); ax[1].set_ylabel("empirical accuracy")
ax[1].set_title("Reliability diagram"); ax[1].legend()
ece = float(np.sum(np.array(cnt) / sum(cnt) * np.abs(np.array(acc_b) - np.array(mid))))
print(f"Expected Calibration Error (10 bins): {ece:.4f}")
plt.tight_layout(); plt.savefig(f"{FIGURES}/15_calibration.png", bbox_inches="tight"); plt.show()
""")

md(r"""
## 5.5 Best-model identification and justification

Selection cannot be a single-number decision — a sound selection requires a comparison
across *parameters, training time and inference time*. A composite score is computed below,
but the reasoning that follows the numbers is what actually determines the recommendation.
""")

code(r"""
sel = comparison[comparison.Model.isin(MODEL_ORDER)].copy().reset_index(drop=True)
def norm(s, higher_better=True):
    r = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else pd.Series(1.0, index=s.index)
    return r if higher_better else 1 - r

score = pd.DataFrame({
    "Model": sel.Model,
    "accuracy (50%)":    norm(sel["Test Accuracy"]) * 0.50,
    "F1 (20%)":          norm(sel["F1 (weighted)"]) * 0.20,
    "train speed (10%)": norm(sel["Training Time (min)"], False) * 0.10,
    "latency (15%)":     norm(sel["Inference (ms/img)"], False) * 0.15,
    "compactness (5%)":  norm(sel["Total Params"], False) * 0.05,
})
score["composite"] = score.iloc[:, 1:].sum(axis=1)
score = score.sort_values("composite", ascending=False).reset_index(drop=True)
display(score.style.format({c: "{:.4f}" for c in score.columns[1:]}).hide(axis="index")
        .background_gradient(subset=["composite"], cmap="Greens"))

winner_acc  = sel.sort_values("Test Accuracy", ascending=False).iloc[0]
winner_comp = score.iloc[0]
print(f"\nHighest raw accuracy : {winner_acc.Model}  ({winner_acc['Test Accuracy']:.4f})")
print(f"Best composite score : {winner_comp.Model}  ({winner_comp['composite']:.4f})")
""")

code(r"""
# --- Statistical significance: are the accuracy differences real? ----------
# McNemar's exact test on paired predictions over the same 10,000 test images.
from scipy.stats import binomtest
preds = {n: np.load(f"{ARTIFACTS}/{n}_probs.npy").argmax(1) for n in MODEL_ORDER
         if os.path.exists(f"{ARTIFACTS}/{n}_probs.npy")}
rows = []
names = list(preds)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = preds[names[i]], preds[names[j]]
        n01 = int(((a == y_test) & (b != y_test)).sum())
        n10 = int(((a != y_test) & (b == y_test)).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
        rows.append({"model A": names[i], "model B": names[j],
                     "A right / B wrong": n01, "A wrong / B right": n10,
                     "p-value": p, "significant (a=0.01)": p < 0.01})
mcn = pd.DataFrame(rows)
display(mcn.style.format({"p-value": "{:.3e}"}).hide(axis="index"))
print("McNemar's exact test on paired predictions over the same 10,000 test images.\n"
      "It conditions on the discordant pairs only, so it measures whether one model is\n"
      "genuinely better rather than whether two independent samples happened to differ.")
""")

code(r"""
# --- The 224-input vs native-32 Simple CNN question ------------------------
if SUPP in EVAL:
    a, b = EVAL["Simple_CNN"], EVAL[SUPP]
    cmp_simple = pd.DataFrame([
        {"variant": "Simple_CNN (224x224 input)", "input": "224x224",
         "params": a["total_params"], "test acc": a["test_accuracy"],
         "F1": a["f1_weighted"], "train min": a["training_time_sec"]/60,
         "ms/img": a["inference_ms_per_image"], "in 500K-1M spec": False},
        {"variant": "Simple_CNN_Native32 (32x32 input)", "input": "32x32",
         "params": b["total_params"], "test acc": b["test_accuracy"],
         "F1": b["f1_weighted"], "train min": b["training_time_sec"]/60,
         "ms/img": b["inference_ms_per_image"], "in 500K-1M spec": False},
    ])
    display(cmp_simple.style.format({"params": "{:,.0f}", "test acc": "{:.4f}", "F1": "{:.4f}",
                                     "train min": "{:.1f}", "ms/img": "{:.3f}"}).hide(axis="index"))
    print(f"Parameter reduction : {a['total_params']/b['total_params']:.1f}x")
    print(f"Training speed-up   : {a['training_time_sec']/b['training_time_sec']:.1f}x")
    print(f"Accuracy difference : {b['test_accuracy']-a['test_accuracy']:+.4f} "
          f"(native-32 minus 224)")
""")

# =========================================================================
md(r"""
---
# 6. Feature Analysis

**Objective:** *Extract features from the last convolutional layer; use visualisation (t-SNE)
to interpret features; discuss feature activation patterns.*

## 6.1 Extraction protocol

| Model | Feature layer | Why this layer |
|-------|---------------|----------------|
| Simple CNN | `conv3_128` | the last `Conv2D`, i.e. the deepest learned spatial representation before the classifier |
| VGG16 | `block5_pool` | the canonical VGG16 tap — the output of the final convolutional stage, the standard "VGG features" used throughout the literature |
| ResNet50 | `conv5_block3_out` | the output of the last residual bottleneck, the tensor GAP normally consumes |
| EfficientNetB0 | `top_activation` | the final Swish activation after the `1×1` expansion, EfficientNet's designated embedding point |

The common principle: for each architecture the tap is the **last tensor that still carries
spatial structure and is still purely convolutional** — after it, only pooling and
classification remain. That makes the four taps functionally equivalent even though their
names and shapes differ, which is exactly what a fair comparison requires. Taking an
arbitrary mid-network layer instead would compare a generic edge/texture code in one model
against a semantic code in another.

Analysis uses a **class-balanced 2,000-image slice of the test set** (200 per class, fixed
seed) so no model gains an advantage from an easier sample.
""")

code(r"""
print(inspect.getsource(C._build_extractor))
print(inspect.getsource(C.extract_features))
""")

code(r"""
FEATS = json.load(open(f"{ARTIFACTS}/feature_analysis.json"))
feat_labels = np.load(f"{ARTIFACTS}/feature_subset_labels.npy")

rows = []
for n in MODEL_ORDER:
    if n not in FEATS: continue
    f = FEATS[n]
    rows.append({
        "Model": n, "Layer": f["layer"],
        "Feature map shape": "x".join(map(str, f["feature_shape"])),
        "Raw dimensionality": f["feature_dim_flat"],
        "Pooled dimensionality": f["pooled_dim"],
        "Sparsity (% zeros)": f["sparsity_pct"],
        "Activation mean": f["activation_mean"],
        "Activation std": f["activation_std"],
        "Activation max": f["activation_max"],
        "Active units / image": f["mean_active_units_per_sample"],
        "PCA-50 var explained": f["pca50_explained_variance"],
    })
feat_tbl = pd.DataFrame(rows)
display(feat_tbl.style.format({
    "Raw dimensionality": "{:,.0f}", "Pooled dimensionality": "{:,.0f}",
    "Sparsity (% zeros)": "{:.2f}", "Activation mean": "{:.4f}", "Activation std": "{:.4f}",
    "Activation max": "{:.3f}", "Active units / image": "{:,.0f}",
    "PCA-50 var explained": "{:.3f}"}).hide(axis="index"))
feat_tbl.to_csv(f"{ARTIFACTS}/feature_dimensions.csv", index=False)
""")

md("## 6.2 t-SNE visualisation of the learned feature spaces")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
pal10 = sns.color_palette("tab10", 10)
for a, n in zip(axes.ravel(), MODEL_ORDER):
    p = f"{ARTIFACTS}/{n}_tsne.npy"
    if not os.path.exists(p): a.axis("off"); continue
    emb = np.load(p)
    for c in range(10):
        m = feat_labels == c
        a.scatter(emb[m, 0], emb[m, 1], s=7, alpha=.65, color=pal10[c], label=C.CLASS_NAMES[c])
    f = FEATS[n]
    a.set_title(f"{n} — {f['layer']}\nsilhouette {f['silhouette_score']:.3f} | "
                f"2-D 10-NN {f['knn10_accuracy_2d']:.3f} | linear probe {f['linear_probe_accuracy']:.3f}",
                fontsize=10)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
axes[0, 0].legend(markerscale=2.5, fontsize=8, ncol=2, loc="best")
fig.suptitle("t-SNE of globally-pooled last-convolutional-layer features (2,000 test images, 200/class)", y=1.0)
plt.tight_layout(); plt.savefig(f"{FIGURES}/16_tsne.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- Quantifying separability, and its relationship to accuracy ------------
sep = pd.DataFrame([{
    "Model": n,
    "silhouette (2-D)": FEATS[n]["silhouette_score"],
    "10-NN accuracy (2-D)": FEATS[n]["knn10_accuracy_2d"],
    "linear probe (pooled)": FEATS[n]["linear_probe_accuracy"],
    "end-to-end test accuracy": EVAL[n]["test_accuracy"],
} for n in MODEL_ORDER if n in FEATS])
display(sep.style.format({c: "{:.4f}" for c in sep.columns[1:]}).hide(axis="index")
        .background_gradient(cmap="Greens", subset=sep.columns[1:].tolist()))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
sep.set_index("Model")[["silhouette (2-D)", "10-NN accuracy (2-D)",
                        "linear probe (pooled)", "end-to-end test accuracy"]].plot(
    kind="bar", ax=ax[0], colormap="Set2", width=.8)
ax[0].set_title("Feature-quality metrics vs realised accuracy"); ax[0].tick_params(axis="x", rotation=20)
ax[0].legend(fontsize=7)
ax[1].scatter(sep["linear probe (pooled)"], sep["end-to-end test accuracy"], s=140, c="tab:blue")
for _, r in sep.iterrows():
    ax[1].annotate(r.Model.replace("_Transfer", ""), (r["linear probe (pooled)"],
                   r["end-to-end test accuracy"]), textcoords="offset points", xytext=(7, 4), fontsize=8)
lims = [min(sep["linear probe (pooled)"].min(), sep["end-to-end test accuracy"].min()) - .05,
        max(sep["linear probe (pooled)"].max(), sep["end-to-end test accuracy"].max()) + .05]
ax[1].plot(lims, lims, "k--", alpha=.5, label="y = x")
ax[1].set_xlabel("linear probe on frozen features"); ax[1].set_ylabel("end-to-end test accuracy")
ax[1].set_title("Does feature quality predict final accuracy?"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIGURES}/17_separability.png", bbox_inches="tight"); plt.show()
""")

md("## 6.3 Activation patterns")

code(r"""
# --- Sparsity and activation-magnitude profiles ----------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
names = [n for n in MODEL_ORDER if n in FEATS]
ax[0].bar(range(len(names)), [FEATS[n]["sparsity_pct"] for n in names],
          color=sns.color_palette("mako", len(names)))
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax[0].set_ylabel("% zero activations"); ax[0].set_title("Feature sparsity")
for i, n in enumerate(names):
    ax[0].text(i, FEATS[n]["sparsity_pct"], f"{FEATS[n]['sparsity_pct']:.1f}%", ha="center", va="bottom", fontsize=8)

w = 0.35; xs = np.arange(len(names))
ax[1].bar(xs - w/2, [FEATS[n]["activation_mean"] for n in names], w, label="mean")
ax[1].bar(xs + w/2, [FEATS[n]["activation_std"] for n in names], w, label="std")
ax[1].set_xticks(xs); ax[1].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax[1].set_title("Activation magnitude"); ax[1].legend(fontsize=8)

ax[2].bar(range(len(names)), [FEATS[n]["feature_dim_flat"] for n in names],
          color=sns.color_palette("flare", len(names)))
ax[2].set_yscale("log"); ax[2].set_xticks(range(len(names)))
ax[2].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax[2].set_ylabel("dimensions (log)"); ax[2].set_title("Raw feature-map dimensionality")
for i, n in enumerate(names):
    ax[2].text(i, FEATS[n]["feature_dim_flat"], f"{FEATS[n]['feature_dim_flat']:,}",
               ha="center", va="bottom", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{FIGURES}/18_activation_stats.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- Distribution of pooled descriptors ------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
for n in names:
    v = np.load(f"{ARTIFACTS}/{n}_pooled.npy").ravel()
    v = v[np.random.RandomState(C.SEED).choice(len(v), min(200000, len(v)), replace=False)]
    ax[0].hist(v, bins=120, alpha=.5, density=True, label=n, log=True)
    ax[1].hist(v[v > 0], bins=120, alpha=.5, density=True, label=n)
ax[0].set_title("Pooled activation distribution (log density)"); ax[0].set_xlabel("activation")
ax[1].set_title("Non-zero activations only"); ax[1].set_xlabel("activation")
for a in ax: a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIGURES}/19_activation_distributions.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- What do individual filters respond to? --------------------------------
C.set_seeds()
sc = C.build_simple_cnn(norm_mean=NORM_MEAN, norm_var=NORM_VAR)
sc.load_weights(f"{ARTIFACTS}/Simple_CNN.weights.h5")
img_i = int(np.where(y_test == 8)[0][0])          # a 'ship'
fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(3, 9)
axi = fig.add_subplot(gs[:, 0]); axi.imshow(x_test[img_i]); axi.axis("off")
axi.set_title(f"input\n({C.CLASS_NAMES[y_test[img_i]]})", fontsize=9)
for r, layer in enumerate(["conv1_32", "conv2_64", "conv3_128"]):
    ex = tf.keras.Model(sc.inputs, sc.get_layer(layer).output)
    fm = ex.predict(x_test[img_i:img_i+1], verbose=0)[0]
    energetic = np.argsort(-fm.mean(axis=(0, 1)))[:8]
    for j, ch in enumerate(energetic):
        a = fig.add_subplot(gs[r, j + 1]); a.imshow(fm[:, :, ch], cmap="viridis"); a.axis("off")
        if j == 0: a.set_ylabel(layer)
        a.set_title(f"{layer}\nch {ch}" if r == 0 else f"ch {ch}", fontsize=7)
fig.suptitle("Simple CNN — 8 highest-energy feature maps per convolutional block "
             "(early = edges/colour, late = parts)", y=1.0)
plt.tight_layout(); plt.savefig(f"{FIGURES}/20_feature_maps.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# --- First-layer learned filters -------------------------------------------
W = sc.get_layer("conv1_32").get_weights()[0]        # (3,3,3,32)
Wn = (W - W.min()) / (W.max() - W.min())
fig, axes = plt.subplots(4, 8, figsize=(11, 5.6))
for i, a in enumerate(axes.ravel()):
    a.imshow(Wn[:, :, :, i], interpolation="nearest"); a.axis("off"); a.set_title(f"f{i}", fontsize=7)
fig.suptitle("Simple CNN — all 32 learned first-layer 3×3 RGB filters", y=1.0)
plt.tight_layout(); plt.savefig(f"{FIGURES}/21_first_layer_filters.png", bbox_inches="tight"); plt.show()
del sc; tf.keras.backend.clear_session()
""")

code(r"""
# --- Class-level feature geometry: how far apart are the class centroids? --
rows = []
for n in names:
    P = np.load(f"{ARTIFACTS}/{n}_pooled.npy")
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    cent = np.stack([P[feat_labels == c].mean(0) for c in range(10)])
    within = float(np.mean([np.linalg.norm(P[feat_labels == c] - cent[c], axis=1).mean()
                            for c in range(10)]))
    D = np.linalg.norm(cent[:, None] - cent[None], axis=-1)
    between = float(D[~np.eye(10, dtype=bool)].mean())
    rows.append({"Model": n, "mean within-class distance": within,
                 "mean between-class distance": between,
                 "separation ratio (between/within)": between / within})
geo = pd.DataFrame(rows)
display(geo.style.format({c: "{:.4f}" for c in geo.columns[1:]}).hide(axis="index"))
print("Computed on L2-normalised pooled descriptors, so the ratio is scale-free and\n"
      "comparable across architectures with different feature dimensionalities.")
""")

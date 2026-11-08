"""
Driver 3 -- Feature Analysis.

For each trained model: tap the last convolutional feature map, record its
dimensionality and activation statistics, then project the globally-pooled
descriptors to 2-D with t-SNE (PCA-50 pre-reduction, the standard protocol)
and quantify class separability.

A fixed, class-balanced 2,000-image slice of the TEST set is used (200 per
class) so every model is compared on identical inputs.
"""
import os, json, time
import numpy as np, tensorflow as tf
import capstone_lib as C
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

A = C.ARTIFACT_DIR
N_PER_CLASS = 200
OUT = f"{A}/feature_analysis.json"

C.set_seeds()
x_train, y_train, x_test, y_test = C.load_cached()
_, _, _, x_test, y_test, y_test_oh = C.prepare_arrays(x_train, y_train, x_test, y_test)

# --- class-balanced evaluation slice --------------------------------------
rng = np.random.RandomState(C.SEED)
sel = np.concatenate([rng.choice(np.where(y_test == c)[0], N_PER_CLASS, replace=False)
                      for c in range(C.NUM_CLASSES)])
rng.shuffle(sel)
xs, ys = x_test[sel], y_test[sel]
np.save(f"{A}/feature_subset_labels.npy", ys)
print(f"[setup] feature slice: {len(xs)} images, {N_PER_CLASS}/class", flush=True)

NORM = np.load(f"{C.DATA_DIR}/train_norm_stats.npy")
BUILD = {
    "Simple_CNN":              (C.build_simple_cnn,   dict(norm_mean=NORM[0], norm_var=NORM[1])),
    "VGG16_Transfer":          (C.build_vgg16,        dict()),
    "ResNet50_Transfer":       (C.build_resnet50,     dict()),
    "EfficientNetB0_Transfer": (C.build_efficientnet, dict()),
}

results = json.load(open(OUT)) if os.path.exists(OUT) else {}

for name, (builder, kw) in BUILD.items():
    if name in results:
        print(f"[skip] {name}"); continue
    wpath = f"{A}/{name}.weights.h5"
    if not os.path.exists(wpath):
        print(f"[wait] {name} weights not ready"); continue

    print(f"\n--- features: {name} ({C.FEATURE_LAYERS[name]}) ---", flush=True)
    C.set_seeds()
    model = builder(**kw)
    model.load_weights(wpath)

    t0 = time.time()
    pooled, stats = C.extract_features(model, C.FEATURE_LAYERS[name], xs, chunk=250)
    stats["extraction_time_sec"] = time.time() - t0
    stats["layer"] = C.FEATURE_LAYERS[name]
    print(f"    raw map {stats['feature_shape']} = {stats['feature_dim_flat']:,} dims | "
          f"pooled {pooled.shape[1]} | sparsity {stats['sparsity_pct']:.1f}%", flush=True)

    # PCA(50) -> t-SNE(2): PCA denoises and makes t-SNE tractable on wide descriptors
    n_comp = min(50, pooled.shape[1], len(pooled))
    pca = PCA(n_components=n_comp, random_state=C.SEED)
    reduced = pca.fit_transform(pooled)
    stats["pca50_explained_variance"] = float(pca.explained_variance_ratio_.sum())

    emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
               random_state=C.SEED, max_iter=1000).fit_transform(reduced)
    np.save(f"{A}/{name}_tsne.npy", emb)
    np.save(f"{A}/{name}_pooled.npy", pooled)

    stats.update(C.separability(emb, ys))
    # linear probe on the pooled descriptors: how linearly separable are they?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=C.SEED))
    stats["linear_probe_accuracy"] = float(cross_val_score(probe, pooled, ys, cv=5).mean())

    results[name] = stats
    C.save_json(results, OUT)
    print(f"    silhouette={stats['silhouette_score']:.4f} "
          f"knn10={stats['knn10_accuracy_2d']:.4f} "
          f"linear_probe={stats['linear_probe_accuracy']:.4f}", flush=True)
    del model; tf.keras.backend.clear_session()

print("\nFEATURE ANALYSIS COMPLETE", flush=True)

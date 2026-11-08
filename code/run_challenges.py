"""
Driver 4 -- Challenge / Advanced-Analysis experiments.

  C1  Fine-tuning        : unfreeze the final block of the best transfer model
  C2  Cross-validation   : 3-fold stratified CV of the Simple CNN
  C3  Ensembling         : soft / weighted / hard voting over the trained models
  C4  Model compression  : TFLite float32 vs float16 vs dynamic-range int8
  C5  Real-time latency  : batch-1 p50/p95 latency and mobile-deployment verdict
"""
import os, json, time, glob
import numpy as np, tensorflow as tf
import capstone_lib as C
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

A = C.ARTIFACT_DIR
OUT = f"{A}/challenges.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

C.set_seeds()
x_train, y_train, x_test, y_test = C.load_cached()
x_train, y_train, y_train_oh, x_test, y_test, y_test_oh = C.prepare_arrays(
    x_train, y_train, x_test, y_test)
NORM = np.load(f"{C.DATA_DIR}/train_norm_stats.npy")

MAIN = {}
for p in glob.glob(f"{A}/*_eval.json"):
    r = json.load(open(p)); MAIN[r["model"]] = r
BUILD = {
    "Simple_CNN":              (C.build_simple_cnn,   dict(norm_mean=NORM[0], norm_var=NORM[1])),
    "VGG16_Transfer":          (C.build_vgg16,        dict()),
    "ResNet50_Transfer":       (C.build_resnet50,     dict()),
    "EfficientNetB0_Transfer": (C.build_efficientnet, dict()),
}
TRANSFER = ["VGG16_Transfer", "ResNet50_Transfer", "EfficientNetB0_Transfer"]


def save(): C.save_json(res, OUT)


# ==========================================================================
# C1 -- FINE-TUNING the best transfer model
# ==========================================================================
if "finetuning" not in res:
    best_t = max(TRANSFER, key=lambda m: MAIN[m]["test_accuracy"])
    print(f"\n=== C1 Fine-tuning {best_t} (best frozen transfer model) ===", flush=True)
    C.set_seeds()
    builder, kw = BUILD[best_t]
    model = builder(**kw)
    model.load_weights(f"{A}/{best_t}.weights.h5")
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))

    # Unfreeze only the final convolutional stage; BatchNorm layers stay frozen so
    # the ImageNet running statistics are not corrupted by small CIFAR batches.
    LAST_STAGE = {"VGG16_Transfer": "block5", "ResNet50_Transfer": "conv5_",
                  "EfficientNetB0_Transfer": "block7"}[best_t]
    base.trainable = True
    n_unfrozen = 0
    for l in base.layers:
        is_bn = isinstance(l, tf.keras.layers.BatchNormalization)
        keep = (LAST_STAGE in l.name or l.name.startswith("top")) and not is_bn
        l.trainable = bool(keep)
        n_unfrozen += int(keep)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),      # 100x lower LR
                  loss="categorical_crossentropy", metrics=["accuracy"])
    pc = C.param_counts(model)
    print(f"    unfroze {n_unfrozen} layers -> trainable params "
          f"{MAIN[best_t]['trainable_params']:,} -> {pc['trainable_params']:,}", flush=True)

    tg, vg, teg = C.make_generators(x_train, y_train_oh, x_test, y_test_oh)
    t0 = time.time()
    h = model.fit(tg, validation_data=vg, epochs=5,
                  callbacks=[tf.keras.callbacks.CSVLogger(f"{A}/finetune_epochs.csv")],
                  verbose=2)
    ft_time = time.time() - t0
    ev = C.evaluate_model(model, teg, y_test, best_t + "_FineTuned")
    np.save(f"{A}/{best_t}_FineTuned_probs.npy", ev.pop("_probs"))
    model.save_weights(f"{A}/{best_t}_FineTuned.weights.h5")

    res["finetuning"] = {
        "base_model": best_t, "unfrozen_stage": LAST_STAGE,
        "layers_unfrozen": n_unfrozen, "learning_rate": 1e-5, "epochs": 5,
        "trainable_params_before": MAIN[best_t]["trainable_params"],
        "trainable_params_after": pc["trainable_params"],
        "accuracy_before": MAIN[best_t]["test_accuracy"],
        "accuracy_after": ev["test_accuracy"],
        "f1_before": MAIN[best_t]["f1_weighted"], "f1_after": ev["f1_weighted"],
        "absolute_gain": ev["test_accuracy"] - MAIN[best_t]["test_accuracy"],
        "finetune_time_sec": ft_time,
        "history": {k: [float(x) for x in v] for k, v in h.history.items()},
        "confusion_matrix": ev["confusion_matrix"],
        "per_class": ev["per_class"],
    }
    save()
    print(f"[done] fine-tune: {MAIN[best_t]['test_accuracy']:.4f} -> {ev['test_accuracy']:.4f}", flush=True)
    del model; tf.keras.backend.clear_session()


# ==========================================================================
# C2 -- K-FOLD CROSS-VALIDATION (Simple CNN)
# ==========================================================================
if "cross_validation" not in res:
    print("\n=== C2 3-fold stratified cross-validation (Simple CNN) ===", flush=True)
    K, N_CV, EP = 3, 12000, 6
    xcv, ycv = x_train[:N_CV], y_train[:N_CV]
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=C.SEED)
    folds = []
    for i, (tr, va) in enumerate(skf.split(xcv, ycv), 1):
        C.set_seeds()
        m = C.build_simple_cnn(norm_mean=NORM[0], norm_var=NORM[1])
        aug = tf.keras.preprocessing.image.ImageDataGenerator(**C.AUG_PARAMS)
        g = aug.flow(xcv[tr], tf.keras.utils.to_categorical(ycv[tr], 10),
                     batch_size=C.BATCH_SIZE, shuffle=True, seed=C.SEED)
        t0 = time.time()
        h = m.fit(g, validation_data=(xcv[va], tf.keras.utils.to_categorical(ycv[va], 10)),
                  epochs=EP, verbose=2)
        pv = m.predict(xcv[va], batch_size=64, verbose=0).argmax(1)
        pt = m.predict(x_test, batch_size=64, verbose=0).argmax(1)
        folds.append({"fold": i, "n_train": len(tr), "n_val": len(va),
                      "val_accuracy": float(accuracy_score(ycv[va], pv)),
                      "test_accuracy": float(accuracy_score(y_test, pt)),
                      "test_f1_weighted": float(f1_score(y_test, pt, average="weighted")),
                      "time_sec": time.time() - t0})
        print(f"    fold {i}: val={folds[-1]['val_accuracy']:.4f} test={folds[-1]['test_accuracy']:.4f}", flush=True)
        del m; tf.keras.backend.clear_session()

    va = np.array([f["val_accuracy"] for f in folds]); te = np.array([f["test_accuracy"] for f in folds])
    res["cross_validation"] = {
        "k": K, "n_samples": N_CV, "epochs_per_fold": EP, "folds": folds,
        "val_accuracy_mean": float(va.mean()), "val_accuracy_std": float(va.std(ddof=1)),
        "test_accuracy_mean": float(te.mean()), "test_accuracy_std": float(te.std(ddof=1)),
        "val_ci95": [float(va.mean() - 1.96 * va.std(ddof=1) / np.sqrt(K)),
                     float(va.mean() + 1.96 * va.std(ddof=1) / np.sqrt(K))],
    }
    save()
    print(f"[done] CV val={va.mean():.4f}+/-{va.std(ddof=1):.4f}", flush=True)


# ==========================================================================
# C3 -- ENSEMBLES
# ==========================================================================
if "ensemble" not in res:
    print("\n=== C3 Ensembles ===", flush=True)
    names = [n for n in BUILD if os.path.exists(f"{A}/{n}_probs.npy")]
    P = {n: np.load(f"{A}/{n}_probs.npy") for n in names}
    val_acc = {n: MAIN[n]["best_val_accuracy"] for n in names}
    ranked = sorted(names, key=lambda n: -MAIN[n]["test_accuracy"])

    def score(p):
        yp = p.argmax(1)
        return {"accuracy": float(accuracy_score(y_test, yp)),
                "f1_weighted": float(f1_score(y_test, yp, average="weighted"))}

    combos = {}
    top2 = ranked[:2]
    combos["top2_soft_voting"] = {"members": top2, **score(np.mean([P[n] for n in top2], 0))}
    combos["all_soft_voting"]  = {"members": ranked, **score(np.mean([P[n] for n in ranked], 0))}
    w = np.array([val_acc[n] for n in ranked]); w = w / w.sum()
    combos["all_weighted_soft"] = {"members": ranked, "weights": w.round(4).tolist(),
                                   **score(np.tensordot(w, np.stack([P[n] for n in ranked]), axes=1))}
    hard = np.stack([P[n].argmax(1) for n in ranked])
    maj = np.array([np.bincount(hard[:, i], minlength=10).argmax() for i in range(hard.shape[1])])
    combos["all_hard_voting"] = {"members": ranked,
                                 "accuracy": float(accuracy_score(y_test, maj)),
                                 "f1_weighted": float(f1_score(y_test, maj, average="weighted"))}
    res["ensemble"] = {"individual": {n: MAIN[n]["test_accuracy"] for n in ranked},
                       "best_individual": ranked[0],
                       "best_individual_accuracy": MAIN[ranked[0]]["test_accuracy"],
                       "combinations": combos}
    save()
    for k, v in combos.items():
        print(f"    {k:22s} acc={v['accuracy']:.4f}", flush=True)


# ==========================================================================
# C4 -- MODEL COMPRESSION (TFLite quantisation)
# ==========================================================================
if "compression" not in res:
    print("\n=== C4 Compression / quantisation ===", flush=True)
    target = max(BUILD, key=lambda m: MAIN[m]["test_accuracy"])
    builder, kw = BUILD[target]
    C.set_seeds(); model = builder(**kw); model.load_weights(f"{A}/{target}.weights.h5")
    n_eval = 300
    xq, yq = x_test[:n_eval], y_test[:n_eval]

    def tflite_eval(tag, configure):
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                          tf.lite.OpsSet.SELECT_TF_OPS]
        configure(conv)
        blob = conv.convert()
        path = f"{A}/{target}_{tag}.tflite"; open(path, "wb").write(blob)
        it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        preds, t0 = [], time.time()
        for i in range(n_eval):
            it.set_tensor(inp["index"], xq[i:i + 1].astype(inp["dtype"]))
            it.invoke(); preds.append(it.get_tensor(out["index"])[0].argmax())
        dt = time.time() - t0
        return {"variant": tag, "size_mb": len(blob) / 1024 ** 2,
                "accuracy": float(accuracy_score(yq, preds)),
                "latency_ms_per_image": dt / n_eval * 1000,
                "n_eval_images": n_eval}

    variants = {}
    try:
        variants["float32"] = tflite_eval("fp32", lambda c: None)
        variants["float16"] = tflite_eval("fp16", lambda c: (
            setattr(c, "optimizations", [tf.lite.Optimize.DEFAULT]),
            setattr(c.target_spec, "supported_types", [tf.float16])))
        variants["dynamic_int8"] = tflite_eval("int8", lambda c:
            setattr(c, "optimizations", [tf.lite.Optimize.DEFAULT]))
    except Exception as e:
        variants["error"] = f"{type(e).__name__}: {e}"
        print("    TFLite conversion issue:", e, flush=True)

    # Post-training magnitude pruning of the dense head (framework-independent)
    prune = []
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    orig = [ [w.copy() for w in l.get_weights()] for l in dense ]
    for frac in (0.0, 0.5, 0.8, 0.9):
        for l, w0 in zip(dense, orig):
            W, b = [w.copy() for w in w0]
            if frac > 0:
                thr = np.quantile(np.abs(W), frac); W[np.abs(W) < thr] = 0.0
            l.set_weights([W, b])
        acc = accuracy_score(yq, model.predict(xq, batch_size=64, verbose=0).argmax(1))
        nz = sum(int((l.get_weights()[0] != 0).sum()) for l in dense)
        prune.append({"sparsity_target": frac, "accuracy": float(acc),
                      "nonzero_dense_weights": nz})
        print(f"    prune {frac:.0%}: acc={acc:.4f}", flush=True)
    for l, w0 in zip(dense, orig):
        l.set_weights(w0)

    fp32_mb = MAIN[target]["total_params"] * 4 / 1024 ** 2
    res["compression"] = {"target_model": target, "keras_fp32_size_mb": fp32_mb,
                          "baseline_accuracy_subset": float(accuracy_score(
                              yq, model.predict(xq, batch_size=64, verbose=0).argmax(1))),
                          "tflite": variants, "dense_head_pruning": prune}
    save()
    del model; tf.keras.backend.clear_session()


# ==========================================================================
# C5 -- REAL-TIME INFERENCE LATENCY
# ==========================================================================
if "latency" not in res:
    print("\n=== C5 Real-time latency (batch size 1) ===", flush=True)
    lat = {}
    for name, (builder, kw) in BUILD.items():
        if not os.path.exists(f"{A}/{name}.weights.h5"): continue
        C.set_seeds(); m = builder(**kw); m.load_weights(f"{A}/{name}.weights.h5")
        f = tf.function(lambda t: m(t, training=False))
        one = x_test[:1]
        for _ in range(20): f(tf.constant(one))                    # warm-up
        ts = []
        for i in range(200):
            t0 = time.perf_counter(); f(tf.constant(x_test[i:i + 1])); ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        t0 = time.perf_counter(); m.predict(x_test[:1024], batch_size=32, verbose=0)
        batch_thr = 1024 / (time.perf_counter() - t0)
        lat[name] = {
            "batch1_mean_ms": float(ts.mean()), "batch1_p50_ms": float(np.percentile(ts, 50)),
            "batch1_p95_ms": float(np.percentile(ts, 95)), "batch1_std_ms": float(ts.std()),
            "batch1_fps": float(1000 / ts.mean()),
            "batch32_throughput_ips": float(batch_thr),
            "total_params": MAIN[name]["total_params"],
            "model_size_mb": MAIN[name]["total_params"] * 4 / 1024 ** 2,
            "test_accuracy": MAIN[name]["test_accuracy"],
            "realtime_30fps_capable": bool(1000 / ts.mean() >= 30),
        }
        print(f"    {name:26s} p50={lat[name]['batch1_p50_ms']:6.2f}ms "
              f"p95={lat[name]['batch1_p95_ms']:6.2f}ms {lat[name]['batch1_fps']:6.1f} fps", flush=True)
        del m; tf.keras.backend.clear_session()
    res["latency"] = lat
    save()

print("\nCHALLENGES COMPLETE", flush=True)

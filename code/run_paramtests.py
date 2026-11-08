"""
Driver 2 -- Parameter-Modification Study (the 5 required ablations).

METHODOLOGY / COMPUTE BUDGET
----------------------------------------------------------------------------
Re-running the full 40k/10k protocol for all 13 variants below would cost
~20 GPU-hours.  Instead every variant is trained on an IDENTICAL, fixed-seed
10,000-image slice of the training pool (8,000 train / 2,000 validation) for a
fixed 6 epochs, and every variant is evaluated on the FULL, untouched 10,000-
image test set.  Because the data, seed, split and epoch budget are held
constant, the *relative* comparison within each ablation is controlled and
valid; only the absolute accuracies are lower than the full-scale runs.
Each ablation changes exactly one factor at a time.
"""
import os, time, json
import numpy as np, tensorflow as tf
import capstone_lib as C

A = C.ARTIFACT_DIR
PT_TRAIN, PT_VAL, PT_EPOCHS = 8000, 2000, 6
OUT = f"{A}/param_tests.json"

x_train, y_train, x_test, y_test = C.load_cached()
x_train, y_train, y_train_oh, x_test, y_test, y_test_oh = C.prepare_arrays(
    x_train, y_train, x_test, y_test)

# Fixed subset drawn from the training pool only -- the test set is untouched.
xs, ys = x_train[:PT_TRAIN + PT_VAL], y_train_oh[:PT_TRAIN + PT_VAL]
NORM = np.load(f"{C.DATA_DIR}/train_norm_stats.npy")
NORM_MEAN, NORM_VAR = NORM[0], NORM[1]

results = json.load(open(OUT)) if os.path.exists(OUT) else {}


def run_variant(key, builder, build_kw, batch_size=C.BATCH_SIZE, epochs=PT_EPOCHS,
                extract_tsne=False):
    if key in results:
        print(f"[skip] {key}"); return results[key]
    print(f"\n--- {key} | batch={batch_size} {build_kw} ---", flush=True)
    C.set_seeds()
    tg, vg, teg = C.make_generators(xs, ys, x_test, y_test_oh, batch_size=batch_size)
    model = builder(**build_kw)
    pc = C.param_counts(model)
    mem = C.estimate_activation_memory(model, batch_size)

    cbs = [tf.keras.callbacks.CSVLogger(f"{A}/pt_{key}_epochs.csv")]   # no early stop: fixed budget
    t0 = time.time()
    h = model.fit(tg, validation_data=vg, epochs=epochs, callbacks=cbs, verbose=2)
    train_time = time.time() - t0

    ev = C.evaluate_model(model, teg, y_test, key)
    probs = ev.pop("_probs")

    rec = {
        "key": key, "batch_size": batch_size, "epochs": epochs,
        "build_kwargs": {k: (v if not isinstance(v, np.ndarray) else "train_stats")
                         for k, v in build_kw.items()},
        "history": {k: [float(x) for x in v] for k, v in h.history.items()},
        "training_time_sec": train_time,
        "sec_per_epoch": train_time / epochs,
        "steps_per_epoch": len(tg),
        "test_accuracy": ev["test_accuracy"], "f1_weighted": ev["f1_weighted"],
        "test_loss": ev["test_loss"],
        "best_val_accuracy": float(max(h.history["val_accuracy"])),
        "final_train_accuracy": float(h.history["accuracy"][-1]),
        "final_val_accuracy": float(h.history["val_accuracy"][-1]),
        "overfit_gap": float(h.history["accuracy"][-1] - h.history["val_accuracy"][-1]),
        "inference_ms_per_image": ev["inference_ms_per_image"],
        "peak_host_rss_mb": C.host_peak_rss_mb(),
        **pc,
        "memory": {k: v for k, v in mem.items() if k != "top_layers"},
    }
    if extract_tsne:
        pooled, fstats = C.extract_features(model, C.FEATURE_LAYERS["Simple_CNN"]
                                            if not build_kw.get("extra_block")
                                            else "conv4_256", x_test[:2000], chunk=256)
        np.save(f"{A}/pt_{key}_pooled.npy", pooled)
        rec["feature_stats"] = fstats
    results[key] = rec
    C.save_json(results, OUT)
    print(f"[done] {key}: acc={rec['test_accuracy']:.4f} f1={rec['f1_weighted']:.4f} "
          f"{rec['sec_per_epoch']:.1f}s/epoch params={pc['total_params']:,}", flush=True)
    del model; tf.keras.backend.clear_session()
    return rec


SIMPLE_KW = dict(norm_mean=NORM_MEAN, norm_var=NORM_VAR)

# --- Q1: batch size 16 / 32 / 64 -----------------------------------------
for bs in (16, 32, 64):
    run_variant(f"q1_batch_{bs}", C.build_simple_cnn, dict(SIMPLE_KW), batch_size=bs)

# --- Q2: learning rate 1e-4 / 1e-3 / 1e-2 --------------------------------
for lr in (1e-4, 1e-3, 1e-2):
    run_variant(f"q2_lr_{lr:g}", C.build_simple_cnn, dict(SIMPLE_KW, lr=lr))

# --- Q3: dropout 0.3 / 0.5 / 0.8 on a transfer model ---------------------
for dr in (0.3, 0.5, 0.8):
    run_variant(f"q3_dropout_{dr:g}", C.build_efficientnet, dict(dropout=dr))

# --- Q4: image size 128 / 224 / 384 --------------------------------------
for sz in (128, 224, 384):
    run_variant(f"q4_imgsize_{sz}", C.build_simple_cnn, dict(SIMPLE_KW, img_size=sz))
for sz in (128, 224):     # transfer-model counterpart (pre-trained at 224)
    run_variant(f"q4_vgg_imgsize_{sz}", C.build_vgg16, dict(img_size=sz))

# --- Q5: architecture depth (+ Conv2D(256) block) ------------------------
run_variant("q5_depth_base",  C.build_simple_cnn, dict(SIMPLE_KW), extract_tsne=True)
run_variant("q5_depth_plus4", C.build_simple_cnn, dict(SIMPLE_KW, extra_block=True),
            extract_tsne=True)

print("\nPARAMETER TESTS COMPLETE", flush=True)

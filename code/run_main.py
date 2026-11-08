"""
Driver 1 -- Main experiment: train and evaluate the four required architectures
(plus one supplementary parameter-budget baseline) on the full 40k/10k/10k split.

Artifacts written per model:
    artifacts/<name>_history.json   per-epoch metrics + timing + parameter counts
    artifacts/<name>_eval.json      test metrics, per-class report, confusion matrix
    artifacts/<name>_probs.npy      softmax outputs on the 10,000 test images
    artifacts/<name>.weights.h5     trained weights (for feature analysis / fine-tuning)
    artifacts/<name>_epochs.csv     raw CSVLogger record
    artifacts/<name>_trainlog.txt   captured stdout of model.fit (verbatim epoch log)
"""
import io, json, os, sys, time, contextlib
import numpy as np, tensorflow as tf
import capstone_lib as C

C.set_seeds()
A = C.ARTIFACT_DIR

x_train, y_train, x_test, y_test = C.load_cached()
x_train, y_train, y_train_oh, x_test, y_test, y_test_oh = C.prepare_arrays(
    x_train, y_train, x_test, y_test)

train_gen, val_gen, test_gen = C.make_generators(x_train, y_train_oh, x_test, y_test_oh)

# Standardisation statistics computed on the TRAINING SUBSET ONLY (leakage control)
tr_idx, va_idx = C.split_indices(len(x_train))
NORM_MEAN = x_train[tr_idx].mean(axis=(0, 1, 2))
NORM_VAR  = x_train[tr_idx].var(axis=(0, 1, 2))
np.save(f"{C.DATA_DIR}/train_norm_stats.npy", np.stack([NORM_MEAN, NORM_VAR]))
print(f"[setup] train={train_gen.n} val={val_gen.n} test={test_gen.n}", flush=True)
print(f"[setup] train-subset mean={NORM_MEAN.round(4).tolist()} var={NORM_VAR.round(4).tolist()}", flush=True)

SPECS = [
    ("Simple_CNN",              C.build_simple_cnn,   dict(norm_mean=NORM_MEAN, norm_var=NORM_VAR)),
    ("VGG16_Transfer",          C.build_vgg16,        dict()),
    ("ResNet50_Transfer",       C.build_resnet50,     dict()),
    ("EfficientNetB0_Transfer", C.build_efficientnet, dict()),
    # Supplementary: identical custom stack at CIFAR-native resolution.  Included
    # because a 224x224 input inflates Flatten->Dense(128) to 12.9M parameters,
    # far above the ~500K-1M envelope in the study protocol.
    ("Simple_CNN_Native32",     C.build_simple_cnn,   dict(img_size=32, norm_mean=NORM_MEAN,
                                                           norm_var=NORM_VAR, name="Simple_CNN_Native32")),
]

for name, builder, kwargs in SPECS:
    if os.path.exists(f"{A}/{name}_eval.json"):
        print(f"[skip] {name} already complete", flush=True)
        continue

    print(f"\n{'='*78}\n[train] {name}\n{'='*78}", flush=True)
    C.set_seeds()                                     # identical init conditions per model
    model = builder(**kwargs)
    pc = C.param_counts(model)
    print(f"[params] {name}: total={pc['total_params']:,} "
          f"trainable={pc['trainable_params']:,} frozen={pc['non_trainable_params']:,}", flush=True)

    with open(f"{A}/{name}_summary.txt", "w") as fh:
        model.summary(print_fn=lambda l: fh.write(l + "\n"), line_length=110)

    buf = io.StringIO()
    class Tee(io.TextIOBase):
        def write(self, s):
            buf.write(s); sys.__stdout__.write(s); sys.__stdout__.flush(); return len(s)
    with contextlib.redirect_stdout(Tee()):
        history, train_time, epochs_run = C.train_model(
            model, train_gen, val_gen, name=name, epochs=C.EPOCHS, verbose=2)
    open(f"{A}/{name}_trainlog.txt", "w").write(buf.getvalue())

    model.save_weights(f"{A}/{name}.weights.h5")

    best_ep = int(np.argmax(history["val_accuracy"]))
    hist_rec = {
        "model": name, "history": history,
        "training_time_sec": train_time,
        "epochs_run": epochs_run, "epochs_configured": C.EPOCHS,
        "sec_per_epoch": train_time / max(epochs_run, 1),
        "best_epoch": best_ep + 1,
        "best_val_accuracy": float(history["val_accuracy"][best_ep]),
        "final_train_accuracy": float(history["accuracy"][-1]),
        "early_stopped": epochs_run < C.EPOCHS,
        "generalization_gap": float(history["accuracy"][best_ep] - history["val_accuracy"][best_ep]),
        **pc,
    }
    C.save_json(hist_rec, f"{A}/{name}_history.json")

    print(f"[eval] {name} ...", flush=True)
    res = C.evaluate_model(model, test_gen, y_test, name)
    np.save(f"{A}/{name}_probs.npy", res.pop("_probs"))
    res.update({k: hist_rec[k] for k in
                ("training_time_sec", "epochs_run", "best_epoch", "best_val_accuracy",
                 "total_params", "trainable_params", "non_trainable_params",
                 "sec_per_epoch", "generalization_gap", "early_stopped")})
    C.save_json(res, f"{A}/{name}_eval.json")

    print(f"[done] {name}: test_acc={res['test_accuracy']:.4f} f1w={res['f1_weighted']:.4f} "
          f"train={train_time/60:.1f}min infer={res['inference_ms_per_image']:.2f}ms/img", flush=True)

    del model; tf.keras.backend.clear_session()

print("\nALL MAIN MODELS COMPLETE", flush=True)

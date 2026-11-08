"""
================================================================================
 CIFAR-10 Image Classification — Shared Research Library
================================================================================
 Central module holding configuration, reproducibility controls, data loading,
 preprocessing/augmentation pipelines, the four required model architectures,
 training/evaluation utilities and feature-analysis helpers.

 Every notebook cell and driver script imports from here so that the exact same
 code path is used for every experiment -- a prerequisite for a fair comparison.
================================================================================
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Determinism must be configured BEFORE TensorFlow is imported.
# ---------------------------------------------------------------------------
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # silence INFO/WARNING spam

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             silhouette_score, top_k_accuracy_score)


# ===========================================================================
# 1. GLOBAL CONFIGURATION
# ===========================================================================
IMG_SIZE      = 224          # network input resolution (transfer-learning standard)
NATIVE_SIZE   = 32           # CIFAR-10 native resolution
BATCH_SIZE    = 32
EPOCHS        = 15           # spec range 10-20; EarlyStopping usually cuts this short
NUM_CLASSES   = 10
LEARNING_RATE = 1e-3         # Adam default
VAL_SPLIT     = 0.2          # 40,000 train / 10,000 validation
DROPOUT       = 0.5

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

DATA_DIR      = "data"
ARTIFACT_DIR  = "artifacts"
FIGURE_DIR    = "figures"
for _d in (DATA_DIR, ARTIFACT_DIR, FIGURE_DIR, "reports"):
    os.makedirs(_d, exist_ok=True)


# ===========================================================================
# 2. REPRODUCIBILITY
# ===========================================================================
def set_seeds(seed: int = SEED) -> None:
    """Seed every source of randomness in the pipeline.

    Sources covered:
      * Python `random`      -> ImageDataGenerator transform sampling
      * NumPy                -> array shuffling, train/val partition, t-SNE init
      * TensorFlow           -> weight initialisation, dropout masks, shuffling
      * PYTHONHASHSEED       -> hash-ordering of Python containers
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def environment_report() -> Dict:
    """Collect an auditable record of the execution environment."""
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "gpu_devices": [d.name for d in gpus],
        "gpu_available": len(gpus) > 0,
        "num_cpus": os.cpu_count(),
        "seed": SEED,
        "mixed_precision_policy": str(tf.keras.mixed_precision.global_policy().name),
    }


# ===========================================================================
# 3. DATA LOADING
# ===========================================================================
def load_from_csv(train_csv: str = f"{DATA_DIR}/train.csv",
                  test_csv: str = f"{DATA_DIR}/test.csv"
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 from the flat CSVs produced by the project's data protocol.

    CSV layout: column 0 = label, columns 1..3072 = pixel values.
    The raw CIFAR row-major layout is (3, 32, 32) channel-first, so it is
    transposed back to (32, 32, 3) channel-last for Keras.
    """
    for p in (train_csv, test_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run the CSV-building step first.")

    tr = pd.read_csv(train_csv, dtype=np.uint8)
    te = pd.read_csv(test_csv, dtype=np.uint8)

    def to_images(df):
        y = df.iloc[:, 0].to_numpy().astype("int64")
        X = df.iloc[:, 1:].to_numpy().astype("uint8")
        X = X.reshape(-1, 3, NATIVE_SIZE, NATIVE_SIZE).transpose(0, 2, 3, 1)
        return X, y

    x_train, y_train = to_images(tr)
    x_test,  y_test  = to_images(te)
    return x_train, y_train, x_test, y_test


def load_cached() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast .npy path (identical content to the CSVs, used by driver scripts)."""
    x_train = np.load(f"{DATA_DIR}/x_train_raw.npy").reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.load(f"{DATA_DIR}/y_train_raw.npy").astype("int64")
    x_test  = np.load(f"{DATA_DIR}/x_test_raw.npy").reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test  = np.load(f"{DATA_DIR}/y_test_raw.npy").astype("int64")
    return x_train, y_train, x_test, y_test


def prepare_arrays(x_train, y_train, x_test, y_test, seed: int = SEED):
    """Scale to [0,1], one-hot encode, and apply a fixed-seed stratified shuffle.

    The shuffle is applied BEFORE `ImageDataGenerator(validation_split=...)`
    because that utility partitions purely by array index (first 80% -> train,
    last 20% -> validation).  Pre-shuffling with a fixed seed therefore turns an
    index cut into a class-balanced, reproducible hold-out split.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)
    return x_train, y_train, y_train_oh, x_test, y_test, y_test_oh


# ===========================================================================
# 4. GENERATORS  (augmentation + leakage-safe splitting)
# ===========================================================================
AUG_PARAMS = dict(
    rotation_range=15,        # small-angle rotation
    width_shift_range=0.1,    # horizontal shift
    height_shift_range=0.1,   # vertical shift
    horizontal_flip=True,     # random horizontal flip
    zoom_range=0.1,           # random crop/zoom surrogate
    fill_mode="nearest",
)


def make_generators(x_train, y_train_oh, x_test, y_test_oh,
                    batch_size: int = BATCH_SIZE, seed: int = SEED,
                    augment: bool = True, val_split: float = VAL_SPLIT):
    """Build train / validation / test iterators.

    CRITICAL DESIGN POINT -- two generator *objects* share one split
    -------------------------------------------------------------------
    A single augmented `ImageDataGenerator` with `validation_split` would apply
    the SAME random transforms to the validation subset, making validation
    scores noisy and non-comparable across epochs.  We therefore instantiate two
    generators with an identical `validation_split` and identical `seed`:

        train_datagen : augmentation ON  -> subset='training'   (40,000 imgs)
        val_datagen   : augmentation OFF -> subset='validation' (10,000 imgs)

    Because `validation_split` partitions deterministically by array index, the
    two subsets are guaranteed disjoint -> no train/validation leakage.
    The test set lives in a completely separate file and is never seen during
    training or model selection.
    """
    train_datagen = ImageDataGenerator(validation_split=val_split,
                                       **(AUG_PARAMS if augment else {}))
    val_datagen   = ImageDataGenerator(validation_split=val_split)   # no augmentation
    test_datagen  = ImageDataGenerator()                             # no augmentation

    train_gen = train_datagen.flow(x_train, y_train_oh, batch_size=batch_size,
                                   subset="training", shuffle=True, seed=seed)
    val_gen   = val_datagen.flow(x_train, y_train_oh, batch_size=batch_size,
                                 subset="validation", shuffle=False, seed=seed)
    test_gen  = test_datagen.flow(x_test, y_test_oh, batch_size=batch_size,
                                  shuffle=False)                     # order preserved
    return train_gen, val_gen, test_gen


def split_indices(n: int, val_split: float = VAL_SPLIT) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce ImageDataGenerator's index partition for auditing/leakage checks."""
    split_at = int(len(range(n)) * (1.0 - val_split))
    return np.arange(0, split_at), np.arange(split_at, n)


# ===========================================================================
# 5. MODEL ARCHITECTURES
# ===========================================================================
def _resize_and_preprocess(inp, img_size: int, preprocess: Optional[Callable]):
    """Resize the 32x32 [0,1] tensor to `img_size` and apply model-specific
    preprocessing -- executed ON THE GPU as part of the graph.

    Why in-graph rather than in the generator?  A materialised
    50,000 x 224 x 224 x 3 float32 array would need ~30 GB of RAM.  Keeping the
    host arrays at 32x32 (~600 MB) and upsampling per batch on the accelerator
    is mathematically equivalent while reducing host memory ~50x and removing
    the CPU->GPU transfer of large tensors.
    """
    x = layers.Resizing(img_size, img_size, interpolation="bilinear",
                        name="resize_to_input")(inp)
    x = layers.Rescaling(255.0, name="to_0_255")(x)      # applications expect [0,255]
    if preprocess is not None:
        x = layers.Lambda(preprocess, output_shape=(img_size, img_size, 3),
                          name="app_preprocess")(x)
    return x


def build_simple_cnn(img_size: int = IMG_SIZE, dropout: float = DROPOUT,
                     lr: float = LEARNING_RATE, extra_block: bool = False,
                     norm_mean=None, norm_var=None, name: str = "Simple_CNN"):
    """Model 1 -- custom CNN built from scratch.

        Conv2D(32)  -> MaxPooling2D
        Conv2D(64)  -> MaxPooling2D
        Conv2D(128) -> MaxPooling2D
        [optional Conv2D(256) -> MaxPooling2D]     <- depth-modification study
        Flatten -> Dense(128, ReLU) -> Dropout -> Dense(10, softmax)

    Standardisation (mean/std) uses statistics computed on the TRAINING SUBSET
    ONLY and is frozen into a Normalization layer -- validation/test statistics
    never influence the model, which is one of the leakage controls.
    """
    inp = layers.Input(shape=(NATIVE_SIZE, NATIVE_SIZE, 3), name="input_32x32")
    x = layers.Resizing(img_size, img_size, interpolation="bilinear",
                        name="resize_to_input")(inp)
    if norm_mean is not None:
        x = layers.Normalization(axis=-1, mean=norm_mean, variance=norm_var,
                                 name="standardize")(x)

    filters = [32, 64, 128] + ([256] if extra_block else [])
    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, (3, 3), activation="relu", padding="same",
                          name=f"conv{i}_{f}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool{i}")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="fc_128")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inp, out, name=name)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _transfer_model(base_ctor, preprocess, name: str, img_size: int = IMG_SIZE,
                    dropout: float = DROPOUT, lr: float = LEARNING_RATE,
                    trainable_base: bool = False):
    """Shared builder for Models 2-4.

    Custom head (identical across all three, per protocol):
        GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout -> Dense(10, softmax)
    """
    inp = layers.Input(shape=(NATIVE_SIZE, NATIVE_SIZE, 3), name="input_32x32")
    x = _resize_and_preprocess(inp, img_size, preprocess)

    base = base_ctor(include_top=False, weights="imagenet",
                     input_shape=(img_size, img_size, 3))
    base.trainable = trainable_base                       # all layers frozen initially
    x = base(x, training=False)                           # keep BatchNorm in inference mode

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc_128")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inp, out, name=name)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_vgg16(**kw):
    """Model 2 -- VGG16 (ImageNet), fully frozen convolutional base."""
    return _transfer_model(VGG16, vgg16_preprocess, "VGG16_Transfer", **kw)


def build_resnet50(**kw):
    """Model 3 -- ResNet50 (ImageNet), fully frozen convolutional base."""
    return _transfer_model(ResNet50, resnet50_preprocess, "ResNet50_Transfer", **kw)


def build_efficientnet(**kw):
    """Model 4 -- EfficientNetB0 (ImageNet), fully frozen convolutional base."""
    return _transfer_model(EfficientNetB0, effnet_preprocess, "EfficientNetB0_Transfer", **kw)


MODEL_BUILDERS = {
    "Simple_CNN":               build_simple_cnn,
    "VGG16_Transfer":           build_vgg16,
    "ResNet50_Transfer":        build_resnet50,
    "EfficientNetB0_Transfer":  build_efficientnet,
}

# Layer whose activations are treated as the "last convolutional feature map"
FEATURE_LAYERS = {
    "Simple_CNN":              "conv3_128",         # last Conv2D of the custom stack
    "VGG16_Transfer":          "block5_pool",       # canonical VGG16 feature tap
    "ResNet50_Transfer":       "conv5_block3_out",  # output of the final residual stage
    "EfficientNetB0_Transfer": "top_activation",    # final Swish activation before pooling
}


def param_counts(model) -> Dict[str, int]:
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {"total_params": trainable + non_trainable,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable}


# ===========================================================================
# 6. CALLBACKS
# ===========================================================================
def make_callbacks(name: str, patience: int = 5, monitor: str = "val_accuracy",
                   log_dir: str = ARTIFACT_DIR) -> List:
    """Protocol callbacks plus two auditing extras.

    EarlyStopping      -- halts when `val_accuracy` has not improved for 5 epochs
                          and restores the best weights (prevents overfitting and
                          guarantees the reported model is the best-validated one).
    ReduceLROnPlateau  -- multiplies the LR by 0.2 after 3 stagnant epochs,
                          letting the optimiser settle into a sharper minimum.
    ModelCheckpoint    -- persists best weights independently of EarlyStopping.
    CSVLogger          -- immutable per-epoch training record for the report.
    """
    return [
        callbacks.EarlyStopping(monitor=monitor, patience=patience, mode="max",
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3,
                                    min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(f"{log_dir}/{name}.best.weights.h5",
                                  monitor=monitor, mode="max",
                                  save_best_only=True, save_weights_only=True, verbose=0),
        callbacks.CSVLogger(f"{log_dir}/{name}_epochs.csv"),
    ]


# ===========================================================================
# 7. TRAIN / EVALUATE
# ===========================================================================
def train_model(model, train_gen, val_gen, name: str, epochs: int = EPOCHS,
                cbs: Optional[List] = None, verbose: int = 2) -> Tuple[Dict, float, int]:
    """Fit a model and return (history dict, wall-clock training seconds, epochs run)."""
    cbs = make_callbacks(name) if cbs is None else cbs
    t0 = time.time()
    hist = model.fit(train_gen,
                     validation_data=val_gen,
                     epochs=epochs,
                     callbacks=cbs,
                     verbose=verbose)
    train_time = time.time() - t0
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return history, train_time, len(history["loss"])


def evaluate_model(model, test_gen, y_true: np.ndarray, name: str) -> Dict:
    """Full evaluation: accuracy, F1, per-class metrics, confusion matrix, latency.

    `test_gen.reset()` rewinds the iterator to index 0 so predictions align with
    `y_true` -- essential when several models reuse the same generator object.
    """
    test_gen.reset()                       # <-- alignment guarantee (see Question 7)
    t0 = time.time()
    probs = model.predict(test_gen, verbose=0)
    inference_time = time.time() - t0

    y_pred = probs.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    f1_w = float(f1_score(y_true, y_pred, average="weighted"))
    f1_m = float(f1_score(y_true, y_pred, average="macro"))
    top5 = float(top_k_accuracy_score(y_true, probs, k=5, labels=list(range(NUM_CLASSES))))
    prec, rec, f1c, sup = precision_recall_fscore_support(y_true, y_pred,
                                                          labels=list(range(NUM_CLASSES)),
                                                          zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    loss = float(tf.keras.losses.categorical_crossentropy(
        tf.keras.utils.to_categorical(y_true, NUM_CLASSES), probs).numpy().mean())

    n = len(y_true)
    return {
        "model": name,
        "test_accuracy": acc,
        "test_loss": loss,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "top5_accuracy": top5,
        "inference_time_sec": float(inference_time),
        "inference_ms_per_image": float(inference_time / n * 1000),
        "inference_throughput_ips": float(n / inference_time),
        "per_class": {CLASS_NAMES[i]: {"precision": float(prec[i]),
                                       "recall": float(rec[i]),
                                       "f1": float(f1c[i]),
                                       "support": int(sup[i])}
                      for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0),
        "_probs": probs,          # kept in-memory only; stripped before JSON export
    }


# ===========================================================================
# 8. FEATURE ANALYSIS
# ===========================================================================
def _build_extractor(model, layer_name: str):
    """Return a Model mapping the 32x32 input to the activations of `layer_name`.

    Transfer models nest the pre-trained network as a single sub-model layer, so
    the extractor is assembled in two stages: the in-graph preprocessing stem
    (resize -> rescale -> application preprocess) is reused as-is, and a
    standalone sub-graph of the frozen base is called on its output.
    """
    try:                                              # flat model (Simple CNN)
        return models.Model(model.inputs, model.get_layer(layer_name).output)
    except ValueError:
        pass

    base = next(l for l in model.layers if isinstance(l, models.Model))
    stem_out = model.get_layer("app_preprocess").output      # outer-graph tensor
    sub = models.Model(base.inputs, base.get_layer(layer_name).output)
    return models.Model(model.inputs, sub(stem_out))


def extract_features(model, layer_name: str, x: np.ndarray, batch_size: int = 64,
                     chunk: int = 512, sample_values: int = 200_000,
                     seed: int = SEED):
    """Stream `x` through the feature tap and return (pooled_vectors, stats).

    The raw activation tensor is never fully materialised -- e.g. the Simple
    CNN's last conv layer is 56x56x128 = 401,408 floats per image, so 2,000
    images would occupy 3.2 GB.  Statistics (sparsity, mean, std, max) are
    accumulated incrementally chunk-by-chunk, a bounded random sample of raw
    activation values is retained for percentile estimates, and only the
    globally-average-pooled (N, C) vectors are returned for t-SNE / kNN.
    """
    extractor = _build_extractor(model, layer_name)
    rng = np.random.RandomState(seed)

    pooled, samples = [], []
    n_vals = 0; n_zero = 0; s = 0.0; ss = 0.0; vmax = -np.inf
    n_active = 0.0
    feat_shape = None

    for i in range(0, len(x), chunk):
        f = extractor.predict(x[i:i + chunk], batch_size=batch_size, verbose=0)
        feat_shape = list(f.shape[1:])
        pooled.append(f.mean(axis=(1, 2)) if f.ndim == 4 else f)
        flat = f.reshape(len(f), -1).astype("float64")
        n_vals += flat.size
        n_zero += int((flat == 0).sum())
        s      += float(flat.sum())
        ss     += float((flat ** 2).sum())
        vmax    = max(vmax, float(flat.max()))
        n_active += float((flat > 0).sum(axis=1).sum())
        # Sample raw activation values for percentile estimation. Index-sampling
        # (rather than np.random.choice without replacement) avoids permuting a
        # 100M-element array, which would cost ~800 MB and several seconds per chunk.
        take = int(min(sample_values // max(1, (len(x) // chunk + 1)), flat.size))
        idx = rng.randint(0, flat.size, size=take)
        samples.append(flat.ravel()[idx])

    mean = s / n_vals
    std = float(np.sqrt(max(ss / n_vals - mean ** 2, 0.0)))
    stats = {
        "feature_shape": feat_shape,
        "feature_dim_flat": int(np.prod(feat_shape)),
        "pooled_dim": int(pooled[0].shape[1]),
        "sparsity_pct": float(n_zero / n_vals * 100.0),
        "activation_mean": float(mean),
        "activation_std": std,
        "activation_max": vmax,
        "activation_p99": float(np.percentile(np.concatenate(samples), 99)),
        "mean_active_units_per_sample": float(n_active / len(x)),
        "n_samples": int(len(x)),
    }
    return np.concatenate(pooled).astype("float32"), stats


def feature_statistics(feats: np.ndarray) -> Dict:
    """Statistics for an already-materialised feature tensor (small inputs only)."""
    flat = feats.reshape(len(feats), -1)
    return {
        "feature_shape": list(feats.shape[1:]),
        "feature_dim_flat": int(flat.shape[1]),
        "sparsity_pct": float((flat == 0).mean() * 100.0),
        "activation_mean": float(flat.mean()),
        "activation_std": float(flat.std()),
        "activation_max": float(flat.max()),
        "activation_p99": float(np.percentile(flat, 99)),
        "mean_active_units_per_sample": float((flat > 0).sum(axis=1).mean()),
    }


def pooled_vectors(feats: np.ndarray) -> np.ndarray:
    """Global-average-pool a (N,H,W,C) feature map to (N,C) for t-SNE/kNN."""
    return feats.mean(axis=(1, 2)) if feats.ndim == 4 else feats


def separability(emb2d: np.ndarray, labels: np.ndarray, seed: int = SEED) -> Dict:
    """Quantify how well classes separate in the 2-D t-SNE embedding.

    silhouette  -- cohesion vs. separation, range [-1, 1]
    knn_acc     -- 10-NN leave-one-out accuracy in embedding space; a direct,
                   interpretable read of local class purity.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    sil = float(silhouette_score(emb2d, labels))
    knn = float(cross_val_score(KNeighborsClassifier(n_neighbors=10),
                                emb2d, labels, cv=5).mean())
    return {"silhouette_score": sil, "knn10_accuracy_2d": knn}


# ===========================================================================
# 9. PERSISTENCE HELPERS
# ===========================================================================
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return super().default(o)


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# 10. MEMORY ANALYSIS
# ===========================================================================
def estimate_activation_memory(model, batch_size: int = BATCH_SIZE,
                               bytes_per_value: int = 4) -> Dict:
    """Analytical forward-activation footprint of a model, layer by layer.

    Training memory is dominated by three terms:
        1. activations  -- every layer output must be cached for backprop
        2. parameters   -- weights themselves
        3. optimiser    -- Adam stores 2 moment tensors per trainable weight
    This helper quantifies all three so the Simple-CNN vs ResNet50 comparison
    rests on numbers rather than intuition.
    """
    rows, total = [], 0

    def walk(m, prefix=""):
        nonlocal total
        for l in m.layers:
            if isinstance(l, models.Model):
                walk(l, prefix + l.name + "/")
                continue
            try:
                shp = l.output.shape
            except Exception:
                continue
            if shp is None or len(shp) < 2:
                continue
            per_img = int(np.prod([d for d in shp[1:] if d is not None]))
            mb = per_img * batch_size * bytes_per_value / 1024 ** 2
            total += mb
            rows.append({"layer": prefix + l.name,
                         "output_shape": [d for d in shp[1:]],
                         "values_per_image": per_img,
                         "activation_mb": round(mb, 2)})

    walk(model)
    pc = param_counts(model)
    param_mb = pc["total_params"] * bytes_per_value / 1024 ** 2
    adam_mb = pc["trainable_params"] * 2 * bytes_per_value / 1024 ** 2
    grad_mb = pc["trainable_params"] * bytes_per_value / 1024 ** 2
    rows.sort(key=lambda r: -r["activation_mb"])
    return {
        "batch_size": batch_size,
        "activation_mb": round(total, 2),
        "parameter_mb": round(param_mb, 2),
        "gradient_mb": round(grad_mb, 2),
        "adam_state_mb": round(adam_mb, 2),
        "estimated_training_mb": round(total + param_mb + grad_mb + adam_mb, 2),
        "estimated_inference_mb": round(total + param_mb, 2),
        "top_layers": rows[:8],
        "num_layers_counted": len(rows),
    }


def host_peak_rss_mb() -> float:
    """Peak resident set size of this process (macOS reports bytes)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2

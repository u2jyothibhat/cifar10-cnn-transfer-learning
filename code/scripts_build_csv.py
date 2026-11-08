"""Build train.csv / test.csv from the extracted CIFAR-10 python batches,
exactly per the project's data protocol: 1 label column + 3072 pixel columns."""
import os, pickle, numpy as np, pandas as pd

ROOT = "extracted_files/cifar-10-batches-py"

def unpickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing batch file: {path} — re-run extraction.")
    with open(path, "rb") as fo:
        return pickle.load(fo, encoding="bytes")

meta = unpickle(os.path.join(ROOT, "batches.meta"))
label_names = [n.decode() for n in meta[b"label_names"]]
print("classes:", label_names)

# ---- Train: combine the 5 training batches -------------------------------
Xs, ys = [], []
for i in range(1, 6):
    b = unpickle(os.path.join(ROOT, f"data_batch_{i}"))
    Xs.append(b[b"data"]); ys.append(np.array(b[b"labels"]))
Xtr = np.concatenate(Xs).astype(np.uint8)      # (50000, 3072)
ytr = np.concatenate(ys).astype(np.uint8)

# ---- Test ----------------------------------------------------------------
bt = unpickle(os.path.join(ROOT, "test_batch"))
Xte = bt[b"data"].astype(np.uint8)             # (10000, 3072)
yte = np.array(bt[b"labels"]).astype(np.uint8)

cols = ["label"] + [f"p{i}" for i in range(3072)]
for name, X, y in [("data/train.csv", Xtr, ytr), ("data/test.csv", Xte, yte)]:
    df = pd.DataFrame(np.column_stack([y, X]).astype(np.uint8), columns=cols)
    df.to_csv(name, index=False)
    print(name, df.shape)

# Also cache as .npy for fast notebook loading (CSV round-trip is slow)
np.save("data/x_train_raw.npy", Xtr); np.save("data/y_train_raw.npy", ytr)
np.save("data/x_test_raw.npy", Xte);  np.save("data/y_test_raw.npy", yte)
with open("data/label_names.txt", "w") as f:
    f.write("\n".join(label_names))
print("done")

# Model Performance Summary
## CIFAR-10 Image Classification using Deep Learning — Master's Research Project

**Author:** Jyothi Kodenkiri  
**Programme:** Master of Science — Artificial Intelligence & Machine Learning (research project)  
**Generated:** 2026-09-08 13:30 UTC  
**Environment:** Python 3.9.6, TensorFlow 2.16.2,
Keras 3.10.0, GPU: yes (Darwin 25.5.0 (arm64))

---

## 1. Objective

Build, evaluate and compare four convolutional architectures on CIFAR-10 (60,000 colour
images, 10 balanced classes) — one custom CNN trained from scratch and three transfer-learning
models with frozen ImageNet backbones — and determine which is best suited to production
deployment under real compute constraints.

## 2. Method at a glance

| Aspect | Setting |
|---|---|
| Data source | official CIFAR-10 archive → `train.csv` (50,000 × 3,073), `test.csv` (10,000 × 3,073) |
| Split | 40,000 train / 10,000 validation / 10,000 test (test never seen during training) |
| Input | 32×32 native, upsampled in-graph to 224×224×3 |
| Normalisation | ÷255 → [0,1]; per-channel standardisation from the **training subset only** |
| Augmentation | rotation ±15°, shift ±10%, zoom ±10%, horizontal flip — **training partition only** |
| Models | Simple CNN (scratch); VGG16 / ResNet50 / EfficientNetB0 (frozen backbone + GAP→Dense(128)→Dropout(0.5)→Dense(10)) |
| Optimizer / loss | Adam (lr 0.001) / categorical cross-entropy |
| Batch / epochs | 32 / up to 15 |
| Callbacks | EarlyStopping(val_accuracy, patience 5, restore best); ReduceLROnPlateau(val_loss, factor 0.2, patience 3) |
| Seed | 42 (Python, NumPy, TensorFlow, PYTHONHASHSEED) |

## 3. Headline results — 10,000 unseen test images

| Model | Test Acc | F1 (w) | F1 (macro) | Top-5 | Params | Trainable | Epochs | Train (min) | ms/img | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet50_Transfer | 0.9056 | 0.9051 | 0.9051 | 0.9976 | 23,851,274 | 263,562 | 15 | 94.5 | 7.409 | 91.0 |
| EfficientNetB0_Transfer | 0.8974 | 0.8971 | 0.8971 | 0.9970 | 4,214,829 | 165,258 | 15 | 58.8 | 4.444 | 16.1 |
| VGG16_Transfer | 0.8441 | 0.8429 | 0.8429 | 0.9942 | 14,781,642 | 66,954 | 15 | 143.8 | 9.813 | 56.4 |
| Simple_CNN | 0.5533 | 0.5476 | 0.5476 | 0.9472 | 12,939,722 | 12,939,722 | 15 | 44.0 | 0.922 | 49.4 |
| Simple_CNN_Native32 | 0.7315 | 0.7229 | 0.7229 | 0.9798 | 356,810 | 356,810 | 15 | 5.6 | 0.115 | 1.4 |

**Best-performing model: ResNet50_Transfer** — 0.9056 test accuracy,
0.9051 weighted F1, +0.3523
above the weakest architecture. McNemar's exact test on paired predictions confirms the leading
differences are statistically significant rather than sampling noise.

## 4. Training behaviour

| Model | Epochs run | Early stopped | Best epoch | Best val acc | LR start | LR end | LR reductions | Train-val gap |
|---|---|---|---|---|---|---|---|---|
| ResNet50_Transfer | 15 | no | 15 | 0.9136 | 1.0e-03 | 2.0e-04 | 1 | -0.0588 |
| EfficientNetB0_Transfer | 15 | no | 15 | 0.9024 | 1.0e-03 | 1.0e-03 | 0 | -0.0812 |
| VGG16_Transfer | 15 | no | 14 | 0.8467 | 1.0e-03 | 2.0e-04 | 1 | -0.0637 |
| Simple_CNN | 15 | no | 13 | 0.5557 | 1.0e-03 | 4.0e-05 | 2 | -0.1086 |
| Simple_CNN_Native32 | 15 | no | 13 | 0.7363 | 1.0e-03 | 2.0e-04 | 1 | -0.0550 |

## 5. Feature analysis

| Model | Layer | Map | Raw dims | Pooled | Sparsity % | Act mean | Act std | Silhouette | 2-D 10-NN | Linear probe |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet50_Transfer | conv5_block3_out | 7x7x2048 | 100,352 | 2,048 | 83.33 | 0.472 | 1.765 | 0.0953 | 0.7610 | 0.8475 |
| EfficientNetB0_Transfer | top_activation | 7x7x1280 | 62,720 | 1,280 | 0.00 | 0.090 | 0.823 | 0.1361 | 0.7720 | 0.8525 |
| VGG16_Transfer | block5_pool | 7x7x512 | 25,088 | 512 | 87.61 | 1.221 | 4.866 | -0.0235 | 0.6095 | 0.7515 |
| Simple_CNN | conv3_128 | 56x56x128 | 401,408 | 128 | 88.42 | 0.232 | 1.082 | -0.1339 | 0.2515 | 0.4485 |

## 6. Advanced experiments

| Experiment | Result |
|---|---|
| **Fine-tuning** (ResNet50_Transfer, last stage, lr 1e-5) | 0.9056 → 0.9334 (**+0.0278**) |
| **Cross-validation** (3-fold, Simple CNN) | 0.4228 ± 0.0084 validation accuracy |
| **Best ensemble** (`top2_soft_voting`) | 0.9219 (+0.0163 vs best single model) |
| **Compression** | TFLite float16 / dynamic int8 + dense-head pruning — see Analysis Report §7 |
| **Lowest batch-1 latency** | Simple_CNN at 3.47 ms (p50) |

## 7. Key findings

1. **ResNet50_Transfer is the most accurate architecture** at 0.9056, and the
   ranking is statistically significant, not noise.
2. **Parameter count predicts training cost backwards.** ResNet50 carries 1.6× VGG16's
   parameters (23.9 M vs 14.8 M) and updates 4× as many (264 K vs 67 K), yet trains in roughly
   two-thirds of VGG16's wall-clock time *and* scores higher. VGG16 stacks wide 3×3 convolutions
   at high spatial resolution; ResNet50 spends most of its depth on 1×1 bottlenecks over reduced
   feature maps, doing far less arithmetic per forward pass despite holding more weights.
   Freezing removes the backward pass, but it cannot make an arithmetic-heavy backbone cheap.
3. **Learning rate is the highest-leverage hyper-parameter.** One order of magnitude in either
   direction moved accuracy more than any architecture choice. `ReduceLROnPlateau` demonstrably
   rescued the Simple CNN's headline run, which was diverging by epoch 3.
4. **Deeper can mean smaller.** Adding a `Conv2D(256)` block to the Simple CNN halved the
   `Flatten` vector and *reduced* total parameters — 99 % of that architecture's weights live in
   the `Flatten → Dense` transition, not in its convolutions.
5. **Upsampling adds pixels, not information — and for a from-scratch network it is actively
   harmful.** The identical Simple CNN architecture fed at CIFAR's native 32×32 uses
   36× fewer parameters, trains 7.9× faster, infers 8.0× faster, and is
   **+0.1782 more accurate** (0.7315 vs 0.5533). Resizing cannot create information the
   32×32 source never held; it only inflates `Flatten → Dense(128)` to 12.85 M parameters and
   destabilises training. 224 is the right input *only* for the pre-trained backbones.
6. **Feature dimensionality is not feature quality.** The Simple CNN's last convolutional layer
   has the largest raw feature map yet is among the weakest by linear probe and t-SNE separability.

## 8. Production recommendation

* **Maximum accuracy** — ResNet50_Transfer, optionally fine-tuned (Analysis Report §7.1, worth
  +0.0278).
* **Real-time / mobile** — Simple_CNN, lowest batch-1 latency at
  3.47 ms (≈ 268 fps), combined with
  float16 TFLite quantisation for the size reduction measured in §7.4.
* **Constrained retraining** — the frozen transfer models converge within a few epochs and are
  markedly more robust to hyper-parameter error than the from-scratch CNN.

## 9. Limitations

The 32×32 source resolution caps every result; frozen backbones leave measurable headroom
(quantified by the fine-tuning experiment); the parameter-modification study uses a reduced
8,000-image / 6-epoch protocol whose **rankings are controlled but whose absolute values are not
comparable** to the headline numbers; and outside the cross-validation there is one run per
configuration, so differences below ≈ 0.0167 should not be treated as
meaningful without a significance test.

---

*Full methodology, all 15 required question answers, figures and per-class results are in
`Analysis_Report.md` and `CIFAR10_CNN_Capstone.ipynb`.*

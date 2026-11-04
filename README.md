# Cats vs Dogs Image Classifier

**A freeCodeCamp Machine Learning with Python certification project.**

A CNN that distinguishes cat from dog photos, trained on the genuine
Kaggle *Dogs vs Cats* dataset.

> **Substitution note (documented):** the certification spec's reference
> implementation is written in TensorFlow/Keras.  No Keras/TF build is
> installable in this environment, so the identical architecture is
> implemented in **PyTorch**.  When the project's venv carries a CUDA build
> of PyTorch, training runs on the **GPU** (RTX 3060 Laptop, 6 GB) — the
> run prints its device line.  Otherwise it falls back to CPU.

## What it does

`main.py`

| function | result |
|---|---|
| `CatsDogsCNN` | 4-conv-block CNN; output layer has **exactly one unit**, so the output shape is `(None, 1)` as the spec requires |
| `build_dataloaders()` | train/val DataLoaders (4/5 – 1/5 split, resize + crop + flip + ImageNet normalisation) |
| `train()` | BCE-with-logits training on the single logit (Adam) |
| `evaluate()` / `test_accuracy_on()` | held-out accuracy, and the spec's 50-image test accuracy |
| `predict_image()` / `predict_dir()` | `cat` / `dog` for any image or folder |

### Spec targets and measured results

| target | measured (full run, 3 epochs, 8.1 min on the RTX 3060 Laptop) |
|---|---|
| output shape `(None, 1)` | `(None, 1)` |
| validation accuracy ≥ 0.63 | **0.7514** (5,000-image validation set) |
| 50-image test accuracy ≥ 0.70 | **0.7800** |

Training on all 19,998 train images, Adam lr 1e-3, batch 256; the loss
drops 0.705 → 0.483 over 3 epochs.  Two zero-byte JPEGs that ship in the
archive (`Cat/666.jpg`, `Dog/11702.jpg`) are skipped so they cannot crash
the dataloader.

## How to run

```bash
pip install -r requirements.txt
python main.py        # full training run; saves cat_dog_cnn.pt
```

GPU when available — the first line of the run is the device report, e.g.
`device: cuda (NVIDIA GeForce RTX 3060 Laptop GPU, 6143 MB total, ...)`.

## How to run the tests

```bash
python test_local.py
```

The suite (14 assertions) verifies the dataset, the output shape, trains a
bounded 2000-image-per-class subset to prove the pipeline end-to-end (learns
far above chance), and — against the `cat_dog_cnn.pt` checkpoint produced by
`main.py` — asserts the spec's accuracy gates: validation accuracy ≥ 0.63
(measured **0.7514**) and 50-image test accuracy ≥ 0.70 (measured
**0.7800**), plus label sanity on real images.  GPU training has a few
points of run-to-run variance, so the smoke-model checks are deliberately
loose and the exact spec numbers are pinned to the saved checkpoint.

## Data

| | |
|---|---|
| **Dataset** | Kaggle *Dogs vs Cats* (PetImages: 12,500 cats + 12,500 dogs, 750×1045 JPEG) |
| **File** | `PetImages/` (~792 MB as `kagglecatsanddogs_5340.zip`) |
| **Source** | `https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip` |
| **Licence** | CDLA-Permissive-2.0 (see `CDLA-Permissive-2.0.pdf` in the folder) |

Fetch: `python ../../fetch_data.py catsdogs`

## Licence

MIT © 2026 FiredMosquito831. Architecture and training code are original,
written from the published project specification; no freeCodeCamp starter or
reference code was used.


## Verification

Everything in this repository was built and tested locally before publication.

| | |
|---|---|
| Tests | **14/14 passing** |
| Interpreter | `projects-ours/.venv` (Python 3.13, torch 2.15.0+cu132) |
| Device | **cuda** |

The full captured test output is committed as [`local-test.log`](local-test.log).

### Data

- **PetImages/** — 24998 rows — <https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip>

### Metrics

- **val_accuracy**: 0.7514
- **test50_accuracy**: 0.78
- **output_shape**: (None,1)

## Source

Written from scratch against the project's published specification. No freeCodeCamp
starter or reference-solution code was used.

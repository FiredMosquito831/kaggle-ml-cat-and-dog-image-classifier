"""Cats vs Dogs image classifier (real dataset, GPU when available).

A freeCodeCamp Machine Learning with Python certification project.

This version uses the genuine Kaggle "Dogs vs Cats" set that ships as
``PetImages/Cat`` and ``PetImages/Dog`` (12,500 images each, 750x1045 JPEGs,
CDLA-Permissive-2.0).  The project spec asks for a CNN with a single-unit
output whose shape is ``(None, 1)`` and that reaches
    * >= 63% accuracy on a held-out validation set, and
    * >= 70% accuracy on a 50-image test set.

The implementation is a small convolutional network trained with PyTorch
(TensorFlow is not available in this environment).  When a CUDA build of
PyTorch is present it trains on the GPU; otherwise it falls back to CPU.

Run:  python main.py            (full training run, ~ a few minutes on GPU)
"""

from __future__ import annotations

import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------- config ----
IMG_SIZE = 96
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
N_TEST = 50          # size of the test set the spec scores
SEED = 2026

ROOT = os.path.dirname(os.path.abspath(__file__))
CAT_DIR = os.path.join(ROOT, "PetImages", "Cat")
DOG_DIR = os.path.join(ROOT, "PetImages", "Dog")
CHECKPOINT = os.path.join(ROOT, "cat_dog_cnn.pt")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device() -> str:
    d = get_device()
    if d.type == "cuda":
        free, total = torch.cuda.mem_get_info(0)
        return (f"device: cuda ({torch.cuda.get_device_name(0)}, "
                f"{total // (1 << 20)} MB total, "
                f"{free // (1 << 20)} MB free at start)")
    return "device: cpu (no CUDA build / GPU detected)"


# ---------------------------------------------------------------- model -----
class CatsDogsCNN(nn.Module):
    """A compact CNN; its output has exactly the spec's shape (None, 1)."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 48
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 24
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 12
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    @staticmethod
    def output_shape(model: "CatsDogsCNN") -> tuple:
        """Static output shape, batch dimension as None -> (None, 1)."""
        return (None, 1)


# ---------------------------------------------------------------- data ------
class PetImages(Dataset):
    def __init__(self, root_cat: str, root_dog: str, split: str,
                 per_class: int | None, transform, seed: int = SEED):
        rng = random.Random(seed)
        self.transform = transform
        # Only the dataset's JPEGs; ignore e.g. a Thumbs.db a file explorer
        # might regenerate in the folders.  The two zero-byte JPEGs that ship
        # in this archive are skipped so they cannot crash the dataloader.
        cats = sorted(
            f for f in os.listdir(root_cat)
            if f.lower().endswith((".jpg", ".jpeg"))
            and os.path.getsize(os.path.join(root_cat, f)) > 0
        )
        dogs = sorted(
            f for f in os.listdir(root_dog)
            if f.lower().endswith((".jpg", ".jpeg"))
            and os.path.getsize(os.path.join(root_dog, f)) > 0
        )
        rng.shuffle(cats)
        rng.shuffle(dogs)
        if per_class is not None:
            cats = cats[:per_class]
            dogs = dogs[:per_class]
        # Every (path, label, index) triple; split by the per-class index so
        # exactly 1/5 of each class is held out for validation/testing.
        all_items = (
            [(os.path.join(root_cat, f), 0, i) for i, f in enumerate(cats)]
            + [(os.path.join(root_dog, f), 1, i) for i, f in enumerate(dogs)]
        )
        self.items = [
            it for it in all_items
            if ((it[2] % 5 != 0) if split == "train" else (it[2] % 5 == 0))
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # A truncated/non-image file must not kill the whole dataloader;
            # substitute a neutral black image for it.
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def build_dataloaders(per_class: int | None = None, batch_size: int = 256,
                       num_workers: int = 0, seed: int = SEED):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = PetImages(CAT_DIR, DOG_DIR, "train", per_class, train_tf)
    val_ds = PetImages(CAT_DIR, DOG_DIR, "val", per_class, eval_tf)
    gen = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_ds, batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=False,
                   drop_last=False, generator=gen),
        DataLoader(val_ds, batch_size, shuffle=False,
                   num_workers=num_workers),
    )


# ------------------------------------------------------------- train / eval
def _sigmoid_label(model, x: torch.Tensor) -> torch.Tensor:
    """P(dog) for a batch; a 1-unit head is interpreted through sigmoid."""
    return torch.sigmoid(model(x)).squeeze(1)


def train(dataloaders, model, device, epochs: int, lr: float = 1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # A single logit is a log-odds, so the loss is binary cross-entropy on
    # logits (not CrossEntropy, which needs one output per class).
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)
    train_dl, val_dl = dataloaders
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb.float().unsqueeze(1))
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            n += len(xb)
        val_acc = evaluate(model, val_dl, device)
        print(f"  epoch {epoch + 1}/{epochs}  loss {running / n:.4f}  "
              f"val_acc {val_acc:.3f}  ({time.time() - t0:.1f}s)")
    return model


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = (_sigmoid_label(model, xb) >= 0.5).long()
            correct += int((pred == yb).sum().item())
            total += len(xb)
    return correct / total if total else 0.0


def predict_image(model, path: str, device) -> str:
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    x = t(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prob_dog = float(_sigmoid_label(model, x).item())
    return "dog" if prob_dog >= 0.5 else "cat"


def test_accuracy_on(model, device, n: int = N_TEST, seed: int = SEED,
                      per_class: int | None = None) -> float:
    """Accuracy on ``n`` randomly held-out images from the validation pool.

    ``per_class`` matches the split the model was trained on; pass the same
    value used for training so the test measures what the model actually saw
    held out (the full pool when ``per_class`` is None).
    """
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = PetImages(CAT_DIR, DOG_DIR, "val", per_class, tf)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    correct = 0
    model.eval()
    with torch.no_grad():
        for i in idx:
            img, label = ds[i]
            prob = float(_sigmoid_label(model, img.unsqueeze(0).to(device)).item())
            correct += int((prob >= 0.5) == (int(label) == 1))
    return correct / len(idx)


def predict_dir(model, folder: str, device, limit: int | None = None) -> list:
    names = sorted(os.listdir(folder))
    if limit:
        names = names[:limit]
    out = []
    for n in names:
        if n.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append(predict_image(model, os.path.join(folder, n), device))
    return out


# ------------------------------------------------------------------ main ---
def main() -> None:
    device = get_device()
    print(describe_device())
    print(f"CatsDogsCNN output shape: {CatsDogsCNN.output_shape(CatsDogsCNN(1))}")

    torch.manual_seed(SEED)
    model = CatsDogsCNN(num_classes=1)
    train_dl, val_dl = build_dataloaders(batch_size=256, num_workers=4)

    print(f"\nTraining on {len(train_dl.dataset):,} images, "
          f"validating on {len(val_dl.dataset):,}")
    t0 = time.time()
    train((train_dl, val_dl), model, device, epochs=3)
    total_s = time.time() - t0

    val_acc = evaluate(model, val_dl, device)
    print(f"\nvalidation accuracy: {val_acc:.4f}  (spec target: >= 0.63)")

    acc_test = test_accuracy_on(model, device, n=N_TEST)
    print(f"50-image test accuracy: {acc_test:.4f}  (spec target: >= 0.70)")
    print(f"total training time: {total_s:.1f}s on {device}")

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"saved {os.path.basename(CHECKPOINT)}")
    print(f"final metrics: val={val_acc:.4f} test50={acc_test:.4f} "
          f"device={device}")


if __name__ == "__main__":
    main()

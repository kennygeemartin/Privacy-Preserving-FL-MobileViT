from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import kagglehub
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_ALIASES = {
    "normal": "normal", "healthy": "normal",
    "pneumonia": "pneumonia",
    "tuberculosis": "tuberculosis", "tb": "tuberculosis",
    "covid": "covid-19", "covid19": "covid-19", "covid-19": "covid-19",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def download_dataset(handle: str) -> Path:
    try:
        path = Path(kagglehub.dataset_download(handle))
    except Exception as exc:
        owner, dataset = handle.split("/", 1)
        versions = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / dataset / "versions"
        cached = sorted((p for p in versions.glob("*") if p.is_dir()),
                        key=lambda p: int(p.name) if p.name.isdigit() else -1)
        if not cached:
            raise RuntimeError(f"Dataset download failed and no cached copy exists: {exc}") from exc
        path = cached[-1]
        print(f"Network unavailable; using cached dataset version at {path}")
    print(f"Path to dataset files: {path}")
    return path


def infer_label(path: Path, root: Path) -> str | None:
    tokens = []
    for part in path.relative_to(root).parts[:-1]:
        tokens.extend(re.split(r"[^a-z0-9-]+", part.lower()))
    stem = re.split(r"[^a-z0-9-]+", path.stem.lower())
    tokens.extend(stem[:2])
    compact = [token.replace("_", "").replace(" ", "") for token in tokens]
    for token in reversed(compact):
        if token in LABEL_ALIASES:
            return LABEL_ALIASES[token]
        if token.startswith("covid"):
            return "covid-19"
        if token.startswith("pneumonia"):
            return "pneumonia"
        if token.startswith("tuberculosis"):
            return "tuberculosis"
        if token == "normal" or token.startswith("normal"):
            return "normal"
    return None


def discover_images(root: Path) -> list[dict]:
    records = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = infer_label(path, root)
            if label is not None:
                records.append({"path": str(path.resolve()), "label": label})
    if not records:
        raise RuntimeError(f"No labelled images found under {root}")
    return records


def stratified_splits(records: list[dict], seed: int) -> dict[str, list[dict]]:
    labels = [r["label"] for r in records]
    train, remainder = train_test_split(records, test_size=0.30, stratify=labels, random_state=seed)
    rem_labels = [r["label"] for r in remainder]
    val, test = train_test_split(remainder, test_size=0.50, stratify=rem_labels, random_state=seed)
    return {"train": train, "val": val, "test": test}


def dirichlet_partition(records: list[dict], classes: list[str], clients: int,
                        alpha: float, seed: int) -> list[list[dict]]:
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in range(clients)]
    for class_name in classes:
        items = [r for r in records if r["label"] == class_name]
        rng.shuffle(items)
        proportions = rng.dirichlet(np.full(clients, alpha))
        cuts = (np.cumsum(proportions)[:-1] * len(items)).astype(int)
        for client_items, partition in zip(np.split(np.array(items, dtype=object), cuts), partitions):
            partition.extend(client_items.tolist())
    for partition in partitions:
        rng.shuffle(partition)
    return partitions


class CXRDataset(Dataset):
    def __init__(self, records: list[dict], class_to_idx: dict[str, int], image_size: int,
                 train: bool, clip_limit: float, grid_size: int):
        self.records = records
        self.class_to_idx = class_to_idx
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        ops = []
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                    transforms.RandomAffine(0, scale=(0.9, 1.1))]
        ops += [transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        gray = cv2.imread(record["path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Unable to read image: {record['path']}")
        clahe = cv2.createCLAHE(self.clip_limit, (self.grid_size, self.grid_size))
        enhanced = clahe.apply(gray)
        image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
        return self.transform(image), self.class_to_idx[record["label"]]


def create_model(num_classes: int, pretrained: bool) -> nn.Module:
    import timm
    try:
        return timm.create_model("mobilevit_xs", pretrained=pretrained, num_classes=num_classes)
    except RuntimeError as exc:
        if pretrained:
            print(f"Warning: pretrained weights unavailable ({exc}); using random initialization")
            return timm.create_model("mobilevit_xs", pretrained=False, num_classes=num_classes)
        raise


def loader(records, cfg, class_to_idx, train, shuffle=None):
    ds = CXRDataset(records, class_to_idx, cfg["data"]["image_size"], train,
                    cfg["data"]["clahe_clip_limit"], cfg["data"]["clahe_grid_size"])
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                      shuffle=train if shuffle is None else shuffle,
                      num_workers=cfg["training"]["num_workers"])


def train_epoch(model, data_loader, optimizer, device, privacy_cfg=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        if privacy_cfg and privacy_cfg.get("enabled"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), privacy_cfg["max_grad_norm"])
            # Simulation-level noisy clipped updates. For formal sample-level accounting,
            # use the supplied Opacus pathway in a dedicated DP experiment.
            sigma = privacy_cfg["noise_multiplier"] * privacy_cfg["max_grad_norm"]
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.add_(torch.randn_like(parameter.grad) * sigma / max(1, len(targets)))
        optimizer.step()
        total_loss += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()
        total += len(targets)
    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.inference_mode()
def predict(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    ys, preds, probs, loss = [], [], [], 0.0
    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss += criterion(logits, targets).item()
        probability = logits.softmax(1)
        ys.extend(targets.cpu().tolist())
        preds.extend(probability.argmax(1).cpu().tolist())
        probs.extend(probability.cpu().tolist())
    return np.array(ys), np.array(preds), np.array(probs), loss / len(ys)


def aggregate_fedavg(states: list[dict], sizes: list[int]) -> dict:
    total = sum(sizes)
    result = copy.deepcopy(states[0])
    for key in result:
        if result[key].is_floating_point():
            result[key] = sum(state[key] * (size / total) for state, size in zip(states, sizes))
        else:
            result[key] = states[0][key]
    return result


def evaluate_metrics(y, pred, prob, classes, loss):
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    metrics = {"loss": float(loss), "accuracy": float(accuracy_score(y, pred)),
               "precision_macro": float(precision), "recall_macro": float(recall),
               "f1_macro": float(f1)}
    try:
        metrics["auroc_macro_ovr"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["auroc_macro_ovr"] = None
    metrics["class_sensitivity"] = {
        name: float(((pred[y == idx]) == idx).mean()) if np.any(y == idx) else None
        for idx, name in enumerate(classes)
    }
    return metrics


def save_manifests(run_dir, splits, partitions):
    for name, records in splits.items():
        with open(run_dir / f"{name}_split.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label"])
            writer.writeheader(); writer.writerows(records)
    with open(run_dir / "client_partitions.json", "w", encoding="utf-8") as handle:
        json.dump({f"client_{i+1}": {"count": len(p), "classes": dict(Counter(r['label'] for r in p))}
                   for i, p in enumerate(partitions)}, handle, indent=2)


def run_training(args, cfg, root):
    seed_everything(cfg["experiment"]["seed"])
    records = discover_images(root)
    if args.max_images:
        limited = []
        by_class = {c: [r for r in records if r["label"] == c] for c in sorted(set(r["label"] for r in records))}
        per_class = max(2, args.max_images // len(by_class))
        for items in by_class.values(): limited.extend(items[:per_class])
        records = limited
    classes = sorted(set(r["label"] for r in records))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    splits = stratified_splits(records, cfg["experiment"]["seed"])
    partitions = dirichlet_partition(splits["train"], classes, cfg["federated"]["clients"],
                                     cfg["federated"]["dirichlet_alpha"], cfg["federated"]["partition_seed"])
    partitions = [p for p in partitions if p]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg["experiment"]["output_dir"]) / f"{args.mode}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_manifests(run_dir, splits, partitions)
    with open(run_dir / "classes.json", "w", encoding="utf-8") as f: json.dump(class_to_idx, f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = cfg["model"]["pretrained"] and not args.no_pretrained
    model = create_model(len(classes), pretrained).to(device)
    val_loader = loader(splits["val"], cfg, class_to_idx, False)
    rounds = args.rounds or cfg["training"]["rounds"]
    local_epochs = args.local_epochs or cfg["training"]["local_epochs"]
    history, best_acc, best_state = [], -1.0, None

    if args.mode == "centralized": partitions = [splits["train"]]
    if args.mode == "local": partitions = [partitions[0]]
    for round_idx in range(1, rounds + 1):
        states, sizes, train_stats = [], [], []
        for client_records in partitions:
            local_model = copy.deepcopy(model)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=cfg["training"]["learning_rate"],
                                           weight_decay=cfg["training"]["weight_decay"])
            client_loader = loader(client_records, cfg, class_to_idx, True)
            stat = None
            for _ in range(local_epochs):
                stat = train_epoch(local_model, client_loader, optimizer, device, cfg["privacy"])
            states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            sizes.append(len(client_records)); train_stats.append(stat)
            del local_model
        model.load_state_dict(aggregate_fedavg(states, sizes)); model.to(device)
        y, pred, prob, val_loss = predict(model, val_loader, device)
        val_metrics = evaluate_metrics(y, pred, prob, classes, val_loss)
        entry = {"round": round_idx, "train_loss": float(np.average([s["loss"] for s in train_stats], weights=sizes)),
                 "train_accuracy": float(np.average([s["accuracy"] for s in train_stats], weights=sizes)),
                 "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]}
        history.append(entry); print(json.dumps(entry))
        if val_metrics["accuracy"] > best_acc:
            best_acc, best_state = val_metrics["accuracy"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_loader = loader(splits["test"], cfg, class_to_idx, False)
    y, pred, prob, test_loss = predict(model, test_loader, device)
    metrics = evaluate_metrics(y, pred, prob, classes, test_loss)
    checkpoint = {"model_name": cfg["model"]["name"], "state_dict": model.cpu().state_dict(),
                  "class_to_idx": class_to_idx, "image_size": cfg["data"]["image_size"], "metrics": metrics}
    torch.save(checkpoint, run_dir / "best_model.pt")
    scripted = torch.jit.trace(model, torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"]))
    scripted.save(str(run_dir / "model_scripted.pt"))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    report = classification_report(y, pred, target_names=classes, output_dict=True, zero_division=0)
    import pandas as pd
    pd.DataFrame(report).transpose().to_csv(run_dir / "classification_report.csv")
    pd.DataFrame(confusion_matrix(y, pred), index=classes, columns=classes).to_csv(run_dir / "confusion_matrix.csv")
    print(f"Artifacts: {run_dir.resolve()}"); print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Nigeria CXR MobileViT training")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("inspect")
    train = sub.add_parser("train")
    train.add_argument("--mode", choices=["federated", "centralized", "local"], default="federated")
    train.add_argument("--rounds", type=int)
    train.add_argument("--local-epochs", type=int)
    train.add_argument("--max-images", type=int)
    train.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(); cfg = load_config(args.config)
    root = Path(cfg["data"]["root"]) if cfg["data"]["root"] else download_dataset(cfg["data"]["kaggle_handle"])
    if args.command == "download": return
    records = discover_images(root)
    if args.command == "inspect":
        print(json.dumps({"root": str(root), "total": len(records), "classes": dict(Counter(r["label"] for r in records))}, indent=2)); return
    run_training(args, cfg, root)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Train teacher model (supervised) to bootstrap distillation workflow.

Usage:
    python train_teacher.py --config config/default.yaml
    python train_teacher.py --config config/default.yaml --data_only
    python train_teacher.py --config config/default.yaml --test_only
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext
from typing import Dict

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TeacherModel
from data import create_data_loaders
from utils import load_config, get_device, clear_gpu_cache


def maybe_compile_model(model, enabled: bool, mode: str, label: str):
    """Compile model when requested, with safe fallback on unsupported options."""
    if not enabled or not hasattr(torch, "compile"):
        return model
    try:
        print(f"Compiling {label} model (mode={mode})...")
        return torch.compile(model, mode=mode)
    except (TypeError, ValueError) as e:
        print(f"Compile mode '{mode}' unsupported for {label}; retrying default mode. ({e})")
        try:
            return torch.compile(model)
        except Exception as inner_e:
            print(f"Compilation skipped for {label}: {inner_e}")
            return model


def get_teacher_save_dir(config) -> str:
    return config.training.get("save_dir", "checkpoints").rstrip("/\\") + "_teacher"


def get_teacher_log_dir(config) -> str:
    return config.logging.get("log_dir", "logs").rstrip("/\\") + "_teacher"


def parse_args():
    parser = argparse.ArgumentParser(description="Train teacher model")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override configuration parameters (e.g., training.batch_size 64)",
    )
    parser.add_argument("--data_only", action="store_true", help="Only test data loading, don't train")
    parser.add_argument("--test_only", action="store_true", help="Only evaluate trained teacher model")
    return parser.parse_args()


def apply_config_overrides(config, overrides):
    if not overrides:
        return config

    print("Applying configuration overrides...")
    overrides_dict = {}
    for i in range(0, len(overrides), 2):
        if i + 1 < len(overrides):
            key = overrides[i]
            value = overrides[i + 1]
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            overrides_dict[key] = value
            print(f"  {key}: {value}")

    config.update(overrides_dict)
    return config


def create_data_loaders_from_config(config, balanced_sampler=None):
    data_config = config.data
    training_config = config.training
    performance_config = training_config.get("performance", {})

    data_loaders, datasets = create_data_loaders(
        train_dir=data_config["train_dir"],
        val_dir=data_config["val_dir"],
        test_dir=data_config.get("test_dir"),
        batch_size=training_config["batch_size"],
        num_workers=training_config.get("num_workers", 4),
        image_size=data_config.get("image_size", 224),
        normalize_mean=data_config.get("normalize", {}).get("mean"),
        normalize_std=data_config.get("normalize", {}).get("std"),
        augmentations=data_config.get("augmentations", {}),
        balanced_sampler=balanced_sampler,
        pin_memory=performance_config.get("pin_memory", True),
        persistent_workers=performance_config.get("persistent_workers"),
        prefetch_factor=performance_config.get("prefetch_factor"),
        drop_last_train=performance_config.get("drop_last_train", True),
    )

    for split, dataset in datasets.items():
        print(f"{split.capitalize()} dataset: {len(dataset)} samples")
        if hasattr(dataset, "get_class_distribution"):
            print(f"  Distribution: {dataset.get_class_distribution()}")

    return data_loaders, datasets


def create_teacher_model(config) -> TeacherModel:
    teacher_config = config.model["teacher"]
    model = TeacherModel(
        num_classes=teacher_config["num_classes"],
        pretrained=teacher_config.get("pretrained", True),
        model_name=teacher_config.get("name", "efficientnet_b0"),
        freeze_all=False,
    )
    print(f"Teacher model: {model.get_model_info()}")
    return model


def set_teacher_trainable(teacher_model: TeacherModel, freeze_backbone: bool) -> None:
    if not freeze_backbone:
        for p in teacher_model.parameters():
            p.requires_grad = True
        return

    for p in teacher_model.parameters():
        p.requires_grad = False

    classifier = None
    if hasattr(teacher_model.model, "get_classifier"):
        classifier = teacher_model.model.get_classifier()

    if isinstance(classifier, nn.Module):
        for p in classifier.parameters():
            p.requires_grad = True
        return

    for attr in ("classifier", "fc", "head"):
        module = getattr(teacher_model.model, attr, None)
        if isinstance(module, nn.Module):
            for p in module.parameters():
                p.requires_grad = True


def create_balanced_sampler(dataset):
    labels = dataset.labels
    num_classes = max(labels) + 1 if labels else 0
    class_counts = [0] * num_classes
    for label in labels:
        class_counts[label] += 1

    class_weights = [0.0] * num_classes
    for i, count in enumerate(class_counts):
        class_weights[i] = (len(labels) / (num_classes * count)) if count > 0 else 0.0

    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def create_class_weights(dataset, device: torch.device):
    labels = dataset.labels
    num_classes = max(labels) + 1 if labels else 0
    class_counts = [0] * num_classes
    for label in labels:
        class_counts[label] += 1

    weights = []
    for count in class_counts:
        weights.append((len(labels) / (num_classes * count)) if count > 0 else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _autocast_ctx(use_amp: bool, amp_dtype: torch.dtype):
    if use_amp:
        return torch.cuda.amp.autocast(dtype=amp_dtype)
    return nullcontext()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    desc: str,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    scaler=None,
    channels_last: bool = True,
    grad_clip_norm=None
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last and images.dim() == 4:
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(use_amp, amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    desc: str,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = True
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last and images.dim() == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        with _autocast_ctx(use_amp, amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
    }


def save_checkpoint(save_dir, model, optimizer, epoch, metrics, phase, is_best):
    checkpoint = {
        "epoch": epoch,
        "teacher_model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "phase": phase,
        "model_info": model.get_model_info(),
    }

    epoch_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_phase_{phase}.pth")
    torch.save(checkpoint, epoch_path)

    latest_path = os.path.join(save_dir, "latest_teacher_model.pth")
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = os.path.join(save_dir, "best_teacher_model.pth")
        torch.save(checkpoint, best_path)
        print(f"New best teacher saved: val_accuracy={metrics['val_accuracy']:.4f}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["teacher_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def test_data_loading(config):
    print("Testing data loading...")
    create_data_loaders_from_config(config)
    print("Data loading test completed successfully!")


def test_model_only(config):
    print("Testing trained teacher model...")
    device = get_device(config.training.get("device", "auto"))
    performance_config = config.training.get("performance", {})
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = performance_config.get("cudnn_benchmark", True)
        torch.backends.cuda.matmul.allow_tf32 = performance_config.get("allow_tf32", True)
        torch.backends.cudnn.allow_tf32 = performance_config.get("allow_tf32", True)
        torch.set_float32_matmul_precision(performance_config.get("float32_matmul_precision", "high"))

    model = create_teacher_model(config).to(device)
    use_amp = bool(performance_config.get("use_amp", True) and device.type == "cuda")
    amp_dtype = torch.float16 if str(performance_config.get("amp_dtype", "float16")).lower() == "float16" else torch.bfloat16
    channels_last = bool(performance_config.get("channels_last", True) and device.type == "cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    save_dir = get_teacher_save_dir(config)
    checkpoint_path = os.path.join(save_dir, "best_teacher_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No teacher checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["teacher_model_state_dict"])

    data_loaders, _ = create_data_loaders_from_config(config)
    criterion = nn.CrossEntropyLoss()
    eval_loader = data_loaders.get("test") or data_loaders["val"]
    metrics = evaluate(
        model,
        eval_loader,
        criterion,
        device,
        "Teacher Evaluation",
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
    )
    print(f"Teacher accuracy: {metrics['accuracy']:.4f}, loss: {metrics['loss']:.4f}")


def train_model(config, resume_path=None):
    print("Starting teacher training...")
    device = get_device(config.training.get("device", "auto"))
    performance_config = config.training.get("performance", {})
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = performance_config.get("cudnn_benchmark", True)
        torch.backends.cuda.matmul.allow_tf32 = performance_config.get("allow_tf32", True)
        torch.backends.cudnn.allow_tf32 = performance_config.get("allow_tf32", True)
        torch.set_float32_matmul_precision(performance_config.get("float32_matmul_precision", "high"))

    model = create_teacher_model(config).to(device)
    use_amp = bool(performance_config.get("use_amp", True) and device.type == "cuda")
    amp_dtype = torch.float16 if str(performance_config.get("amp_dtype", "float16")).lower() == "float16" else torch.bfloat16
    channels_last = bool(performance_config.get("channels_last", True) and device.type == "cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    compile_mode = performance_config.get("compile_mode", "default")
    model = maybe_compile_model(
        model,
        enabled=performance_config.get("compile_teacher", False),
        mode=compile_mode,
        label="teacher",
    )

    # Optional class imbalance handling
    class_imbalance = config.training.get("class_imbalance", {})
    use_class_weights = class_imbalance.get("use_class_weights", False)
    use_balanced_sampler = class_imbalance.get("use_balanced_sampler", False)

    data_loaders, datasets = create_data_loaders_from_config(config)
    if use_balanced_sampler:
        sampler = create_balanced_sampler(datasets["train"])
        data_loaders, datasets = create_data_loaders_from_config(config, balanced_sampler=sampler)

    class_weights = create_class_weights(datasets["train"], device) if use_class_weights else None

    log_dir = get_teacher_log_dir(config)
    save_dir = get_teacher_save_dir(config)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    phases_config = config.training["phases"]
    best_val_accuracy = 0.0
    start_epoch = 0
    resume_loaded = False

    for phase_name, phase_config in phases_config.items():
        freeze_backbone = phase_config.get("freeze_backbone", False)
        set_teacher_trainable(model, freeze_backbone=freeze_backbone)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        adamw_kwargs = {
            "lr": float(phase_config["learning_rate"]),
            "weight_decay": float(phase_config["weight_decay"]),
        }
        if device.type == "cuda" and performance_config.get("adamw_fused", True):
            adamw_kwargs["fused"] = True
        try:
            optimizer = optim.AdamW(trainable_params, **adamw_kwargs)
        except TypeError:
            adamw_kwargs.pop("fused", None)
            optimizer = optim.AdamW(trainable_params, **adamw_kwargs)

        if resume_path and (not resume_loaded) and os.path.exists(resume_path):
            checkpoint = load_checkpoint(model, optimizer, resume_path, device)
            start_epoch = checkpoint["epoch"] + 1
            best_val_accuracy = checkpoint.get("metrics", {}).get("val_accuracy", 0.0)
            resume_loaded = True
            print(f"Resumed from epoch {start_epoch}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        phase_epochs = phase_config["epochs"]
        save_every = config.training.get("save_every", 5)

        for epoch in range(start_epoch, start_epoch + phase_epochs):
            epoch_start = time.time()
            train_metrics = run_epoch(
                model=model,
                loader=data_loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                desc=f"Epoch {epoch} [{phase_name}] Train",
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                channels_last=channels_last,
                grad_clip_norm=performance_config.get("grad_clip_norm"),
            )
            val_metrics = evaluate(
                model=model,
                loader=data_loaders["val"],
                criterion=criterion,
                device=device,
                desc=f"Epoch {epoch} [{phase_name}] Val",
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
            )

            combined = {
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "epoch_time_sec": time.time() - epoch_start,
            }

            writer.add_scalar("Teacher/TrainLoss", combined["train_loss"], epoch)
            writer.add_scalar("Teacher/TrainAccuracy", combined["train_accuracy"], epoch)
            writer.add_scalar("Teacher/ValLoss", combined["val_loss"], epoch)
            writer.add_scalar("Teacher/ValAccuracy", combined["val_accuracy"], epoch)

            is_best = combined["val_accuracy"] > best_val_accuracy
            if is_best:
                best_val_accuracy = combined["val_accuracy"]

            if (epoch % save_every == 0) or (epoch == start_epoch + phase_epochs - 1) or is_best:
                save_checkpoint(
                    save_dir=save_dir,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=combined,
                    phase=phase_name,
                    is_best=is_best,
                )

            print(
                f"Epoch {epoch} [{phase_name}] "
                f"train_acc={combined['train_accuracy']:.4f} "
                f"val_acc={combined['val_accuracy']:.4f}"
            )

            if device.type == "cuda" and performance_config.get("empty_cache_each_epoch", False):
                clear_gpu_cache()

        start_epoch += phase_epochs

    if "test" in data_loaders:
        best_path = os.path.join(save_dir, "best_teacher_model.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["teacher_model_state_dict"])

        test_metrics = evaluate(
            model=model,
            loader=data_loaders["test"],
            criterion=nn.CrossEntropyLoss(),
            device=device,
            desc="Teacher Final Test",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        print(f"Final teacher test accuracy: {test_metrics['accuracy']:.4f}")

    writer.close()
    print(f"Teacher training complete. Best val accuracy: {best_val_accuracy:.4f}")
    print(f"Teacher checkpoints: {save_dir}")
    print(f"Teacher logs: {log_dir}")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.override:
        config = apply_config_overrides(config, args.override)
    config.validate()

    if args.data_only:
        test_data_loading(config)
    elif args.test_only:
        test_model_only(config)
    else:
        train_model(config, resume_path=args.resume)


if __name__ == "__main__":
    main()

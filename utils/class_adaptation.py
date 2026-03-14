import os
from typing import Any, Dict, Optional


CLASS_PROFILES: Dict[int, Dict[str, Any]] = {
    10: {
        "distillation": {"alpha": 0.75, "temperature": 3.5, "label_smoothing": 0.02},
        "training": {
            "batch_size": 64,
            "num_workers": 6,
            "label_smoothing": 0.02,
            "phases": {
                "phase1": {"epochs": 10, "learning_rate": 0.0011, "weight_decay": 1.2e-4},
                "phase2": {"epochs": 8, "learning_rate": 0.00018, "weight_decay": 1.2e-5},
            },
        },
        "model": {"student": {"unfreeze_last_blocks": 2}},
        "data": {
            "augmentations": {
                "horizontal_flip": 0.5,
                "rotation_degrees": 12,
                "color_jitter": {
                    "brightness": 0.18,
                    "contrast": 0.18,
                    "saturation": 0.18,
                    "hue": 0.06,
                },
            }
        },
    },
    40: {
        "distillation": {"alpha": 0.6, "temperature": 5.0, "label_smoothing": 0.05},
        "training": {
            "batch_size": 32,
            "num_workers": 8,
            "label_smoothing": 0.05,
            "class_imbalance": {"use_class_weights": True},
            "phases": {
                "phase1": {"epochs": 14, "learning_rate": 0.0009, "weight_decay": 2.5e-4},
                "phase2": {"epochs": 12, "learning_rate": 0.00012, "weight_decay": 6e-5},
            },
        },
        "model": {"student": {"unfreeze_last_blocks": 3}},
        "data": {
            "augmentations": {
                "horizontal_flip": 0.5,
                "rotation_degrees": 16,
                "color_jitter": {
                    "brightness": 0.22,
                    "contrast": 0.22,
                    "saturation": 0.22,
                    "hue": 0.09,
                },
            }
        },
    },
}


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def infer_class_count(train_dir: str) -> Optional[int]:
    if not train_dir or not os.path.isdir(train_dir):
        return None

    class_dirs = [
        name
        for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    ]
    return len(class_dirs) if class_dirs else None


def adapt_config_for_class_count(config, class_profile: str = "auto") -> Optional[int]:
    training_cfg = config.training if isinstance(config.training, dict) else {}
    adaptation_cfg = training_cfg.get("class_adaptation", {})

    mode = str(class_profile).lower().strip()
    if mode == "off":
        return None

    if not adaptation_cfg.get("enabled", True):
        return None

    forced_count = None
    if mode in {"10", "40"}:
        forced_count = int(mode)

    detected_count = infer_class_count(config.data.get("train_dir", ""))
    class_count = forced_count if forced_count is not None else detected_count
    if class_count is None:
        return None

    if adaptation_cfg.get("auto_set_num_classes", True):
        config.model.setdefault("student", {})["num_classes"] = class_count
        config.model.setdefault("teacher", {})["num_classes"] = class_count

    if adaptation_cfg.get("apply_profiles", True):
        profile = CLASS_PROFILES.get(class_count)
        if profile:
            merged = _deep_merge(config.to_dict(), profile)
            config.model = merged.get("model", config.model)
            config.distillation = merged.get("distillation", config.distillation)
            config.training = merged.get("training", config.training)
            config.data = merged.get("data", config.data)
            config.logging = merged.get("logging", config.logging)

    return class_count

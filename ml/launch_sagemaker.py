"""Create or submit the configured SageMaker Hugging Face training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REQUIRED_CONFIG = {
    "instance_type",
    "transformers_version",
    "pytorch_version",
    "py_version",
}


def validate_config(config: object) -> dict:
    if not isinstance(config, dict):
        raise ValueError("SageMaker config must be a mapping")
    missing = sorted(REQUIRED_CONFIG - config.keys())
    if missing:
        raise ValueError(f"SageMaker config is missing: {missing}")
    hyperparameters = config.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        raise ValueError("hyperparameters must be a mapping")
    tags = config.get("tags", [])
    if not isinstance(tags, list) or not all(
        isinstance(tag, dict)
        and isinstance(tag.get("Key"), str)
        and isinstance(tag.get("Value"), str)
        for tag in tags
    ):
        raise ValueError("tags must be a list of Key/Value mappings")
    return config


def estimator_kwargs(config: dict, role_arn: str) -> dict:
    kwargs = {
        "entry_point": "train.py",
        "source_dir": str(MODULE_DIR),
        "role": role_arn,
        "instance_type": config["instance_type"],
        "instance_count": config.get("instance_count", 1),
        "transformers_version": config["transformers_version"],
        "pytorch_version": config["pytorch_version"],
        "py_version": config["py_version"],
        "hyperparameters": config.get("hyperparameters", {}),
        "output_path": config.get("output_path"),
        "base_job_name": config.get("base_job_name", "contract-classifier"),
    }
    for optional_key in ("checkpoint_s3_uri", "tags"):
        if config.get(optional_key):
            kwargs[optional_key] = config[optional_key]
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("ml/configs/sagemaker.yaml"))
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--training-s3-uri", required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--checkpoint-s3-uri")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Without this flag, print the resolved job configuration",
    )
    args = parser.parse_args()
    try:
        import yaml
    except ImportError as error:
        raise SystemExit(
            "launcher requires PyYAML; install ml/requirements-sagemaker.txt"
        ) from error
    config = dict(validate_config(yaml.safe_load(args.config.read_text(encoding="utf-8"))))
    config["output_path"] = args.output_s3_uri
    if args.checkpoint_s3_uri:
        config["checkpoint_s3_uri"] = args.checkpoint_s3_uri
    resolved = {
        "role": args.role_arn,
        "training_s3_uri": args.training_s3_uri,
        **config,
    }
    if not args.submit:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    try:
        from sagemaker.huggingface import HuggingFace
    except ImportError as error:
        raise SystemExit(
            "submission requires the SageMaker SDK; install ml/requirements-sagemaker.txt"
        ) from error
    estimator = HuggingFace(**estimator_kwargs(config, args.role_arn))
    estimator.fit({"training": args.training_s3_uri})


if __name__ == "__main__":
    main()

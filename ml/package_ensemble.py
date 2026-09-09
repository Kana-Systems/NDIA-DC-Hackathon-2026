"""Build a hash-verified, self-contained Llama LoRA ensemble model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

DEFAULT_MEMBERS = (
    "full--Llama-3.1-8B-r128-lr075-cap30-d03--seed-17",
    "full--Llama-3.1-8B-r128-lr075-cap30-d03--seed-29",
    "full--Llama-3.1-8B-r128-lr075-cap30-d03--seed-43",
)
MEMBER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "base_model_provenance.json",
    "label_mapping.json",
    "training_provenance.json",
    "window_config.json",
)
TOKENIZER_FILES = (
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: str, destination: str) -> str:
    resolved = str(Path(source).resolve())
    try:
        os.link(resolved, destination)
    except OSError:
        shutil.copy2(resolved, destination)
    return destination


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        copy_function=_link_or_copy,
    )


def _manifest_files(root: Path) -> dict[str, dict[str, int | str]]:
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return files


def _load_backup(backup_root: Path) -> tuple[dict, dict[str, dict]]:
    manifest_path = backup_root / "BACKUP_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("immutable_copy") is not True:
        raise ValueError("backup manifest is not marked immutable")
    members = manifest.get("members")
    if not isinstance(members, list):
        raise ValueError("backup manifest members are invalid")
    by_name = {member.get("name"): member for member in members if isinstance(member, dict)}
    return manifest, by_name


def _verify_and_copy_member(
    backup_root: Path,
    member: dict,
    destination: Path,
    deployed_name: str,
) -> dict:
    source = backup_root / str(member["name"])
    destination.mkdir(parents=True)
    recorded_files = member.get("files")
    if not isinstance(recorded_files, dict):
        raise ValueError("backup member file manifest is invalid")
    for filename in MEMBER_FILES:
        source_path = source / filename
        record = recorded_files.get(filename)
        if (
            not source_path.is_file()
            or not isinstance(record, dict)
            or source_path.stat().st_size != record.get("bytes")
            or sha256(source_path) != record.get("sha256")
        ):
            raise ValueError(f"backup member failed verification: {member['name']}/{filename}")
        _link_or_copy(str(source_path), str(destination / filename))
    return {
        "name": deployed_name,
        "path": f"members/{deployed_name}",
        "source_backup_name": member["name"],
        "model_id": member["model_id"],
        "weights_sha256": member["weights_sha256"],
    }


def _deployed_member_name(source_name: str) -> str:
    match = re.search(r"--seed-(\d+)$", source_name)
    if not match:
        raise ValueError(f"backup member name has no seed suffix: {source_name}")
    return f"seed-{match.group(1)}"


def build_ensemble(
    backup_root: Path,
    base_model_dir: Path,
    thresholds_path: Path,
    output: Path,
    member_names: tuple[str, ...] = DEFAULT_MEMBERS,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if not (base_model_dir / "config.json").is_file():
        raise FileNotFoundError("base model snapshot is missing config.json")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    backup_manifest, backup_members = _load_backup(backup_root)
    selected = []
    for name in member_names:
        if name not in backup_members:
            raise ValueError(f"selected member is absent from backup: {name}")
        selected.append(backup_members[name])
    if [member["model_id"] for member in selected] != thresholds.get("member_model_ids"):
        raise ValueError("threshold artifact model members do not match the backup")
    if [member["weights_sha256"] for member in selected] != thresholds.get("member_weights_sha256"):
        raise ValueError("threshold artifact hashes do not match the backup")

    first = backup_root / member_names[0]
    base_provenance = json.loads((first / "base_model_provenance.json").read_text())
    module_root = Path(__file__).resolve().parent
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        for filename in TOKENIZER_FILES:
            source_path = first / filename
            record = selected[0]["files"].get(filename)
            if (
                not source_path.is_file()
                or not isinstance(record, dict)
                or source_path.stat().st_size != record.get("bytes")
                or sha256(source_path) != record.get("sha256")
            ):
                raise ValueError(f"backup tokenizer failed verification: {filename}")
            _link_or_copy(str(source_path), str(staging / filename))
        code = staging / "code"
        code.mkdir()
        for filename in ("inference.py", "heuristic.py", "windowing.py"):
            _link_or_copy(str(module_root / filename), str(code / filename))
        deployed_members = []
        for member in selected:
            deployed_name = _deployed_member_name(member["name"])
            deployed_members.append(
                _verify_and_copy_member(
                    backup_root,
                    member,
                    staging / "members" / deployed_name,
                    deployed_name,
                )
            )
        _copy_tree(base_model_dir, staging / "base_model")
        config = {
            "schema_version": "1.0",
            "artifact_type": "lora_probability_ensemble",
            "model_id": thresholds["model_id"],
            "aggregation": "mean_probabilities",
            "promotion_authorized": thresholds.get("promotion_authorized", False),
            "selection_metrics": thresholds.get("selection_metrics", {}),
            "threshold_artifact_sha256": sha256(thresholds_path),
            "backup_manifest_sha256": sha256(backup_root / "BACKUP_MANIFEST.json"),
            "base_model": {
                "model_id": base_provenance["model_id"],
                "revision": base_provenance["exact_revision"],
                "path": "base_model",
                "files": _manifest_files(staging / "base_model"),
            },
            "members": deployed_members,
            "source_backup_created_at": backup_manifest.get("created_at"),
        }
        (staging / "ensemble_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = build_ensemble(
        args.backup_root.resolve(),
        args.base_model_dir.resolve(),
        args.thresholds.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "model_id": config["model_id"],
                "members": len(config["members"]),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

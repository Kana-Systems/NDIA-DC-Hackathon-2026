import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

adapter = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/prepare-sagemaker-image.py")
)


def test_registry_digest_can_have_multiple_tags_but_not_different_content() -> None:
    raw = json.dumps({"schemaVersion": 2})
    digest = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
    images = [
        {"imageId": {"imageDigest": digest, "imageTag": tag}, "imageManifest": raw}
        for tag in ("build-run", "classifier-cache")
    ]
    assert adapter["verified_manifest"]({"images": images}, digest) == {"schemaVersion": 2}
    images[1]["imageManifest"] = "{}"
    with pytest.raises(ValueError, match="unambiguous"):
        adapter["verified_manifest"]({"images": images}, digest)
    with pytest.raises(ValueError, match="requested digest"):
        adapter["verified_manifest"]({"images": images[:1]}, "sha256:wrong")


def test_manifest_conversion_preserves_image_bytes_and_input() -> None:
    source = {
        "schemaVersion": 2,
        "mediaType": adapter["OCI_MANIFEST"],
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": "sha256:config",
            "size": 123,
        },
        "layers": [
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "digest": "sha256:layer",
                "size": 456,
            }
        ],
    }
    original = copy.deepcopy(source)
    result = adapter["docker_manifest"](source)
    assert source == original
    assert result["mediaType"] == adapter["DOCKER_MANIFEST"]
    for before, after in zip(
        [source["config"], *source["layers"]], [result["config"], *result["layers"]], strict=True
    ):
        assert before["digest"] == after["digest"]
        assert before["size"] == after["size"]
    source["layers"][0]["mediaType"] = "application/vnd.oci.image.layer.v1.tar+zstd"
    with pytest.raises(ValueError, match="gzip"):
        adapter["docker_manifest"](source)


def test_runtime_selection_excludes_attestations_and_rejects_ambiguity() -> None:
    runtime = {"digest": "sha256:runtime", "platform": {"os": "linux", "architecture": "amd64"}}
    attestation = {
        "digest": "sha256:attestation",
        "platform": {"os": "unknown", "architecture": "unknown"},
    }
    assert adapter["platform_digest"]({"manifests": [attestation, runtime]}) == "sha256:runtime"
    for manifests in ([attestation], [runtime, runtime]):
        with pytest.raises(ValueError, match="exactly one"):
            adapter["platform_digest"]({"manifests": manifests})

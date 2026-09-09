"""Publish a Docker V2 manifest for SageMaker without downloading image layers.

Buildx keeps the original OCI index, SBOM, and provenance in ECR. This adapter
selects its Linux/amd64 image and changes only compatible manifest media types.
Config and compressed layer bytes keep their original content-addressed digests.
"""

import argparse
import hashlib
import json
import subprocess

DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"
DOCKER_LAYER = "application/vnd.docker.image.rootfs.diff.tar.gzip"


def docker_manifest(manifest: dict) -> dict:
    if manifest.get("schemaVersion") != 2 or manifest.get("mediaType") not in (
        DOCKER_MANIFEST,
        OCI_MANIFEST,
    ):
        raise ValueError("Expected a single-platform image manifest")
    config = dict(manifest["config"])
    if config["mediaType"] not in (DOCKER_CONFIG, "application/vnd.oci.image.config.v1+json"):
        raise ValueError("Unsupported image configuration type")
    config["mediaType"] = DOCKER_CONFIG
    layers = []
    for descriptor in manifest["layers"]:
        if descriptor["mediaType"] not in (
            DOCKER_LAYER,
            "application/vnd.oci.image.layer.v1.tar+gzip",
        ):
            raise ValueError("Only gzip-compressed image layers can be translated")
        layers.append({**descriptor, "mediaType": DOCKER_LAYER})
    return {"schemaVersion": 2, "mediaType": DOCKER_MANIFEST, "config": config, "layers": layers}


def platform_digest(index: dict) -> str:
    matches = [
        item["digest"]
        for item in index["manifests"]
        if item.get("platform", {}).get("os") == "linux"
        and item.get("platform", {}).get("architecture") == "amd64"
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one Linux/amd64 runtime, excluding attestations")
    return matches[0]


def verified_manifest(result: dict, digest: str) -> dict:
    # ECR may return the same digest once per tag. Aliases are not ambiguity;
    # different manifest bytes or a mismatched content hash are.
    images = result.get("images", [])
    manifests = {item["imageManifest"] for item in images}
    if result.get("failures") or len(manifests) != 1:
        raise ValueError("Unable to retrieve one unambiguous pinned image manifest")
    raw = manifests.pop()
    if any(item["imageId"]["imageDigest"] != digest for item in images) or (
        "sha256:" + hashlib.sha256(raw.encode()).hexdigest() != digest
    ):
        raise ValueError("Registry manifest does not match its requested digest")
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("region", "repository-name", "source-digest", "runtime-tag"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()

    def ecr(operation: str, **parameters):
        request = {"repositoryName": args.repository_name, **parameters}
        result = subprocess.run(
            [
                "aws",
                "ecr",
                operation,
                "--region",
                args.region,
                "--cli-input-json",
                json.dumps(request),
                "--output",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def fetch(digest: str) -> dict:
        result = ecr("batch-get-image", imageIds=[{"imageDigest": digest}])
        return verified_manifest(result, digest)

    source = fetch(args.source_digest)
    if "manifests" in source:
        source = fetch(platform_digest(source))
    runtime = json.dumps(docker_manifest(source), separators=(",", ":"), sort_keys=True)
    digest = "sha256:" + hashlib.sha256(runtime.encode()).hexdigest()
    cached = ecr("batch-get-image", imageIds=[{"imageTag": args.runtime_tag}])
    if cached.get("images"):
        if cached["images"][0]["imageId"]["imageDigest"] != digest:
            raise ValueError("Immutable runtime tag points to different image content")
    else:
        if any(item["failureCode"] != "ImageNotFound" for item in cached.get("failures", [])):
            raise ValueError("Unable to check the runtime image tag")
        ecr(
            "put-image",
            imageTag=args.runtime_tag,
            imageDigest=digest,
            imageManifest=runtime,
            imageManifestMediaType=DOCKER_MANIFEST,
        )
    if fetch(digest).get("mediaType") != DOCKER_MANIFEST:
        raise ValueError("Published runtime does not use the SageMaker manifest format")
    print(digest)


if __name__ == "__main__":
    main()

"""Create or terminate a Runpod Pod with a server-side TTL.

The installed runpodctl release does not yet expose GraphQL's terminateAfter
field. This small wrapper keeps the API key in RUNPOD_API_KEY and prints only
non-secret API responses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

API_URL = "https://api.runpod.io/graphql"


def request(query: str, variables: dict | None = None) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is required")
    payload = json.dumps(
        {"query": query, "variables": variables or {}},
        separators=(",", ":"),
    ).encode()
    http_request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "runpodctl/2.12.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"Runpod HTTP {error.code}: {body}") from error
    if result.get("errors"):
        raise RuntimeError("Runpod GraphQL error: " + json.dumps(result["errors"]))
    return result["data"]


def input_fields() -> int:
    data = request(
        """
        query {
          __type(name: "PodFindAndDeployOnDemandInput") {
            inputFields { name }
          }
        }
        """
    )
    fields = sorted(row["name"] for row in data["__type"]["inputFields"])
    print(json.dumps(fields, indent=2))
    return 0


def launch(args: argparse.Namespace) -> int:
    if not 0 < args.ttl_hours <= 12:
        raise ValueError("ttl-hours must be above zero and at most 12")
    terminate_after = datetime.now(UTC) + timedelta(hours=args.ttl_hours)
    pod_input = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeId": args.gpu_id,
        "gpuCount": 1,
        "cloudType": "SECURE",
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": 0,
        "networkVolumeId": args.network_volume_id,
        "volumeMountPath": args.volume_mount_path,
        "ports": "22/tcp",
        "startSsh": True,
        "supportPublicIp": True,
        "terminateAfter": terminate_after.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "minCudaVersion": args.min_cuda_version,
    }
    data = request(
        """
        mutation Launch($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            imageName
            desiredStatus
          }
        }
        """,
        {"input": pod_input},
    )
    print(
        json.dumps(
            {
                **data["podFindAndDeployOnDemand"],
                "terminateAfter": pod_input["terminateAfter"],
                "networkVolumeId": args.network_volume_id,
            },
            indent=2,
        )
    )
    return 0


def terminate(args: argparse.Namespace) -> int:
    data = request(
        """
        mutation Terminate($input: PodTerminateInput!) {
          podTerminate(input: $input)
        }
        """,
        {"input": {"podId": args.pod_id}},
    )
    print(json.dumps(data, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    fields = commands.add_parser("input-fields")
    fields.set_defaults(func=input_fields)

    create = commands.add_parser("launch")
    create.add_argument("--name", required=True)
    create.add_argument("--image", required=True)
    create.add_argument("--gpu-id", required=True)
    create.add_argument("--network-volume-id", required=True)
    create.add_argument("--ttl-hours", type=float, required=True)
    create.add_argument("--container-disk-gb", type=int, default=50)
    create.add_argument("--volume-mount-path", default="/workspace")
    create.add_argument("--min-cuda-version", default="13.0")
    create.set_defaults(func=launch)

    delete = commands.add_parser("terminate")
    delete.add_argument("pod_id")
    delete.set_defaults(func=terminate)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "input-fields":
        return args.func()
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

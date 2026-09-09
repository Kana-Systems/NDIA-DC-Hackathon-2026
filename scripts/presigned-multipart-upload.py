"""Upload a large remote artifact through narrowly scoped S3 presigned part URLs."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCHEMA_VERSION = "1.0"


def _s3_client(region: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version="s3v4"),
    )


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported multipart plan schema")
    return payload


def _write_private(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(temporary, 0o600)
    temporary.replace(path)


def create_plan(args: argparse.Namespace) -> None:
    if not 8 <= args.part_size_mb <= 1024:
        raise ValueError("part size must be between 8 and 1024 MiB")
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    if not args.sha256 or len(args.sha256) != 64:
        raise ValueError("a hexadecimal SHA-256 is required")
    part_size = args.part_size_mb * 1024 * 1024
    count = math.ceil(args.size / part_size)
    if count < 1 or count > 10_000:
        raise ValueError("multipart upload requires between 1 and 10,000 parts")
    client = _s3_client(args.region)
    created = client.create_multipart_upload(
        Bucket=args.bucket,
        Key=args.key,
        ContentType="application/gzip",
        ExpectedBucketOwner=args.expected_bucket_owner,
        Metadata={"sha256": args.sha256, "model-id": args.model_id},
        ServerSideEncryption="aws:kms",
        SSEKMSKeyId=args.kms_key,
    )
    upload_id = created["UploadId"]
    try:
        parts = []
        for number in range(1, count + 1):
            offset = (number - 1) * part_size
            size = min(part_size, args.size - offset)
            url = client.generate_presigned_url(
                "upload_part",
                Params={
                    "Bucket": args.bucket,
                    "Key": args.key,
                    "UploadId": upload_id,
                    "PartNumber": number,
                },
                ExpiresIn=args.expires_seconds,
            )
            parts.append(
                {
                    "part_number": number,
                    "offset": offset,
                    "size": size,
                    "url": url,
                }
            )
        _write_private(
            args.plan,
            {
                "schema_version": SCHEMA_VERSION,
                "bucket": args.bucket,
                "key": args.key,
                "upload_id": upload_id,
                "size": args.size,
                "sha256": args.sha256,
                "model_id": args.model_id,
                "expected_bucket_owner": args.expected_bucket_owner,
                "part_size": part_size,
                "workers": args.workers,
                "parts": parts,
            },
        )
    except Exception:
        client.abort_multipart_upload(
            Bucket=args.bucket,
            Key=args.key,
            UploadId=upload_id,
            ExpectedBucketOwner=args.expected_bucket_owner,
        )
        raise


def _upload_part(archive: Path, part: dict) -> dict:
    with archive.open("rb") as stream:
        stream.seek(part["offset"])
        data = stream.read(part["size"])
    if len(data) != part["size"]:
        raise OSError(f"could not read complete part {part['part_number']}")
    request = urllib.request.Request(part["url"], data=data, method="PUT")
    request.add_header("Content-Length", str(len(data)))
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(
                request,
                timeout=900,
            ) as response:
                etag = response.headers.get("ETag")
                if response.status != 200 or not etag:
                    raise RuntimeError(
                        f"part {part['part_number']} returned HTTP {response.status}"
                    )
                return {"PartNumber": part["part_number"], "ETag": etag}
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"part {part['part_number']} failed after retries") from last_error


def upload_parts(args: argparse.Namespace) -> None:
    plan = _load(args.plan)
    if args.archive.stat().st_size != plan["size"]:
        raise ValueError("remote archive size does not match the multipart plan")
    results = []
    if args.results.is_file():
        existing = _load(args.results)
        if (
            existing.get("bucket") != plan["bucket"]
            or existing.get("key") != plan["key"]
            or existing.get("upload_id") != plan["upload_id"]
        ):
            raise ValueError("existing multipart results do not match the upload plan")
        results = existing.get("parts", [])
    completed = {item["PartNumber"] for item in results}
    pending = [part for part in plan["parts"] if part["part_number"] not in completed]

    def save() -> None:
        _write_private(
            args.results,
            {
                "schema_version": SCHEMA_VERSION,
                "bucket": plan["bucket"],
                "key": plan["key"],
                "upload_id": plan["upload_id"],
                "parts": sorted(results, key=lambda item: item["PartNumber"]),
            },
        )

    with ThreadPoolExecutor(max_workers=plan["workers"]) as executor:
        futures = {
            executor.submit(_upload_part, args.archive, part): part["part_number"]
            for part in pending
        }
        for future in as_completed(futures):
            results.append(future.result())
            save()
            print(
                f"uploaded part {futures[future]}/{len(plan['parts'])}; "
                f"complete={len(results)}",
                flush=True,
            )
    save()


def complete_upload(args: argparse.Namespace) -> None:
    plan = _load(args.plan)
    results = _load(args.results)
    if (
        results.get("bucket") != plan["bucket"]
        or results.get("key") != plan["key"]
        or results.get("upload_id") != plan["upload_id"]
        or len(results.get("parts", [])) != len(plan["parts"])
    ):
        raise ValueError("multipart results do not match the upload plan")
    client = _s3_client(args.region)
    completed = client.complete_multipart_upload(
        Bucket=plan["bucket"],
        Key=plan["key"],
        UploadId=plan["upload_id"],
        MultipartUpload={"Parts": results["parts"]},
        ExpectedBucketOwner=plan["expected_bucket_owner"],
    )
    head = client.head_object(
        Bucket=plan["bucket"],
        Key=plan["key"],
        ExpectedBucketOwner=plan["expected_bucket_owner"],
    )
    if (
        head["ContentLength"] != plan["size"]
        or head.get("Metadata", {}).get("sha256") != plan["sha256"]
        or head.get("Metadata", {}).get("model-id") != plan["model_id"]
        or head.get("ServerSideEncryption") != "aws:kms"
    ):
        raise RuntimeError("completed S3 object failed metadata verification")
    print(
        json.dumps(
            {
                "etag": completed["ETag"],
                "s3_uri": f"s3://{plan['bucket']}/{plan['key']}",
                "sha256": plan["sha256"],
                "size": plan["size"],
                "version_id": completed.get("VersionId"),
            },
            sort_keys=True,
        )
    )


def abort_upload(args: argparse.Namespace) -> None:
    plan = _load(args.plan)
    _s3_client(args.region).abort_multipart_upload(
        Bucket=plan["bucket"],
        Key=plan["key"],
        UploadId=plan["upload_id"],
        ExpectedBucketOwner=plan["expected_bucket_owner"],
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--bucket", required=True)
    create.add_argument("--key", required=True)
    create.add_argument("--size", type=int, required=True)
    create.add_argument("--sha256", required=True)
    create.add_argument("--model-id", required=True)
    create.add_argument("--kms-key", required=True)
    create.add_argument("--expected-bucket-owner", required=True)
    create.add_argument("--region", default="us-gov-west-1")
    create.add_argument("--part-size-mb", type=int, default=16)
    create.add_argument("--workers", type=int, default=4)
    create.add_argument("--expires-seconds", type=int, default=7200)
    create.add_argument("--plan", type=Path, required=True)
    create.set_defaults(handler=create_plan)

    upload = commands.add_parser("upload")
    upload.add_argument("--archive", type=Path, required=True)
    upload.add_argument("--plan", type=Path, required=True)
    upload.add_argument("--results", type=Path, required=True)
    upload.set_defaults(handler=upload_parts)

    complete = commands.add_parser("complete")
    complete.add_argument("--plan", type=Path, required=True)
    complete.add_argument("--results", type=Path, required=True)
    complete.add_argument("--region", default="us-gov-west-1")
    complete.set_defaults(handler=complete_upload)

    abort = commands.add_parser("abort")
    abort.add_argument("--plan", type=Path, required=True)
    abort.add_argument("--region", default="us-gov-west-1")
    abort.set_defaults(handler=abort_upload)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

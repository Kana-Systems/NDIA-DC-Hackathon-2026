"""Download unchanged publisher CUAD files for a real-document ingestion demo.

No synthetic data, rewording, splitting, training or SharePoint writes. The
size-bounded subset is an ingestion demonstration, NOT an unbiased ML benchmark.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

REVISION = "a3c393f5d103fd0c516374e4fdff676c8176dcb1"
DATASET = "theatticusproject/cuad"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if not 100 <= args.count <= 510:
        parser.error("Choose 100–510 complete original documents")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folder = output / "CUAD-original-contracts"
    folder.mkdir(exist_ok=True)
    url = (
        f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
        "/CUAD_v1/full_contract_txt?recursive=true&limit=1000"
    )
    rows, visited = [], set()
    while url:
        if url in visited or urlsplit(url).hostname != "huggingface.co":
            raise ValueError("Unexpected publisher pagination")
        visited.add(url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        rows.extend(response.json())
        url = response.links.get("next", {}).get("url")
    eligible = sorted(
        (
            row
            for row in rows
            if row["type"] == "file"
            and row["path"].endswith(".txt")
            and 80 <= row["size"] <= 100000
            and len(Path(row["path"]).name) <= 200
        ),
        key=lambda row: (row["size"], row["path"]),
    )
    if len(eligible) < args.count:
        raise ValueError(f"Only {len(eligible)} whole contracts fit the current app limits")
    selected = eligible[: args.count]
    if len({Path(row["path"]).name for row in selected}) != args.count:
        raise ValueError("Duplicate filenames; no files overwritten")

    def download(row):
        source = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
        source += quote(row["path"], safe="/")
        target = folder / Path(row["path"]).name
        if target.exists():
            raw = target.read_bytes()
        else:
            result = requests.get(source, timeout=60)
            result.raise_for_status()
            raw = result.content
        blob_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if len(raw) != row["size"] or blob_hash != row["oid"]:
            raise ValueError("Publisher checksum mismatch; no file overwritten")
        if not 80 <= len(raw.decode("utf-8-sig")) <= 100000:
            raise ValueError("Original document does not meet application limits")
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(raw)
        return {
            "filename": target.name,
            "source_url": source,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "publisher_git_blob": blob_hash,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, selected))
    if len({row["sha256"] for row in files}) != args.count:
        raise ValueError("Duplicate source content; not counted as distinct documents")
    manifest = {
        "dataset": "CUAD v1",
        "publisher": "The Atticus Project",
        "source_url": "https://www.atticusprojectai.org/cuad/",
        "revision": REVISION,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": (
            "Hendrycks et al., CUAD: An Expert-Annotated NLP Dataset "
            "for Legal Contract Review, NeurIPS 2021"
        ),
        "selection": (
            f"{args.count} shortest complete TXT originals; "
            "length-biased ingestion demo, not ML evaluation"
        ),
        "modifications": "None: downloaded original bytes; no splitting or generated text",
        "documents": len(files),
        "files": files,
    }
    (output / "CUAD-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Keep attribution with the redistributed originals. JSON is not ingested or
    # counted as an additional document by the Lens connector.
    (folder / "_CUAD-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(files)} distinct complete original contracts at {folder}")
    print("Publisher checksums verified; attribution JSON accompanies the 100 source TXT files.")
    print("Not uploaded or ingested yet; do not count this as a successful SharePoint run.")


if __name__ == "__main__":
    main()

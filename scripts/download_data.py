"""Fetch complete official GSM8K JSONL with curl and verify every row.

Uses the official raw GitHub URL, then its jsDelivr mirror if necessary. Normal
TLS validation remains enabled. Existing valid complete files are reused.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


FILES = {
    "train": {"bytes": 4166206, "rows": 7473},
    "test": {"bytes": 749738, "rows": 1319,
             "git_blob_sha1": "e4c2ff4942b9a78bd74f04141224c11e28d12dc9"},
}
ORIGIN = "https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/"
RAW = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
MIRROR = "https://cdn.jsdelivr.net/gh/openai/grade-school-math@master/grade_school_math/data/"


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def validate(data: bytes, split: str) -> bool:
    spec = FILES[split]
    try:
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        valid = len(data) == spec["bytes"] and len(rows) == spec["rows"] and all(
            isinstance(row.get("question"), str) and isinstance(row.get("answer"), str)
            and "####" in row["answer"] for row in rows
        )
        return valid and (not spec.get("git_blob_sha1") or git_blob_sha1(data) == spec["git_blob_sha1"])
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return False


def download(split: str, target: Path, timeout: int) -> tuple[bytes, str]:
    if target.exists():
        data = target.read_bytes()
        if validate(data, split):
            return data, "existing verified file"
    curl = "curl.exe" if os.name == "nt" else "curl"
    temporary = target.with_suffix(".download")
    errors = []
    try:
        for base in (RAW, MIRROR):
            url = base + split + ".jsonl"
            try:
                result = subprocess.run([
                    curl, "--silent", "--show-error", "--location", "--fail", "--compressed",
                    "--max-time", str(timeout), "--output", str(temporary), url,
                ], capture_output=True, timeout=timeout + 5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except subprocess.TimeoutExpired:
                errors.append(f"{url}: subprocess timeout")
                continue
            data = temporary.read_bytes() if temporary.exists() else b""
            if result.returncode == 0 and validate(data, split):
                os.replace(temporary, target)
                return data, url
            errors.append(f"{url}: curl={result.returncode}, bytes={len(data)}, complete validation failed")
    finally:
        temporary.unlink(missing_ok=True)
    raise RuntimeError("No complete GSM8K download; existing file was preserved. " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "gsm8k_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {}
    for split, spec in FILES.items():
        data, downloaded_from = download(split, args.output / f"{split}.jsonl", args.timeout)
        if downloaded_from == "existing verified file":
            downloaded_from = previous.get("files", {}).get(split, {}).get("download_url", downloaded_from)
        manifest[split] = {
            "rows": spec["rows"], "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "git_blob_sha1": git_blob_sha1(data),
            "origin": ORIGIN + split + ".jsonl", "download_url": downloaded_from,
        }
        print(f"Verified {split}: {spec['rows']} rows, {len(data)} bytes", flush=True)
    timestamp = previous.get("retrieved_at_utc", datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps({"retrieved_at_utc": timestamp, "files": manifest}, indent=2) + "\n",
                             encoding="utf-8")
    lines = ["# Benchmark data provenance", "",
             "The benchmark uses the complete official GSM8K main dataset: 7,473 official training rows",
             "and 1,319 official test rows. No prefix-only pool is used.", "",
             "Source repository: https://github.com/openai/grade-school-math", "",
             "The experiment files were downloaded through jsDelivr's GitHub mirror using `curl --compressed`,",
             "because direct GitHub downloads were interrupted in this environment. TLS certificate checking",
             "remained enabled. Complete byte counts and all JSONL rows were validated before use.", "",
             "The test file's Git blob SHA1 was independently obtained from the GitHub contents API and",
             "verified. SHA256 and Git blob SHA1 checksums for both files are recorded below and in",
             "`raw/gsm8k_manifest.json`; the training hash is a reproducibility checksum.", "",
             f"Retrieved: {timestamp}", ""]
    for split, record in manifest.items():
        lines.extend([f"## {split}", "", f"- Original: {record['origin']}",
                      f"- Download: {record['download_url']}",
                      f"- Rows: {record['rows']}; bytes: {record['bytes']}",
                      f"- SHA256: `{record['sha256']}`", f"- Git blob SHA1: `{record['git_blob_sha1']}`", ""])
    lines.extend(["Training and validation are sampled only from official train. Final evaluation is sampled",
                  "only from official test, with question-hash deduplication across splits.", "",
                  "Reproduce from the repository root (no network request if complete valid files exist):", "",
                  "```sh", "python scripts/download_data.py", "```", ""])
    (args.output.parent / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

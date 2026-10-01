"""Prepare the pinned local emotion2vec model and isolated runtime.

This helper uses only public ModelScope file endpoints and a hash-locked set of
new Python wheels. It never resolves model names from an account or downloads
anything during inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


MODEL_ID = "iic/emotion2vec_plus_base"
MODEL_REVISION = "58354f1e5b6c118cf8b175b5b1955d6b07df42c2"
API_FILES_URL = "https://modelscope.cn/api/v1/models/iic/emotion2vec_plus_base/repo/files"
API_FILE_URL = "https://modelscope.cn/api/v1/models/iic/emotion2vec_plus_base/repo"
DEFAULT_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
LOCK_NAME = "requirements-tone-lock.txt"
MODEL_MANIFEST_NAME = "emotion2vec.manifest.json"
RUNTIME_MANIFEST_NAME = "emotion-runtime.manifest.json"
EXPECTED_FILES = {
    "model.pt": (1_118_245_678, "60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd"),
    "config.yaml": (3_170, "50674ec187406b4f1208925594ceaf1f28f16a22333b5f2ac5b3d0cbffdd670e"),
    "configuration.json": (343, "8b6a745213025c7d4565f9f074cfef1b5cd5ef76e38e2a8f7f8a3db271e735b2"),
    "tokens.txt": (120, "654625c18ef0f0583441a3b811c20904f2cf30febb716f32e14810b8e6bb715c"),
}
PUBLIC_TEST_AUDIO = (
    "test.wav",
    131_376,
    "a4839eaaa3d54bd2db6eb48aa3d40def1b5c5004df3fd163a8dcd045097f8a23",
)


class PreparationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_file_inventory() -> dict[str, dict[str, Any]]:
    response = requests.get(
        API_FILES_URL,
        params={"Revision": MODEL_REVISION, "Recursive": "true"},
        timeout=(15, 60),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("Success") is False:
        raise PreparationError(f"ModelScope metadata request failed: {payload.get('Message', 'unknown error')}")
    data = payload.get("Data") or {}
    entries = data.get("Files") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise PreparationError("ModelScope returned no pinned file inventory")
    result: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = item.get("Name") or item.get("name")
        if isinstance(name, str):
            result[name] = item
    for name, (size, digest) in EXPECTED_FILES.items():
        entry = result.get(name)
        upstream_size = entry.get("Size", entry.get("size")) if entry else None
        upstream_hash = entry.get("Sha256", entry.get("sha256")) if entry else None
        if upstream_size != size or upstream_hash != digest:
            raise PreparationError(f"pinned ModelScope metadata does not match the verified manifest: {name}")
    sample_entry = result.get(PUBLIC_TEST_AUDIO[0])
    if sample_entry:
        sample_size = sample_entry.get("Size", sample_entry.get("size"))
        sample_hash = sample_entry.get("Sha256", sample_entry.get("sha256"))
        if sample_size != PUBLIC_TEST_AUDIO[1] or sample_hash != PUBLIC_TEST_AUDIO[2]:
            raise PreparationError("pinned public sample metadata does not match the verified sample manifest")
    return result


def _download_verified_file(
    model_dir: Path,
    filename: str,
    expected_size: int,
    expected_hash: str,
    *,
    sample_output: Path | None = None,
) -> dict[str, Any]:
    destination = sample_output if sample_output is not None else model_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size and _sha256(destination) == expected_hash:
        return {"size": expected_size, "sha256": expected_hash, "cached": True}

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= expected_size:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = requests.get(
        API_FILE_URL,
        params={"Revision": MODEL_REVISION, "FilePath": filename},
        headers=headers,
        stream=True,
        timeout=(20, 120),
    )
    try:
        if offset and response.status_code == 416:
            partial.unlink(missing_ok=True)
            response.close()
            return _download_verified_file(
                model_dir,
                filename,
                expected_size,
                expected_hash,
                sample_output=sample_output,
            )
        response.raise_for_status()
        append = bool(offset and response.status_code == 206)
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        received = 0
        with partial.open(mode) as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                received += len(chunk)
                if offset + received > expected_size:
                    raise PreparationError(f"download exceeded the pinned size: {filename}")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    finally:
        response.close()

    if partial.stat().st_size != expected_size:
        raise PreparationError(
            f"download size mismatch for {filename}: expected {expected_size}, got {partial.stat().st_size}"
        )
    actual_hash = _sha256(partial)
    if actual_hash != expected_hash:
        raise PreparationError(f"SHA-256 mismatch for downloaded public file: {filename}")
    os.replace(partial, destination)
    return {"size": expected_size, "sha256": expected_hash, "cached": False}


def _read_lock(lock_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    pending = ""
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pending += " " + line.removesuffix("\\").strip()
        if line.endswith("\\"):
            continue
        match = re.search(r"([A-Za-z0-9_.-]+)==([^\s]+)", pending)
        hash_match = re.search(r"--hash=sha256:([0-9a-f]{64})", pending)
        if not match or not hash_match:
            raise PreparationError(f"invalid hash-locked requirement: {pending.strip()}")
        records.append({"name": match.group(1).lower().replace("_", "-"), "version": match.group(2), "hash": hash_match.group(1)})
        pending = ""
    if pending.strip():
        raise PreparationError("incomplete final requirement in tone lock")
    if not records:
        raise PreparationError("tone dependency lock is empty")
    return records


def _installed_target_versions(runtime_dir: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for metadata in runtime_dir.glob("*.dist-info/METADATA"):
        name = version = None
        try:
            for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Name: "):
                    name = line[6:].strip().lower().replace("_", "-")
                elif line.startswith("Version: "):
                    version = line[9:].strip()
                if name and version:
                    break
        except OSError:
            continue
        if name and version:
            versions[name] = version
    return versions


def _install_runtime(project_root: Path, runtime_dir: Path, temp_dir: Path, index_url: str) -> dict[str, str]:
    lock_path = project_root / LOCK_NAME
    records = _read_lock(lock_path)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    installed = _installed_target_versions(runtime_dir)
    wrong = [record for record in records if record["name"] in installed and installed[record["name"]] != record["version"]]
    if wrong:
        raise PreparationError("isolated runtime contains a conflicting pinned package version; refusing to overwrite it")
    missing = [record for record in records if installed.get(record["name"]) != record["version"]]
    if missing:
        temp_dir.mkdir(parents=True, exist_ok=True)
        requirement_file = temp_dir / "tone-requirements-pending.txt"
        lines = [f"{item['name']}=={item['version']} --hash=sha256:{item['hash']}" for item in missing]
        requirement_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pip_env = os.environ.copy()
        pip_env["TEMP"] = str(temp_dir)
        pip_env["TMP"] = str(temp_dir)
        pip_env["PIP_CACHE_DIR"] = str(temp_dir / "pip-cache")
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--only-binary=:all:",
            "--require-hashes",
            "--disable-pip-version-check",
            "--no-input",
            "--no-warn-script-location",
            "--index-url",
            index_url,
            "--target",
            str(runtime_dir),
            "-r",
            str(requirement_file),
        ]
        completed = subprocess.run(command, cwd=str(project_root), env=pip_env, check=False)
        if completed.returncode:
            raise PreparationError(f"pinned isolated runtime installation failed with exit code {completed.returncode}")

    installed = _installed_target_versions(runtime_dir)
    expected = {item["name"]: item["version"] for item in records}
    if any(installed.get(name) != version for name, version in expected.items()):
        raise PreparationError("isolated runtime is missing one or more locked package versions after installation")
    runtime_manifest = {
        "schema": 1,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "funasr_version": expected.get("funasr"),
        "requirements_sha256": _sha256(lock_path),
        "packages": expected,
    }
    _write_json_atomic(runtime_dir / RUNTIME_MANIFEST_NAME, runtime_manifest)
    return expected


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def prepare_assets(model_dir: Path, temp_dir: Path, public_sample_path: Path | None = None) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    inventory = _request_file_inventory()
    files: dict[str, dict[str, Any]] = {}
    for name, (size, digest) in EXPECTED_FILES.items():
        files[name] = _download_verified_file(model_dir, name, size, digest)
    if public_sample_path is not None:
        sample_name, sample_size, sample_hash = PUBLIC_TEST_AUDIO
        files[f"verification/{sample_name}"] = _download_verified_file(
            temp_dir,
            sample_name,
            sample_size,
            sample_hash,
            sample_output=public_sample_path,
        )
    manifest = {
        "schema": 1,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": "Apache-2.0",
        "source": API_FILE_URL,
        "source_revision": MODEL_REVISION,
        "files": {
            name: {"size": size, "sha256": digest, "source": "pinned ModelScope public file API"}
            for name, (size, digest) in EXPECTED_FILES.items()
        },
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if public_sample_path is not None:
        sample_name, sample_size, sample_hash = PUBLIC_TEST_AUDIO
        manifest["verification_sample"] = {"name": sample_name, "size": sample_size, "sha256": sample_hash}
    _write_json_atomic(model_dir / MODEL_MANIFEST_NAME, manifest)
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": files,
        "metadata_file_count": len(inventory),
        "manifest": str(model_dir / MODEL_MANIFEST_NAME),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=project_root / "models" / "emotion2vec_plus_base")
    parser.add_argument("--runtime-dir", type=Path, default=project_root / ".runtime" / "tone")
    parser.add_argument("--temp-dir", type=Path, default=Path(tempfile.gettempdir()) / "consult-asr-tone")
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--skip-dependencies", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--public-sample-path", type=Path, help="optional verified public test.wav destination for offline smoke testing")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    runtime_dir = args.runtime_dir.expanduser().resolve()
    temp_dir = args.temp_dir.expanduser().resolve()
    try:
        if not args.skip_dependencies:
            packages = _install_runtime(project_root, runtime_dir, temp_dir, args.index_url)
            print(json.dumps({"runtime_packages": packages}, ensure_ascii=False))
        if not args.skip_model:
            result = prepare_assets(model_dir, temp_dir, args.public_sample_path.expanduser().resolve() if args.public_sample_path else None)
            print(json.dumps(result, ensure_ascii=False))
        return 0
    except (PreparationError, requests.RequestException, OSError, ValueError) as exc:
        print(f"emotion preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Build Windows deployment assets from a validated environment (build host only).

Build requirements: conda-pack and pefile. Neither is needed by end users.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.5.0"
REPOSITORY = "hexiovo/consult_asr"
CHUNK_SIZE = 1500 * 1024 * 1024


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_tree(archive: zipfile.ZipFile, source: Path, destination: str) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(p in {"__pycache__", ".cache", ".git"} for p in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Release input must not contain symlinks: {path}")
        archive.write(path, (Path(destination) / path.relative_to(source)).as_posix())


def ffmpeg_closure(executable: Path) -> list[Path]:
    import pefile

    bin_dir = executable.parent
    prefix = bin_dir.parent.parent
    search = (bin_dir, prefix)
    pending = [executable]
    found: dict[str, Path] = {}
    while pending:
        path = pending.pop()
        key = path.name.casefold()
        if key in found:
            continue
        found[key] = path
        with pefile.PE(str(path), fast_load=True) as pe:
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
            ])
            imports = list(getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
            imports += list(getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []))
            for record in imports:
                name = record.dll.decode("ascii")
                local = next((directory / name for directory in search if (directory / name).is_file()), None)
                if local:
                    pending.append(local)
                elif not name.lower().startswith(("api-ms-", "ext-ms-")) and not (
                    Path(os.environ["WINDIR"]) / "System32" / name
                ).is_file():
                    raise RuntimeError(f"Unresolved FFmpeg dependency: {name}")
    return sorted(found.values())


def archive_record(path: Path, destination: str) -> dict:
    digest = sha256(path)
    parts = []
    if path.stat().st_size <= CHUNK_SIZE:
        parts.append({"name": path.name, "size": path.stat().st_size, "sha256": digest})
    else:
        with path.open("rb") as source:
            index = 1
            while source.tell() < path.stat().st_size:
                part = path.with_name(f"{path.name}.{index:03d}")
                part_hash = hashlib.sha256()
                size = 0
                with part.open("wb") as target:
                    while size < CHUNK_SIZE and (block := source.read(min(8 * 1024 * 1024, CHUNK_SIZE - size))):
                        target.write(block)
                        part_hash.update(block)
                        size += len(block)
                parts.append({"name": part.name, "size": size, "sha256": part_hash.hexdigest()})
                index += 1
        path.unlink()
    return {"name": path.name, "destination": destination, "sha256": digest, "parts": parts}


def write_batch(path: Path, content: str) -> None:
    path.write_bytes(content.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))


def validate_environment(prefix: Path) -> None:
    # These two Conda packages were intentionally replaced by the application's
    # validated pip pins. Do not silently tolerate any other missing Conda files.
    expected = {"packaging": "24.2", "setuptools": "80.9.0"}
    code = "import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in ['packaging','setuptools']}))"
    actual = json.loads(subprocess.check_output([str(prefix / "python.exe"), "-B", "-c", code]))
    if actual != expected:
        raise RuntimeError(f"Runtime compatibility pins differ: {actual}")
    for metadata in (prefix / "conda-meta").glob("*.json"):
        package = json.loads(metadata.read_text(encoding="utf-8"))
        missing = [name for name in package.get("files", []) if not (prefix / name).exists()]
        if missing and package["name"] not in expected:
            raise RuntimeError(f"Unexpected missing Conda files: {package['name']}: {missing[:5]}")
    subprocess.run([str(prefix / "python.exe"), "-B", "-m", "pip", "check"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-prefix", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / f"v{VERSION}")
    parser.add_argument("--refresh-app", action="store_true", help="Rebuild only application/installer assets; preserve existing verified runtime/model archives")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not args.refresh_app:
        raise RuntimeError("Use an empty output directory; existing artifacts are never overwritten.")
    preserved = []
    if args.refresh_app:
        previous = json.loads((output / "release-manifest.json").read_text(encoding="utf-8"))
        if previous["version"] != VERSION:
            raise RuntimeError("Cannot reuse data from a different application version.")
        preserved = [record for record in previous["archives"] if record["destination"] != "."]
        for record in preserved:
            for part in record["parts"]:
                if sha256(output / part["name"]) != part["sha256"]:
                    raise RuntimeError(f"Reused release asset checksum mismatch: {part['name']}")
    python_prefix = args.python_prefix.resolve()
    validate_environment(python_prefix)
    records = []
    with tempfile.TemporaryDirectory(prefix="consult-asr-release-") as temporary:
        stage = Path(temporary)
        print("Building relative-path launcher ...", flush=True)
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(ROOT / "packaging" / "build_exe.ps1"), "-OutputDirectory", str(stage),
                        "-PythonExe", str(python_prefix / "python.exe"), "-Portable", "-SkipChecks"], check=True)
        print("Packaging app, isolated emotion runtime and FFmpeg ...", flush=True)
        app_archive = output / f"ConsultASR-{VERSION}-app.zip"
        with zipfile.ZipFile(app_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for directory in ("app", "core", "utils", "tools", "assets", "packaging"):
                # Only tracked files enter the code bundle; no recordings, logs or local settings.
                paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", directory], cwd=ROOT)
                for raw in paths.split(b"\0"):
                    if raw:
                        relative = raw.decode("utf-8")
                        path = ROOT / relative
                        if path.is_file():
                            archive.write(path, relative)
            for name in ("consult_asr.py", "config.py", "README.md", "USER_GUIDE.md", "DEPLOYMENT.md",
                         "version.md", "requirements-lock.txt", "requirements-tone-lock.txt",
                         "run_consult_asr.bat", "prepare_models.bat", "diagnose.bat"):
                archive.write(ROOT / name, name)
            archive.write(stage / "咨询录音转写.exe", "咨询录音转写.exe")
            guide = ROOT / "output/pdf/咨询录音转写_图文使用指南.pdf"
            if guide.is_file():
                archive.write(guide, "output/pdf/咨询录音转写_图文使用指南.pdf")
            archive.writestr("local_models/pyannote/speaker-diarization-3.1/", b"")
            add_tree(archive, ROOT / ".runtime" / "tone", ".runtime/tone")
            ffmpeg_files = ffmpeg_closure(args.ffmpeg.resolve())
            for path in ffmpeg_files:
                archive.write(path, f".runtime/ffmpeg/{path.name}")
            ffmpeg_prefix = args.ffmpeg.resolve().parent.parent.parent
            copied_names = {p.name.casefold() for p in ffmpeg_files}
            for metadata in (ffmpeg_prefix / "conda-meta").glob("*.json"):
                package = json.loads(metadata.read_text(encoding="utf-8"))
                if not any(Path(name).name.casefold() in copied_names for name in package.get("files", [])):
                    continue
                package_dir = Path(package.get("link", {}).get("source", ""))
                license_root = f".runtime/ffmpeg/package-info/{package['name']}"
                archive.write(metadata, f"{license_root}/conda-package.json")
                for directory in ("info/licenses", "info/recipe"):
                    if (package_dir / directory).is_dir():
                        add_tree(archive, package_dir / directory, f"{license_root}/{directory}")
            for directory in ("Library/licenses", "Library/share/licenses"):
                source = ffmpeg_prefix / directory
                if source.is_dir():
                    add_tree(archive, source, f".runtime/ffmpeg/{directory}")
            for name in ("COPYING.txt", "README.txt"):
                path = ffmpeg_prefix / "Library" / name
                if path.is_file():
                    archive.write(path, f".runtime/ffmpeg/{name}")
            notice = (
                "ConsultASR bundled third-party components\n\n"
                "Python, PySide6/Qt, PyTorch/CUDA/cuDNN, pyannote.audio, faster-whisper/CTranslate2,\n"
                "FFmpeg and other dependencies retain their own license terms.\n"
                "Runtime license texts and Python distribution METADATA/licenses are retained.\n"
                "FFmpeg binary source: conda-forge ffmpeg 8.1.1 (GPL build); recipe and sources:\n"
                "https://github.com/conda-forge/ffmpeg-feedstock/tree/main/recipe\n"
                "https://ffmpeg.org/releases/ffmpeg-8.1.1.tar.gz\n"
                "Public ASR weights: https://huggingface.co/Systran/faster-whisper-large-v3 (MIT).\n"
                "Emotion weights: https://modelscope.cn/models/iic/emotion2vec_plus_base (Apache-2.0).\n"
                "Gated pyannote weights are NOT redistributed; obtain access with your own account.\n"
                "Bundled FFmpeg files:\n" + "\n".join(p.name for p in ffmpeg_files) + "\n"
            )
            archive.writestr("THIRD_PARTY_NOTICES.txt", notice)
        records.append(archive_record(app_archive, "."))
        if args.refresh_app:
            records.extend(preserved)
        else:
            print("Packing relocatable Python environment (this may take several minutes) ...", flush=True)
            import conda_pack

            runtime_archive = output / f"ConsultASR-{VERSION}-runtime.zip"
            conda_pack.pack(prefix=str(python_prefix), output=str(runtime_archive), format="zip",
                            compress_level=1, n_threads=2, ignore_missing_files=True,
                            filters=[("exclude", "*/__pycache__/*"), ("exclude", "*.pyc"),
                                     ("exclude", "*.pyo"), ("exclude", "*.pdb")])
            records.append(archive_record(runtime_archive, ".runtime/python"))
            for label, source, destination in (
                ("asr", ROOT / "local_models" / "faster-whisper-large-v3", "local_models/faster-whisper-large-v3"),
                ("emotion", ROOT / "models" / "emotion2vec_plus_base", "models/emotion2vec_plus_base"),
            ):
                print(f"Packaging public {label} model ...", flush=True)
                path = output / f"ConsultASR-{VERSION}-{label}.zip"
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                    add_tree(archive, source, ".")
                records.append(archive_record(path, destination))
        manifest = {"schema": 1, "version": VERSION,
                    "base_url": f"https://github.com/{REPOSITORY}/releases/download/v{VERSION}",
                    "archives": records}
        (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        installer = stage / "installer"
        installer.mkdir()
        shutil.copy2(output / "release-manifest.json", installer)
        # Windows PowerShell 5.1 requires BOM to decode Chinese literals reliably.
        (installer / "setup.ps1").write_text((ROOT / "packaging/setup.ps1").read_text(encoding="utf-8"), encoding="utf-8-sig")
        write_batch(installer / "setup.bat", '@echo off\nsetlocal\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*\nset "SETUP_EXIT=%ERRORLEVEL%"\necho.\npause\nexit /b %SETUP_EXIT%\n')
        shutil.copy2(ROOT / "DEPLOYMENT.md", installer / "INSTALL.txt")
        installer_archive = output / f"ConsultASR-{VERSION}-Windows-x64-Setup.zip"
        with zipfile.ZipFile(installer_archive, "w", zipfile.ZIP_DEFLATED) as archive:
            add_tree(archive, installer, ".")
    checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file() and path.name != "SHA256SUMS.txt")
    (output / "SHA256SUMS.txt").write_text(checksums, encoding="ascii")
    print(f"Release assets ready: {output}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ASR_MODEL_DIR, DIARIZATION_MODEL_DIR, MODEL_ROOT


ASR_REPOSITORY = "Systran/faster-whisper-large-v3"
ASR_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
DIARIZATION_REPOSITORY = "pyannote/speaker-diarization-3.1"
_REFERENCE = re.compile(r"(?<![A-Za-z0-9_.-])(pyannote/[A-Za-z0-9_.-]+)(?:@([A-Za-z0-9][A-Za-z0-9._-]*))?")
_ASR_REQUIRED_FILES = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    "vocabulary.json",
    "preprocessor_config.json",
)
_PYANNOTE_CHECKPOINT_NAME = "pytorch_model.bin"


class ModelPreparationError(RuntimeError):
    pass


def _local_checkpoint_path(model_directory: Path) -> Path:
    checkpoint = model_directory / _PYANNOTE_CHECKPOINT_NAME
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ModelPreparationError(
            f"pyannote 关联模型缺少本地 {_PYANNOTE_CHECKPOINT_NAME}：{checkpoint}"
        )
    if not any(part.casefold() == "pyannote" for part in checkpoint.resolve().parts):
        raise ModelPreparationError(
            f"pyannote 本地检查点路径缺少 pyannote 标识：{checkpoint}"
        )
    return checkpoint.resolve()


def _write_yaml_ascii(yaml_path: Path, content: Any, yaml_module: Any) -> None:
    """Write UTF-8 YAML with ASCII escapes for pyannote.audio 3.1.1's default open()."""
    temp = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
    temp.write_text(
        yaml_module.safe_dump(content, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temp, yaml_path)


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def _safe_message(exc: Exception, token: str | None) -> str:
    message = str(exc)
    if token:
        message = message.replace(token, "[hidden]")
    return re.sub(r"\bhf_[A-Za-z0-9]{16,}\b", "[hidden]", message)[:500]


def _write_manifest(entries: dict[str, dict[str, str]]) -> None:
    path = MODEL_ROOT / "model-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    old: dict[str, Any] = {}
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = {}
    old.setdefault("schema_version", 1)
    old.setdefault("models", {}).update(entries)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _download_repo(repo_id: str, revision: str | None, target: Path, token: str | None):
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise ModelPreparationError("缺少 huggingface_hub；请安装 requirements-lock.txt。") from exc

    try:
        auth = token if token else False
        info = HfApi(token=auth).model_info(repo_id, revision=revision, token=auth)
        resolved_revision = info.sha
        snapshot_download(
            repo_id=repo_id,
            revision=resolved_revision,
            token=auth,
            local_dir=str(target),
            max_workers=4,
        )
        return resolved_revision
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403}:
            raise ModelPreparationError(
                f"模型访问被拒绝：{repo_id}（HTTP {status}）。如果模型需要访问许可，请由用户在模型页面完成授权；工具不会接受条款。"
            ) from None
        detail = _safe_message(exc, token)
        raise ModelPreparationError(f"模型下载失败：{repo_id}（HTTP {status or 'unknown'}）；{detail}") from None
    except Exception as exc:
        detail = _safe_message(exc, token)
        raise ModelPreparationError(f"模型下载失败：{repo_id}；{detail}") from None


def prepare_asr(token: str | None, revision: str = ASR_REVISION) -> dict[str, dict[str, str]]:
    files_present = ASR_MODEL_DIR.is_dir() and all(
        (ASR_MODEL_DIR / name).is_file() for name in _ASR_REQUIRED_FILES
    ) and (ASR_MODEL_DIR / "model.bin").stat().st_size > 0
    if files_present and revision == ASR_REVISION:
        print(f"ASR 模型已存在：{ASR_MODEL_DIR}")
        resolved_revision = revision
    else:
        resolved_revision = _download_repo(ASR_REPOSITORY, revision, ASR_MODEL_DIR, token)
    return {"asr_large_v3": {"repository": ASR_REPOSITORY, "revision": resolved_revision, "path": str(ASR_MODEL_DIR)}}


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        match = _REFERENCE.fullmatch(value.strip())
        if match:
            return replacements.get(match.group(1), value)
        return value
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_strings(item, replacements) for item in value)
    return value


def _model_reference_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in {"segmentation", "embedding"} and isinstance(item, str):
                yield str(key), item
            yield from _model_reference_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _model_reference_values(item)


def _checkpoint_problem(field: str, reference: Any) -> str | None:
    if not isinstance(reference, str) or not reference.strip():
        return f"{field}: 必须配置本地 pytorch_model.bin 检查点"

    configured = Path(reference).expanduser()
    if not configured.is_absolute():
        return f"{field}: 检查点必须使用绝对路径（{reference}）"

    checkpoint = configured.resolve()
    model_root = DIARIZATION_MODEL_DIR.expanduser().resolve()
    try:
        checkpoint.relative_to(model_root)
    except ValueError:
        return f"{field}: 检查点必须位于本地 pyannote 模型目录中（{reference}）"

    if not any(part.casefold() == "pyannote" for part in checkpoint.parts):
        return f"{field}: 检查点路径必须包含 pyannote 目录段（{reference}）"
    if checkpoint.name.casefold() != _PYANNOTE_CHECKPOINT_NAME.casefold():
        return f"{field}: 检查点文件名必须为 {_PYANNOTE_CHECKPOINT_NAME}（{reference}）"
    if not checkpoint.is_file():
        return f"{field}: 检查点文件不存在（{reference}）"
    if checkpoint.stat().st_size == 0:
        return f"{field}: 检查点文件为空（{reference}）"
    return None


def _diarization_config_problems(config: Any) -> list[str]:
    if not isinstance(config, dict):
        return ["pyannote config.yaml 顶层必须是 YAML 映射"]

    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict):
        return ["pyannote config.yaml 必须包含 pipeline 映射"]

    problems: list[str] = []
    name = pipeline.get("name")
    if not isinstance(name, str) or not name.strip():
        problems.append("pyannote pipeline 必须包含非空 name")
    params = pipeline.get("params")
    if not isinstance(params, dict):
        problems.append("pyannote pipeline 必须包含 params 映射")
        return problems

    for field in ("segmentation", "embedding"):
        if field not in params:
            problems.append(f"pyannote pipeline.params 缺少 {field} 检查点")
            continue
        problem = _checkpoint_problem(field, params[field])
        if problem:
            problems.append(f"pyannote 配置无效：{problem}")
    return problems


def prepare_diarization(token: str | None) -> dict[str, dict[str, str]]:
    try:
        import yaml
    except ImportError as exc:
        raise ModelPreparationError("缺少 PyYAML；请安装 requirements-lock.txt。") from exc

    manifest: dict[str, dict[str, str]] = {}
    queue: list[tuple[str, str | None, Path]] = [
        (DIARIZATION_REPOSITORY, None, DIARIZATION_MODEL_DIR)
    ]
    downloaded: dict[str, tuple[Path, str]] = {}
    while queue:
        repo_id, revision, target = queue.pop(0)
        if repo_id in downloaded:
            continue
        target.mkdir(parents=True, exist_ok=True)
        resolved = _download_repo(repo_id, revision, target, token)
        downloaded[repo_id] = (target, resolved)
        manifest[f"pyannote:{repo_id}"] = {
            "repository": repo_id,
            "revision": resolved,
            "path": str(target),
        }
        yaml_files = list(target.rglob("*.yaml")) + list(target.rglob("*.yml"))
        for yaml_path in yaml_files:
            try:
                content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            for text in _strings(content):
                for ref in _REFERENCE.finditer(text):
                    child_repo, child_revision = ref.group(1), ref.group(2)
                    if child_repo == repo_id or child_repo in downloaded:
                        continue
                    local_path = DIARIZATION_MODEL_DIR / "_nested" / Path(*child_repo.split("/"))
                    queue.append((child_repo, child_revision, local_path))

    replacements: dict[str, str] = {}
    for repo_id, (path, _) in downloaded.items():
        if repo_id != DIARIZATION_REPOSITORY:
            replacements[repo_id] = str(_local_checkpoint_path(path))

    for path, _ in downloaded.values():
        for yaml_path in list(path.rglob("*.yaml")) + list(path.rglob("*.yml")):
            try:
                content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            rewritten = _replace_strings(content, replacements)
            if rewritten != content:
                # Keep it ASCII so Windows ANSI defaults can parse paths under Chinese directories.
                _write_yaml_ascii(yaml_path, rewritten, yaml)

    config_path = DIARIZATION_MODEL_DIR / "config.yaml"
    if not config_path.is_file():
        raise ModelPreparationError(f"pyannote 模型缺少 config.yaml：{config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract_problems = _diarization_config_problems(config)
    if contract_problems:
        raise ModelPreparationError("pyannote pipeline 配置无效：" + "; ".join(contract_problems))
    unresolved = sorted({
        match.group(1)
        for text in _strings(config)
        for match in [_REFERENCE.fullmatch(text.strip())]
        if match and match.group(1) != DIARIZATION_REPOSITORY
    })
    if unresolved:
        raise ModelPreparationError("以下 pyannote 关联模型未能解析为本地路径：" + ", ".join(unresolved))
    return manifest


def verify_models() -> list[str]:
    problems = []
    if (
        not ASR_MODEL_DIR.is_dir()
        or any(not (ASR_MODEL_DIR / name).is_file() for name in _ASR_REQUIRED_FILES)
        or (ASR_MODEL_DIR / "model.bin").stat().st_size == 0
    ):
        problems.append(f"缺少 faster-whisper 模型文件：{ASR_MODEL_DIR}")
    config_path = DIARIZATION_MODEL_DIR / "config.yaml"
    if not config_path.is_file():
        problems.append(f"缺少 pyannote config.yaml：{config_path}")
        return problems
    try:
        import yaml
        yaml_files = list(DIARIZATION_MODEL_DIR.rglob("*.yaml")) + list(DIARIZATION_MODEL_DIR.rglob("*.yml"))
        remotes: set[str] = set()
        missing_paths: list[str] = []
        for yaml_path in yaml_files:
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            is_pipeline_config = yaml_path.resolve() == config_path.resolve()
            if is_pipeline_config:
                problems.extend(_diarization_config_problems(config))
            for text in _strings(config):
                match = _REFERENCE.fullmatch(text.strip())
                if match:
                    remotes.add(match.group(1))
            if not is_pipeline_config:
                for field, reference in _model_reference_values(config):
                    problem = _checkpoint_problem(field, reference)
                    if problem:
                        missing_paths.append(f"pyannote 配置无效：{problem}")
        if remotes:
            problems.append("pyannote 配置仍有非本地模型引用：" + ", ".join(sorted(remotes)))
        if missing_paths:
            problems.append("pyannote 关联模型本地路径不存在：" + "; ".join(missing_paths))
    except ImportError as exc:
        problems.append(f"无法读取 pyannote 配置：{type(exc).__name__}")
    except (OSError, ValueError) as exc:
        problems.append(f"无法读取 pyannote 配置：{type(exc).__name__}")
    except yaml.YAMLError as exc:
        problems.append(f"无法读取 pyannote 配置：{type(exc).__name__}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="准备并校验本地离线模型资源。")
    parser.add_argument("--asr", action="store_true", help="准备 large-v3 ASR 模型")
    parser.add_argument("--diarization", action="store_true", help="准备 pyannote 双人分离及所有关联模型")
    parser.add_argument("--verify", action="store_true", help="只检查模型和本地路径")
    parser.add_argument(
        "--prompt-token",
        action="store_true",
        help="在当前控制台隐藏输入 Hugging Face 令牌；令牌只保存在本进程内存中",
    )
    parser.add_argument("--asr-revision", default=ASR_REVISION, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    if args.verify:
        problems = verify_models()
        if problems:
            for problem in problems:
                print(f"未就绪：{problem}")
            return 1
        print("离线模型文件与 pyannote 本地引用检查通过。")
        return 0

    selected = args.asr or args.diarization
    do_asr = args.asr if selected else True
    do_diarization = args.diarization if selected else True
    token = _token()
    if args.prompt_token:
        try:
            token = getpass.getpass("Hugging Face token（输入隐藏，不会保存）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("未读取令牌，模型准备已取消。")
            return 2
        if not token:
            print("未提供令牌，模型准备已取消。")
            return 2
    entries: dict[str, dict[str, str]] = {}
    try:
        if do_asr:
            entries.update(prepare_asr(token, args.asr_revision))
        if do_diarization:
            entries.update(prepare_diarization(token))
        _write_manifest(entries)
    except ModelPreparationError as exc:
        print(f"准备未完成：{exc}", file=sys.stderr)
        return 3 if "HTTP 401" in str(exc) or "HTTP 403" in str(exc) else 1
    for name, item in entries.items():
        print(f"{name}: revision={item['revision']} path={item['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

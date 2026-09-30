from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from config import DIARIZATION_MODEL_DIR, NUM_SPEAKERS


def _enable_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Diarizer:
    """Lazy local-only pyannote speaker diarization."""

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | os.PathLike[str] = DIARIZATION_MODEL_DIR,
    ) -> None:
        self.device = device
        self.model_path = Path(model_path).expanduser().resolve()
        self.pipeline: Any = None

    @property
    def is_ready(self) -> bool:
        return (self.model_path / "config.yaml").is_file()

    def load(self) -> None:
        if self.pipeline is not None:
            return
        _enable_offline_mode()
        if not self.is_ready:
            raise FileNotFoundError(
                "未找到本地双人说话人分离模型。请先运行离线模型准备脚本；推理不会自动联网下载模型。"
            )
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError("缺少 pyannote.audio 或 PyTorch 运行依赖，请按部署说明安装。") from exc
        try:
            self.pipeline = Pipeline.from_pretrained(
                str(self.model_path / "config.yaml"),
                use_auth_token=False,
            )
            self.pipeline.to(torch.device(self.device))
        except Exception as exc:
            self.pipeline = None
            raise RuntimeError(
                "本地说话人分离模型加载失败。请检查离线模型是否完整，以及模型目录中的关联模型路径。"
            ) from exc

    def diarize(self, audio_path: str | os.PathLike[str], cancel_event=None) -> Any:
        if cancel_event and cancel_event.is_set():
            from utils.audio_utils import ProcessingCancelled
            raise ProcessingCancelled("已在说话人分离前取消。")
        audio = Path(audio_path).expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError("用于说话人分离的规范化音频不存在。")
        self.load()
        result = self.pipeline(str(audio), min_speakers=1, max_speakers=NUM_SPEAKERS)
        if cancel_event and cancel_event.is_set():
            from utils.audio_utils import ProcessingCancelled
            raise ProcessingCancelled("已在说话人分离安全边界取消。")
        return result[0] if isinstance(result, tuple) else result

    def release(self) -> None:
        self.pipeline = None
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

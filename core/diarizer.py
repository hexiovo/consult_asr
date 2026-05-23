import os
import subprocess
from pyannote.audio import Pipeline
import torch
from config import HF_TOKEN


class Diarizer:

    def __init__(self, device="cuda"):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=HF_TOKEN
        )
        self.pipeline.to(torch.device(device))

    def _convert_to_wav(self, file_path):
        """如果不是 wav，使用 ffmpeg 转换为单声道 16kHz wav"""
        wav_path = file_path.rsplit(".", 1)[0] + "_converted.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", file_path,
            "-ac", "1",
            "-ar", "16000",
            wav_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 转换失败: {result.stderr.decode()}")

        return wav_path

    def diarize(self, file_path, num_speakers=2):
        """对音频进行说话人分离，返回 Annotation 和临时 wav 文件路径（可删除）"""
        temp_wav = None

        if not file_path.lower().endswith(".wav"):
            temp_wav = self._convert_to_wav(file_path)
            file_path = temp_wav

        result = self.pipeline(file_path, num_speakers=num_speakers)

        # 兼容 Pyannote Audio 3.x 返回 tuple (annotation, scores)
        if isinstance(result, tuple):
            diarization = result[0]
        else:
            diarization = result

        return diarization, temp_wav
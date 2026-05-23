import whisper
import torch

class Transcriber:

    def __init__(self, device="cuda"):
        self.model = whisper.load_model("large-v3", device=device)

    def transcribe_chunks(self, chunks, progress_callback=None):
        """
        chunks: [(chunk_path, offset), ...]
        progress_callback: function(percent: float)
        """

        all_segments = []
        total = len(chunks)

        for idx, (chunk_path, offset) in enumerate(chunks):

            result = self.model.transcribe(
                chunk_path,
                language="zh"
            )

            for seg in result["segments"]:
                seg["start"] += offset
                seg["end"] += offset
                all_segments.append(seg)

            # 🔥 更新百分比
            if progress_callback:
                percent = (idx + 1) / total * 100
                progress_callback(percent)

        return all_segments

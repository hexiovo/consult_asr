import whisper
import torch

class Transcriber:

    def __init__(self, device="cuda"):
        self.model = whisper.load_model("large-v3", device=device)

    def transcribe_chunks(self, chunks):
        all_segments = []

        for chunk_path, offset in chunks:
            result = self.model.transcribe(chunk_path, language="zh")

            for seg in result["segments"]:
                seg["start"] += offset
                seg["end"] += offset
                all_segments.append(seg)

        return all_segments

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

    def diarize(self, file_path):
        return self.pipeline(file_path, num_speakers=2)

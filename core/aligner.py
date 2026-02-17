def align_segments(diarization, segments):

    aligned = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):

        for seg in segments:
            overlap = max(
                0,
                min(seg["end"], turn.end) - max(seg["start"], turn.start)
            )

            duration = seg["end"] - seg["start"]

            if duration > 0 and overlap / duration > 0.5:
                aligned.append({
                    "speaker": speaker,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                })

    return aligned

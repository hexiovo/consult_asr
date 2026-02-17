from pydub import AudioSegment

AudioSegment.converter = r"G:\conda\.conda\envs\consult_asr\Library\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"G:\conda\.conda\envs\consult_asr\Library\bin\ffprobe.exe"


def split_audio(file_path, chunk_minutes=15):
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)

    chunk_length = chunk_minutes * 60 * 1000
    chunks = []

    for i in range(0, duration_ms, chunk_length):
        chunk = audio[i:i + chunk_length]
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append((chunk_path, i / 1000))

    return chunks

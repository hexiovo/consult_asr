from pydub import AudioSegment
import os

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

def is_audio_file(path):
    return os.path.splitext(path)[1].lower() in {
        ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"
    }


def collect_audio_files(input_dir):
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if is_audio_file(full_path):
                audio_files.append(full_path)
    return audio_files
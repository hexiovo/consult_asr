import os
import time
import threading
import torch

from utils.logger import setup_logger
from utils.audio_utils import split_audio
from core.transcriber import Transcriber
from core.diarizer import Diarizer
from core.aligner import align_segments
from core.role_identifier import identify_roles
from core.exporter import export_word
from config import CHUNK_MINUTES
from app.gui import (
    select_audio,
    ask_filename,
    show_success,
    show_error
)
from app.progress_window import ProgressWindow


def run():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    logger, log_path = setup_logger(BASE_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=========== ConsultASR 启动 ===========")
    logger.info(f"运行设备: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    progress = ProgressWindow()

    def task():

        try:
            total_start = time.time()

            progress.update(5, "选择音频文件")
            file_path = select_audio()
            if not file_path:
                return

            progress.update(10, "输入输出文件名")
            filename = ask_filename()
            if not filename:
                return

            logger.info(f"音频路径: {file_path}")
            logger.info(f"输出文件名: {filename}")

            # ========================
            # 音频分块
            # ========================
            start = time.time()
            progress.update(20, "音频分块处理中...")
            chunks = split_audio(file_path, chunk_minutes=CHUNK_MINUTES)
            logger.info(f"音频分块完成，共 {len(chunks)} 块")
            logger.info(f"分块耗时: {time.time() - start:.2f} 秒")

            # ========================
            # Whisper 识别
            # ========================
            start = time.time()
            progress.update(40, "语音识别中...")
            transcriber = Transcriber(device)
            segments = transcriber.transcribe_chunks(chunks)
            logger.info(f"语音识别完成，文本段数: {len(segments)}")
            logger.info(f"识别耗时: {time.time() - start:.2f} 秒")

            # ========================
            # 说话人分离
            # ========================
            start = time.time()
            progress.update(65, "说话人分离中...")
            diarizer = Diarizer(device)
            diarization = diarizer.diarize(file_path)
            logger.info("说话人分离完成")
            logger.info(f"分离耗时: {time.time() - start:.2f} 秒")

            # ========================
            # 时间对齐
            # ========================
            start = time.time()
            progress.update(80, "时间对齐优化中...")
            aligned = align_segments(diarization, segments)
            logger.info("时间对齐完成")
            logger.info(f"对齐耗时: {time.time() - start:.2f} 秒")

            # ========================
            # 角色识别
            # ========================
            start = time.time()
            progress.update(90, "自动角色识别中...")
            roles = identify_roles(aligned)
            logger.info("角色识别完成")
            logger.info(f"角色识别耗时: {time.time() - start:.2f} 秒")

            # ========================
            # 导出
            # ========================
            start = time.time()
            progress.update(95, "导出文档中...")
            output_path = export_word(aligned, roles, OUTPUT_DIR, filename)
            logger.info("文档导出完成")
            logger.info(f"导出耗时: {time.time() - start:.2f} 秒")

            total_time = time.time() - total_start
            logger.info(f"总耗时: {total_time:.2f} 秒")

            progress.update(100, "处理完成")
            show_success(output_path, log_path)

        except Exception as e:
            logger.exception("系统运行异常")
            show_error(e)

    threading.Thread(target=task, daemon=True).start()
    progress.root.mainloop()

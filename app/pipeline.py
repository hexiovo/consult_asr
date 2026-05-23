import os
import threading
import torch
from tkinter import messagebox


from utils.logger import setup_logger
from utils.audio_utils import split_audio
from utils.audio_utils import collect_audio_files
from core.transcriber import Transcriber
from core.diarizer import Diarizer
from core.aligner import align_segments
from core.role_identifier import identify_roles
from core.exporter import export_word
from core.doc_postprocessor import merge_speaker_text
from config import CHUNK_MINUTES
from app.gui import (
    select_audio,
    ask_filename,
    show_success,
    show_error,
    ProgressWindow,
    select_folder,
    select_output_folder
)


def run_single():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    logger, log_path = setup_logger(BASE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("选择了单一文件模式")
    logger.info(f"运行设备: {device}")

    file_path = select_audio()

    if not file_path:
        return

    # 自动提取文件名
    filename = os.path.splitext(os.path.basename(file_path))[0]

    if not filename:
        return

    logger.info(f"选择音频: {file_path}")
    logger.info(f"输出文件名: {filename}")

    # 输出目录 = 输入目录
    OUTPUT_DIR = os.path.dirname(file_path)

    progress = ProgressWindow("语音处理进度")

    # ===== 模型只初始化一次 =====
    transcriber = Transcriber(device)
    diarizer = Diarizer(device)

    def worker():

        try:

            _task(
                file_path=file_path,
                filename=filename,
                OUTPUT_DIR=OUTPUT_DIR,
                device=device,
                logger=logger,
                log_path=log_path,
                progress=progress,
                transcriber=transcriber,
                diarizer=diarizer
            )

            # ===== 主线程关闭窗口 =====
            progress.root.after(0, progress.close)

        except Exception as e:

            import traceback

            logger.error(
                "单文件处理失败:\n%s",
                traceback.format_exc()
            )

            def show_err():
                show_error(str(e))

            progress.root.after(0, show_err)
            progress.root.after(0, progress.close)

    threading.Thread(
        target=worker,
        daemon=False
    ).start()

    # ===== 主线程事件循环 =====
    progress.root.mainloop()


def run_pipeline():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    logger, log_path = setup_logger(BASE_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"运行设备: {device}")

    # ===== 选择输入目录 =====
    input_dir = select_folder()

    if not input_dir:
        return

    # ===== 选择输出目录 =====
    output_root = select_output_folder()

    if not output_root:
        return

    # ===== 安全检查 =====
    if os.path.abspath(input_dir) == os.path.abspath(output_root):

        messagebox.showerror(
            "错误",
            "输入目录和输出目录不能相同"
        )

        return

    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_root}")

    # ===== 收集音频 =====
    audio_files = collect_audio_files(input_dir)

    if not audio_files:

        logger.warning("未找到音频文件")

        messagebox.showwarning(
            "提示",
            "未找到音频文件"
        )

        return

    total = len(audio_files)

    logger.info(f"共检测到 {total} 个音频文件")

    progress = ProgressWindow("批量语音处理进度")

    # ===== 模型只初始化一次 =====
    transcriber = Transcriber(device)
    diarizer = Diarizer(device)

    def batch_task():

        try:

            for idx, file_path in enumerate(audio_files, 1):

                try:

                    # ===== 相对路径 =====
                    rel_path = os.path.relpath(
                        file_path,
                        input_dir
                    )

                    # 去扩展名
                    rel_no_ext = os.path.splitext(rel_path)[0]

                    # ===== 输出目录 =====
                    output_dir = os.path.join(
                        output_root,
                        os.path.dirname(rel_no_ext)
                    )

                    os.makedirs(output_dir, exist_ok=True)

                    filename = os.path.basename(rel_no_ext)

                    logger.info(
                        f"[{idx}/{total}] 处理: {file_path}"
                    )

                    logger.info(f"输出目录: {output_dir}")

                    # ===== 主线程更新UI =====
                    progress.root.after(
                        0,
                        lambda i=idx, t=total, f=filename:
                        progress.update(
                            f"[{i}/{t}] 处理:\n{f}"
                        )
                    )

                    # ===== 执行任务 =====
                    _task(
                        file_path=file_path,
                        filename=filename,
                        OUTPUT_DIR=output_dir,
                        device=device,
                        logger=logger,
                        log_path=log_path,
                        progress=progress,
                        transcriber=transcriber,
                        diarizer=diarizer
                    )

                except Exception as e:

                    import traceback

                    logger.error(
                        f"失败: {file_path}\n"
                        f"{traceback.format_exc()}"
                    )

            logger.info("批处理完成")

            # ===== 主线程UI =====
            def finish():

                progress.update("全部处理完成")

                show_success(
                    output_root,
                    log_path
                )

                progress.close()

            progress.root.after(0, finish)

        except Exception as e:

            import traceback

            logger.error(
                "批处理异常:\n%s",
                traceback.format_exc()
            )

            def show_err():

                show_error(str(e))

                progress.close()

            progress.root.after(0, show_err)

    threading.Thread(
        target=batch_task,
        daemon=False
    ).start()

    progress.root.mainloop()

def _task(
        file_path,
        filename,
        OUTPUT_DIR,
        device,
        logger,
        log_path,
        progress,
        transcriber,
        diarizer
):

    chunks = []
    temp_wav = None

    try:

        display_name = filename

        # =========================================================
        # 1. 切分音频
        # =========================================================

        progress.root.after(
            0,
            lambda:
            progress.update(
                f"[{display_name}] 1/7 正在切分音频..."
            )
        )

        logger.info("开始音频切分")

        chunks = split_audio(
            file_path,
            chunk_minutes=CHUNK_MINUTES
        )

        logger.info(
            f"音频切分完成，共 {len(chunks)} 个片段"
        )

        # =========================================================
        # 2. 语音识别
        # =========================================================

        progress.root.after(
            0,
            lambda:
            progress.update(
                f"[{display_name}] 2/7 正在进行语音识别..."
            )
        )

        logger.info("开始语音识别")

        segments = transcriber.transcribe_chunks(chunks)

        logger.info(
            f"语音识别完成，共 {len(segments)} 条文本"
        )

        # =========================================================
        # 3. 说话人分离
        # =========================================================

        progress.root.after(
            0,
            lambda:
            progress.update(
                f"[{display_name}] 3/7 正在进行说话人分离..."
            )
        )

        logger.info("开始说话人分离")

        diarization, temp_wav = diarizer.diarize(file_path)

        logger.info("说话人分离完成")

        # =========================================================
        # 4. 时间对齐
        # =========================================================

        progress.root.after(
            0,
            lambda:
            progress.update(
                f"[{display_name}] 4/7 正在对齐时间戳..."
            )
        )

        logger.info("开始时间对齐")

        aligned = align_segments(
            diarization,
            segments
        )

        logger.info(
            f"对齐完成，共 {len(aligned)} 条段落"
        )

        # =========================================================
        # 5. 角色识别
        # =========================================================

        progress.root.after(
            0,
            lambda:
            progress.update(
                f"[{display_name}] 5/7 正在识别角色..."
            )
        )

        logger.info("开始角色识别")

        roles = identify_roles(aligned)

        logger.info("角色识别完成")

        # =========================================================
        # 6. 导出Word
        # =========================================================

        progress.root.after(
            0,
            lambda:
            progress.update(
                f"[{display_name}] 6/7 正在导出Word文档..."
            )
        )

        logger.info("开始导出文档")

        output_path = export_word(
            aligned,
            roles,
            OUTPUT_DIR,
            filename
        )

        logger.info(f"导出完成: {output_path}")

        # =========================================================
        # 7. 修正说话人
        # =========================================================

        progress.root.after(
            0,
            lambda:
            progress.update(
                f"[{display_name}] 7/7 正在修正说话人..."
            )
        )

        logger.info("开始修正说话人")

        merge_speaker_text(output_path)

        logger.info(f"修正完成: {output_path}")

        # =========================================================
        # 完成
        # =========================================================

        logger.info("全部处理完成")

        progress.root.after(
            0,
            lambda:
            progress.update(
                f"[{display_name}] 处理完成"
            )
        )

        # ===== 成功提示（主线程）=====
        progress.root.after(
            0,
            lambda:
            show_success(
                output_path,
                log_path
            )
        )

    except Exception as e:

        import traceback

        logger.error(
            "系统运行异常:\n%s",
            traceback.format_exc()
        )

        # ===== 主线程弹窗 =====
        progress.root.after(
            0,
            lambda:
            show_error(str(e))
        )

    finally:

        # =========================================================
        # 清理临时文件
        # =========================================================

        logger.info("开始清理临时文件")

        try:

            # ===== 删除 chunks =====
            for chunk_path, _ in chunks:

                if os.path.exists(chunk_path):

                    os.remove(chunk_path)

                    logger.info(
                        f"删除chunk: {chunk_path}"
                    )

            # ===== 删除 wav =====
            if temp_wav:

                if os.path.exists(temp_wav):

                    os.remove(temp_wav)

                    logger.info(
                        f"删除临时wav: {temp_wav}"
                    )

            logger.info("临时文件清理完成")

        except Exception as cleanup_error:

            logger.warning(
                f"清理临时文件失败: {cleanup_error}"
            )



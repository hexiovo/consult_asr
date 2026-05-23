import logging
import os
import sys
import warnings
from datetime import datetime


class StreamToLogger:
    """
    劫持 stdout / stderr 到 logger
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):

        message = message.strip()

        if message:
            self.logger.log(self.level, message)

    def flush(self):
        pass


def setup_logger(base_dir):
    log_dir = os.path.join(base_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{timestamp}.log")

    logger = logging.getLogger("ConsultASR")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 防止重复添加 handler

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # ===== 文件输出 =====
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ===== 劫持 stdout/stderr =====
    class StreamToLogger:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
        def write(self, msg):
            msg = msg.strip()
            if msg:
                self.logger.log(self.level, msg)
        def flush(self):
            pass

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # ===== 捕获 warnings =====
    logging.captureWarnings(True)
    warnings.simplefilter("always")  # 保证所有警告都输出
    # warnings.showwarning 会自动被 logging 捕获

    logger.info("系统启动")
    logger.info(f"日志文件: {log_path}")

    return logger, log_path
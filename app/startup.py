"""Early startup feedback and the native launcher's ready handshake."""

import os
from pathlib import Path


def show_startup_window():
    # The EXE already displays its window before Python and Qt are imported.
    if os.environ.get("CONSULT_ASR_STARTUP_EVENT"):
        return None
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QFrame, QLabel, QProgressBar, QVBoxLayout

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("ConsultASR")
    app.setOrganizationName("ConsultASR")
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1] / "assets" / "consult_asr.ico")))
    window = QFrame(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
    window.setObjectName("startupWindow")
    window.setWindowTitle("咨询录音转写 · 正在启动")
    window.setFixedSize(440, 210)
    window.setStyleSheet(
        "QFrame#startupWindow { background: #fffefa; border: 1px solid #b2cdc5; }"
        "QLabel { color: #24574c; border: none; font-size: 14px; }"
        "QLabel#startupTitle { font-size: 24px; font-weight: 600; }"
        "QProgressBar { background: #e0eeea; border: none; border-radius: 3px; }"
        "QProgressBar::chunk { background: #386c55; border-radius: 3px; }"
    )
    layout = QVBoxLayout(window)
    layout.setContentsMargins(30, 26, 30, 26)
    title = QLabel("咨询录音转写")
    title.setObjectName("startupTitle")
    layout.addWidget(title)
    layout.addWidget(QLabel("正在启动，请稍候…"))
    progress = QProgressBar()
    progress.setRange(0, 0)
    progress.setTextVisible(False)
    progress.setFixedHeight(6)
    layout.addWidget(progress)
    layout.addWidget(QLabel("正在准备界面与本地资源"))
    screen = app.primaryScreen()
    if screen is not None:
        window.move(screen.availableGeometry().center() - window.rect().center())
    # Keep QApplication alive until run_app takes over the event loop.
    window._application = app
    window.show()
    app.processEvents()
    return window


def finish_startup(startup_window=None) -> bool:
    if startup_window is not None:
        startup_window.close()
    name = os.environ.pop("CONSULT_ASR_STARTUP_EVENT", "")
    if os.name != "nt" or not name:
        return False
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    kernel.OpenEventW.restype = ctypes.c_void_p
    kernel.SetEvent.argtypes = [ctypes.c_void_p]
    kernel.SetEvent.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.OpenEventW(0x0002, False, name)  # EVENT_MODIFY_STATE
    if not handle:
        return False
    try:
        return bool(kernel.SetEvent(handle))
    finally:
        kernel.CloseHandle(handle)

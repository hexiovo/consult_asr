"""Offline launcher self-check; creates no real session or transcription."""
import os
import tempfile
from unittest.mock import patch

from PySide6.QtCore import QSettings, QStandardPaths, QTimer
from PySide6.QtWidgets import QApplication

import config
import app.gui as gui

assert os.environ.get("HF_HUB_OFFLINE") == "1"
assert config.FFMPEG_EXECUTABLE and os.path.isfile(config.FFMPEG_EXECUTABLE)
assert config.ASR_MODEL_DIR.is_dir() and config.DIARIZATION_MODEL_DIR.is_dir()
assert config.TONE_MODEL_DIR.is_dir() and config.TONE_RUNTIME_DIR.is_dir()
with tempfile.TemporaryDirectory(prefix="consult-asr-launch-check-") as temp:
    QStandardPaths.setTestModeEnabled(True)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, temp)
    QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, temp)
    app = QApplication([])
    gui._configure_application_icon(app)
    assert not app.windowIcon().isNull()
    with patch.object(gui.MainWindow, "_restore_session"), patch.object(
        gui.MainWindow, "_persist_session"
    ), patch.object(gui.MainWindow, "_refresh_model_state"):
        window = gui.MainWindow()
        assert not window.windowIcon().isNull()
        window.show()
        QTimer.singleShot(250, app.quit)
        assert app.exec() == 0
        window.close()

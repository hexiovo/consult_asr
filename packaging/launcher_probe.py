"""Offline launcher self-check; creates no real session or transcription."""
import os
import tempfile
from unittest.mock import patch

from PySide6.QtCore import QSettings, QStandardPaths, QTimer, QPoint
from PySide6.QtWidgets import QApplication

import config
import app.gui as gui
from utils.windows_shell import set_process_identity, APP_ID

assert os.environ.get("HF_HUB_OFFLINE") == "1"
assert config.FFMPEG_EXECUTABLE and os.path.isfile(config.FFMPEG_EXECUTABLE)
assert config.ASR_MODEL_DIR.is_dir() and config.DIARIZATION_MODEL_DIR.is_dir()
assert config.TONE_MODEL_DIR.is_dir() and config.TONE_RUNTIME_DIR.is_dir()
with tempfile.TemporaryDirectory(prefix="consult-asr-launch-check-") as temp:
    QStandardPaths.setTestModeEnabled(True)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, temp)
    QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, temp)
    set_process_identity()
    app = QApplication([])
    gui._configure_application_icon(app)
    assert not app.windowIcon().isNull()
    with patch.object(
        gui.MainWindow, "_restore_session", side_effect=AssertionError("Startup must not restore old results")
    ), patch.object(
        gui.MainWindow, "_persist_session"
    ), patch.object(gui.MainWindow, "_refresh_model_state"):
        window = gui.MainWindow()
        assert not window.input_files and not window.analyses and not window.role_choices
        assert window.table.rowCount() == 0 and window.table.currentRow() == -1
        assert not window.save_button.isEnabled() and not window.open_word_button.isEnabled()
        assert not window.windowIcon().isNull()
        assert window._media_player is None and window._audio_output is None
        if os.environ.get("CONSULT_ASR_CHECK_STARTUP") == "1":
            from app.startup import finish_startup

            window.show()
            app.processEvents()
            assert window.isVisible()
            assert finish_startup()
        if app.platformName() == "windows":
            import ctypes as c
            from utils.windows_shell import _Key, _Variant, _guid, _PROPERTY_GUID

            identity = gui._configure_window_identity(window)
            assert identity == {"icons": True, "shell": True}, identity
            hwnd = int(window.winId())
            send = c.windll.user32.SendMessageW
            send.argtypes = [c.c_void_p, c.c_uint, c.c_size_t, c.c_ssize_t]
            send.restype = c.c_ssize_t
            assert send(hwnd, 0x7f, 0, 0) and send(hwnd, 0x7f, 1, 0)
            iid = _guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99")
            store = c.c_void_p()
            acquire = c.windll.shell32.SHGetPropertyStoreForWindow
            acquire.argtypes = [c.c_void_p, c.POINTER(type(iid)), c.POINTER(c.c_void_p)]
            acquire.restype = c.c_int32
            assert acquire(hwnd, c.byref(iid), c.byref(store)) >= 0
            table = c.cast(store, c.POINTER(c.POINTER(c.c_void_p))).contents
            get_value = c.WINFUNCTYPE(c.c_int32, c.c_void_p, c.POINTER(_Key), c.POINTER(_Variant))(table[5])
            clear = c.windll.ole32.PropVariantClear
            clear.argtypes = [c.POINTER(_Variant)]
            try:
                launcher = os.environ["CONSULT_ASR_LAUNCHER_PATH"]
                for pid, expected in ((2, f'"{launcher}"'), (3, f"{launcher},0"), (5, APP_ID)):
                    variant = _Variant()
                    key = _Key(_guid(_PROPERTY_GUID), pid)
                    try:
                        assert get_value(store, c.byref(key), c.byref(variant)) >= 0
                        assert variant.vt == 31 and c.wstring_at(variant.value.pointer) == expected
                    finally:
                        clear(c.byref(variant))
            finally:
                c.WINFUNCTYPE(c.c_uint32, c.c_void_p)(table[2])(store)
        else:
            window.show()
            window._suppress_sentence_events = True
            window.sentence_combo.addItem("00:12 · 说话人1 · " + "这是一段合成测试的较长发言。" * 30)
            window._suppress_sentence_events = False
            for advanced in (False, True):
                window.advanced_toggle.setChecked(advanced)
                for width, height in ((900, 600), (1160, 760), (1280, 700)):
                    window.resize(width, height)
                    app.processEvents()
                    box = window.correction_box
                    controls = (window.sentence_combo, window.sentence_speaker_combo, window.sentence_editor, window.sentence_precision_note)
                    for control in controls:
                        origin = control.mapTo(box, QPoint(0, 0))
                        assert origin.y() >= 0 and origin.y() + control.height() <= box.height()
                        assert origin.x() >= 0 and origin.x() + control.width() <= box.width()
                    assert window.sentence_combo.height() >= window.sentence_combo.minimumSizeHint().height()
                    assert window.sentence_editor.height() >= 64
                    for first, second in zip(controls, controls[1:]):
                        assert first.y() + first.height() <= second.y()
                    window.review_scroll.ensureWidgetVisible(window.save_button)
                    app.processEvents()
                    position = window.save_button.mapTo(window.review_scroll.viewport(), QPoint(0, 0))
                    assert position.y() >= 0 and position.y() + window.save_button.height() <= window.review_scroll.viewport().height()
        QTimer.singleShot(250, app.quit)
        assert app.exec() == 0
        window.close()

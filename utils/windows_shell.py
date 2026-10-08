"""Windows identity for the Python window launched by the native desktop EXE."""
from __future__ import annotations

import ctypes as c
import os
import uuid
from pathlib import Path

APP_ID = "ConsultASR.Desktop"
_PROPERTY_GUID = "9f4c2855-9f79-4b39-a8d0-e1d42de1d5f3"


class _GUID(c.Structure):
    _fields_ = [("data1", c.c_uint32), ("data2", c.c_uint16),
                ("data3", c.c_uint16), ("data4", c.c_ubyte * 8)]


class _Key(c.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", c.c_uint32)]


class _Value(c.Union):
    _fields_ = [("pointer", c.c_void_p), ("alignment", c.c_uint64 * 2)]


class _Variant(c.Structure):
    _fields_ = [("vt", c.c_uint16), ("reserved1", c.c_uint16),
                ("reserved2", c.c_uint16), ("reserved3", c.c_uint16), ("value", _Value)]


def _guid(value: str) -> _GUID:
    return _GUID.from_buffer_copy(uuid.UUID(value).bytes_le)


def set_process_identity() -> bool:
    if os.name != "nt":
        return False
    try:
        function = c.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        function.argtypes = [c.c_wchar_p]
        function.restype = c.c_int32
        return function(APP_ID) >= 0
    except (OSError, AttributeError):
        return False


def _set_properties(hwnd: int, properties: dict[int, str]) -> None:
    iid = _guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99")
    store = c.c_void_p()
    acquire = c.windll.shell32.SHGetPropertyStoreForWindow
    acquire.argtypes = [c.c_void_p, c.POINTER(_GUID), c.POINTER(c.c_void_p)]
    acquire.restype = c.c_int32
    if acquire(hwnd, c.byref(iid), c.byref(store)) < 0:
        raise OSError("Windows window property store is unavailable")
    table = c.cast(store, c.POINTER(c.POINTER(c.c_void_p))).contents
    release = c.WINFUNCTYPE(c.c_uint32, c.c_void_p)(table[2])
    try:
        set_value = c.WINFUNCTYPE(c.c_int32, c.c_void_p, c.POINTER(_Key), c.POINTER(_Variant))(table[6])
        commit = c.WINFUNCTYPE(c.c_int32, c.c_void_p)(table[7])
        for pid, text in properties.items():
            buffer = c.create_unicode_buffer(text)
            variant = _Variant()
            variant.vt = 31  # VT_LPWSTR; SetValue copies this caller-owned string.
            variant.value.pointer = c.cast(buffer, c.c_void_p)
            key = _Key(_guid(_PROPERTY_GUID), pid)
            if set_value(store, c.byref(key), c.byref(variant)) < 0:
                raise OSError("Windows window property could not be set")
        if commit(store) < 0:
            raise OSError("Windows window properties could not be committed")
    finally:
        release(store)


def release_icons(handles: list[int]) -> None:
    if os.name != "nt":
        return
    destroy = c.windll.user32.DestroyIcon
    destroy.argtypes = [c.c_void_p]
    destroy.restype = c.c_int
    for handle in handles:
        destroy(handle)
    handles.clear()


def set_window_identity(hwnd: int, icon_path: Path, launcher_path: str = "") -> tuple[dict[str, bool], list[int]]:
    result = {"icons": False, "shell": False}
    handles: list[int] = []
    if os.name != "nt":
        return result, handles
    try:
        user32 = c.windll.user32
        load = user32.LoadImageW
        load.argtypes = [c.c_void_p, c.c_wchar_p, c.c_uint, c.c_int, c.c_int, c.c_uint]
        load.restype = c.c_void_p
        send = user32.SendMessageW
        send.argtypes = [c.c_void_p, c.c_uint, c.c_size_t, c.c_ssize_t]
        send.restype = c.c_ssize_t
        for kind, size in ((0, 32), (1, 256)):
            handle = load(None, str(icon_path), 1, size, size, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
            if not handle:
                raise OSError("Application icon could not be loaded")
            handles.append(handle)
            send(hwnd, 0x80, kind, handle)  # WM_SETICON
        result["icons"] = True
    except (OSError, AttributeError):
        # Any handle already assigned to this HWND stays alive until window destruction.
        pass
    try:
        properties = {5: APP_ID}
        launcher = Path(launcher_path) if launcher_path else None
        if launcher is not None and launcher.is_file():
            properties.update({2: f'"{launcher}"', 3: f"{launcher},0", 4: "咨询录音转写"})
        elif icon_path.is_file():
            properties[3] = str(icon_path)
        _set_properties(hwnd, properties)
        result["shell"] = True
    except (OSError, AttributeError):
        pass
    return result, handles

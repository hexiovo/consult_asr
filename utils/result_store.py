"""Bounded, atomic persistence for local JSON result records."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ResultStoreError(Exception):
    """A generic result-store failure that never includes stored payload data."""


class AtomicJsonStore:
    """Read and atomically replace a bounded UTF-8 JSON object file."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.path = Path(path).expanduser()
        self.max_bytes = max_bytes

    def read(self) -> dict[str, Any] | None:
        """Return the stored JSON object, or ``None`` when the file is absent."""

        try:
            with self.path.open("rb") as source:
                payload = source.read(self.max_bytes + 1)
        except FileNotFoundError:
            return None
        except OSError:
            raise ResultStoreError("Unable to read result store") from None

        if len(payload) > self.max_bytes:
            raise ResultStoreError("Result store exceeds configured size limit")

        try:
            decoded = payload.decode("utf-8")
            value = json.loads(
                decoded,
                parse_constant=self._reject_constant,
                parse_float=self._parse_finite_float,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OverflowError, RecursionError):
            raise ResultStoreError("Invalid result store") from None

        if not isinstance(value, dict):
            raise ResultStoreError("Result store must contain a JSON object")
        return value

    def write(self, value: Mapping[str, Any]) -> None:
        """Atomically replace the result object, enforcing the byte limit."""

        if not isinstance(value, Mapping):
            raise ResultStoreError("Result data must be a mapping")

        temporary_path: Path | None = None
        descriptor: int | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary_path = Path(temporary_name)

            encoder = json.JSONEncoder(
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            encoded_size = 0
            with os.fdopen(descriptor, "wb") as target:
                descriptor = None
                for chunk in encoder.iterencode(dict(value)):
                    encoded = chunk.encode("utf-8")
                    encoded_size += len(encoded)
                    if encoded_size > self.max_bytes:
                        raise ResultStoreError(
                            "Result data exceeds configured size limit"
                        )
                    written = target.write(encoded)
                    if written != len(encoded):
                        raise OSError("short write")
                target.flush()
                os.fsync(target.fileno())

            os.replace(temporary_path, self.path)
            temporary_path = None
        except ResultStoreError:
            raise
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise ResultStoreError("Result data is not valid JSON") from None
        except OSError:
            raise ResultStoreError("Unable to write result store") from None
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _reject_constant(_value: str) -> None:
        raise ValueError("non-standard JSON number")

    @staticmethod
    def _parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed

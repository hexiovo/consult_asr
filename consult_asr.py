"""Launch the local consultation transcription desktop application."""


def main() -> int:
    try:
        from app.gui import run_app
    except ImportError as exc:
        raise SystemExit(
            "界面依赖尚未安装。请先按 README.md 创建隔离环境并安装 requirements-lock.txt。"
        ) from exc
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())

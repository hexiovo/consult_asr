from app.pipeline import run_single
from app.pipeline import run_pipeline
from app.gui import select_mode

if __name__ == "__main__":
    mode = select_mode()

    if mode == 1:
        input_path = run_single()
    elif mode == 2:
        input_path = run_pipeline()


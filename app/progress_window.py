import threading
import tkinter as tk
from tkinter import ttk


class ProgressWindow:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ConsultASR 运行中")
        self.root.geometry("400x150")

        self.label = tk.Label(self.root, text="初始化中...")
        self.label.pack(pady=10)

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=350,
            mode="determinate"
        )
        self.progress.pack(pady=10)

        self.percent_label = tk.Label(self.root, text="0%")
        self.percent_label.pack()

    def update(self, value, text=""):
        self.progress["value"] = value
        self.percent_label.config(text=f"{int(value)}%")
        if text:
            self.label.config(text=text)
        self.root.update_idletasks()

    def start(self, target_func):
        thread = threading.Thread(target=target_func)
        thread.start()
        self.root.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import os

def select_audio():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择音频文件",
        filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac *.aac")]
    )
    root.destroy()
    return file_path


def ask_filename():
    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("输出文件名", "请输入导出的文件名（不含扩展名）:")
    root.destroy()
    return name


def show_success(output_path, log_path):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "完成",
        f"导出成功:\n{output_path}\n\n日志文件:\n{log_path}"
    )
    root.destroy()


def show_error(error):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("错误", str(error))
    root.destroy()


# ================= 进度窗口 =================

class ProgressWindow:

    def __init__(self, title="处理中"):

        self.root = tk.Tk()

        self.root.title(title)

        self.root.geometry("420x160")

        self.root.resizable(False, False)

        # ===== 防止用户误关闭 =====
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.disable_close
        )

        # ===== 主容器 =====
        frame = ttk.Frame(
            self.root,
            padding=20
        )

        frame.pack(
            fill="both",
            expand=True
        )

        # ===== 状态文本 =====
        self.label = ttk.Label(
            frame,
            text="准备中...",
            anchor="center",
            justify="center",
            font=("微软雅黑", 11)
        )

        self.label.pack(
            pady=(10, 20),
            fill="x"
        )

        # ===== 进度条 =====
        self.progressbar = ttk.Progressbar(
            frame,
            mode="indeterminate",
            length=320
        )

        self.progressbar.pack(
            pady=(0, 10)
        )

        self.progressbar.start(10)

        # ===== 状态 =====
        self.closed = False

    # =====================================================
    # 禁止手动关闭
    # =====================================================

    def disable_close(self):

        pass

    # =====================================================
    # 更新文本（线程安全调用）
    # =====================================================

    def update(self, text):

        if self.closed:
            return

        try:

            self.label.config(text=text)

        except Exception:

            pass

    # =====================================================
    # 安全关闭
    # =====================================================

    def close(self):

        if self.closed:
            return

        self.closed = True

        try:

            self.progressbar.stop()

        except Exception:

            pass

        try:

            self.root.destroy()

        except Exception:

            pass

def select_mode():
    root = tk.Tk()
    root.title("选择输入模式")

    # ===== 尺寸（建议稍微放大一点更协调）=====
    width = 360
    height = 200

    # ===== 计算居中位置 =====
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    root.geometry(f"{width}x{height}+{x}+{y}")
    root.resizable(False, False)

    result = {"mode": None}

    # ===== 中间容器 =====
    frame = tk.Frame(root)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(frame, text="请选择处理模式", font=("Arial", 12)).pack(pady=5)

    mode_var = tk.StringVar()
    combo = ttk.Combobox(
        frame,
        textvariable=mode_var,
        state="readonly",
        values=["单个音频文件", "文件夹（批量处理）"]
    )
    combo.current(0)
    combo.pack(pady=5)

    def confirm():
        if mode_var.get() == "单个音频文件":
            result["mode"] = 1
        else:
            result["mode"] = 2
        root.destroy()

    # ===== 关键：点 X 时退出 =====
    def on_close():
        result["mode"] = None
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    tk.Button(frame, text="确定", width=15, command=confirm).pack(pady=10)

    root.mainloop()
    return result["mode"]

def select_folder():
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(
        title="选择音频文件夹"
    )

    root.destroy()

    if not folder_path:
        return None

    if not os.listdir(folder_path):
        messagebox.showwarning("提示", "该文件夹为空")
        return None

    return folder_path

def select_output_folder():
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(
        title="选择输出文件夹（处理结果将保存到这里）"
    )

    root.destroy()

    if not folder_path:
        return None

    return folder_path
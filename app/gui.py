import os
from tkinter import Tk, filedialog, simpledialog, messagebox


def select_audio():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="选择咨询音频",
        filetypes=[("Audio Files", "*.wav *.mp3 *.m4a")]
    )

    return file_path


def ask_filename():
    return simpledialog.askstring(
        "输出文件名",
        "请输入输出文件名（不含扩展名）："
    )


def show_success(output_path, log_path):
    messagebox.showinfo(
        "完成",
        f"文件已生成:\n{output_path}\n\n日志:\n{log_path}"
    )


def show_error(error):
    messagebox.showerror("错误", str(error))

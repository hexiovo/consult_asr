"""Local, searchable user guide shared with the repository documentation."""

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)


GUIDE_PATH = Path(__file__).resolve().parents[1] / "USER_GUIDE.md"
PDF_GUIDE_PATH = GUIDE_PATH.parent / "output" / "pdf" / "咨询录音转写_图文使用指南.pdf"


class UserGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("使用指南 · 咨询录音转写")
        self.resize(900, 700)
        self.setMinimumSize(640, 480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(QLabel("本地使用指南 · 可在处理录音时查看 · 不会上传数据"))

        navigation = QHBoxLayout()
        self.sections = QComboBox()
        self.sections.setAccessibleName("指南章节")
        self.sections.setMinimumContentsLength(12)
        self.sections.currentIndexChanged.connect(self._jump_to_section)
        navigation.addWidget(self.sections)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索指南，按回车查找")
        self.search.setAccessibleName("搜索使用指南")
        self.search.returnPressed.connect(self._find_next)
        navigation.addWidget(self.search, 1)
        self.find_button = QPushButton("查找下一个")
        self.find_button.setObjectName("secondaryButton")
        self.find_button.setAutoDefault(False)
        self.find_button.clicked.connect(self._find_next)
        navigation.addWidget(self.find_button)
        layout.addLayout(navigation)

        self.viewer = QTextBrowser()
        self.viewer.setAccessibleName("使用指南正文")
        self.viewer.setOpenLinks(False)
        self.viewer.setOpenExternalLinks(False)
        self.viewer.document().setDefaultStyleSheet(
            "h1, h2, h3 { color: #315d49; } "
            "p, li { line-height: 140%; } "
            "pre { background-color: #edf2ed; }"
        )
        layout.addWidget(self.viewer, 1)

        footer = QHBoxLayout()
        self.search_status = QLabel()
        self.search_status.setObjectName("mutedLabel")
        footer.addWidget(self.search_status, 1)
        self.pdf_button = QPushButton("打开 PDF 指南")
        self.pdf_button.setObjectName("primaryButton")
        self.pdf_button.setAutoDefault(False)
        self.pdf_button.clicked.connect(self._open_pdf)
        footer.addWidget(self.pdf_button)
        close_button = QPushButton("关闭指南")
        close_button.setObjectName("quietButton")
        close_button.setAutoDefault(False)
        close_button.clicked.connect(self.close)
        footer.addWidget(close_button)
        layout.addLayout(footer)
        self.reload_guide()

    def _open_pdf(self):
        if not PDF_GUIDE_PATH.is_file():
            QMessageBox.information(
                self, "PDF 指南缺失", "请保留 output/pdf 中的图文使用指南 PDF，或继续阅读本窗口的文字指南。"
            )
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(PDF_GUIDE_PATH))):
            QMessageBox.warning(
                self, "无法打开 PDF", "系统未能打开 PDF，请使用本地 PDF 阅读器打开 output/pdf 中的图文使用指南。"
            )

    def reload_guide(self):
        self.sections.blockSignals(True)
        self.sections.clear()
        self.sections.addItem("章节跳转…", None)
        self.search.clear()
        self.search_status.clear()
        try:
            text = GUIDE_PATH.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            self.viewer.setPlainText(
                "使用指南文件缺失或无法读取。\n"
                "请将 USER_GUIDE.md 保留在程序根目录，恢复文件后重新点击“使用指南”。\n"
                "此问题不影响转录、校对和导出。"
            )
            available = False
        else:
            self.viewer.setMarkdown(text)
            block = self.viewer.document().begin()
            while block.isValid():
                if block.blockFormat().headingLevel() == 2:
                    self.sections.addItem(block.text(), block.position())
                block = block.next()
            available = True
        self.sections.setEnabled(available)
        self.search.setEnabled(available)
        self.find_button.setEnabled(available)
        self.sections.blockSignals(False)
        self.viewer.moveCursor(QTextCursor.MoveOperation.Start)

    def _jump_to_section(self, index):
        position = self.sections.itemData(index)
        if position is not None:
            cursor = self.viewer.textCursor()
            cursor.setPosition(position)
            self.viewer.setTextCursor(cursor)
            self.viewer.ensureCursorVisible()

    def _find_next(self):
        query = self.search.text().strip()
        if not query:
            self.search_status.clear()
            return
        found = self.viewer.find(query)
        if not found:
            self.viewer.moveCursor(QTextCursor.MoveOperation.Start)
            found = self.viewer.find(query)
        self.search_status.setText("" if found else "未找到匹配文字，请换一个关键词。")

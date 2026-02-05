"""视频列表面板组件"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTreeWidget, QTreeWidgetItem, QFileDialog
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor


class VideoListPanel(QWidget):
    """视频列表面板"""
    video_selected = pyqtSignal(str, str)

    def __init__(self, data_manager=None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.video_pairs: List[Tuple] = []
        self.current_index = -1
        self.annotations_ref: Dict = {}  # 引用主窗口的标注数据
        self._setup_ui()

    def set_data_manager(self, data_manager):
        """设置数据管理器"""
        self.data_manager = data_manager

    def set_annotations_ref(self, annotations: Dict):
        """设置标注数据引用"""
        self.annotations_ref = annotations

    def get_last_folder(self) -> str:
        """获取上次打开的文件夹"""
        if self.data_manager:
            return self.data_manager.get_last_folder()
        return ""

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 选择按钮
        btn = QPushButton("📁 选择文件夹")
        btn.setStyleSheet("""
            QPushButton {
                background: #1976D2; color: white; border: none;
                padding: 16px; font-size: 18px; font-weight: bold; border-radius: 8px;
            }
            QPushButton:hover { background: #1565C0; }
        """)
        btn.clicked.connect(self.select_folder)
        layout.addWidget(btn)

        # 路径
        self.path_label = QLabel("未选择文件夹")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #666; font-size: 14px; padding: 5px;")
        layout.addWidget(self.path_label)

        # 列表
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["测试者", "动作", "次数", "状态"])
        self.tree.setColumnWidth(0, 70)
        self.tree.setColumnWidth(1, 70)
        self.tree.setColumnWidth(2, 70)
        self.tree.setColumnWidth(3, 80)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background: white; border: 2px solid #E0E0E0; border-radius: 8px;
                font-size: 15px;
            }
            QTreeWidget::item { padding: 8px; }
            QTreeWidget::item:selected { background: #BBDEFB; color: #1565C0; }
            QHeaderView::section { background: #F5F5F5; padding: 8px; font-weight: bold; font-size: 14px; }
        """)
        self.tree.itemDoubleClicked.connect(lambda item, _: self.select_video(self.tree.indexOfTopLevelItem(item)))
        layout.addWidget(self.tree, 1)

        # 导航
        nav = QHBoxLayout()
        for text, slot in [("◀ 上一个", self.prev_video), ("下一个 ▶", self.next_video)]:
            b = QPushButton(text)
            b.setStyleSheet("""
                QPushButton {
                    background: #607D8B; color: white; border: none;
                    padding: 12px 22px; font-size: 16px; border-radius: 6px;
                }
                QPushButton:hover { background: #455A64; }
            """)
            b.clicked.connect(slot)
            nav.addWidget(b)
        layout.addLayout(nav)

        # 进度
        self.progress = QLabel("0 / 0")
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet("font-size: 18px; font-weight: bold; color: #333; padding: 10px;")
        layout.addWidget(self.progress)

    def auto_load_last_folder(self):
        """自动加载上次的文件夹"""
        last_folder = self.get_last_folder()
        if last_folder and Path(last_folder).exists():
            self._load(last_folder)

    def select_folder(self):
        dialog = QFileDialog(self, "选择视频文件夹")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        # 设置初始目录为上次打开的文件夹
        last_folder = self.get_last_folder()
        if last_folder and Path(last_folder).exists():
            dialog.setDirectory(last_folder)
        # 设置更大的对话框尺寸
        dialog.resize(900, 600)
        if dialog.exec_():
            folders = dialog.selectedFiles()
            if folders:
                if self.data_manager:
                    self.data_manager.set_last_folder(folders[0])
                self._load(folders[0])

    def _load(self, folder: str):
        self.path_label.setText(folder)
        self.video_pairs = []
        self.tree.clear()

        # 解析文件夹结构: RecSync-Archive\s01\m01\e1\r0000\front_xxx.mp4
        for d in Path(folder).rglob("*"):
            if d.is_dir():
                mp4s = list(d.glob("*.mp4"))
                if mp4s:
                    front, side = None, None
                    for f in mp4s:
                        n = f.name.lower()
                        if 'front' in n or 'font' in n:
                            front = str(f)
                        elif 'side' in n:
                            side = str(f)
                    if not front:
                        front = str(mp4s[0])
                    if front:
                        # 解析路径获取测试者、动作、次数信息
                        rel_path = d.relative_to(folder)
                        parts = rel_path.parts

                        # 尝试解析: s01/m01/e1/r0000 结构
                        subject = ""  # 测试者
                        motion = ""   # 动作
                        trial = ""    # 次数

                        for p in parts:
                            p_lower = p.lower()
                            if p_lower.startswith('s') and len(p) >= 2 and p[1:].replace('0', '').isdigit():
                                subject = p.upper()  # S01
                            elif p_lower.startswith('m') and len(p) >= 2 and p[1:].replace('0', '').isdigit():
                                motion = p.upper()   # M01
                            elif p_lower.startswith('e') and len(p) >= 2:
                                # e1 是实验编号，可以合并到motion
                                motion = f"{motion}/{p}" if motion else p
                            elif p_lower.startswith('r') and len(p) >= 2:
                                trial = p.upper()    # R0000

                        # 如果无法解析，使用相对路径
                        if not subject:
                            subject = parts[0] if len(parts) > 0 else "?"
                        if not motion:
                            motion = parts[1] if len(parts) > 1 else "?"
                        if not trial:
                            trial = parts[-1] if len(parts) > 0 else "?"

                        vid = str(rel_path).replace("\\", "_").replace("/", "_")
                        self.video_pairs.append((front, side, vid, subject, motion, trial))

                        # 检查是否已标注
                        status = "待标注"
                        if vid in self.annotations_ref:
                            status = "✓ 已标注"

                        item = QTreeWidgetItem([subject, motion, trial, status])
                        if status == "✓ 已标注":
                            item.setForeground(3, QColor("#4CAF50"))
                        self.tree.addTopLevelItem(item)

        self._update()
        if self.video_pairs:
            self.select_video(0)

    def select_video(self, idx: int):
        if 0 <= idx < len(self.video_pairs):
            self.current_index = idx
            for i in range(self.tree.topLevelItemCount()):
                self.tree.topLevelItem(i).setSelected(i == idx)
            self.tree.scrollToItem(self.tree.topLevelItem(idx))
            # video_pairs: (front, side, vid, subject, motion, trial)
            self.video_selected.emit(self.video_pairs[idx][0], self.video_pairs[idx][1] or "")
            self._update()

    def prev_video(self):
        if self.current_index > 0:
            self.select_video(self.current_index - 1)

    def next_video(self):
        if self.current_index < len(self.video_pairs) - 1:
            self.select_video(self.current_index + 1)

    def get_current_id(self) -> Optional[str]:
        # video_pairs: (front, side, vid, subject, motion, trial)
        return self.video_pairs[self.current_index][2] if 0 <= self.current_index < len(self.video_pairs) else None

    def get_current_info(self) -> Optional[Dict]:
        """获取当前视频的详细信息"""
        if 0 <= self.current_index < len(self.video_pairs):
            front, side, vid, subject, motion, trial = self.video_pairs[self.current_index]
            return {'subject': subject, 'motion': motion, 'trial': trial, 'vid': vid}
        return None

    def mark_done(self, vid: str):
        for i, item in enumerate(self.video_pairs):
            if item[2] == vid:  # vid is at index 2
                tree_item = self.tree.topLevelItem(i)
                if tree_item:
                    tree_item.setText(3, "✓ 已完成")  # status is column 3
                    tree_item.setForeground(3, QColor("#4CAF50"))

    def _update(self):
        t = len(self.video_pairs)
        c = self.current_index + 1 if self.current_index >= 0 else 0
        self.progress.setText(f"{c} / {t}")


# 需要导入Qt
from PyQt5.QtCore import Qt

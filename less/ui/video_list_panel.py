"""视频列表面板组件 - 可折叠层级树结构"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont, QIcon


class VideoListPanel(QWidget):
    """视频列表面板 - 支持层级折叠"""
    video_selected = pyqtSignal(str, str)

    def __init__(self, data_manager=None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.video_pairs: List[Tuple] = []  # (front, side, vid, subject, motion)
        self.current_index = -1
        self.subject_items: Dict[str, QTreeWidgetItem] = {}  # 测试者节点缓存
        self.motion_items: Dict[str, QTreeWidgetItem] = {}  # 动作节点缓存 {vid: item}
        self._setup_ui()

    def set_data_manager(self, data_manager):
        self.data_manager = data_manager

    def get_last_folder(self) -> str:
        if self.data_manager:
            return self.data_manager.get_last_folder()
        return ""

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 标题栏
        header = QWidget()
        header.setStyleSheet("background: #1976D2; border-radius: 6px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)

        title = QLabel("📁 视频列表")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        # 折叠/展开按钮
        self.collapse_btn = QPushButton("折叠全部")
        self.collapse_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.2); color: white; border: none;
                padding: 4px 10px; border-radius: 4px; font-size: 12px;
            }
            QPushButton:hover { background: rgba(255,255,255,0.3); }
        """)
        self.collapse_btn.clicked.connect(self._toggle_collapse)
        header_layout.addWidget(self.collapse_btn)

        layout.addWidget(header)

        # 选择文件夹按钮
        btn = QPushButton("选择文件夹...")
        btn.setStyleSheet("""
            QPushButton {
                background: #E3F2FD; color: #1976D2; border: 2px dashed #1976D2;
                padding: 12px; font-size: 14px; border-radius: 6px;
            }
            QPushButton:hover { background: #BBDEFB; }
        """)
        btn.clicked.connect(self.select_folder)
        layout.addWidget(btn)

        # 路径显示
        self.path_label = QLabel("未选择文件夹")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px;")
        layout.addWidget(self.path_label)

        # 树形列表 - 可折叠层级结构
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["名称", "状态"])
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 60)
        self.tree.header().setStretchLastSection(True)
        self.tree.setAnimated(True)  # 启用折叠动画
        self.tree.setIndentation(20)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background: white; border: 1px solid #E0E0E0; border-radius: 6px;
                font-size: 13px;
            }
            QTreeWidget::item { padding: 6px 4px; }
            QTreeWidget::item:selected { background: #E3F2FD; color: #1565C0; }
            QTreeWidget::item:hover { background: #F5F5F5; }
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                image: url(:/icons/branch-closed.png);
            }
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {
                image: url(:/icons/branch-open.png);
            }
            QHeaderView::section {
                background: #F5F5F5; padding: 6px; font-weight: bold;
                border: none; border-bottom: 1px solid #E0E0E0;
            }
        """)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree, 1)

        # 导航按钮
        nav = QHBoxLayout()
        nav.setSpacing(6)
        for text, slot in [("◀ 上一个", self.prev_video), ("下一个 ▶", self.next_video)]:
            b = QPushButton(text)
            b.setStyleSheet("""
                QPushButton {
                    background: #607D8B; color: white; border: none;
                    padding: 10px 16px; font-size: 13px; border-radius: 5px;
                }
                QPushButton:hover { background: #455A64; }
            """)
            b.clicked.connect(slot)
            nav.addWidget(b)
        layout.addLayout(nav)

        # 进度显示
        self.progress = QLabel("0 / 0")
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet("""
            font-size: 14px; font-weight: bold; color: #333;
            padding: 8px; background: #F5F5F5; border-radius: 5px;
        """)
        layout.addWidget(self.progress)

    def _toggle_collapse(self):
        """切换折叠/展开状态"""
        if self.collapse_btn.text() == "折叠全部":
            self.tree.collapseAll()
            self.collapse_btn.setText("展开全部")
        else:
            self.tree.expandAll()
            self.collapse_btn.setText("折叠全部")

    def auto_load_last_folder(self):
        last_folder = self.get_last_folder()
        if last_folder and Path(last_folder).exists():
            self._load(last_folder)

    def select_folder(self):
        dialog = QFileDialog(self, "选择视频文件夹")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        last_folder = self.get_last_folder()
        if last_folder and Path(last_folder).exists():
            dialog.setDirectory(last_folder)
        dialog.resize(900, 600)
        if dialog.exec_():
            folders = dialog.selectedFiles()
            if folders:
                if self.data_manager:
                    self.data_manager.set_last_folder(folders[0])
                self._load(folders[0])

    def _load(self, folder: str):
        """加载文件夹 - 支持新目录结构: archive/s01/m01_e1/front.mp4"""
        self.path_label.setText(folder)
        self.video_pairs = []
        self.tree.clear()
        self.subject_items = {}
        self.motion_items = {}

        root_path = Path(folder)

        # 遍历所有包含视频的目录
        for d in root_path.rglob("*"):
            if not d.is_dir():
                continue

            # 查找mp4文件
            mp4s = list(d.glob("*.mp4"))
            if not mp4s:
                continue

            # 识别front和side视频
            front, side = None, None
            for f in mp4s:
                name_lower = f.name.lower()
                if 'front' in name_lower or 'font' in name_lower:
                    front = str(f)
                elif 'side' in name_lower:
                    side = str(f)

            # 如果没找到front，使用第一个视频
            if not front and mp4s:
                front = str(mp4s[0])

            if not front:
                continue

            # 解析路径结构
            try:
                rel_path = d.relative_to(root_path)
                parts = rel_path.parts
            except ValueError:
                continue

            # 解析: s01/m01_e1 结构
            subject = ""  # 测试者 (s01)
            motion = ""   # 动作 (m01_e1)

            for p in parts:
                p_lower = p.lower()
                # 匹配 s01, s02 等测试者编号
                if p_lower.startswith('s') and len(p) >= 2:
                    try:
                        int(p[1:])  # 验证是数字
                        subject = p.upper()
                    except ValueError:
                        pass
                # 匹配 m01_e1, m02_e2 等动作编号
                elif p_lower.startswith('m') and len(p) >= 2:
                    motion = p.upper()
                # 也支持旧格式: m01/e1 分开的情况
                elif p_lower.startswith('e') and len(p) >= 2 and motion:
                    motion = f"{motion}_{p.upper()}"

            # 如果无法解析，使用路径作为标识
            if not subject:
                subject = parts[0].upper() if parts else "未知"
            if not motion:
                motion = parts[-1].upper() if parts else "未知"

            # 生成视频ID
            vid = f"{subject}_{motion}".replace("/", "_").replace("\\", "_")

            self.video_pairs.append((front, side, vid, subject, motion))

            # 创建或获取测试者节点
            if subject not in self.subject_items:
                subject_item = QTreeWidgetItem([f"👤 {subject}", ""])
                subject_item.setFlags(subject_item.flags() & ~Qt.ItemIsSelectable)
                font = QFont()
                font.setBold(True)
                subject_item.setFont(0, font)
                subject_item.setExpanded(True)
                self.tree.addTopLevelItem(subject_item)
                self.subject_items[subject] = subject_item

            # 检查标注状态
            status, status_color = self._get_video_status(vid)

            # 创建动作节点
            motion_item = QTreeWidgetItem([f"🎬 {motion}", status])
            motion_item.setData(0, Qt.UserRole, len(self.video_pairs) - 1)  # 存储索引
            motion_item.setForeground(1, QColor(status_color))
            self.subject_items[subject].addChild(motion_item)
            self.motion_items[vid] = motion_item  # 缓存动作节点

        # 更新进度
        self._update()

        # 自动选择第一个视频
        if self.video_pairs:
            self.select_video(0)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """单击项目"""
        # 如果是测试者节点（父节点），切换展开状态
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """双击项目 - 选择视频"""
        idx = item.data(0, Qt.UserRole)
        if idx is not None:
            self.select_video(idx)

    def select_video(self, idx: int):
        if 0 <= idx < len(self.video_pairs):
            self.current_index = idx
            front, side, vid, subject, motion = self.video_pairs[idx]

            # 更新选中状态
            self.tree.clearSelection()
            if subject in self.subject_items:
                parent = self.subject_items[subject]
                for i in range(parent.childCount()):
                    child = parent.child(i)
                    if child.data(0, Qt.UserRole) == idx:
                        child.setSelected(True)
                        self.tree.scrollToItem(child)
                        parent.setExpanded(True)
                        break

            self.video_selected.emit(front, side or "")
            self._update()

    def prev_video(self):
        if self.current_index > 0:
            self.select_video(self.current_index - 1)

    def next_video(self):
        if self.current_index < len(self.video_pairs) - 1:
            self.select_video(self.current_index + 1)

    def get_current_id(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.video_pairs):
            return self.video_pairs[self.current_index][2]
        return None

    def get_current_info(self) -> Optional[Dict]:
        if 0 <= self.current_index < len(self.video_pairs):
            front, side, vid, subject, motion = self.video_pairs[self.current_index]
            return {'subject': subject, 'motion': motion, 'trial': '', 'vid': vid}
        return None

    def mark_done(self, vid: str):
        """标记视频状态已更新"""
        if vid in self.motion_items:
            status, status_color = self._get_video_status(vid)
            item = self.motion_items[vid]
            item.setText(1, status)
            item.setForeground(1, QColor(status_color))
        self._update()

    def _get_video_status(self, vid: str) -> Tuple[str, str]:
        """获取视频标注状态和颜色

        三种状态：
        - 未标注 (灰色): 无数据
        - 待完善 (橙色): 有数据但未完成
        - 已标注 (绿色): 全部完成
        """
        if not self.data_manager:
            return "未标注", "#999"

        ann = self.data_manager.get_annotation(vid)
        if not ann:
            return "未标注", "#999"

        if ann.get('completed', False):
            return "✓ 已标注", "#4CAF50"  # 绿色
        else:
            return "○ 待完善", "#FF9800"  # 橙色

    def refresh_all_status(self):
        """刷新所有视频的状态显示"""
        for vid, item in self.motion_items.items():
            status, status_color = self._get_video_status(vid)
            item.setText(1, status)
            item.setForeground(1, QColor(status_color))
        self._update()

    def _update(self):
        total = len(self.video_pairs)
        current = self.current_index + 1 if self.current_index >= 0 else 0

        # 统计已完成和暂存数量
        if self.data_manager:
            completed = self.data_manager.get_completed_count()
            self.progress.setText(f"{current} / {total}  (已完成: {completed})")
        else:
            self.progress.setText(f"{current} / {total}")

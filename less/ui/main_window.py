"""主窗口模块"""

import sys
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QSpinBox,
    QMessageBox, QDesktopWidget, QSplitter, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from ..config import MEDIAPIPE_AVAILABLE
from ..pose_detector import PoseDetector
from ..data_manager import DataManager
from .video_player import VideoPlayer
from .scoring_panel import ScoringPanel
from .video_list_panel import VideoListPanel
from .styles import (
    BUTTON_STYLE_PRIMARY, POSE_BUTTON_STYLE, SLIDER_STYLE, MAIN_WINDOW_STYLE
)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.data_manager = DataManager()
        self.pose_detector = PoseDetector() if MEDIAPIPE_AVAILABLE else None
        self.current_id = None
        self.playing = False
        self.mark_step = 0  # 空格键标记步骤: 0=开始帧, 1=IC, 2=MKF, 3=结束帧
        self.play_start_time = 0  # 播放开始时间
        self.play_start_frame = 0  # 播放开始帧
        self._loading = False  # 加载数据时阻止自动保存
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self._setup_ui()
        # 连接标注数据引用，使视频列表可以显示标注状态
        self.video_list.set_data_manager(self.data_manager)
        # 检查专家姓名
        self._check_expert()
        # 自动加载上次打开的文件夹
        self.video_list.auto_load_last_folder()

    def _check_expert(self):
        """检查并设置专家姓名"""
        if not self.data_manager.expert_id:
            self._request_expert_name()
        else:
            self._update_expert_display()

    def _request_expert_name(self):
        """请求用户输入专家姓名"""
        while True:
            name, ok = QInputDialog.getText(
                self, "专家信息",
                "请输入您的姓名（用于标识标注数据）:",
                text=""
            )
            if ok and name.strip():
                self.data_manager.set_expert(name.strip())
                self._update_expert_display()
                break
            elif not ok:
                # 用户点击取消，退出程序
                QMessageBox.warning(self, "警告", "必须输入专家姓名才能使用本软件")
                # 继续循环，强制输入

    def _update_expert_display(self):
        """更新专家姓名显示"""
        expert_name = self.data_manager.expert_id or "未设置"
        self.expert_label.setText(f"👤 专家: {expert_name}")

        # 设置提示词显示当前文件路径
        if self.data_manager.annotation_path:
            self.expert_label.setToolTip(f"标注文件: {self.data_manager.annotation_path}")
        else:
            self.expert_label.setToolTip("")

        # 强制刷新显示
        self.expert_label.update()
        self.expert_label.repaint()
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

    def _edit_expert_name(self):
        """修改专家姓名"""
        current_name = self.data_manager.expert_id or ""
        name, ok = QInputDialog.getText(
            self, "修改专家姓名",
            "请输入新的专家姓名:",
            text=current_name
        )
        if ok and name.strip():
            new_name = name.strip()
            if new_name != current_name:
                self.data_manager.set_expert(new_name)
                self._update_expert_display()
                self._update_stats()
                self.video_list.refresh_all_status()
                # 重新加载当前视频的标注数据（新专家的数据）
                self._load_current()
                self.statusBar().showMessage(f"专家姓名已修改为: {new_name}")

    def _setup_ui(self):
        self.setWindowTitle("LESS视频标注工具")

        # 获取屏幕大小，设置窗口为屏幕的90%
        screen = QDesktopWidget().availableGeometry()
        w, h = int(screen.width() * 0.92), int(screen.height() * 0.88)
        self.setGeometry((screen.width() - w) // 2, (screen.height() - h) // 2, w, h)
        self.setMinimumSize(1200, 700)

        self.setStyleSheet(MAIN_WINDOW_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setSpacing(0)
        main.setContentsMargins(10, 10, 10, 10)

        # 使用QSplitter实现可调整大小的布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #E0E0E0;
                border-radius: 3px;
            }
            QSplitter::handle:hover {
                background: #2196F3;
            }
        """)

        # 左侧：视频列表（可折叠树形结构）
        self.video_list = VideoListPanel(data_manager=self.data_manager)
        self.video_list.setMinimumWidth(200)
        self.video_list.video_selected.connect(self._on_video)
        splitter.addWidget(self.video_list)

        # 中间：视频区域
        video_panel = QWidget()
        video_panel.setStyleSheet("background: white; border-radius: 12px;")
        vl = QVBoxLayout(video_panel)
        vl.setSpacing(12)
        vl.setContentsMargins(15, 15, 15, 15)

        # 专家信息栏
        expert_bar = QWidget()
        expert_bar.setStyleSheet("background: #E8F5E9; border-radius: 8px; padding: 8px;")
        expert_layout = QHBoxLayout(expert_bar)
        expert_layout.setContentsMargins(12, 8, 12, 8)
        expert_layout.setSpacing(10)

        self.expert_label = QLabel("👤 专家: ")
        self.expert_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E7D32;")
        expert_layout.addWidget(self.expert_label)

        # 修改姓名按钮
        edit_expert_btn = QPushButton("修改")
        edit_expert_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50; color: white; border: none;
                padding: 4px 12px; border-radius: 4px; font-size: 12px;
            }
            QPushButton:hover { background: #388E3C; }
        """)
        edit_expert_btn.setToolTip("修改专家姓名")
        edit_expert_btn.clicked.connect(self._edit_expert_name)
        expert_layout.addWidget(edit_expert_btn)

        expert_layout.addStretch()

        # 统计信息
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #666; font-size: 12px;")
        expert_layout.addWidget(self.stats_label)

        vl.addWidget(expert_bar)

        # 双视频 - 占据更多空间
        videos = QHBoxLayout()
        videos.setSpacing(15)
        self.front = VideoPlayer("正面视频 (Front)")
        self.side = VideoPlayer("侧面视频 (Side)")
        if self.pose_detector:
            self.front.set_pose_detector(self.pose_detector)
            self.side.set_pose_detector(self.pose_detector)
        self.front.frame_changed.connect(self._on_frame)
        videos.addWidget(self.front, 1)
        videos.addWidget(self.side, 1)
        vl.addLayout(videos, 1)

        # 控制栏
        ctrl = QWidget()
        ctrl.setStyleSheet("background: #F5F5F5; border-radius: 10px; padding: 12px;")
        cl = QHBoxLayout(ctrl)
        cl.setSpacing(10)

        # 控制按钮: (文字, 函数, 提示词)
        ctrl_btns = [
            ("⏮", lambda: self._step(-5), "快退5帧"),
            ("◀", lambda: self._step(-1), "后退1帧 (←)"),
            ("▶", self._play, "播放/暂停 (P)"),
            ("▶", lambda: self._step(1), "前进1帧 (→)"),
            ("⏭", lambda: self._step(5), "快进5帧"),
        ]
        for txt, fn, tip in ctrl_btns:
            b = QPushButton(txt)
            b.setStyleSheet(BUTTON_STYLE_PRIMARY)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            if txt == "▶" and fn == self._play:
                self.play_btn = b
            cl.addWidget(b)

        cl.addSpacing(30)

        self.frame_lbl = QLabel("帧: 0 / 0")
        self.frame_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        cl.addWidget(self.frame_lbl)

        self.time_lbl = QLabel("时间: 0.000s")
        self.time_lbl.setStyleSheet("font-size: 18px; color: #666;")
        cl.addWidget(self.time_lbl)

        cl.addStretch()

        cl.addWidget(QLabel("速度:"))
        self.speed = QComboBox()
        self.speed.addItems(["0.25x", "0.5x", "1x", "1.5x", "2x"])
        self.speed.setCurrentIndex(2)
        self.speed.setStyleSheet("padding: 10px; font-size: 16px; min-width: 90px;")
        self.speed.setToolTip("调整视频播放速度")
        self.speed.currentIndexChanged.connect(self._on_speed_changed)
        cl.addWidget(self.speed)

        self.pose_btn = QPushButton("骨骼: 开" if self.pose_detector else "骨骼: 不可用")
        self.pose_btn.setCheckable(True)
        self.pose_btn.setChecked(True)
        self.pose_btn.setEnabled(self.pose_detector is not None)
        self.pose_btn.setToolTip("开启/关闭MediaPipe骨骼检测显示\n显示关节角度辅助评分")
        self.pose_btn.setStyleSheet(POSE_BUTTON_STYLE)
        self.pose_btn.clicked.connect(self._toggle_pose)
        cl.addWidget(self.pose_btn)

        vl.addWidget(ctrl)

        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet(SLIDER_STYLE)
        self.slider.valueChanged.connect(self._seek)
        vl.addWidget(self.slider)

        # 关键帧标注区域
        kf = QWidget()
        kf.setStyleSheet("background: #E3F2FD; border-radius: 10px; padding: 15px;")
        kfl = QHBoxLayout(kf)
        kfl.setSpacing(20)

        # 按时间顺序排列：开始帧 -> IC帧 -> MKF帧 -> 结束帧
        kfl.addWidget(self._create_frame_group("开始帧", "#9C27B0", "start_frame", self._mark_start, "S",
                                                "动作开始的帧\n起跳离地或动作开始的时刻"))
        kfl.addWidget(self._create_frame_group("IC帧", "#4CAF50", "ic", self._mark_ic, "I",
                                                "Initial Contact - 初始触地帧\n足部首次接触地面的时刻"))
        kfl.addWidget(self._create_frame_group("MKF帧", "#2196F3", "mkf", self._mark_mkf, "M",
                                                "Maximum Knee Flexion - 最大屈膝帧\n膝关节屈曲角度最大的时刻"))
        kfl.addWidget(self._create_frame_group("结束帧", "#FF5722", "end_frame", self._mark_end, "E",
                                                "动作结束的帧\n身体稳定或动作完成的时刻"))

        vl.addWidget(kf)

        splitter.addWidget(video_panel)

        # 右侧：评分 - 加宽以使每个评分项占一行
        right = QWidget()
        right.setMinimumWidth(420)
        rl = QVBoxLayout(right)
        rl.setSpacing(12)
        rl.setContentsMargins(0, 0, 0, 0)

        self.scoring = ScoringPanel()
        self.scoring.score_changed.connect(self._on_score)
        rl.addWidget(self.scoring, 1)

        # 总分 (最高19分: 15项×1分 + 2项×2分)
        score_box = QWidget()
        score_box.setStyleSheet("background: white; border-radius: 10px; padding: 15px;")
        score_box.setToolTip("LESS总分 (0-19分)\n分数越低表示落地动作质量越好\n0-4分: 优秀\n5-6分: 良好\n≥7分: 需要改进")
        sl = QHBoxLayout(score_box)
        sl.addWidget(QLabel("总分:"))
        self.total = QLabel("0")
        self.total.setStyleSheet("font-size: 36px; font-weight: bold; color: #4CAF50;")
        sl.addWidget(self.total)
        sl.addWidget(QLabel("/ 19"))
        sl.addStretch()
        rl.addWidget(score_box)

        # 导出
        exp = QHBoxLayout()
        open_file_btn = QPushButton("📂 打开标注文件")
        open_file_btn.setStyleSheet("""
            QPushButton { background: #607D8B; color: white; padding: 14px 24px;
            font-size: 16px; border: none; border-radius: 6px; }
            QPushButton:hover { background: #455A64; }
        """)
        open_file_btn.setToolTip("打开标注文件所在的文件夹")
        open_file_btn.clicked.connect(self._open_annotation_file)
        exp.addWidget(open_file_btn)

        rl.addLayout(exp)

        # 切换专家
        load_exp = QHBoxLayout()
        load_expert_btn = QPushButton("切换专家")
        load_expert_btn.setStyleSheet("""
            QPushButton { background: #FF5722; color: white; padding: 14px 24px;
            font-size: 16px; border: none; border-radius: 6px; }
            QPushButton:hover { background: #E64A19; }
        """)
        load_expert_btn.setToolTip("切换到其他专家查看或修改其标注")
        load_expert_btn.clicked.connect(self._load_expert_scores)
        load_exp.addWidget(load_expert_btn)
        rl.addLayout(load_exp)

        splitter.addWidget(right)

        # 设置splitter初始比例 (视频列表:视频区域:评分 = 1:4:2)
        splitter.setSizes([250, 800, 450])
        main.addWidget(splitter)

        # 快捷键
        QShortcut(QKeySequence(Qt.Key_Space), self, self._mark_next)  # 空格键顺序标记
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence("P"), self, self._play)  # P键播放/暂停
        QShortcut(QKeySequence("S"), self, self._mark_start)
        QShortcut(QKeySequence("I"), self, self._mark_ic)
        QShortcut(QKeySequence("M"), self, self._mark_mkf)
        QShortcut(QKeySequence("E"), self, self._mark_end)

        self.statusBar().showMessage("就绪 - 请选择视频文件夹")
        self.statusBar().setStyleSheet("font-size: 16px; padding: 6px;")

    def _create_frame_group(self, title, color, spinbox_attr, mark_fn, shortcut_hint, description):
        """创建关键帧标注组"""
        group = QWidget()
        group.setStyleSheet(f"background: white; border: 2px solid {color}; border-radius: 8px; padding: 10px;")
        group.setToolTip(description)
        gl = QVBoxLayout(group)
        gl.setSpacing(8)
        gl.setContentsMargins(12, 10, 12, 10)

        # 标题
        title_lbl = QLabel(f"{title} ({shortcut_hint})")
        title_lbl.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color}; background: transparent;")
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setToolTip(f"{description}\n快捷键: {shortcut_hint}")
        gl.addWidget(title_lbl)

        # SpinBox
        spin = QSpinBox()
        spin.setRange(0, 99999)
        spin.setStyleSheet(f"padding: 10px; font-size: 18px; min-width: 100px; border: 2px solid {color}; border-radius: 6px;")
        spin.setAlignment(Qt.AlignCenter)
        spin.setToolTip(f"当前{title}的帧号")
        setattr(self, spinbox_attr, spin)
        gl.addWidget(spin)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        mark_btn = QPushButton("标记")
        mark_btn.setStyleSheet(f"background: {color}; color: white; padding: 8px 16px; border: none; border-radius: 6px; font-size: 14px; font-weight: bold;")
        mark_btn.setToolTip(f"将当前帧标记为{title} ({shortcut_hint})")
        mark_btn.clicked.connect(mark_fn)
        btn_row.addWidget(mark_btn)

        goto_btn = QPushButton("跳转")
        goto_btn.setStyleSheet("background: #FF9800; color: white; padding: 8px 16px; border: none; border-radius: 6px; font-size: 14px; font-weight: bold;")
        goto_btn.setToolTip(f"跳转到{title}所在帧")
        goto_btn.clicked.connect(lambda: self._goto(spin.value()))
        btn_row.addWidget(goto_btn)

        gl.addLayout(btn_row)
        return group

    # ---- 视频控制 ----

    def _on_video(self, front, side):
        # 停止播放
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("▶")

        if self.current_id:
            self._auto_save()

        self.front.load_video(front)
        if side:
            self.side.load_video(side)
        self.slider.setMaximum(max(1, self.front.total_frames - 1))
        self.slider.setValue(0)
        self.current_id = self.video_list.get_current_id()
        self._load_current()

        # 显示当前视频信息
        info = self.video_list.get_current_info()
        if info:
            msg = f"已加载: 测试者 {info['subject']} | 动作 {info['motion']}"
            if not side:
                msg += " (仅正面视频)"
        else:
            msg = f"已加载: {self.current_id}"
        self.statusBar().showMessage(msg)

    def _on_frame(self, f, t):
        self.frame_lbl.setText(f"帧: {f} / {self.front.total_frames}")
        self.time_lbl.setText(f"时间: {t:.3f}s")
        self.slider.blockSignals(True)
        self.slider.setValue(f)
        self.slider.blockSignals(False)

    def _seek(self, v):
        self._goto(v)

    def _show_frame(self, f):
        """显示指定帧（不暂停播放）"""
        self.front.show_frame(f)
        if self.side.cap:
            self.side.show_frame(f)

    def _goto(self, f):
        """跳转到指定帧并暂停播放"""
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("▶")
            self.play_btn.setToolTip("播放视频")
        self._show_frame(f)

    def _step(self, d):
        """步进帧（播放时不暂停）"""
        self._show_frame(self.front.current_frame + d)

    def _play(self):
        if self.playing:
            self.timer.stop()
            self.play_btn.setText("▶")
            self.play_btn.setToolTip("播放视频 (从当前位置)")
        else:
            # 如果视频已结束，从头开始播放
            if self.front.current_frame >= self.front.total_frames - 1:
                self._show_frame(0)
            # 记录播放开始时间和帧
            self.play_start_time = time.time()
            self.play_start_frame = self.front.current_frame
            # 使用固定的刷新间隔（约30fps），实际帧由时间计算
            self.timer.start(33)
            self.play_btn.setText("⏸")
            self.play_btn.setToolTip("暂停播放")
        self.playing = not self.playing

    def _tick(self):
        # 基于时间计算应该显示的帧
        speeds = {0: 0.25, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}
        speed = speeds.get(self.speed.currentIndex(), 1.0)
        elapsed = time.time() - self.play_start_time
        target_frame = self.play_start_frame + int(elapsed * self.front.fps * speed)

        if target_frame >= self.front.total_frames - 1:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("▶")
            self.play_btn.setToolTip("重新播放视频")
            self._show_frame(self.front.total_frames - 1)
            self.statusBar().showMessage("视频播放完毕，点击播放按钮重新播放")
            return

        # 只有当目标帧变化时才更新显示
        if target_frame != self.front.current_frame:
            self._show_frame(target_frame)

    def _on_speed_changed(self, index):
        """播放速度改变时重置时间基准"""
        if self.playing:
            # 重置时间基准，从当前帧继续播放
            self.play_start_time = time.time()
            self.play_start_frame = self.front.current_frame

    def _toggle_pose(self):
        en = self.pose_btn.isChecked()
        self.front.show_pose = en
        self.side.show_pose = en
        self.pose_btn.setText(f"骨骼: {'开' if en else '关'}")
        self.front.show_frame(self.front.current_frame)
        if self.side.cap:
            self.side.show_frame(self.side.current_frame)

    # ---- 关键帧标记 ----

    def _mark_start(self):
        self.start_frame.setValue(self.front.current_frame)
        self.mark_step = 1
        self._auto_save()

    def _mark_end(self):
        self.end_frame.setValue(self.front.current_frame)
        self.mark_step = 0
        self._auto_save()

    def _mark_ic(self):
        self.ic.setValue(self.front.current_frame)
        self.mark_step = 2
        self._auto_save()

    def _mark_mkf(self):
        self.mkf.setValue(self.front.current_frame)
        self.mark_step = 3
        self._auto_save()

    def _mark_next(self):
        """空格键顺序标记: 开始帧 -> IC帧 -> MKF帧 -> 结束帧"""
        if self.mark_step == 0:
            self._mark_start()
        elif self.mark_step == 1:
            self._mark_ic()
        elif self.mark_step == 2:
            self._mark_mkf()
        elif self.mark_step == 3:
            self._mark_end()

    # ---- 评分 ----

    def _on_score(self, t):
        self.total.setText(str(t))
        # 自动保存
        self._auto_save()

    # ---- 数据管理 ----

    def _get_data(self):
        return DataManager.create_annotation_data(
            video_id=self.current_id,
            keyframes={
                'start': self.start_frame.value(),
                'ic': self.ic.value(),
                'mkf': self.mkf.value(),
                'end': self.end_frame.value()
            },
            scores=self.scoring.get_scores(),
            total_score=self.scoring.get_total_score()
        )

    def _load_current(self):
        """加载当前视频的标注数据"""
        self._loading = True  # 阻止自动保存

        # 先清空所有UI
        self.start_frame.setValue(0)
        self.ic.setValue(0)
        self.mkf.setValue(0)
        self.end_frame.setValue(0)
        self.mark_step = 0
        self.scoring.clear_scores()
        self.total.setText("0")

        # 更新统计信息
        self._update_stats()

        # 加载当前视频的标注
        if self.current_id:
            d = self.data_manager.get_annotation(self.current_id)
            if d:
                kf = d.get('keyframes', {})
                self.start_frame.setValue(kf.get('start', 0))
                self.ic.setValue(kf.get('ic', 0))
                self.mkf.setValue(kf.get('mkf', 0))
                self.end_frame.setValue(kf.get('end', 0))

                # 设置评分
                scores = d.get('scores', {})
                self.scoring.set_scores(scores)

                # 设置总分 - 从评分面板计算，确保与显示一致
                self.total.setText(str(self.scoring.get_total_score()))

                if kf.get('end', 0) > 0:
                    self.mark_step = 0

        self._loading = False  # 恢复自动保存

        # 强制刷新UI
        self.scoring.repaint()
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

    def _update_stats(self):
        """更新统计信息显示"""
        completed = self.data_manager.get_completed_count()
        draft = self.data_manager.get_draft_count()
        total = len(self.video_list.video_pairs) if hasattr(self.video_list, 'video_pairs') else 0
        self.stats_label.setText(f"已标注: {completed} | 待完善: {draft} | 总计: {total}")

    def _auto_save(self):
        """自动保存 - 自动判断完成状态"""
        if self._loading:  # 加载数据时不保存
            return
        if not self.current_id or not self.data_manager.expert_id:
            return

        data = self._get_data()
        # 自动判断是否完成
        data['completed'] = DataManager.is_annotation_complete(data)

        self.data_manager.set_annotation(self.current_id, data)
        self.data_manager.save_annotations()
        self.video_list.mark_done(self.current_id)
        self._update_stats()

        # 状态栏显示
        if data['completed']:
            self.statusBar().showMessage(f"已标注: {self.current_id}")
        else:
            self.statusBar().showMessage(f"待完善: {self.current_id}")

    def _open_annotation_file(self):
        """打开标注文件所在文件夹"""
        import os
        import subprocess

        file_path = self.data_manager.annotation_path
        if not file_path:
            QMessageBox.information(self, "提示", "请先选择视频文件夹并设置专家姓名")
            return

        if not file_path.exists():
            # 文件不存在，打开文件夹
            folder = file_path.parent
            if folder.exists():
                if os.name == 'nt':
                    os.startfile(str(folder))
                elif sys.platform == 'darwin':
                    subprocess.run(['open', str(folder)])
                else:
                    subprocess.run(['xdg-open', str(folder)])
            else:
                QMessageBox.information(self, "提示", "标注文件夹尚未创建")
            return

        # 打开文件所在文件夹并选中文件
        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', '/select,', str(file_path)])
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', '-R', str(file_path)])
        else:  # Linux
            subprocess.run(['xdg-open', str(file_path.parent)])

    def _load_expert_scores(self):
        """切换到其他专家"""
        # 获取可用的专家评分文件
        expert_files = self.data_manager.get_expert_score_files()

        # 过滤掉当前专家的文件
        current_expert = self.data_manager.expert_id
        if current_expert:
            expert_files = [(name, fname, path) for name, fname, path in expert_files
                           if name.strip() != current_expert.strip()]

        if not expert_files:
            QMessageBox.information(
                self, "提示",
                "没有找到其他专家的评分文件"
            )
            return

        # 让用户选择专家
        if len(expert_files) > 1:
            items = [name for name, fname, _ in expert_files]
            item, ok = QInputDialog.getItem(
                self, "切换专家",
                "请选择要切换到的专家:",
                items, 0, False
            )
            if not ok:
                return
            selected_expert = item
        else:
            selected_expert = expert_files[0][0]

        # 确认切换
        reply = QMessageBox.question(
            self, "切换专家",
            f"确定要切换到专家「{selected_expert}」吗？\n\n当前专家的数据已自动保存。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # 切换专家
        success, message = self.data_manager.switch_to_expert(selected_expert)

        if success:
            # 更新UI
            self._update_expert_display()
            self._update_stats()
            self.video_list.refresh_all_status()

            # 重新加载当前视频的标注数据
            self._load_current()

            # 显示切换成功信息
            QMessageBox.information(self, "切换成功", message)
            self.statusBar().showMessage(f"已切换到专家: {selected_expert}")
        else:
            QMessageBox.warning(self, "切换失败", message)

    def closeEvent(self, e):
        self._auto_save()
        self.data_manager.save_annotations()
        self.front.release()
        self.side.release()
        if self.pose_detector:
            self.pose_detector.release()
        e.accept()

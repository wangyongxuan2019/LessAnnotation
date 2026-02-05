"""主窗口模块"""

from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QSpinBox,
    QMessageBox, QFileDialog, QDesktopWidget
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
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self._setup_ui()
        # 连接标注数据引用，使视频列表可以显示标注状态
        self.video_list.set_annotations_ref(self.data_manager.annotations)
        # 自动加载上次打开的文件夹
        self.video_list.auto_load_last_folder()

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
        main.setSpacing(15)
        main.setContentsMargins(15, 15, 15, 15)

        # 左侧：视频列表 - 加宽
        self.video_list = VideoListPanel(data_manager=self.data_manager)
        self.video_list.setMinimumWidth(280)
        self.video_list.setMaximumWidth(320)
        self.video_list.video_selected.connect(self._on_video)
        main.addWidget(self.video_list)

        # 中间：视频区域
        video_panel = QWidget()
        video_panel.setStyleSheet("background: white; border-radius: 12px;")
        vl = QVBoxLayout(video_panel)
        vl.setSpacing(12)
        vl.setContentsMargins(15, 15, 15, 15)

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

        main.addWidget(video_panel, 1)

        # 右侧：评分 - 加宽以使每个评分项占一行
        right = QWidget()
        right.setMinimumWidth(480)
        right.setMaximumWidth(550)
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

        # 按钮
        btns = QHBoxLayout()
        save_btns = [
            ("💾 保存", self._save, "#2196F3", "保存当前标注 (Ctrl+S)"),
            ("保存并下一个 ▶", self._save_next, "#4CAF50", "保存当前标注并自动切换到下一个视频")
        ]
        for txt, fn, color, tip in save_btns:
            b = QPushButton(txt)
            b.setStyleSheet(f"""
                QPushButton {{ background: {color}; color: white; padding: 16px 30px;
                font-size: 18px; font-weight: bold; border: none; border-radius: 8px; }}
                QPushButton:hover {{ opacity: 0.9; }}
            """)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            btns.addWidget(b)
        rl.addLayout(btns)

        # 导出
        exp = QHBoxLayout()
        export_btns = [
            ("导出CSV", self._csv, "将所有标注数据导出为CSV表格文件"),
            ("导出JSON", self._json, "将所有标注数据导出为JSON文件")
        ]
        for txt, fn, tip in export_btns:
            b = QPushButton(txt)
            b.setStyleSheet("""
                QPushButton { background: #607D8B; color: white; padding: 14px 24px;
                font-size: 16px; border: none; border-radius: 6px; }
                QPushButton:hover { background: #455A64; }
            """)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            exp.addWidget(b)
        rl.addLayout(exp)

        main.addWidget(right)

        # 快捷键
        QShortcut(QKeySequence(Qt.Key_Space), self, self._mark_next)  # 空格键顺序标记
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence("P"), self, self._play)  # P键播放/暂停
        QShortcut(QKeySequence("S"), self, self._mark_start)
        QShortcut(QKeySequence("I"), self, self._mark_ic)
        QShortcut(QKeySequence("M"), self, self._mark_mkf)
        QShortcut(QKeySequence("E"), self, self._mark_end)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save)

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
        if self.current_id:
            self._auto_save()
        self.front.load_video(front)
        if side:
            self.side.load_video(side)
        self.slider.setMaximum(max(1, self.front.total_frames - 1))
        self.current_id = self.video_list.get_current_id()
        self._load_current()

        # 显示当前视频信息
        info = self.video_list.get_current_info()
        if info:
            msg = f"已加载: 测试者 {info['subject']} | 动作 {info['motion']} | 次数 {info['trial']}"
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
                self._goto(0)
            speeds = {0: 0.25, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}
            interval = int(1000 / (self.front.fps * speeds.get(self.speed.currentIndex(), 1)))
            self.timer.start(interval)
            self.play_btn.setText("⏸")
            self.play_btn.setToolTip("暂停播放")
        self.playing = not self.playing

    def _tick(self):
        if self.front.current_frame >= self.front.total_frames - 1:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("▶")
            self.play_btn.setToolTip("重新播放视频")
            self.statusBar().showMessage("视频播放完毕，点击播放按钮重新播放")
            return
        self._step(1)

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
        self.statusBar().showMessage(f"开始帧已标记: {self.front.current_frame} | 下一步: 按空格标记IC帧")
        self.mark_step = 1

    def _mark_end(self):
        self.end_frame.setValue(self.front.current_frame)
        self.statusBar().showMessage(f"结束帧已标记: {self.front.current_frame} | 所有关键帧已标记完成")
        self.mark_step = 0

    def _mark_ic(self):
        self.ic.setValue(self.front.current_frame)
        self.statusBar().showMessage(f"IC帧已标记: {self.front.current_frame} | 下一步: 按空格标记MKF帧")
        self.mark_step = 2

    def _mark_mkf(self):
        self.mkf.setValue(self.front.current_frame)
        self.statusBar().showMessage(f"MKF帧已标记: {self.front.current_frame} | 下一步: 按空格标记结束帧")
        self.mark_step = 3

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
        self.start_frame.setValue(0)
        self.ic.setValue(0)
        self.mkf.setValue(0)
        self.end_frame.setValue(0)
        self.mark_step = 0
        self.scoring.clear_scores()
        if self.current_id:
            d = self.data_manager.get_annotation(self.current_id)
            if d:
                kf = d.get('keyframes', {})
                self.start_frame.setValue(kf.get('start', 0))
                self.ic.setValue(kf.get('ic', 0))
                self.mkf.setValue(kf.get('mkf', 0))
                self.end_frame.setValue(kf.get('end', 0))
                self.scoring.set_scores(d.get('scores', {}))
                self.total.setText(str(d.get('total_score', 0)))
                if kf.get('end', 0) > 0:
                    self.mark_step = 0

    def _auto_save(self):
        if self.current_id:
            self.data_manager.set_annotation(self.current_id, self._get_data())

    def _save(self):
        if not self.current_id:
            QMessageBox.warning(self, "警告", "请先选择视频")
            return
        self.data_manager.set_annotation(self.current_id, self._get_data())
        self.data_manager.save_annotations()
        self.video_list.mark_done(self.current_id)
        self.statusBar().showMessage(f"已保存: {self.current_id}")

    def _save_next(self):
        self._save()
        self.video_list.next_video()

    def _csv(self):
        if not self.data_manager.annotations:
            QMessageBox.information(self, "提示", "没有数据")
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出", f"less_{datetime.now():%Y%m%d_%H%M%S}.csv", "CSV (*.csv)")
        if p:
            count = self.data_manager.export_csv(p)
            QMessageBox.information(self, "成功", f"已导出 {count} 条")

    def _json(self):
        if not self.data_manager.annotations:
            QMessageBox.information(self, "提示", "没有数据")
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出", f"less_{datetime.now():%Y%m%d_%H%M%S}.json", "JSON (*.json)")
        if p:
            count = self.data_manager.export_json(p)
            QMessageBox.information(self, "成功", f"已导出 {count} 条")

    def closeEvent(self, e):
        self._auto_save()
        self.data_manager.save_annotations()
        self.front.release()
        self.side.release()
        if self.pose_detector:
            self.pose_detector.release()
        e.accept()

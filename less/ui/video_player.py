"""视频播放器组件"""

import cv2
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class VideoPlayer(QWidget):
    """视频播放器"""
    frame_changed = pyqtSignal(int, float)

    def __init__(self, title="视频", parent=None):
        super().__init__(parent)
        self.title = title
        self.cap = None
        self.fps = 30
        self.total_frames = 0
        self.current_frame = 0
        self.pose_detector = None
        self.show_pose = True
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 标题
        title = QLabel(self.title)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 18px; font-weight: bold; color: #1565C0;
            padding: 10px; background: #E3F2FD; border-radius: 6px;
        """)
        layout.addWidget(title)

        # 视频显示 - 设置更大的最小尺寸
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background: #263238; border: 3px solid #455A64; border-radius: 8px;
            color: #90A4AE; font-size: 18px;
        """)
        self.video_label.setText("点击左侧选择文件夹加载视频")
        layout.addWidget(self.video_label, 1)

    def load_video(self, path: str) -> bool:
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            return False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.show_frame(0)
        return True

    def set_pose_detector(self, detector):
        self.pose_detector = detector

    def show_frame(self, frame_num: int):
        if not self.cap:
            return
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, raw_frame = self.cap.read()
        if ret:
            self.current_frame = frame_num

            # 关键：创建完全独立的帧副本（C连续内存）
            frame = np.array(raw_frame, dtype=np.uint8, copy=True, order='C')

            if self.pose_detector and self.show_pose:
                frame, _ = self.pose_detector.detect(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape

            # 缩放适应标签大小
            lw, lh = self.video_label.width() - 6, self.video_label.height() - 6
            if lw > 0 and lh > 0:
                scale = min(lw / w, lh / h)
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w > 0 and new_h > 0:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    h, w = new_h, new_w

            img = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(img))
            self.frame_changed.emit(frame_num, frame_num / self.fps)

    def release(self):
        if self.cap:
            self.cap.release()

"""
LESS视频标注工具 - Windows客户端
用于Landing Error Scoring System (LESS) 评分的双视角视频标注工具
版本: 2.5 (2026-02-04) - 完全隔离内存，修复MediaPipe绘制问题
"""

import sys
import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QSpinBox, QGroupBox,
    QRadioButton, QButtonGroup, QFileDialog, QTreeWidget, QTreeWidgetItem,
    QScrollArea, QMessageBox, QSizePolicy, QDesktopWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence, QColor
from PyQt5.QtWidgets import QShortcut, QProxyStyle, QStyle

# ============ MediaPipe 兼容性处理 ============
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_MODE = None  # 'legacy' or 'tasks'

# 尝试导入MediaPipe
try:
    import mediapipe as mp

    # 先尝试旧版API (mp.solutions)
    try:
        test_pose = mp.solutions.pose
        test_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_MODE = 'legacy'
        print("MediaPipe已加载 (Legacy API)")
    except AttributeError:
        pass

    # 如果旧版不行，尝试新版Tasks API
    if not MEDIAPIPE_AVAILABLE:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            MEDIAPIPE_AVAILABLE = True
            MEDIAPIPE_MODE = 'tasks'
            print("MediaPipe已加载 (Tasks API)")
        except ImportError:
            pass

except ImportError as e:
    print(f"MediaPipe导入失败: {e}")
    print("请运行: pip install mediapipe")

# 模型文件路径（仅Tasks API需要）
MODEL_PATH = Path(__file__).parent / "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"


def download_model():
    """下载MediaPipe模型文件"""
    if MODEL_PATH.exists():
        return True
    try:
        print(f"正在下载模型: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("模型下载完成")
        return True
    except Exception as e:
        print(f"模型下载失败: {e}")
        return False


class PoseDetector:
    """MediaPipe姿态检测器 - 自动适配新旧API"""

    LANDMARKS = {
        'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }

    CONNECTIONS = [
        (0, 11), (0, 12),  # 鼻子到肩膀
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32)
    ]

    def __init__(self):
        self.enabled = False
        self.pose = None
        self.detector = None
        self._init()

    def _init(self):
        if not MEDIAPIPE_AVAILABLE:
            print("MediaPipe不可用，骨骼检测已禁用")
            return

        try:
            if MEDIAPIPE_MODE == 'legacy':
                # 旧版API
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.enabled = True
                print("姿态检测器初始化成功 (Legacy)")

            elif MEDIAPIPE_MODE == 'tasks':
                # 新版Tasks API
                if not MODEL_PATH.exists():
                    if not download_model():
                        return

                base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
                options = mp_vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector = mp_vision.PoseLandmarker.create_from_options(options)
                self.enabled = True
                print("姿态检测器初始化成功 (Tasks)")

        except Exception as e:
            print(f"姿态检测器初始化失败: {e}")
            self.enabled = False

    def detect(self, input_frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """检测姿态并绘制 - 完全隔离，避免MediaPipe污染输出"""
        angles = {}

        # 立即创建完全独立的输出帧（C连续内存布局）
        output_frame = np.array(input_frame, dtype=np.uint8, copy=True, order='C')

        if not self.enabled:
            return output_frame, angles

        try:
            h, w = input_frame.shape[:2]
            landmarks_list = None

            # 为MediaPipe创建完全独立的输入图像
            mp_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            mp_input = np.array(mp_rgb, dtype=np.uint8, copy=True, order='C')

            if MEDIAPIPE_MODE == 'legacy' and self.pose:
                results = self.pose.process(mp_input)
                if results.pose_landmarks:
                    landmarks_list = [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]

            elif MEDIAPIPE_MODE == 'tasks' and self.detector:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_input)
                results = self.detector.detect(mp_image)
                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    landmarks_list = [(lm.x, lm.y, getattr(lm, 'visibility', 1.0))
                                     for lm in results.pose_landmarks[0]]

            # 只在output_frame上绘制（它与mp_input完全无关）
            if landmarks_list:
                points = [(int(x * w), int(y * h), v) for x, y, v in landmarks_list]

                # 只绘制这17个点：鼻子(0) + 上肢(11-16) + 下肢(23-32)
                DRAW_POINTS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}

                # 绘制骨架线（黄色）
                for i1, i2 in self.CONNECTIONS:
                    if i1 in DRAW_POINTS and i2 in DRAW_POINTS:
                        if i1 < len(points) and i2 < len(points):
                            p1, p2 = points[i1], points[i2]
                            if p1[2] > 0.5 and p2[2] > 0.5:
                                cv2.line(output_frame, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 255), 2)

                # 绘制关键点
                for idx in DRAW_POINTS:
                    if idx >= len(points):
                        continue
                    px, py, vis = points[idx]
                    if vis <= 0.5:
                        continue
                    if idx == 0:
                        col = (0, 0, 255)  # 红-头
                    elif idx >= 23:
                        col = (0, 255, 0)  # 绿-下肢
                    else:
                        col = (255, 0, 255)  # 紫-上肢
                    cv2.circle(output_frame, (px, py), 5, col, -1)

                angles = self._calc_angles(landmarks_list, w, h)
                self._draw_angles(output_frame, angles, landmarks_list, w, h)

            return output_frame, angles

        except Exception as e:
            print(f"检测错误: {e}")
            return output_frame, angles

    def _calc_angles(self, landmarks, w, h) -> Dict:
        angles = {}

        def pt(name):
            idx = self.LANDMARKS.get(name, 0)
            if idx < len(landmarks):
                return np.array([landmarks[idx][0] * w, landmarks[idx][1] * h])
            return np.array([0, 0])

        def visibility(name):
            idx = self.LANDMARKS.get(name, 0)
            if idx < len(landmarks):
                return landmarks[idx][2]
            return 0

        def angle(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

        def foot_rotation(heel, toe, hip, ankle):
            """计算足部旋转角度 (外旋为正，内旋为负)"""
            # 足部方向向量 (从脚跟到脚趾)
            foot_vec = toe - heel
            # 参考方向: 从髋到踝的垂直投影方向 (前后方向)
            forward_vec = ankle - hip
            # 计算足部与前进方向的夹角
            # 使用叉积判断方向 (外旋/内旋)
            cross = foot_vec[0] * forward_vec[1] - foot_vec[1] * forward_vec[0]
            dot = np.dot(foot_vec, forward_vec)
            angle_rad = np.arctan2(abs(cross), dot)
            angle_deg = np.degrees(angle_rad)
            # 根据叉积符号判断外旋/内旋
            return angle_deg if cross > 0 else -angle_deg

        try:
            # 膝屈曲角度 (180° - 实际弯曲角度，显示屈曲程度)
            left_knee_angle = angle(pt('left_hip'), pt('left_knee'), pt('left_ankle'))
            right_knee_angle = angle(pt('right_hip'), pt('right_knee'), pt('right_ankle'))
            angles['左膝屈曲'] = 180 - left_knee_angle
            angles['右膝屈曲'] = 180 - right_knee_angle

            # 髋屈曲角度
            angles['左髋屈曲'] = 180 - angle(pt('left_shoulder'), pt('left_hip'), pt('left_knee'))
            angles['右髋屈曲'] = 180 - angle(pt('right_shoulder'), pt('right_hip'), pt('right_knee'))

            # 躯干屈曲角度
            mid_s = (pt('left_shoulder') + pt('right_shoulder')) / 2
            mid_h = (pt('left_hip') + pt('right_hip')) / 2
            angles['躯干屈曲'] = angle(mid_s, mid_h, mid_h + np.array([0, -100]))

            # 足部旋转角度 (需要侧面视角或足部关键点可见)
            if visibility('left_heel') > 0.5 and visibility('left_foot_index') > 0.5:
                left_rot = foot_rotation(pt('left_heel'), pt('left_foot_index'),
                                         pt('left_hip'), pt('left_ankle'))
                if left_rot > 0:
                    angles['左足外旋'] = abs(left_rot)
                else:
                    angles['左足内旋'] = abs(left_rot)

            if visibility('right_heel') > 0.5 and visibility('right_foot_index') > 0.5:
                right_rot = foot_rotation(pt('right_heel'), pt('right_foot_index'),
                                          pt('right_hip'), pt('right_ankle'))
                if right_rot > 0:
                    angles['右足外旋'] = abs(right_rot)
                else:
                    angles['右足内旋'] = abs(right_rot)

        except:
            pass

        return angles

    def _draw_angles(self, frame, angles, landmarks, w, h):
        """使用PIL在关节位置绘制角度信息"""
        if not angles or not landmarks:
            return

        # 关节位置映射 (角度名称 -> 关键点索引)
        joint_positions = {
            '左膝屈曲': 25,   # left_knee
            '右膝屈曲': 26,   # right_knee
            '左髋屈曲': 23,   # left_hip
            '右髋屈曲': 24,   # right_hip
            '左足外旋': 27,   # left_ankle
            '左足内旋': 27,
            '右足外旋': 28,   # right_ankle
            '右足内旋': 28,
        }

        # 转换为PIL图像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 加载字体
        font = None
        font_small = None
        try:
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
            ]
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, 20)
                    font_small = ImageFont.truetype(fp, 16)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
                font_small = font
        except:
            font = ImageFont.load_default()
            font_small = font

        # 在关节位置绘制角度
        drawn_positions = set()  # 避免重叠
        for name, val in angles.items():
            joint_idx = joint_positions.get(name)
            if joint_idx and joint_idx < len(landmarks):
                lm = landmarks[joint_idx]
                x, y = int(lm[0] * w), int(lm[1] * h)

                # 避免同一位置重复绘制，稍微偏移
                key = (x // 30, y // 30)
                offset_y = 0
                while key in drawn_positions:
                    offset_y += 22
                    key = (x // 30, (y + offset_y) // 30)
                drawn_positions.add(key)

                # 简化显示：只显示数值
                text = f"{val:.0f}°"
                tx, ty = x + 12, y - 10 + offset_y

                # 绘制描边
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0))
                # 主文本 (黄色)
                draw.text((tx, ty), text, font=font, fill=(255, 255, 0))
            else:
                # 躯干屈曲等没有对应关节的，显示在左上角
                if '躯干' in name:
                    text = f"{name}: {val:.0f}°"
                    y_pos = 20
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        draw.text((12 + dx, y_pos + dy), text, font=font_small, fill=(0, 0, 0))
                    draw.text((12, y_pos), text, font=font_small, fill=(255, 255, 0))

        # 转换回OpenCV格式
        frame_rgb = np.array(pil_image)
        frame[:] = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def release(self):
        if self.pose:
            self.pose.close()
        if self.detector:
            self.detector.close()


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


class ScoringPanel(QScrollArea):
    """LESS评分面板"""
    score_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.button_groups = {}
        self._setup_ui()

    def _setup_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(460)
        self.setStyleSheet("""
            QScrollArea { background: #FAFAFA; border: none; }
            QGroupBox {
                font-size: 17px; font-weight: bold;
                border: 2px solid #BDBDBD; border-radius: 8px;
                margin-top: 14px; padding-top: 12px; background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px;
                padding: 2px 10px; color: white; border-radius: 4px;
            }
            QRadioButton { font-size: 15px; padding: 3px; spacing: 4px; }
            QRadioButton::indicator { width: 18px; height: 18px; }
            QLabel { font-size: 15px; }
        """)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # IC时刻评分 (Initial Contact) - A-J, N
        layout.addWidget(self._create_group("IC时刻评分 (Initial Contact)", "#2196F3", [
            ("itemA", "A. 膝屈曲角度", ["0-是(>30°)", "1-否"],
             "Knee flexion angle at initial contact >30 degrees\n初始触地时膝关节屈曲角度是否>30°\n0分: 是，屈曲>30°(良好)\n1分: 否，屈曲≤30°"),
            ("itemB", "B. 膝外翻", ["0-是(膝过中足)", "1-否"],
             "Knee valgus at initial contact, knees over mid-foot\n初始触地时膝关节是否在中足上方\n0分: 是，膝盖在中足上方(良好)\n1分: 否，膝盖内扣"),
            ("itemC", "C. 躯干屈曲", ["0-屈曲", "1-未屈曲"],
             "Trunk flexion angle at contact\n初始触地时躯干是否屈曲\n0分: 躯干屈曲(良好)\n1分: 躯干未屈曲"),
            ("itemD", "D. 躯干侧屈", ["0-垂直", "1-不垂直"],
             "Lateral trunk flexion at contact\n初始触地时躯干是否保持垂直\n0分: 躯干垂直(良好)\n1分: 躯干不垂直/侧屈"),
            ("itemE", "E. 踝跖屈", ["0-趾到跟", "1-否"],
             "Ankle plantar flexion at contact (toe to heel)\n初始触地时是否脚趾先着地再到脚跟\n0分: 是，趾-跟顺序(良好)\n1分: 否"),
            ("itemF", "F. 足外旋>30°", ["0-否", "1-是"],
             "Foot position at initial contact, toes >30° external rotation\n初始触地时足尖外旋是否>30°\n0分: 否，外旋≤30°(良好)\n1分: 是，外旋>30°"),
            ("itemG", "G. 足内旋>30°", ["0-否", "1-是"],
             "Foot position at initial contact, toes >30° internal rotation\n初始触地时足尖内旋是否>30°\n0分: 否，内旋≤30°(良好)\n1分: 是，内旋>30°"),
            ("itemH", "H. 站距<肩宽", ["0-否", "1-是"],
             "Stance width at initial contact < shoulder width\n初始触地时站立宽度是否小于肩宽\n0分: 否，站距≥肩宽(良好)\n1分: 是，站距<肩宽"),
            ("itemI", "I. 站距>肩宽", ["0-否", "1-是"],
             "Stance width at initial contact > shoulder width\n初始触地时站立宽度是否大于肩宽\n0分: 否，站距≤肩宽(良好)\n1分: 是，站距>肩宽"),
            ("itemJ", "J. 双足对称着地", ["0-是", "1-否"],
             "Initial foot contact symmetric\n初始触地时双足是否对称着地\n0分: 是，对称(良好)\n1分: 否，不对称"),
            ("itemN", "N. 髋屈曲(IC)", ["0-是", "1-否"],
             "Hip flexion angle at initial contact, hips flexed\n初始触地时髋关节是否屈曲\n0分: 是，髋屈曲(良好)\n1分: 否，髋未屈曲"),
        ]))

        # MKF时刻评分 (Maximum Knee Flexion) - K, L, M, O
        layout.addWidget(self._create_group("MKF时刻评分 (Max Knee Flexion)", "#FF9800", [
            ("itemK", "K. 膝屈曲位移>45°", ["0-是", "1-否"],
             "Knee flexion displacement >45 degrees\n膝关节屈曲位移是否>45°\n0分: 是，位移>45°(良好)\n1分: 否，位移≤45°"),
            ("itemL", "L. 膝外翻位移", ["0-否", "1-是"],
             "Knee valgus displacement, knee inside great toe\n膝关节是否内移至大脚趾内侧\n0分: 否，膝盖未内移(良好)\n1分: 是，膝盖在大脚趾内侧"),
            ("itemM", "M. 躯干屈曲(MKF)", ["0-是", "1-否"],
             "Trunk flexion at maximal knee angle\n最大屈膝时躯干屈曲是否比IC时更多\n0分: 是，屈曲更多(良好)\n1分: 否"),
            ("itemO", "O. 髋屈曲(MKF)", ["0-是", "1-否"],
             "Hip flexion at maximal knee angle\n最大屈膝时髋屈曲是否比IC时更多\n0分: 是，屈曲更多(良好)\n1分: 否"),
        ]))

        # 整体评分 - P, Q
        layout.addWidget(self._create_group("整体评分 (Overall)", "#4CAF50", [
            ("itemP", "P. 矢状面关节位移", ["0-柔软", "1-中等", "2-僵硬"],
             "Joint displacement, sagittal plane\n矢状面整体关节位移/缓冲质量\n0分: 柔软(Soft)，缓冲良好\n1分: 中等(Average)\n2分: 僵硬(Stiff)，缓冲差"),
            ("itemQ", "Q. 整体印象", ["0-优秀", "1-中等", "2-较差"],
             "Overall impression\n整体落地动作质量\n0分: 优秀(Excellent)\n1分: 中等(Average)\n2分: 较差(Poor)"),
        ]))

        layout.addStretch()
        self.setWidget(container)

    def _create_group(self, title, color, items):
        group = QGroupBox(title)
        group.setStyleSheet(group.styleSheet() + f"QGroupBox::title {{ background: {color}; }}")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)  # 增大行距

        for item_data in items:
            # 支持带提示词和不带提示词两种格式
            if len(item_data) == 4:
                item_id, label, options, tooltip = item_data
            else:
                item_id, label, options = item_data
                tooltip = ""

            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setSpacing(8)
            hl.setContentsMargins(8, 6, 8, 6)  # 增大上下边距

            # 标签 - 固定宽度，带提示词（仅在标签上显示）
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: 500; color: #333;")
            lbl.setMinimumWidth(120)
            if tooltip:
                lbl.setToolTip(tooltip)
                lbl.setCursor(Qt.WhatsThisCursor)  # 显示问号光标提示用户悬停查看
            hl.addWidget(lbl)

            # 选项按钮 - 不设置提示词
            bg = QButtonGroup(w)
            self.button_groups[item_id] = bg

            for i, opt in enumerate(options):
                rb = QRadioButton(opt)
                # 选项上不显示提示词
                bg.addButton(rb, i)
                hl.addWidget(rb)

            bg.buttonClicked.connect(lambda: self.score_changed.emit(self.get_total_score()))
            hl.addStretch()
            layout.addWidget(w)

        return group

    def get_total_score(self):
        return sum(g.id(g.checkedButton()) for g in self.button_groups.values() if g.checkedButton())

    def get_scores(self):
        return {k: (g.id(g.checkedButton()) if g.checkedButton() else None) for k, g in self.button_groups.items()}

    def set_scores(self, scores):
        for k, v in scores.items():
            if k in self.button_groups and v is not None:
                btn = self.button_groups[k].button(v)
                if btn:
                    btn.setChecked(True)

    def clear_scores(self):
        for g in self.button_groups.values():
            g.setExclusive(False)
            for b in g.buttons():
                b.setChecked(False)
            g.setExclusive(True)

    def is_complete(self):
        return all(g.checkedButton() for g in self.button_groups.values())


class VideoListPanel(QWidget):
    """视频列表面板"""
    video_selected = pyqtSignal(str, str)
    CONFIG_FILE = Path(__file__).parent / "less_config.json"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_pairs = []
        self.current_index = -1
        self.annotations_ref = {}  # 引用主窗口的标注数据
        self.last_folder = self._load_last_folder()
        self._setup_ui()

    def set_annotations_ref(self, annotations):
        """设置标注数据引用"""
        self.annotations_ref = annotations

    def _load_last_folder(self):
        """加载上次打开的文件夹路径"""
        try:
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    folder = config.get('last_folder', '')
                    if folder and Path(folder).exists():
                        return folder
        except:
            pass
        return ""

    def _save_last_folder(self, folder):
        """保存文件夹路径"""
        try:
            config = {}
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            config['last_folder'] = folder
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            self.last_folder = folder
        except:
            pass

    def auto_load_last_folder(self):
        """自动加载上次的文件夹"""
        if self.last_folder and Path(self.last_folder).exists():
            self._load(self.last_folder)

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

    def select_folder(self):
        dialog = QFileDialog(self, "选择视频文件夹")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        # 设置初始目录为上次打开的文件夹
        if self.last_folder and Path(self.last_folder).exists():
            dialog.setDirectory(self.last_folder)
        # 设置更大的对话框尺寸
        dialog.resize(900, 600)
        if dialog.exec_():
            folders = dialog.selectedFiles()
            if folders:
                self._save_last_folder(folders[0])
                self._load(folders[0])

    def _load(self, folder):
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

    def select_video(self, idx):
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

    def get_current_id(self):
        # video_pairs: (front, side, vid, subject, motion, trial)
        return self.video_pairs[self.current_index][2] if 0 <= self.current_index < len(self.video_pairs) else None

    def get_current_info(self):
        """获取当前视频的详细信息"""
        if 0 <= self.current_index < len(self.video_pairs):
            front, side, vid, subject, motion, trial = self.video_pairs[self.current_index]
            return {'subject': subject, 'motion': motion, 'trial': trial, 'vid': vid}
        return None

    def mark_done(self, vid):
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


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.pose_detector = PoseDetector() if MEDIAPIPE_AVAILABLE else None
        self.annotations = {}
        self.current_id = None
        self.playing = False
        self.mark_step = 0  # 空格键标记步骤: 0=开始帧, 1=IC, 2=MKF, 3=结束帧
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self._setup_ui()
        self._load_data()
        # 连接标注数据引用，使视频列表可以显示标注状态
        self.video_list.set_annotations_ref(self.annotations)
        # 自动加载上次打开的文件夹
        self.video_list.auto_load_last_folder()

    def _setup_ui(self):
        self.setWindowTitle("LESS视频标注工具")

        # 获取屏幕大小，设置窗口为屏幕的90%
        screen = QDesktopWidget().availableGeometry()
        w, h = int(screen.width() * 0.92), int(screen.height() * 0.88)
        self.setGeometry((screen.width() - w) // 2, (screen.height() - h) // 2, w, h)
        self.setMinimumSize(1200, 700)

        self.setStyleSheet("""
            QMainWindow { background: #ECEFF1; }
            QWidget { font-family: "Microsoft YaHei", "Segoe UI"; }
            QLabel { font-size: 16px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setSpacing(15)
        main.setContentsMargins(15, 15, 15, 15)

        # 左侧：视频列表 - 加宽
        self.video_list = VideoListPanel()
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

        btn_style = """
            QPushButton {
                background: #2196F3; color: white; border: none;
                padding: 14px 26px; font-size: 20px; border-radius: 8px; min-width: 65px;
            }
            QPushButton:hover { background: #1976D2; }
        """
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
            b.setStyleSheet(btn_style)
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
        self.pose_btn.setStyleSheet("""
            QPushButton { background: #4CAF50; color: white; border: none; padding: 12px 20px; border-radius: 6px; font-size: 16px; }
            QPushButton:checked { background: #4CAF50; }
            QPushButton:!checked { background: #9E9E9E; }
            QPushButton:disabled { background: #BDBDBD; }
        """)
        self.pose_btn.clicked.connect(self._toggle_pose)
        cl.addWidget(self.pose_btn)

        vl.addWidget(ctrl)

        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 10px; background: #E0E0E0; border-radius: 5px; }
            QSlider::handle:horizontal { width: 22px; height: 22px; background: #2196F3; border-radius: 11px; margin: -6px 0; }
            QSlider::sub-page:horizontal { background: #2196F3; border-radius: 5px; }
        """)
        self.slider.valueChanged.connect(self._seek)
        vl.addWidget(self.slider)

        # 关键帧标注区域
        kf = QWidget()
        kf.setStyleSheet("background: #E3F2FD; border-radius: 10px; padding: 15px;")
        kfl = QHBoxLayout(kf)
        kfl.setSpacing(20)

        # 创建四个关键帧标注组
        def create_frame_group(title, color, spinbox_attr, mark_fn, shortcut_hint, description):
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

        # 按时间顺序排列：开始帧 -> IC帧 -> MKF帧 -> 结束帧
        kfl.addWidget(create_frame_group("开始帧", "#9C27B0", "start_frame", self._mark_start, "S",
                                         "动作开始的帧\n起跳离地或动作开始的时刻"))
        kfl.addWidget(create_frame_group("IC帧", "#4CAF50", "ic", self._mark_ic, "I",
                                         "Initial Contact - 初始触地帧\n足部首次接触地面的时刻"))
        kfl.addWidget(create_frame_group("MKF帧", "#2196F3", "mkf", self._mark_mkf, "M",
                                         "Maximum Knee Flexion - 最大屈膝帧\n膝关节屈曲角度最大的时刻"))
        kfl.addWidget(create_frame_group("结束帧", "#FF5722", "end_frame", self._mark_end, "E",
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
        # 暂停播放
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText("▶")
            self.play_btn.setToolTip("播放视频")
        # 跳转到指定帧
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

    def _mark_start(self):
        self.start_frame.setValue(self.front.current_frame)
        self.statusBar().showMessage(f"开始帧已标记: {self.front.current_frame} | 下一步: 按空格标记IC帧")
        self.mark_step = 1  # 下一步标记IC帧

    def _mark_end(self):
        self.end_frame.setValue(self.front.current_frame)
        self.statusBar().showMessage(f"结束帧已标记: {self.front.current_frame} | 所有关键帧已标记完成")
        self.mark_step = 0  # 重置，下一次从开始帧开始

    def _mark_ic(self):
        self.ic.setValue(self.front.current_frame)
        self.statusBar().showMessage(f"IC帧已标记: {self.front.current_frame} | 下一步: 按空格标记MKF帧")
        self.mark_step = 2  # 下一步标记MKF帧

    def _mark_mkf(self):
        self.mkf.setValue(self.front.current_frame)
        self.statusBar().showMessage(f"MKF帧已标记: {self.front.current_frame} | 下一步: 按空格标记结束帧")
        self.mark_step = 3  # 下一步标记结束帧

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

    def _on_score(self, t):
        self.total.setText(str(t))

    def _get_data(self):
        return {
            'video_id': self.current_id,
            'keyframes': {
                'start': self.start_frame.value(),
                'ic': self.ic.value(),
                'mkf': self.mkf.value(),
                'end': self.end_frame.value()
            },
            'scores': self.scoring.get_scores(),
            'total_score': self.scoring.get_total_score(),
            'metadata': {'timestamp': datetime.now().isoformat()}
        }

    def _load_current(self):
        self.start_frame.setValue(0)
        self.ic.setValue(0)
        self.mkf.setValue(0)
        self.end_frame.setValue(0)
        self.mark_step = 0  # 重置标记步骤
        self.scoring.clear_scores()
        if self.current_id and self.current_id in self.annotations:
            d = self.annotations[self.current_id]
            kf = d.get('keyframes', {})
            self.start_frame.setValue(kf.get('start', 0))
            self.ic.setValue(kf.get('ic', 0))
            self.mkf.setValue(kf.get('mkf', 0))
            self.end_frame.setValue(kf.get('end', 0))
            self.scoring.set_scores(d.get('scores', {}))
            self.total.setText(str(d.get('total_score', 0)))
            # 如果已有标注，标记步骤设为完成
            if kf.get('end', 0) > 0:
                self.mark_step = 0

    def _auto_save(self):
        if self.current_id:
            self.annotations[self.current_id] = self._get_data()

    def _save(self):
        if not self.current_id:
            QMessageBox.warning(self, "警告", "请先选择视频")
            return
        self.annotations[self.current_id] = self._get_data()
        self._save_file()
        self.video_list.mark_done(self.current_id)
        self.statusBar().showMessage(f"已保存: {self.current_id}")

    def _save_next(self):
        self._save()
        self.video_list.next_video()

    def _load_data(self):
        p = Path("less_annotations.json")
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    self.annotations = {a['video_id']: a for a in json.load(f).get('annotations', [])}
            except:
                pass

    def _save_file(self):
        with open("less_annotations.json", 'w', encoding='utf-8') as f:
            json.dump({'annotations': list(self.annotations.values())}, f, ensure_ascii=False, indent=2)

    def _csv(self):
        if not self.annotations:
            QMessageBox.information(self, "提示", "没有数据")
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出", f"less_{datetime.now():%Y%m%d_%H%M%S}.csv", "CSV (*.csv)")
        if p:
            # LESS 17项: A-Q
            item_ids = ['itemA', 'itemB', 'itemC', 'itemD', 'itemE', 'itemF', 'itemG',
                       'itemH', 'itemI', 'itemJ', 'itemK', 'itemL', 'itemM', 'itemN',
                       'itemO', 'itemP', 'itemQ']
            with open(p, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['video_id', 'start_frame', 'ic_frame', 'mkf_frame', 'end_frame'] +
                          [f'item_{c}' for c in 'ABCDEFGHIJKLMNOPQ'] + ['total_score', 'timestamp'])
                for d in self.annotations.values():
                    kf = d.get('keyframes', {})
                    row = [d['video_id'], kf.get('start', ''), kf.get('ic', ''), kf.get('mkf', ''), kf.get('end', '')]
                    row += [d.get('scores', {}).get(item_id, '') for item_id in item_ids]
                    row += [d.get('total_score', ''), d.get('metadata', {}).get('timestamp', '')]
                    w.writerow(row)
            QMessageBox.information(self, "成功", f"已导出 {len(self.annotations)} 条")

    def _json(self):
        if not self.annotations:
            QMessageBox.information(self, "提示", "没有数据")
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出", f"less_{datetime.now():%Y%m%d_%H%M%S}.json", "JSON (*.json)")
        if p:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump({'annotations': list(self.annotations.values())}, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", f"已导出 {len(self.annotations)} 条")

    def closeEvent(self, e):
        self._auto_save()
        self._save_file()
        self.front.release()
        self.side.release()
        if self.pose_detector:
            self.pose_detector.release()
        e.accept()


class TooltipDelayStyle(QProxyStyle):
    """自定义样式：设置提示词延迟时间"""
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_ToolTip_WakeUpDelay:
            return 1000  # 1秒延迟
        return super().styleHint(hint, option, widget, returnData)


def main():
    print("=" * 50)
    print("LESS视频标注工具 v2.5")
    print("修复: 完全隔离内存布局，防止MediaPipe污染")
    print("=" * 50)

    app = QApplication(sys.argv)

    # 使用自定义样式设置提示词延迟
    app.setStyle(TooltipDelayStyle('Fusion'))

    # 更大的默认字体
    font = QFont("Microsoft YaHei", 14)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

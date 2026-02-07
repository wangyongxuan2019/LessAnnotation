"""姿态检测器模块 - 使用 MediaPipe"""

from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    MEDIAPIPE_AVAILABLE, MEDIAPIPE_MODE, MODEL_PATH,
    mp, mp_python, mp_vision, download_model
)


class PoseDetector:
    """姿态检测器 - MediaPipe"""

    LANDMARKS = {
        'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }

    CONNECTIONS = [
        (0, 11), (0, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32)
    ]

    def __init__(self):
        """初始化姿态检测器"""
        self.enabled = False
        self.pose = None
        self.detector = None
        self._init()

    def _init(self):
        """初始化检测器"""
        if not MEDIAPIPE_AVAILABLE:
            print("MediaPipe 不可用!")
            return

        try:
            if MEDIAPIPE_MODE == 'legacy':
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.enabled = True
                print("姿态检测器初始化成功 (MediaPipe Legacy)")

            elif MEDIAPIPE_MODE == 'tasks':
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
                print("姿态检测器初始化成功 (MediaPipe Tasks)")

        except Exception as e:
            print(f"姿态检测器初始化失败: {e}")
            self.enabled = False

    def _detect(self, frame: np.ndarray) -> Optional[List]:
        """检测姿态"""
        try:
            mp_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_input = np.array(mp_rgb, dtype=np.uint8, copy=True, order='C')

            if MEDIAPIPE_MODE == 'legacy' and self.pose:
                results = self.pose.process(mp_input)
                if results.pose_landmarks:
                    return [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]

            elif MEDIAPIPE_MODE == 'tasks' and self.detector:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_input)
                results = self.detector.detect(mp_image)
                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    return [(lm.x, lm.y, getattr(lm, 'visibility', 1.0))
                            for lm in results.pose_landmarks[0]]
            return None

        except Exception as e:
            print(f"检测错误: {e}")
            return None

    def detect(self, input_frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """检测姿态并绘制骨骼"""
        angles = {}
        output_frame = np.array(input_frame, dtype=np.uint8, copy=True, order='C')

        if not self.enabled:
            return output_frame, angles

        try:
            h, w = input_frame.shape[:2]
            landmarks_list = self._detect(input_frame)

            if landmarks_list:
                self._draw_skeleton(output_frame, landmarks_list, w, h)
                angles = self._calc_angles(landmarks_list, w, h)
                self._draw_angles(output_frame, angles, landmarks_list, w, h)

            return output_frame, angles

        except Exception as e:
            print(f"检测错误: {e}")
            return output_frame, angles

    def _draw_skeleton(self, frame: np.ndarray, landmarks: List, w: int, h: int):
        """绘制骨骼"""
        points = [(int(x * w), int(y * h), v) for x, y, v in landmarks]
        DRAW_POINTS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}

        for i1, i2 in self.CONNECTIONS:
            if i1 in DRAW_POINTS and i2 in DRAW_POINTS:
                if i1 < len(points) and i2 < len(points):
                    p1, p2 = points[i1], points[i2]
                    if p1[2] > 0.5 and p2[2] > 0.5:
                        cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 255), 2)

        for idx in DRAW_POINTS:
            if idx >= len(points):
                continue
            px, py, vis = points[idx]
            if vis <= 0.5:
                continue
            if idx == 0:
                col = (0, 0, 255)
            elif idx >= 23:
                col = (0, 255, 0)
            else:
                col = (255, 0, 255)
            cv2.circle(frame, (px, py), 5, col, -1)

    def _calc_angles(self, landmarks, w, h) -> Dict:
        """计算关节角度"""
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
            foot_vec = toe - heel
            forward_vec = ankle - hip
            cross = foot_vec[0] * forward_vec[1] - foot_vec[1] * forward_vec[0]
            dot = np.dot(foot_vec, forward_vec)
            angle_rad = np.arctan2(abs(cross), dot)
            angle_deg = np.degrees(angle_rad)
            return angle_deg if cross > 0 else -angle_deg

        try:
            left_knee_angle = angle(pt('left_hip'), pt('left_knee'), pt('left_ankle'))
            right_knee_angle = angle(pt('right_hip'), pt('right_knee'), pt('right_ankle'))
            angles['左膝屈曲'] = 180 - left_knee_angle
            angles['右膝屈曲'] = 180 - right_knee_angle

            angles['左髋屈曲'] = 180 - angle(pt('left_shoulder'), pt('left_hip'), pt('left_knee'))
            angles['右髋屈曲'] = 180 - angle(pt('right_shoulder'), pt('right_hip'), pt('right_knee'))

            mid_s = (pt('left_shoulder') + pt('right_shoulder')) / 2
            mid_h = (pt('left_hip') + pt('right_hip')) / 2
            angles['躯干屈曲'] = angle(mid_s, mid_h, mid_h + np.array([0, -100]))

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

        joint_positions = {
            '左膝屈曲': 25, '右膝屈曲': 26,
            '左髋屈曲': 23, '右髋屈曲': 24,
            '左足外旋': 27, '左足内旋': 27,
            '右足外旋': 28, '右足内旋': 28,
        }

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

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

        drawn_positions = set()
        for name, val in angles.items():
            joint_idx = joint_positions.get(name)
            if joint_idx and joint_idx < len(landmarks):
                lm = landmarks[joint_idx]
                x, y = int(lm[0] * w), int(lm[1] * h)

                key = (x // 30, y // 30)
                offset_y = 0
                while key in drawn_positions:
                    offset_y += 22
                    key = (x // 30, (y + offset_y) // 30)
                drawn_positions.add(key)

                text = f"{val:.0f}°"
                tx, ty = x + 12, y - 10 + offset_y

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0))
                draw.text((tx, ty), text, font=font, fill=(255, 255, 0))
            else:
                if '躯干' in name:
                    text = f"{name}: {val:.0f}°"
                    y_pos = 20
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        draw.text((12 + dx, y_pos + dy), text, font=font_small, fill=(0, 0, 0))
                    draw.text((12, y_pos), text, font=font_small, fill=(255, 255, 0))

        frame_rgb = np.array(pil_image)
        frame[:] = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def release(self):
        """释放资源"""
        if self.pose:
            self.pose.close()
        if self.detector:
            self.detector.close()

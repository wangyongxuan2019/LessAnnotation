"""姿态检测配置和兼容性处理"""

import urllib.request
from pathlib import Path

# ============ MediaPipe 配置 ============
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_MODE = None  # 'legacy' or 'tasks'

mp = None
mp_python = None
mp_vision = None

try:
    import mediapipe as _mp
    mp = _mp

    # 先尝试旧版API (mp.solutions)
    try:
        _test_pose = mp.solutions.pose
        _test_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
        MEDIAPIPE_MODE = 'legacy'
        print("MediaPipe 已加载 (Legacy API)")
    except AttributeError:
        pass

    # 如果旧版不行，尝试新版Tasks API
    if not MEDIAPIPE_AVAILABLE:
        try:
            from mediapipe.tasks import python as _mp_python
            from mediapipe.tasks.python import vision as _mp_vision
            mp_python = _mp_python
            mp_vision = _mp_vision
            MEDIAPIPE_AVAILABLE = True
            MEDIAPIPE_MODE = 'tasks'
            print("MediaPipe 已加载 (Tasks API)")
        except ImportError:
            pass

except ImportError:
    print("MediaPipe 未安装")

if not MEDIAPIPE_AVAILABLE:
    print(">>> 警告: MediaPipe 不可用!")

# ============ 模型配置 ============
# MediaPipe 模型（仅Tasks API需要）
MODEL_PATH = Path(__file__).parent.parent / "pose_landmarker_heavy.task"
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

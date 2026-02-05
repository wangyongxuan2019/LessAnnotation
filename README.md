# LESS Annotation Tool

LESS (Landing Error Scoring System) 视频标注工具 - 用于跳跃落地动作评分的双视角视频标注系统。

## 功能特点

- **双视角同步播放**: 支持正面和侧面视频同步播放
- **MediaPipe姿态检测**: 实时检测33个关节点，计算关节角度辅助评分
- **逐帧控制**: 精确的帧级控制，支持键盘快捷键
- **关键帧标注**: 开始帧、IC帧（初始触地）、MKF帧（最大屈膝）、结束帧
- **LESS 17项评分**: 完整的评分面板，一页内完成所有操作
- **数据导出**: 支持CSV和JSON格式导出

## 安装

### 环境要求

- Python 3.8+
- Windows 10/11

### 安装依赖

```bash
pip install -r requirements.txt
```

首次运行时会自动下载MediaPipe模型文件(约4MB)。

## 使用方法

### 启动程序

```bash
python main.py
```

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| 空格 | 顺序标记关键帧 |
| P | 播放/暂停 |
| ←/→ | 后退/前进1帧 |
| S | 标记开始帧 |
| I | 标记IC帧 |
| M | 标记MKF帧 |
| E | 标记结束帧 |
| Ctrl+S | 保存 |

## LESS 评分标准

LESS 包含 17 个评分项目，总分范围 0-19 分

**评分解读**:
- 0-4分: 优秀
- 5-6分: 良好
- 7分以上: 需要改进

## 技术栈

- **GUI**: PyQt5
- **视频处理**: OpenCV
- **姿态检测**: MediaPipe Pose Landmarker

## 许可证

MIT License

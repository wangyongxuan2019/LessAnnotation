"""UI样式模块"""

from PyQt5.QtWidgets import QProxyStyle, QStyle


class TooltipDelayStyle(QProxyStyle):
    """自定义样式：设置提示词延迟时间"""
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_ToolTip_WakeUpDelay:
            return 1000  # 1秒延迟
        return super().styleHint(hint, option, widget, returnData)


# 按钮样式常量
BUTTON_STYLE_PRIMARY = """
    QPushButton {
        background: #2196F3; color: white; border: none;
        padding: 14px 26px; font-size: 20px; border-radius: 8px; min-width: 65px;
    }
    QPushButton:hover { background: #1976D2; }
"""

BUTTON_STYLE_SUCCESS = """
    QPushButton {
        background: #4CAF50; color: white; border: none;
        padding: 16px 30px; font-size: 18px; font-weight: bold; border-radius: 8px;
    }
    QPushButton:hover { opacity: 0.9; }
"""

BUTTON_STYLE_SECONDARY = """
    QPushButton {
        background: #607D8B; color: white; border: none;
        padding: 14px 24px; font-size: 16px; border-radius: 6px;
    }
    QPushButton:hover { background: #455A64; }
"""

POSE_BUTTON_STYLE = """
    QPushButton { background: #4CAF50; color: white; border: none; padding: 12px 20px; border-radius: 6px; font-size: 16px; }
    QPushButton:checked { background: #4CAF50; }
    QPushButton:!checked { background: #9E9E9E; }
    QPushButton:disabled { background: #BDBDBD; }
"""

SLIDER_STYLE = """
    QSlider::groove:horizontal { height: 10px; background: #E0E0E0; border-radius: 5px; }
    QSlider::handle:horizontal { width: 22px; height: 22px; background: #2196F3; border-radius: 11px; margin: -6px 0; }
    QSlider::sub-page:horizontal { background: #2196F3; border-radius: 5px; }
"""

MAIN_WINDOW_STYLE = """
    QMainWindow { background: #ECEFF1; }
    QWidget { font-family: "Microsoft YaHei", "Segoe UI"; }
    QLabel { font-size: 16px; }
"""

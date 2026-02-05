"""
LESS视频标注工具 - Windows客户端
用于Landing Error Scoring System (LESS) 评分的双视角视频标注工具
版本: 2.5 (2026-02-04) - 完全隔离内存，修复MediaPipe绘制问题
"""

import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from less import __version__
from less.ui.main_window import MainWindow
from less.ui.styles import TooltipDelayStyle


def main():
    print("=" * 50)
    print(f"LESS视频标注工具 v{__version__}")
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

"""LESS评分面板组件"""

from PyQt5.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal


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

"""数据管理模块 - 标注数据的加载、保存和导出"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class DataManager:
    """标注数据管理器"""

    ANNOTATION_FILE = "less_annotations.json"
    CONFIG_FILE = "less_config.json"

    # LESS 17项评分ID
    ITEM_IDS = [
        'itemA', 'itemB', 'itemC', 'itemD', 'itemE', 'itemF', 'itemG',
        'itemH', 'itemI', 'itemJ', 'itemK', 'itemL', 'itemM', 'itemN',
        'itemO', 'itemP', 'itemQ'
    ]

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.annotations: Dict = {}
        self.load_annotations()

    @property
    def annotation_path(self) -> Path:
        return self.base_path / self.ANNOTATION_FILE

    @property
    def config_path(self) -> Path:
        return self.base_path / self.CONFIG_FILE

    def load_annotations(self):
        """加载标注数据"""
        if self.annotation_path.exists():
            try:
                with open(self.annotation_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.annotations = {a['video_id']: a for a in data.get('annotations', [])}
            except Exception as e:
                print(f"加载标注数据失败: {e}")

    def save_annotations(self):
        """保存标注数据"""
        try:
            with open(self.annotation_path, 'w', encoding='utf-8') as f:
                json.dump({'annotations': list(self.annotations.values())}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存标注数据失败: {e}")

    def get_annotation(self, video_id: str) -> Optional[Dict]:
        """获取指定视频的标注"""
        return self.annotations.get(video_id)

    def set_annotation(self, video_id: str, data: Dict):
        """设置指定视频的标注"""
        self.annotations[video_id] = data

    def has_annotation(self, video_id: str) -> bool:
        """检查是否有指定视频的标注"""
        return video_id in self.annotations

    def export_csv(self, file_path: str) -> int:
        """导出为CSV格式，返回导出条数"""
        if not self.annotations:
            return 0

        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # 表头
            writer.writerow(
                ['video_id', 'start_frame', 'ic_frame', 'mkf_frame', 'end_frame'] +
                [f'item_{c}' for c in 'ABCDEFGHIJKLMNOPQ'] +
                ['total_score', 'timestamp']
            )
            # 数据行
            for d in self.annotations.values():
                kf = d.get('keyframes', {})
                row = [
                    d['video_id'],
                    kf.get('start', ''),
                    kf.get('ic', ''),
                    kf.get('mkf', ''),
                    kf.get('end', '')
                ]
                row += [d.get('scores', {}).get(item_id, '') for item_id in self.ITEM_IDS]
                row += [d.get('total_score', ''), d.get('metadata', {}).get('timestamp', '')]
                writer.writerow(row)

        return len(self.annotations)

    def export_json(self, file_path: str) -> int:
        """导出为JSON格式，返回导出条数"""
        if not self.annotations:
            return 0

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'annotations': list(self.annotations.values())}, f, ensure_ascii=False, indent=2)

        return len(self.annotations)

    def load_config(self) -> Dict:
        """加载配置"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def save_config(self, config: Dict):
        """保存配置"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def get_last_folder(self) -> str:
        """获取上次打开的文件夹"""
        config = self.load_config()
        folder = config.get('last_folder', '')
        if folder and Path(folder).exists():
            return folder
        return ""

    def set_last_folder(self, folder: str):
        """设置上次打开的文件夹"""
        config = self.load_config()
        config['last_folder'] = folder
        self.save_config(config)

    @staticmethod
    def create_annotation_data(video_id: str, keyframes: Dict, scores: Dict, total_score: int) -> Dict:
        """创建标注数据结构"""
        return {
            'video_id': video_id,
            'keyframes': keyframes,
            'scores': scores,
            'total_score': total_score,
            'metadata': {'timestamp': datetime.now().isoformat()}
        }

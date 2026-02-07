"""数据管理模块 - 标注数据直接保存到专家文件"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple


class DataManager:
    """标注数据管理器 - 直接使用 score_by_expert_{name}.json"""

    CONFIG_FILE = "less_config.json"

    # LESS 17项评分ID
    ITEM_IDS = [
        'itemA', 'itemB', 'itemC', 'itemD', 'itemE', 'itemF', 'itemG',
        'itemH', 'itemI', 'itemJ', 'itemK', 'itemL', 'itemM', 'itemN',
        'itemO', 'itemP', 'itemQ'
    ]

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.expert_id: str = ""  # 专家姓名
        self.created_at: str = ""  # 数据创建时间
        self.annotations: Dict[str, Dict] = {}  # {video_id: annotation}
        self._video_folder: str = ""  # 当前视频文件夹路径
        self._load_config_and_expert()

    def _load_config_and_expert(self):
        """加载配置和专家数据"""
        config = self.load_config()
        self._video_folder = config.get('last_folder', '')
        saved_expert = config.get('expert_id', '')
        if saved_expert:
            self.expert_id = saved_expert
            self._load_expert_data()

    @property
    def experts_dir(self) -> Optional[Path]:
        """获取experts_score文件夹路径"""
        if self._video_folder:
            return Path(self._video_folder) / "experts_score"
        return None

    @property
    def annotation_path(self) -> Optional[Path]:
        """获取当前专家的标注文件路径"""
        if not self.expert_id or not self.experts_dir:
            return None
        return self.experts_dir / f"score_by_expert_{self.expert_id}.json"

    @property
    def config_path(self) -> Path:
        return self.base_path / self.CONFIG_FILE

    def _load_expert_data(self):
        """加载当前专家的数据"""
        if not self.annotation_path or not self.annotation_path.exists():
            self.annotations = {}
            self.created_at = ""
            return

        try:
            with open(self.annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.created_at = data.get('created_at', '')
                self.annotations = data.get('annotations', {})
                # 确保expert_id一致
                file_expert = data.get('expert_id', '')
                if file_expert and file_expert != self.expert_id:
                    for ann in self.annotations.values():
                        ann['expert_id'] = self.expert_id
        except Exception as e:
            print(f"加载专家数据失败: {e}")
            self.annotations = {}

    def save_annotations(self):
        """保存标注数据到专家文件"""
        if not self.expert_id:
            print("警告: 未设置专家，无法保存")
            return
        if not self.experts_dir:
            print("警告: 未设置视频文件夹，无法保存")
            return

        try:
            # 确保文件夹存在
            self.experts_dir.mkdir(parents=True, exist_ok=True)

            data = {
                'expert_id': self.expert_id,
                'created_at': self.created_at or datetime.now().isoformat(),
                'annotations': self.annotations
            }
            with open(self.annotation_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存标注数据失败: {e}")

    def set_expert(self, expert_name: str):
        """设置专家姓名"""
        old_expert = self.expert_id
        self.expert_id = expert_name

        # 保存到配置
        config = self.load_config()
        config['expert_id'] = expert_name
        self.save_config(config)

        # 如果专家变了，加载新专家的数据
        if old_expert != expert_name:
            self._load_expert_data()

        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def set_video_folder(self, folder: str):
        """设置视频文件夹并加载数据"""
        self._video_folder = folder
        config = self.load_config()
        config['last_folder'] = folder
        self.save_config(config)
        # 重新加载当前专家的数据
        self._load_expert_data()

    def get_annotation(self, video_id: str) -> Optional[Dict]:
        """获取指定视频的标注"""
        return self.annotations.get(video_id)

    def set_annotation(self, video_id: str, data: Dict):
        """设置指定视频的标注"""
        if not self.expert_id:
            print("警告: 未设置专家，无法保存标注")
            return

        # 添加专家标识到数据中
        data['expert_id'] = self.expert_id
        self.annotations[video_id] = data

    def has_annotation(self, video_id: str) -> bool:
        """检查是否有指定视频的标注"""
        return video_id in self.annotations

    def get_completed_count(self) -> int:
        """获取已完成的标注数量"""
        return sum(1 for ann in self.annotations.values() if ann.get('completed', False))

    def get_draft_count(self) -> int:
        """获取暂存的标注数量"""
        return sum(1 for ann in self.annotations.values() if not ann.get('completed', False))

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
        return self._video_folder

    def set_last_folder(self, folder: str):
        """设置上次打开的文件夹（别名）"""
        self.set_video_folder(folder)

    @staticmethod
    def is_annotation_complete(data: Dict) -> bool:
        """检查标注是否完成"""
        if not data:
            return False

        kf = data.get('keyframes', {})
        if kf.get('start') is None:
            return False
        if not all(kf.get(k, 0) > 0 for k in ['ic', 'mkf', 'end']):
            return False

        scores = data.get('scores', {})
        if not all(scores.get(item) is not None for item in DataManager.ITEM_IDS):
            return False

        return True

    @staticmethod
    def create_annotation_data(video_id: str, keyframes: Dict, scores: Dict, total_score: int) -> Dict:
        """创建标注数据结构"""
        return {
            'video_id': video_id,
            'keyframes': keyframes,
            'scores': scores,
            'total_score': total_score,
            'completed': False,
            'metadata': {'timestamp': datetime.now().isoformat()}
        }

    def get_expert_score_files(self) -> List[Tuple[str, str, Path]]:
        """获取experts_score文件夹中的所有专家评分文件"""
        if not self.experts_dir or not self.experts_dir.exists():
            return []

        result = []
        for f in self.experts_dir.glob("score_by_expert_*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    expert_name = data.get('expert_id', f.stem.replace('score_by_expert_', ''))
                    result.append((expert_name, f.name, f))
            except:
                expert_name = f.stem.replace('score_by_expert_', '')
                result.append((expert_name, f.name, f))

        return result

    def switch_to_expert(self, expert_name: str) -> Tuple[bool, str]:
        """切换到其他专家（加载其数据）"""
        self.expert_id = expert_name
        self._load_expert_data()

        # 更新配置
        config = self.load_config()
        config['expert_id'] = expert_name
        self.save_config(config)

        completed_count = self.get_completed_count()
        total_count = len(self.annotations)

        return True, f"已切换到专家: {expert_name}\n标注数量: {total_count} 条\n已标注: {completed_count} 条"

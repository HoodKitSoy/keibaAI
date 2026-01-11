# modules/rl/__init__.py
"""
強化学習モジュール

競馬の馬券購入を強化学習で最適化するためのモジュール群。
全7馬券種（単勝、複勝、馬連、馬単、ワイド、三連複、三連単）およびWIN5に対応。

モデル構成:
- LightGBMモデル: models/lightgbm/
- 強化学習モデル: models/rl/{bet_type}/
"""

from ._agent import DQNAgent, ReplayBuffer, clear_cuda_memory, get_safe_device
from ._environment import BET_TYPES, BettingEnvironment, WIN5Environment
from ._policy import (
    RLBetPolicy,
    RLWIN5Policy,
)
from ._trainer import RLTrainer, WIN5Trainer

__all__ = [
    # 環境
    "BettingEnvironment",
    "WIN5Environment",
    "BET_TYPES",
    # エージェント
    "DQNAgent",
    "ReplayBuffer",
    # ユーティリティ
    "clear_cuda_memory",
    "get_safe_device",
    # ポリシー
    "RLBetPolicy",
    "RLWIN5Policy",
    # トレーナー
    "RLTrainer",
    "WIN5Trainer",
]

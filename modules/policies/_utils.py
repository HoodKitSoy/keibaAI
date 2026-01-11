"""
ポリシー共通ユーティリティ（後方互換性のため維持）

注意: このモジュールの機能は modules.core に統合されました。
新しいコードでは modules.core を直接使用してください。
"""

import pandas as pd

from modules.core import build_race_id
from modules.core.utils import iter_race_groups

# 後方互換性のためのエクスポート
build_race_id_series = build_race_id

__all__ = ["build_race_id_series", "iter_race_groups"]


def _zfill_series(s: pd.Series, w: int) -> pd.Series:
    """後方互換性のため維持"""
    return s.astype(str).str.zfill(w)

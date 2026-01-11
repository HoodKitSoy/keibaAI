"""
設定と定数モジュール（後方互換性のため維持）

注意: このモジュールの機能は modules.core に統合されました。
新しいコードでは modules.core を直接使用してください。
"""

from modules.core import (
    DB_DIR_STR as DB_DIR,
)
from modules.core import (
    EXCLUDE_DATA_KUBUNS,
    FINAL_COLUMN_ORDER,
    TARGET_JYO_CDS,
    TARGET_YEARS,
)
from modules.core import (
    LOG_PATH_STR as LOG_PATH,
)
from modules.core import (
    TMP_DIR_STR as TMP_DIR,
)

# 後方互換性のため、modules.coreから再エクスポート
# すべての定数は modules.core で定義されています

__all__ = [
    "DB_DIR",
    "TMP_DIR",
    "LOG_PATH",
    "TARGET_YEARS",
    "TARGET_JYO_CDS",
    "FINAL_COLUMN_ORDER",
    "EXCLUDE_DATA_KUBUNS",
]

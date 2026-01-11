"""
共通モジュール (modules.core)

プロジェクト全体で使用する共通機能を提供します。

このモジュールは、以下の4つのサブモジュールから構成されます:
- constants: 定数定義（RACE_KEY_COLS, FINAL_COLUMN_ORDERなど）
- config: パス設定、シード設定
- data_loading: Parquetファイルの検索・読み込み
- utils: レースID生成、グルーピングなどのユーティリティ

使用例:
    >>> from modules.core import RACE_KEY_COLS, load_preprocessed
    >>> df = load_preprocessed()
    >>> print(RACE_KEY_COLS)
"""

# 定数のエクスポート
# 設定のエクスポート
from .config import (
    DB_DIR,
    DB_DIR_STR,
    LOG_PATH,
    LOG_PATH_STR,
    MODELS_DIR,
    MODELS_DIR_STR,
    PREDICTIONS_DIR,
    PREDICTIONS_DIR_STR,
    SEED,
    TMP_DIR,
    TMP_DIR_STR,
    ensure_directories,
    get_prediction_path,
    get_preprocessed_path,
)
from .constants import (
    EXCLUDE_DATA_KUBUNS,
    EXCLUDE_FEATURE_COLS,
    FINAL_COLUMN_ORDER,
    NUM_CLASSES_RANK,
    RACE_KEY_COLS,
    RECORD_KEY_COLS,
    TARGET_COL,
    TARGET_JYO_CDS,
    TARGET_YEARS,
)

# データローディングのエクスポート
from .data_loading import (
    exclude_data_kubun,
    filter_by_date_range,
    filter_by_jyo_cd,
    find_parquet_file,
    load_db_tables,
    load_preprocessed,
    read_csv_or_parquet,
    read_parquet,
)

# ユーティリティのエクスポート
from .utils import (
    add_race_id_column,
    build_race_id,
    iter_race_groups,
    normalize_race_keys,
    validate_race_keys,
    zfill_series,
)

__all__ = [
    # 定数
    "RACE_KEY_COLS",
    "RECORD_KEY_COLS",
    "TARGET_COL",
    "FINAL_COLUMN_ORDER",
    "TARGET_YEARS",
    "TARGET_JYO_CDS",
    "EXCLUDE_DATA_KUBUNS",
    "NUM_CLASSES_RANK",
    # 設定
    "DB_DIR",
    "DB_DIR_STR",
    "TMP_DIR",
    "TMP_DIR_STR",
    "MODELS_DIR",
    "MODELS_DIR_STR",
    "PREDICTIONS_DIR",
    "PREDICTIONS_DIR_STR",
    "LOG_PATH",
    "LOG_PATH_STR",
    "SEED",
    "ensure_directories",
    "get_preprocessed_path",
    "get_prediction_path",
    # データローディング
    "find_parquet_file",
    "read_parquet",
    "read_csv_or_parquet",
    "load_preprocessed",
    "load_db_tables",
    "filter_by_date_range",
    "filter_by_jyo_cd",
    "exclude_data_kubun",
    # ユーティリティ
    "zfill_series",
    "build_race_id",
    "iter_race_groups",
    "add_race_id_column",
    "validate_race_keys",
    "normalize_race_keys",
]

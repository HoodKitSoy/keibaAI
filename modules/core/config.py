"""
共通設定モジュール

プロジェクト全体で使用するパス設定とシード設定を提供します。
.envファイルからの設定はルートのconfig.pyを通じて読み込みます。
"""

import os
from pathlib import Path

# ルートのconfig.pyから設定を読み込む
import config as root_config

# ========================================
# ディレクトリパス設定（.envから読み込み）
# ========================================

# データベースディレクトリ（Parquetファイルの格納先）
DB_DIR = Path(root_config.PREPROCESSING_DB_DIR)
DB_DIR_STR = root_config.PREPROCESSING_DB_DIR  # 後方互換用

# 一時ファイルディレクトリ（前処理済みファイルの出力先）
TMP_DIR = Path(root_config.PREPROCESSING_OUTPUT_DIR)
TMP_DIR_STR = root_config.PREPROCESSING_OUTPUT_DIR  # 後方互換用

# モデル保存ディレクトリ
MODELS_DIR = Path(root_config.TRAIN_OUTPUT_DIR)
MODELS_DIR_STR = root_config.TRAIN_OUTPUT_DIR  # 後方互換用

# 予測結果出力ディレクトリ
PREDICTIONS_DIR = Path(root_config.PREDICT_OUTPUT_DIR)
PREDICTIONS_DIR_STR = root_config.PREDICT_OUTPUT_DIR  # 後方互換用

# ログファイルパス
LOG_PATH = TMP_DIR / "preprocessing.log"
LOG_PATH_STR = os.path.join(TMP_DIR_STR, "preprocessing.log")  # 後方互換用

# ========================================
# デフォルトファイル名設定
# ========================================

# 前処理済みファイル（学習用）
DEFAULT_PREPROCESSED_FILE = "db_preprocessed_2016_2024"

# 当日前処理済みファイルのパターン
TODAY_PREPROCESSED_PATTERN = "today_preprocessed_{date}"

# 予測結果ファイルのパターン
PREDICTION_FILENAME_PATTERN = "prediction_{date}_{task}"

# ========================================
# 再現性設定（.envから読み込み）
# ========================================

# ランダムシード
SEED = root_config.SEED

# ========================================
# テーブル名設定
# ========================================

# 学習用テーブル（n_プレフィックス）
TRAIN_UMA_RACE_TABLE = "n_uma_race"
TRAIN_RACE_TABLE = "n_race"
TRAIN_UMA_TABLE = "n_uma"
TRAIN_HANRO_TABLE = "n_hanro"
TRAIN_CHIP_TABLE = "n_chip"

# 当日予測用テーブル（s_プレフィックス、調教データはn_）
TODAY_UMA_RACE_TABLE = "s_uma_race"
TODAY_RACE_TABLE = "s_race"
TODAY_UMA_TABLE = "s_uma"

# 払戻・WIN5テーブル
HARAI_TABLE = "n_harai"
WIN5_HEAD_TABLE = "n_jyusyosiki_head"
WIN5_TABLE = "n_jyusyosiki"

# 当日WIN5テーブル（s_プレフィックス）
TODAY_WIN5_HEAD_TABLE = "s_jyusyosiki_head"
TODAY_WIN5_TABLE = "s_jyusyosiki"

# ========================================
# ヘルパー関数
# ========================================


def ensure_directories() -> None:
    """
    必要なディレクトリが存在しない場合は作成します。
    """
    for directory in [DB_DIR, TMP_DIR, MODELS_DIR, PREDICTIONS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_preprocessed_path(
    date: str | None = None, as_string: bool = False
) -> Path | str:
    """
    前処理済みファイルのパスを取得します。

    Args:
        date: 日付文字列（YYYYMMDD形式）。Noneの場合は学習用ファイル
        as_string: Trueの場合は文字列で返す

    Returns:
        Path | str: ファイルパス
    """
    if date:
        filename = TODAY_PREPROCESSED_PATTERN.format(date=date) + ".parquet"
    else:
        filename = DEFAULT_PREPROCESSED_FILE + ".parquet"

    path = TMP_DIR / filename
    return str(path) if as_string else path


def get_prediction_path(date: str, task: str, as_string: bool = False) -> Path | str:
    """
    予測結果ファイルのパスを取得します。

    Args:
        date: 日付文字列（YYYYMMDD形式）
        task: タスク名（"win", "top3", "rank"）
        as_string: Trueの場合は文字列で返す

    Returns:
        Path | str: ファイルパス
    """
    filename = PREDICTION_FILENAME_PATTERN.format(date=date, task=task) + ".parquet"
    path = PREDICTIONS_DIR / filename
    return str(path) if as_string else path

# config.py
"""
設定モジュール

.envファイルから各種パラメータを読み込みます。
各スクリプト（preparing.py, preprocessing.py, train.py, test.py, predict.py）の
設定を一元管理します。
"""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


# =====================================
# ヘルパー関数
# =====================================
def _get_int_env(name: str, default: int) -> int:
    """整数型の環境変数を取得"""
    try:
        v = os.getenv(name)
        return int(v) if v is not None and str(v).strip() != "" else default
    except Exception:
        return default


def _get_float_env(name: str, default: float) -> float:
    """浮動小数点型の環境変数を取得"""
    try:
        v = os.getenv(name)
        return float(v) if v is not None and str(v).strip() != "" else default
    except Exception:
        return default


def _get_str_env(name: str, default: str) -> str:
    """文字列型の環境変数を取得"""
    v = os.getenv(name)
    return v if v is not None and str(v).strip() != "" else default


def _get_bool_env(name: str, default: bool) -> bool:
    """ブール型の環境変数を取得"""
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).lower() in ("true", "1", "yes", "on")


def _get_list_int_env(name: str, default: List[int]) -> List[int]:
    """カンマ区切りの整数リスト環境変数を取得"""
    try:
        v = os.getenv(name)
        if v is not None and str(v).strip() != "":
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return default
    except Exception:
        return default


def _get_list_str_env(name: str, default: List[str]) -> List[str]:
    """カンマ区切りの文字列リスト環境変数を取得"""
    try:
        v = os.getenv(name)
        if v is not None and str(v).strip() != "":
            return [x.strip() for x in v.split(",") if x.strip()]
        return default
    except Exception:
        return default


# =====================================
# [共通設定] 全スクリプト共通
# =====================================
SEED = _get_int_env("SEED", 42)
DEVICE = _get_str_env("DEVICE", "")  # 空欄の場合は自動検出


# =====================================
# [preparing.py] データベースエクスポート設定
# =====================================
DB_HOST = _get_str_env("DB_HOST", "localhost")
DB_PORT = _get_int_env("DB_PORT", 3306)
DB_USER = _get_str_env("DB_USER", "root")
DB_PASSWORD = _get_str_env("DB_PASSWORD", "")
DB_NAME = _get_str_env("DB_NAME", "everydb2")
PREPARING_OUTPUT_DIR = _get_str_env("PREPARING_OUTPUT_DIR", "data/DB")


# =====================================
# [preprocessing.py] データ前処理設定
# =====================================
# 処理モード: train（学習用）、today（当日）、date（日付指定）
PREPROCESSING_MODE = _get_str_env("PREPROCESSING_MODE", "train")
# 対象日付（MODE=date時に使用、YYYYMMDD形式）
PREPROCESSING_TARGET_DATE = _get_str_env("PREPROCESSING_TARGET_DATE", "")
PREPROCESSING_DB_DIR = _get_str_env("PREPROCESSING_DB_DIR", "data/DB")
PREPROCESSING_OUTPUT_DIR = _get_str_env("PREPROCESSING_OUTPUT_DIR", "data/tmp")
PREPROCESSING_TARGET_YEARS = _get_list_int_env(
    "PREPROCESSING_TARGET_YEARS", [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
)
PREPROCESSING_TARGET_JYO_CDS = _get_list_str_env(
    "PREPROCESSING_TARGET_JYO_CDS",
    ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
)


# =====================================
# [train.py] モデル学習設定
# =====================================
TRAIN_INPUT_FILE = _get_str_env(
    "TRAIN_INPUT_FILE", "data/tmp/db_preprocessed_2016_2024.parquet"
)
TRAIN_RETURN_PATH = _get_str_env("TRAIN_RETURN_PATH", "data/DB/n_harai.parquet")
TRAIN_OUTPUT_DIR = _get_str_env("TRAIN_OUTPUT_DIR", "models")
TRAIN_TASKS = _get_list_str_env("TRAIN_TASKS", ["win", "top3", "rank"])

# データ分割（年）
TRAIN_YEARS = _get_list_int_env(
    "TRAIN_YEARS", [2016, 2017, 2018, 2019, 2020, 2021, 2022]
)
VALID_YEARS = _get_list_int_env("VALID_YEARS", [2023])
TEST_YEARS = _get_list_int_env("TEST_YEARS", [2024])

# --- LightGBM設定 ---
LGBM_NUM_BOOST_ROUND = _get_int_env("LGBM_NUM_BOOST_ROUND", 2000)
LGBM_EARLY_STOPPING_ROUNDS = _get_int_env("LGBM_EARLY_STOPPING_ROUNDS", 200)
LGBM_N_TRIALS = _get_int_env("LGBM_N_TRIALS", 50)
LGBM_CUSTOM_LOSS = _get_str_env("LGBM_CUSTOM_LOSS", "none")
LGBM_FOCAL_GAMMA = _get_float_env("LGBM_FOCAL_GAMMA", 2.0)
LGBM_FOCAL_ALPHA = _get_float_env("LGBM_FOCAL_ALPHA", 0.25)
LGBM_RANK_NUM_CLASS = _get_int_env("LGBM_RANK_NUM_CLASS", 18)
LGBM_SAMPLE_FRAC = _get_float_env("LGBM_SAMPLE_FRAC", 1.0)
LGBM_SKIP = _get_bool_env("LGBM_SKIP", False)
LGBM_PREDS_PATH = _get_str_env("LGBM_PREDS_PATH", "")
TRAIN_VERBOSE = _get_bool_env("TRAIN_VERBOSE", False)

# --- 学習データフィルタ設定 ---
# TrueにするとWIN5対象レースのみで学習（デフォルト: False）
TRAIN_WIN5_RACES_ONLY = _get_bool_env("TRAIN_WIN5_RACES_ONLY", False)


# 後方互換用エイリアス
MODEL_SAVE_DIR = TRAIN_OUTPUT_DIR
NUM_BOOST_ROUND = LGBM_NUM_BOOST_ROUND
EARLY_STOPPING_ROUNDS = LGBM_EARLY_STOPPING_ROUNDS
N_TRIALS = LGBM_N_TRIALS

# LightGBMモデル用サブディレクトリ
LIGHTGBM_MODEL_DIR = os.path.join(TRAIN_OUTPUT_DIR, "lightgbm")

# --- 強化学習設定 ---
RL_ENABLED = _get_bool_env("RL_ENABLED", False)
RL_BET_TYPES = _get_list_str_env("RL_BET_TYPES", [])  # 空欄の場合は全券種+WIN5
RL_NUM_EPISODES = _get_int_env("RL_NUM_EPISODES", 100)
RL_LEARNING_RATE = _get_float_env("RL_LEARNING_RATE", 1e-4)
RL_GAMMA = _get_float_env("RL_GAMMA", 0.99)
RL_EPSILON_START = _get_float_env("RL_EPSILON_START", 1.0)
RL_EPSILON_END = _get_float_env("RL_EPSILON_END", 0.05)
RL_EPSILON_DECAY = _get_float_env("RL_EPSILON_DECAY", 0.995)
RL_BUFFER_SIZE = _get_int_env("RL_BUFFER_SIZE", 100000)
RL_BATCH_SIZE = _get_int_env("RL_BATCH_SIZE", 64)
RL_TARGET_UPDATE_FREQ = _get_int_env("RL_TARGET_UPDATE_FREQ", 100)
RL_HIDDEN_DIMS = _get_list_int_env("RL_HIDDEN_DIMS", [256, 256, 128])
RL_BET_AMOUNT = _get_float_env("RL_BET_AMOUNT", 100.0)
RL_EVAL_INTERVAL = _get_int_env("RL_EVAL_INTERVAL", 10)
RL_SAVE_INTERVAL = _get_int_env("RL_SAVE_INTERVAL", 50)

# 強化学習モデル用サブディレクトリ
RL_MODEL_DIR = os.path.join(TRAIN_OUTPUT_DIR, "rl")


# =====================================
# [test.py] モデル評価・シミュレーション設定
# =====================================
TEST_PREDS_DIR = _get_str_env("TEST_PREDS_DIR", "models/lightgbm")
TEST_INPUT_FILE = _get_str_env(
    "TEST_INPUT_FILE", ""
)  # 空欄の場合はpreds_*.parquetを使用
TEST_REGENERATE_PREDS = _get_bool_env(
    "TEST_REGENERATE_PREDS", False
)  # Trueの場合はTEST_INPUT_FILEから予測を再生成
TEST_TASKS = _get_list_str_env("TEST_TASKS", ["win"])
TEST_OUTPUT_DIR = _get_str_env("TEST_OUTPUT_DIR", "models")
TEST_USE_POLICY_CONFIG = _get_bool_env("TEST_USE_POLICY_CONFIG", True)
TEST_POLICY_CONFIG_PATH = _get_str_env("TEST_POLICY_CONFIG_PATH", "policy_config.json")
TEST_POLICY = _get_str_env("TEST_POLICY", "all")
TEST_RANK_THRESHOLD = _get_int_env("TEST_RANK_THRESHOLD", 1)
TEST_SHOW_PLOT = _get_bool_env("TEST_SHOW_PLOT", False)
TEST_TIERED_COVERAGE_THRESHOLD = _get_float_env("TEST_TIERED_COVERAGE_THRESHOLD", 0.5)
TEST_MIN_TOP_SCORE = _get_float_env("TEST_MIN_TOP_SCORE", 0.35)
TEST_MIN_SCORE_GAP = _get_float_env("TEST_MIN_SCORE_GAP", 0.10)
TEST_USE_OR_CONDITION = _get_bool_env("TEST_USE_OR_CONDITION", False)

# --- WIN5シミュレーション設定 ---
TEST_WIN5_ENABLED = _get_bool_env("TEST_WIN5_ENABLED", True)
TEST_WIN5_THRESHOLD_MIN = _get_float_env("TEST_WIN5_THRESHOLD_MIN", 0.01)
TEST_WIN5_THRESHOLD_MAX = _get_float_env("TEST_WIN5_THRESHOLD_MAX", 0.10)
TEST_WIN5_THRESHOLD_STEP = _get_float_env("TEST_WIN5_THRESHOLD_STEP", 0.01)
TEST_WIN5_AMOUNT = _get_float_env("TEST_WIN5_AMOUNT", 100.0)


# =====================================
# [predict.py] 当日予測設定
# =====================================
# 予測モード: today（当日）、date（日付指定）
PREDICT_MODE = _get_str_env("PREDICT_MODE", "today")
# 対象日付（MODE=date時に使用、YYYYMMDD形式）
PREDICT_TARGET_DATE = _get_str_env("PREDICT_TARGET_DATE", "")
PREDICT_MODELS_DIR = _get_str_env("PREDICT_MODELS_DIR", "models")
PREDICT_PREPROCESSED_DIR = _get_str_env("PREDICT_PREPROCESSED_DIR", "data/tmp")
PREDICT_OUTPUT_DIR = _get_str_env("PREDICT_OUTPUT_DIR", "predictions")
PREDICT_TASKS = _get_list_str_env("PREDICT_TASKS", ["win", "top3", "rank"])
PREDICT_USE_POLICY_CONFIG = _get_bool_env("PREDICT_USE_POLICY_CONFIG", True)
PREDICT_POLICY_CONFIG_PATH = _get_str_env(
    "PREDICT_POLICY_CONFIG_PATH", "policy_config.json"
)
PREDICT_VERBOSE = _get_bool_env("PREDICT_VERBOSE", False)
PREDICT_FILTER_UNSTARTED = _get_bool_env("PREDICT_FILTER_UNSTARTED", False)

# --- WIN5予測設定 ---
PREDICT_WIN5_ENABLED = _get_bool_env("PREDICT_WIN5_ENABLED", True)
PREDICT_WIN5_THRESHOLD = _get_float_env("PREDICT_WIN5_THRESHOLD", 0.3)


# =====================================
# 馬券種リスト
# =====================================
BET_TYPES = ["tansho", "fukusho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]
BET_TYPES_ALL = BET_TYPES + ["win5"]

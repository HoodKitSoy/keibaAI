"""
評価モジュールパッケージ

競馬予測モデルの評価とポリシー比較機能を提供します。

設定はすべて.envファイルから読み込まれます（argparseは廃止）。
"""

from ._data_loader import load_model_and_data
from ._evaluator import evaluate_strategies, run_win5_simulation
from ._output_manager import (
    get_comparison_output_dir,
    get_output_dir,
    get_strategy_output_dir,
    sanitize_name_for_path,
)
from ._result_saver import (
    save_comparison_by_bet_type,
    save_comparison_results,
    save_strategy_details,
    save_win5_details,
    save_win5_results,
)
from ._strategy_manager import prepare_strategies, simulate_odds_table
from ._utils import (
    build_race_id_from_key,
    build_win5_candidates,
    build_win5_thresholds,
    clamp,
    find_latest_preds_dir,
    format_threshold_dir_name,
    format_threshold_value,
    load_preds_score_table,
)

__all__ = [
    # データローダー
    "load_model_and_data",
    # 評価機能
    "evaluate_strategies",
    "run_win5_simulation",
    # 結果保存
    "save_strategy_details",
    "save_comparison_results",
    "save_comparison_by_bet_type",
    "save_win5_details",
    "save_win5_results",
    # 戦略マネージャー
    "prepare_strategies",
    "simulate_odds_table",
    # 出力マネージャー
    "get_output_dir",
    "get_strategy_output_dir",
    "get_comparison_output_dir",
    "sanitize_name_for_path",
    # ユーティリティ
    "find_latest_preds_dir",
    "load_preds_score_table",
    "clamp",
    "format_threshold_value",
    "format_threshold_dir_name",
    "build_win5_thresholds",
    "build_win5_candidates",
    "build_race_id_from_key",
]

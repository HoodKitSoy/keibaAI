"""
予測モジュール

当日のレースデータを用いた予測機能を提供します。

注意: 設定関連は modules.core に統合されました。
設定はすべて.envファイルから読み込まれます（argparseは廃止）。
"""

from ._data_loader_predict import load_preprocessed_data
from ._data_processor_predict import align_to_training_format
from ._policy_recommender import (
    apply_policy_to_predictions,
    format_recommendations,
    generate_policy_recommendations,
)
from ._predictor import predict_with_models, save_predictions
from ._result_reporter import (
    print_prediction_summary,
    save_policy_report,
    save_win5_report,
)
from ._win5_predictor import (
    format_win5_recommendations,
    generate_win5_predictions,
)

__all__ = [
    # data loader
    "load_preprocessed_data",
    # data processor
    "align_to_training_format",
    # predictor
    "predict_with_models",
    "save_predictions",
    # policy recommender
    "generate_policy_recommendations",
    "apply_policy_to_predictions",
    "format_recommendations",
    # win5 predictor
    "generate_win5_predictions",
    "format_win5_recommendations",
    # result reporter
    "save_policy_report",
    "save_win5_report",
    "print_prediction_summary",
]

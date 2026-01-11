# modules/training/__init__.py
"""
訓練モジュール

注意: ほとんどの定数と共通機能は modules.core に統合されました。
このモジュールは訓練固有の機能のみを提供します。

設定はすべて.envファイルから読み込まれます（argparseは廃止）。
"""

# 後方互換性のため、modules.coreから再エクスポート
# trainモジュール固有の設定
from modules.core import (
    FINAL_COLUMN_ORDER,
    NUM_CLASSES_RANK,
    RACE_KEY_COLS,
    RECORD_KEY_COLS,
    SEED,
    TARGET_COL,
)

from ._config_handler import get_effective_config
from ._feature_engineering import (
    coerce_numeric_or_categorical,
    select_features,
    split_by_year,
)
from ._file_utils import (
    ensure_dir,
    safe_save_df,
    safe_save_json,
    safe_save_model,
)
from ._model_trainer import (
    build_lgb_dataset,
    evaluate,
    make_binary_accuracy_metric,
    make_focal_loss,
    suggest_params,
    train_one_task,
)
from ._phase_runner import (
    load_existing_predictions,
    load_return_data,
    run_lightgbm_phase,
    run_rl_phase,
    split_by_years,
)

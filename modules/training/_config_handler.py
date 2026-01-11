"""
設定ハンドリングモジュール

.env設定から実効設定を生成します。
"""

from __future__ import annotations

from typing import Any, Dict

import config


def get_effective_config(settings: Any) -> Dict:
    """
    設定オブジェクトとconfig値をマージして実効設定を返す

    設定オブジェクト（SimpleNamespace等）で指定された値を優先し、
    未指定の場合はconfigのデフォルト値を使用します。

    Args:
        settings: 設定オブジェクト（SimpleNamespace等）

    Returns:
        Dict: マージされた実効設定
    """
    return {
        # LightGBM
        "n_trials": getattr(settings, "n_trials", None) or config.N_TRIALS,
        "num_boost_round": getattr(settings, "num_boost_round", None)
        or config.NUM_BOOST_ROUND,
        "early_stopping_rounds": getattr(settings, "early_stopping_rounds", None)
        or config.EARLY_STOPPING_ROUNDS,
        # 強化学習
        "rl_num_episodes": getattr(settings, "rl_num_episodes", None)
        or config.RL_NUM_EPISODES,
        "rl_learning_rate": getattr(settings, "rl_learning_rate", None)
        or config.RL_LEARNING_RATE,
        "rl_gamma": getattr(settings, "rl_gamma", None) or config.RL_GAMMA,
        "rl_epsilon_start": getattr(settings, "rl_epsilon_start", None)
        or config.RL_EPSILON_START,
        "rl_epsilon_end": getattr(settings, "rl_epsilon_end", None)
        or config.RL_EPSILON_END,
        "rl_epsilon_decay": getattr(settings, "rl_epsilon_decay", None)
        or config.RL_EPSILON_DECAY,
        "rl_buffer_size": getattr(settings, "rl_buffer_size", None)
        or config.RL_BUFFER_SIZE,
        "rl_batch_size": getattr(settings, "rl_batch_size", None)
        or config.RL_BATCH_SIZE,
        "rl_target_update_freq": getattr(settings, "rl_target_update_freq", None)
        or config.RL_TARGET_UPDATE_FREQ,
        "rl_hidden_dims": (
            [int(d) for d in getattr(settings, "rl_hidden_dims", "").split(",")]
            if getattr(settings, "rl_hidden_dims", None)
            else config.RL_HIDDEN_DIMS
        ),
        "rl_bet_amount": getattr(settings, "rl_bet_amount", None)
        or config.RL_BET_AMOUNT,
        "rl_eval_interval": getattr(settings, "rl_eval_interval", None)
        or config.RL_EVAL_INTERVAL,
        "rl_save_interval": getattr(settings, "rl_save_interval", None)
        or config.RL_SAVE_INTERVAL,
        # 共通
        "device": getattr(settings, "device", None)
        or (config.DEVICE if config.DEVICE else None),
        "seed": getattr(settings, "seed", None) or config.SEED,
        "train_years": config.TRAIN_YEARS,
        "valid_years": config.VALID_YEARS,
        "test_years": config.TEST_YEARS,
        "bet_types": (
            [t.strip() for t in getattr(settings, "rl_bet_types", "").split(",")]
            if getattr(settings, "rl_bet_types", None)
            else config.BET_TYPES_ALL
        ),
    }

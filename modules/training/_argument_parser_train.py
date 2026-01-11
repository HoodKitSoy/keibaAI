"""
コマンドライン引数パーサーモジュール

訓練スクリプトのコマンドライン引数を処理します。
すべての設定は.envから読み込まれたconfig値をデフォルトとして使用。
引数で指定された場合はそちらを優先。
"""

from __future__ import annotations

import argparse
import os

import config


def parse_train_arguments() -> argparse.Namespace:
    """
    コマンドライン引数をパース

    すべての設定は.envから読み込まれたconfig値をデフォルトとして使用。
    引数で指定された場合はそちらを優先。

    Returns:
        argparse.Namespace: パースされた引数
    """
    parser = argparse.ArgumentParser(
        description="LightGBM + 強化学習 統合学習スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # =====================================
    # 入力・出力設定
    # =====================================
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "tmp", "db_preprocessed_2016_2024.parquet"),
        help="前処理済みデータのパス（parquet/csv）",
    )
    parser.add_argument(
        "--return-path",
        type=str,
        default=os.path.join("data", "DB", "n_harai.parquet"),
        help="払戻データのパス（強化学習用）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=config.MODEL_SAVE_DIR,
        help="出力ベースディレクトリ",
    )

    # =====================================
    # フェーズ制御
    # =====================================
    parser.add_argument(
        "--skip-lgbm",
        action="store_true",
        help="LightGBM学習をスキップ（既存の予測を使用）",
    )
    parser.add_argument(
        "--enable-rl",
        action="store_true",
        help="強化学習を有効化",
    )
    parser.add_argument(
        "--preds-path",
        type=str,
        default=None,
        help="既存の予測ファイルパス（--skip-lgbm時に使用）",
    )

    # =====================================
    # LightGBM設定
    # =====================================
    lgbm_group = parser.add_argument_group("LightGBM設定")
    lgbm_group.add_argument(
        "--tasks",
        type=str,
        default=",".join(config.TRAIN_TASKS),
        help="学習タスク（カンマ区切り: win,top3,rank）（.env: TRAIN_TASKS）",
    )
    lgbm_group.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "win", "top3", "rank"],
        help="(後方互換) 単一タスク指定。--tasksが指定されていれば無視",
    )
    lgbm_group.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=f"Optuna試行回数（未指定は.envから: {config.LGBM_N_TRIALS}）",
    )
    lgbm_group.add_argument(
        "--num-boost-round",
        type=int,
        default=None,
        help=f"LightGBM学習回数（未指定は.envから: {config.LGBM_NUM_BOOST_ROUND}）",
    )
    lgbm_group.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help=f"early stoppingラウンド（未指定は.envから: {config.LGBM_EARLY_STOPPING_ROUNDS}）",
    )
    lgbm_group.add_argument(
        "--custom-loss",
        type=str,
        default="none",
        choices=["none", "focal"],
        help="2値タスク用カスタム損失（focal）",
    )
    lgbm_group.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="focal lossのgamma",
    )
    lgbm_group.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="focal lossのalpha",
    )
    lgbm_group.add_argument(
        "--rank-num-class",
        type=int,
        default=None,
        help="rankタスクのクラス数（未指定時はデータ内最大着順を使用）",
    )
    lgbm_group.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="学習用にランダムサンプリングする比率（0<frac<=1）。動作確認用",
    )

    # =====================================
    # 強化学習設定
    # =====================================
    rl_group = parser.add_argument_group("強化学習設定")
    rl_group.add_argument(
        "--rl-bet-types",
        type=str,
        default=None,
        help="強化学習の対象券種（カンマ区切り）。未指定時は全券種+WIN5",
    )
    rl_group.add_argument(
        "--rl-num-episodes",
        type=int,
        default=None,
        help=f"強化学習のエピソード数（未指定は.envから: {config.RL_NUM_EPISODES}）",
    )
    rl_group.add_argument(
        "--rl-learning-rate",
        type=float,
        default=None,
        help=f"強化学習の学習率（未指定は.envから: {config.RL_LEARNING_RATE}）",
    )
    rl_group.add_argument(
        "--rl-gamma",
        type=float,
        default=None,
        help=f"割引率（未指定は.envから: {config.RL_GAMMA}）",
    )
    rl_group.add_argument(
        "--rl-epsilon-start",
        type=float,
        default=None,
        help=f"初期ε（未指定は.envから: {config.RL_EPSILON_START}）",
    )
    rl_group.add_argument(
        "--rl-epsilon-end",
        type=float,
        default=None,
        help=f"最終ε（未指定は.envから: {config.RL_EPSILON_END}）",
    )
    rl_group.add_argument(
        "--rl-epsilon-decay",
        type=float,
        default=None,
        help=f"ε減衰率（未指定は.envから: {config.RL_EPSILON_DECAY}）",
    )
    rl_group.add_argument(
        "--rl-buffer-size",
        type=int,
        default=None,
        help=f"リプレイバッファサイズ（未指定は.envから: {config.RL_BUFFER_SIZE}）",
    )
    rl_group.add_argument(
        "--rl-batch-size",
        type=int,
        default=None,
        help=f"バッチサイズ（未指定は.envから: {config.RL_BATCH_SIZE}）",
    )
    rl_group.add_argument(
        "--rl-target-update-freq",
        type=int,
        default=None,
        help=f"ターゲットネットワーク更新頻度（未指定は.envから: {config.RL_TARGET_UPDATE_FREQ}）",
    )
    rl_group.add_argument(
        "--rl-hidden-dims",
        type=str,
        default=None,
        help=f"隠れ層のユニット数（カンマ区切り、未指定は.envから: {config.RL_HIDDEN_DIMS}）",
    )
    rl_group.add_argument(
        "--rl-bet-amount",
        type=float,
        default=None,
        help=f"1点あたりの賭け金（未指定は.envから: {config.RL_BET_AMOUNT}）",
    )
    rl_group.add_argument(
        "--rl-eval-interval",
        type=int,
        default=None,
        help=f"評価間隔（未指定は.envから: {config.RL_EVAL_INTERVAL}）",
    )
    rl_group.add_argument(
        "--rl-save-interval",
        type=int,
        default=None,
        help=f"保存間隔（未指定は.envから: {config.RL_SAVE_INTERVAL}）",
    )
    rl_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="計算デバイス（cuda/cpu、未指定は自動検出）",
    )

    # =====================================
    # その他
    # =====================================
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細出力",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"乱数シード（未指定は.envから: {config.SEED}）",
    )

    # 後方互換
    parser.add_argument(
        "--use-custom-loss",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


# 後方互換のために旧名も残す
parse_arguments = parse_train_arguments

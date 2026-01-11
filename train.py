#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py - LightGBM + 強化学習 統合学習スクリプト

JRA-DB前処理出力（Parquet/CSV）を読み込んで、以下の2段階で学習を行います。

【Phase 1: LightGBM学習】
- 2016-2022年: 学習データ
- 2023年: 検証データ（ハイパーパラメータチューニング）
- 2024年: テストデータ（最終評価）
- 全データに対して予測スコア（proba）を生成

【Phase 2: 強化学習】（オプション、.env RL_ENABLED=true時）
- LightGBMの予測スコアを状態の一部として使用
- 全7馬券種（単勝/複勝/馬連/馬単/ワイド/三連複/三連単）およびWIN5に対応
- 2016-2022年: 強化学習の学習データ
- 2023年: 強化学習の検証データ
- 2024年: 強化学習のテストデータ

出力物：
- ./models/lightgbm/lgbm_{task}.model
- ./models/lightgbm/best_params_{task}.json
- ./models/lightgbm/feature_importance_{task}.csv
- ./models/lightgbm/metrics_{task}.json
- ./models/lightgbm/preds_{task}.parquet
- ./models/rl/{bet_type}/best_model.pt（強化学習時）

使用方法:
    # .envファイルで設定を行い、実行
    python train.py

設定（.env）:
    TRAIN_TASKS: 学習タスク（win,top3,rank）
    RL_ENABLED: 強化学習の有効化（true/false）
    LGBM_SKIP: LightGBM学習をスキップ（true/false）
    RL_BET_TYPES: 強化学習の対象券種（カンマ区切り、空欄で全券種）
"""

from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np

import config
from modules.core import load_preprocessed
from modules.training import (
    get_effective_config,
    load_existing_predictions,
    load_return_data,
    run_lightgbm_phase,
    run_rl_phase,
)


def _build_settings() -> SimpleNamespace:
    """
    .envから設定を読み込み、SimpleNamespaceとして返す

    Returns:
        SimpleNamespace: 設定オブジェクト（argsの代替）
    """
    return SimpleNamespace(
        # 入出力
        input=config.TRAIN_INPUT_FILE,
        return_path=config.TRAIN_RETURN_PATH,
        out_dir=config.TRAIN_OUTPUT_DIR,
        # フェーズ制御
        skip_lgbm=config.LGBM_SKIP,
        enable_rl=config.RL_ENABLED,
        preds_path=config.LGBM_PREDS_PATH if config.LGBM_PREDS_PATH else None,
        # LightGBM設定
        tasks=",".join(config.TRAIN_TASKS),
        task="all",  # 後方互換
        n_trials=config.LGBM_N_TRIALS,
        num_boost_round=config.LGBM_NUM_BOOST_ROUND,
        early_stopping_rounds=config.LGBM_EARLY_STOPPING_ROUNDS,
        custom_loss=config.LGBM_CUSTOM_LOSS,
        focal_gamma=config.LGBM_FOCAL_GAMMA,
        focal_alpha=config.LGBM_FOCAL_ALPHA,
        rank_num_class=config.LGBM_RANK_NUM_CLASS
        if config.LGBM_RANK_NUM_CLASS > 0
        else None,
        sample_frac=config.LGBM_SAMPLE_FRAC if config.LGBM_SAMPLE_FRAC < 1.0 else None,
        # 強化学習設定
        rl_bet_types=",".join(config.RL_BET_TYPES) if config.RL_BET_TYPES else None,
        rl_num_episodes=config.RL_NUM_EPISODES,
        rl_learning_rate=config.RL_LEARNING_RATE,
        rl_gamma=config.RL_GAMMA,
        rl_epsilon_start=config.RL_EPSILON_START,
        rl_epsilon_end=config.RL_EPSILON_END,
        rl_epsilon_decay=config.RL_EPSILON_DECAY,
        rl_buffer_size=config.RL_BUFFER_SIZE,
        rl_batch_size=config.RL_BATCH_SIZE,
        rl_target_update_freq=config.RL_TARGET_UPDATE_FREQ,
        rl_hidden_dims=None,  # configから直接取得
        rl_bet_amount=config.RL_BET_AMOUNT,
        rl_eval_interval=config.RL_EVAL_INTERVAL,
        rl_save_interval=config.RL_SAVE_INTERVAL,
        # 共通
        device=config.DEVICE if config.DEVICE else None,
        seed=config.SEED,
        verbose=config.TRAIN_VERBOSE,
    )


def main() -> int:
    """
    メイン関数

    LightGBM学習と強化学習を統合的に実行します。

    Returns:
        int: 終了コード（0=成功、非0=エラー）
    """
    settings = _build_settings()

    # 実効設定を取得
    eff_config = get_effective_config(settings)

    # シード設定
    np.random.seed(eff_config["seed"])

    # CUDA環境の初期化（強化学習を使う場合）
    if settings.enable_rl:
        try:
            import torch

            if torch.cuda.is_available():
                # CUDAコンテキストをリセット
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"\n[CUDA初期化] GPU: {torch.cuda.get_device_name(0)}")
                print(
                    f"  メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                )
            else:
                raise RuntimeError("CUDAが利用できません")
        except Exception as e:
            print(f"\n❌ CUDA初期化エラー: {e}")
            print("GPUを使用した強化学習には以下が必要です:")
            print("  1. NVIDIAドライバがインストールされていること")
            print("  2. PyTorchのCUDA版がインストールされていること")
            print("  3. nvidia-smi コマンドでGPUが認識されていること")
            return 1

    print("\n" + "=" * 70)
    print("競馬予測AI 統合学習スクリプト")
    print("=" * 70)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"モード: {'LightGBM + 強化学習' if settings.enable_rl else 'LightGBMのみ'}")
    if settings.skip_lgbm:
        print("  ※ LightGBM学習はスキップ（既存予測を使用）")
    print(
        f"データ分割: 学習={eff_config['train_years']}, "
        f"検証={eff_config['valid_years']}, テスト={eff_config['test_years']}"
    )
    print("=" * 70)

    # データ読み込み
    print("\n[データ読み込み]")
    df_raw = load_preprocessed(settings.input)
    print(f"入力: {settings.input} -> shape={df_raw.shape}")

    if settings.sample_frac is not None:
        if not (0 < settings.sample_frac <= 1):
            raise ValueError(
                "LGBM_SAMPLE_FRAC は 0 より大きく 1 以下で指定してください"
            )
        df_raw = df_raw.sample(
            frac=settings.sample_frac, random_state=eff_config["seed"]
        )
        print(f"サンプル抽出: frac={settings.sample_frac} -> shape={df_raw.shape}")

    # 年の分布確認
    if "Year" in df_raw.columns:
        year_counts = df_raw["Year"].value_counts().sort_index()
        print(f"年別レコード数:\n{year_counts}")

    # Phase 1: LightGBM
    if settings.skip_lgbm:
        df_with_scores = load_existing_predictions(settings, df_raw)
    else:
        df_with_scores = run_lightgbm_phase(settings, df_raw, eff_config)

    # Phase 2: 強化学習
    if settings.enable_rl:
        return_table = load_return_data(settings.return_path)
        rl_results = run_rl_phase(settings, df_with_scores, return_table, eff_config)

        # 結果サマリ
        print("\n" + "=" * 60)
        print("強化学習結果サマリ")
        print("=" * 60)
        for bet_type, history in rl_results.items():
            if history:
                print(f"  {bet_type}: {len(history)} episodes completed")
    else:
        print(
            "\n[強化学習スキップ] RL_ENABLED=true を設定すると強化学習が有効になります"
        )

    print(f"\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
フェーズ実行モジュール

LightGBMおよび強化学習の各フェーズの実行ロジックを提供します。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

import config
from modules.core import RACE_KEY_COLS, RECORD_KEY_COLS, TARGET_COL

from ._feature_engineering import coerce_numeric_or_categorical, select_features
from ._model_trainer import train_one_task


def split_by_years(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """
    年でデータを分割

    Args:
        df: 分割対象のDataFrame
        years: 抽出する年のリスト

    Returns:
        pd.DataFrame: 指定年のデータ
    """
    if "Year" not in df.columns:
        return df

    df_year = df["Year"]
    if pd.api.types.is_numeric_dtype(df_year):
        return df[df_year.isin(years)].copy()
    else:
        years_str = [str(y) for y in years]
        return df[df_year.astype(str).isin(years_str)].copy()


def run_lightgbm_phase(
    args: argparse.Namespace,
    df_raw: pd.DataFrame,
    eff_config: Dict,
) -> pd.DataFrame:
    """
    Phase 1: LightGBM学習

    全データに対して予測スコアを生成し、元のデータフレームに追加して返します。

    Args:
        args: コマンドライン引数
        df_raw: 前処理済みの元データ
        eff_config: 実効設定

    Returns:
        pd.DataFrame: 予測スコアを追加したデータフレーム
    """
    print("\n" + "=" * 60)
    print("Phase 1: LightGBM 学習")
    print("=" * 60)

    # 設定の上書き
    config.N_TRIALS = eff_config["n_trials"]
    config.NUM_BOOST_ROUND = eff_config["num_boost_round"]
    config.EARLY_STOPPING_ROUNDS = eff_config["early_stopping_rounds"]

    # 特徴量抽出
    df_sub, feature_cols = select_features(df_raw)
    df_conv, categorical_cols = coerce_numeric_or_categorical(
        df_sub, feature_cols + RECORD_KEY_COLS
    )

    # タスク決定
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = ["win", "top3", "rank"] if args.task == "all" else [args.task]

    invalid = [t for t in tasks if t not in ("win", "top3", "rank")]
    if invalid:
        raise ValueError(f"不正なタスク指定: {invalid}")

    # rankクラス数動的決定
    kj_all = pd.to_numeric(df_conv[TARGET_COL], errors="coerce")
    dynamic_rank_num_class = args.rank_num_class
    if dynamic_rank_num_class is None:
        dynamic_rank_num_class = (
            int(kj_all[kj_all > 0].max()) if (kj_all > 0).any() else 18
        )

    print(f"rank num_class = {dynamic_rank_num_class}")

    # LightGBMモデル出力ディレクトリ
    lgbm_out_dir = os.path.join(args.out_dir, "lightgbm")
    os.makedirs(lgbm_out_dir, exist_ok=True)

    # 各タスクを学習
    preds_dict = {}
    for task in tasks:
        print(f"\n--- LightGBM学習: {task} (custom_loss={args.custom_loss}) ---")

        # train_one_taskを呼び出し
        train_one_task(
            task,
            df_conv,
            feature_cols,
            categorical_cols,
            lgbm_out_dir,
            custom_loss=args.custom_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            dynamic_num_class=dynamic_rank_num_class,
        )

        # 予測結果を読み込み
        preds_path = Path(lgbm_out_dir) / f"preds_{task}.parquet"
        if preds_path.exists():
            preds_df = pd.read_parquet(preds_path)
            preds_dict[task] = preds_df
            print(f"予測結果を読み込み: {preds_path} ({len(preds_df)} records)")

    # 予測スコアを元データにマージ
    result_df = df_raw.copy()

    for task, preds_df in preds_dict.items():
        merge_keys = RACE_KEY_COLS + ["Umaban"]
        available_keys = [
            k for k in merge_keys if k in preds_df.columns and k in result_df.columns
        ]

        if "proba" in preds_df.columns:
            for key in available_keys:
                if key in preds_df.columns and key in result_df.columns:
                    preds_df[key] = preds_df[key].astype(str)
                    result_df[key] = result_df[key].astype(str)

            score_col = f"lgbm_score_{task}"
            preds_df = preds_df[available_keys + ["proba"]].rename(
                columns={"proba": score_col}
            )
            result_df = result_df.merge(preds_df, on=available_keys, how="left")
            print(f"スコア列追加: {score_col}")

    return result_df


def load_existing_predictions(
    args: argparse.Namespace, df_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    既存の予測結果を読み込んでマージ

    Args:
        args: コマンドライン引数
        df_raw: 前処理済みの元データ

    Returns:
        pd.DataFrame: 予測スコアをマージしたデータフレーム
    """
    print("\n[LightGBMスキップ] 既存の予測を使用します")

    result_df = df_raw.copy()

    # 指定されたパス、またはデフォルトのパスから読み込み
    if args.preds_path:
        preds_paths = [Path(args.preds_path)]
    else:
        lgbm_dir = Path(args.out_dir) / "lightgbm"
        preds_paths = [
            lgbm_dir / "preds_win.parquet",
            lgbm_dir / "preds_top3.parquet",
            lgbm_dir / "preds_rank.parquet",
        ]
        # 後方互換: 旧パス（lightgbmサブディレクトリなし）
        if not any(p.exists() for p in preds_paths):
            preds_paths = [
                Path(args.out_dir) / "preds_win.parquet",
                Path(args.out_dir) / "preds_top3.parquet",
                Path(args.out_dir) / "preds_rank.parquet",
            ]

    for preds_path in preds_paths:
        if not preds_path.exists():
            print(f"警告: 予測ファイルが見つかりません: {preds_path}")
            continue

        preds_df = pd.read_parquet(preds_path)
        task = preds_path.stem.replace("preds_", "")

        merge_keys = RACE_KEY_COLS + ["Umaban"]
        available_keys = [
            k for k in merge_keys if k in preds_df.columns and k in result_df.columns
        ]

        if "proba" in preds_df.columns:
            for key in available_keys:
                if key in preds_df.columns and key in result_df.columns:
                    preds_df[key] = preds_df[key].astype(str)
                    result_df[key] = result_df[key].astype(str)

            score_col = f"lgbm_score_{task}"
            preds_df = preds_df[available_keys + ["proba"]].rename(
                columns={"proba": score_col}
            )
            result_df = result_df.merge(preds_df, on=available_keys, how="left")
            print(f"スコア列追加: {score_col} from {preds_path}")

    return result_df


def load_return_data(return_path: str) -> dict:
    """
    払戻データを読み込み

    Args:
        return_path: 払戻データファイルのパス

    Returns:
        dict: 払戻データを含む辞書
    """
    path = Path(return_path)
    if not path.exists():
        print(f"警告: 払戻ファイルが見つかりません: {path}")
        return {}

    try:
        from modules.preprocessing._return_processor import ReturnProcessor

        processor = ReturnProcessor(str(path))
        return {"harai_df": processor.preprocessed_data}
    except Exception as e:
        print(f"警告: 払戻データ読み込みエラー: {e}")
        return {}


def run_rl_phase(
    args: argparse.Namespace,
    df_with_scores: pd.DataFrame,
    return_table: dict,
    eff_config: Dict,
) -> Dict:
    """
    Phase 2: 強化学習

    LightGBMの予測スコアを特徴量として使用して強化学習を実行します。
    全7馬券種とWIN5に対応。

    Args:
        args: コマンドライン引数
        df_with_scores: LightGBMスコアを含むDataFrame
        return_table: 払戻データ
        eff_config: 実効設定

    Returns:
        Dict: 各馬券種の学習結果
    """
    print("\n" + "=" * 60)
    print("Phase 2: 強化学習")
    print("=" * 60)

    from modules.rl import RLTrainer

    # スコア列の確認
    score_cols = [c for c in df_with_scores.columns if c.startswith("lgbm_score_")]
    if not score_cols:
        if "proba" in df_with_scores.columns:
            df_with_scores["lgbm_score_win"] = df_with_scores["proba"]
            score_cols = ["lgbm_score_win"]
        else:
            raise ValueError(
                "LightGBMのスコア列が見つかりません。--skip-lgbmを使用する場合は--preds-pathを指定してください。"
            )

    print(f"使用するスコア列: {score_cols}")

    # 主要なスコア列を'score'として設定
    primary_score = score_cols[0]
    df_with_scores["score"] = df_with_scores[primary_score]

    # データ分割
    train_df = split_by_years(df_with_scores, eff_config["train_years"])
    valid_df = split_by_years(df_with_scores, eff_config["valid_years"])
    test_df = split_by_years(df_with_scores, eff_config["test_years"])

    print("\n強化学習データ分割:")
    print(f"  学習: {len(train_df)} records (Years: {eff_config['train_years']})")
    print(f"  検証: {len(valid_df)} records (Years: {eff_config['valid_years']})")
    print(f"  テスト: {len(test_df)} records (Years: {eff_config['test_years']})")

    if len(train_df) == 0:
        raise ValueError("学習データが空です")

    # 券種の決定
    bet_types = eff_config["bet_types"]
    print(f"\n対象券種: {bet_types}")

    # RL出力ディレクトリ
    rl_out_dir = os.path.join(args.out_dir, "rl")

    # 設定表示
    print("\n強化学習設定:")
    print(f"  エピソード数: {eff_config['rl_num_episodes']}")
    print(f"  学習率: {eff_config['rl_learning_rate']}")
    print(f"  割引率: {eff_config['rl_gamma']}")
    print(
        f"  ε範囲: {eff_config['rl_epsilon_start']} -> {eff_config['rl_epsilon_end']}"
    )
    print(f"  隠れ層: {eff_config['rl_hidden_dims']}")
    print(f"  出力先: {rl_out_dir}")

    # 学習実行
    results = {}

    # GPUメモリクリア用の関数をインポート
    from modules.rl import clear_cuda_memory

    for bet_type in bet_types:
        # 券種ごとにGPUメモリをクリア
        clear_cuda_memory()

        if bet_type == "win5":
            print(f"\n{'=' * 60}")
            print("  WIN5 の学習")
            print(f"{'=' * 60}")
            # WIN5は特殊な処理が必要なため、別途実装
            # 現時点ではスキップして警告を出す
            print("情報: WIN5の強化学習は現在開発中です。スキップします。")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {bet_type.upper()} の強化学習を開始")
        print(f"{'=' * 60}")

        try:
            trainer = RLTrainer(
                train_score_table=train_df,
                valid_score_table=valid_df if len(valid_df) > 0 else None,
                return_table=return_table,
                bet_amount=eff_config["rl_bet_amount"],
                max_horses=18,
                bet_type=bet_type,
                model_save_dir=rl_out_dir,
                hidden_dims=eff_config["rl_hidden_dims"],
                learning_rate=eff_config["rl_learning_rate"],
                gamma=eff_config["rl_gamma"],
                epsilon_start=eff_config["rl_epsilon_start"],
                epsilon_end=eff_config["rl_epsilon_end"],
                epsilon_decay=eff_config["rl_epsilon_decay"],
                buffer_size=eff_config["rl_buffer_size"],
                batch_size=eff_config["rl_batch_size"],
                target_update_freq=eff_config["rl_target_update_freq"],
                device=eff_config["device"],
            )

            history = trainer.train(
                num_episodes=eff_config["rl_num_episodes"],
                eval_interval=eff_config["rl_eval_interval"],
                save_interval=eff_config["rl_save_interval"],
                verbose=args.verbose or True,
            )

            results[bet_type] = history

        except Exception as e:
            print(f"\n❌ {bet_type.upper()} の学習中にエラーが発生しました: {e}")
            print("次の券種に進みます...")
            clear_cuda_memory()
            results[bet_type] = {"error": str(e)}
            continue

    return results

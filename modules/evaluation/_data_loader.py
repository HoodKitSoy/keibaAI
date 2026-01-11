"""
データローダーモジュール

モデル・予測データ・払戻データの読み込みを行います。

パス構成:
- LightGBMモデル・予測: models/lightgbm/
- 強化学習モデル: models/rl/{bet_type}/
- 旧パス（後方互換）: models/
"""

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import pandas as pd

import config
from modules import preprocessing, training
from modules.core import DB_DIR_STR, RECORD_KEY_COLS
from modules.training._feature_engineering import coerce_numeric_or_categorical

from ._utils import load_preds_score_table


def generate_predictions_from_model(
    input_file: str,
    test_years: List[int],
    task_list: List[str],
    model_dir: str,
) -> Dict[str, pd.DataFrame]:
    """
    前処理済みデータを読み込み、学習済みモデルで予測を生成する

    Args:
        input_file: 前処理済みファイルパス (例: data/tmp/db_preprocessed_2016_2025.parquet)
        test_years: テスト対象年のリスト (例: [2024, 2025])
        task_list: タスクリスト (例: ["win", "top3", "rank"])
        model_dir: モデルディレクトリ (例: models/lightgbm)

    Returns:
        Dict[str, pd.DataFrame]: タスク名をキーとしたスコアテーブル辞書
    """
    print("\n前処理済みデータから予測を生成します...")
    print(f"  入力ファイル: {input_file}")
    print(f"  テスト対象年: {test_years}")
    print(f"  タスク: {task_list}")

    # データ読み込み
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    df = pd.read_parquet(input_file)
    print(f"  全データ: {len(df)} 行")

    # Year列でフィルタリング
    if "Year" not in df.columns:
        raise ValueError("データに'Year'列がありません")

    # Year列を整数に変換
    df["Year"] = df["Year"].astype(str).str[:4].astype(int)
    test_df = df[df["Year"].isin(test_years)].copy()
    print(f"  テストデータ: {len(test_df)} 行 (年: {test_years})")

    if len(test_df) == 0:
        raise ValueError(f"テスト対象年 {test_years} のデータが見つかりません")

    score_tables: Dict[str, pd.DataFrame] = {}
    model_path = Path(model_dir)

    for task in task_list:
        print(f"\n  --- タスク: {task} ---")

        # モデルファイルを検索
        model_file = model_path / f"lgbm_{task}.model"
        if not model_file.exists():
            print(f"    モデルファイルが見つかりません: {model_file}")
            continue

        # 特徴量情報を読み込み
        feature_info_file = model_path / f"feature_info_{task}.json"
        if not feature_info_file.exists():
            print(f"    特徴量情報ファイルが見つかりません: {feature_info_file}")
            continue

        with open(feature_info_file, "r", encoding="utf-8") as f:
            feature_info = json.load(f)

        feature_cols = feature_info.get("feature_names", [])
        categorical_cols = feature_info.get("categorical_features", [])

        print(f"    モデル: {model_file}")
        print(f"    特徴量数: {len(feature_cols)}")
        print(f"    カテゴリ列数: {len(categorical_cols)}")

        # モデル読み込み
        gbm = lgb.Booster(model_file=str(model_file))

        # 特徴量の準備
        # 必要な列が存在するかチェック
        missing_cols = [c for c in feature_cols if c not in test_df.columns]
        if missing_cols:
            print(f"    警告: 不足している特徴量があります: {missing_cols[:5]}...")
            # 不足列を0で埋める
            for col in missing_cols:
                test_df[col] = 0

        # 型変換（train.pyと同じロジックを使用）
        X_test, _ = coerce_numeric_or_categorical(test_df, feature_cols)
        X_test = X_test[feature_cols].copy()

        # 予測実行
        pred = gbm.predict(X_test)
        print(f"    予測完了: {len(pred)} 件")

        # スコアテーブルの作成
        key_cols_present = [k for k in RECORD_KEY_COLS if k in test_df.columns]
        preds_df = test_df[key_cols_present].copy()

        if task in ("win", "top3"):
            preds_df["proba"] = pred
            preds_df["pred_label"] = (pred >= 0.5).astype(int)
            # スコアテーブル形式に変換
            preds_df["score"] = pred
        else:
            # rank (multiclass)
            for i in range(pred.shape[1]):
                preds_df[f"p{i + 1}"] = pred[:, i]
            preds_df["pred_label"] = pred.argmax(axis=1) + 1
            # スコアテーブル形式: 1位確率をスコアとして使用
            preds_df["score"] = pred[:, 0] if pred.shape[1] > 0 else 0

        # Umaban列の追加（スコアテーブルで必要）
        if "Umaban" in test_df.columns and "Umaban" not in preds_df.columns:
            preds_df["Umaban"] = test_df["Umaban"].values

        score_tables[task] = preds_df
        print(f"    スコアテーブル生成完了: {len(preds_df)} 行")

    return score_tables


def load_preds_for_task(
    task: str, preds_dir: str, model_save_dir: str
) -> Optional[pd.DataFrame]:
    """
    指定タスクの予測結果を読み込む

    Args:
        task: タスク名（win, top3, rank）
        preds_dir: 明示的に指定されたpredsディレクトリ
        model_save_dir: モデル保存ディレクトリ

    Returns:
        Optional[pd.DataFrame]: スコアテーブル（見つからない場合はNone）
    """
    # 検索順序: 1) 指定ディレクトリ → 2) models/lightgbm/ → 3) models/直下
    search_paths = []

    if preds_dir and os.path.isdir(preds_dir):
        search_paths.append(preds_dir)

    lightgbm_dir = os.path.join(model_save_dir, "lightgbm")
    if os.path.isdir(lightgbm_dir):
        search_paths.append(lightgbm_dir)

    search_paths.append(model_save_dir)

    for search_dir in search_paths:
        # glob で再帰探索
        cand = glob.glob(
            os.path.join(search_dir, f"**/preds_{task}.parquet"), recursive=True
        )
        if not cand:
            # 直下も試す
            direct = os.path.join(search_dir, f"preds_{task}.parquet")
            if os.path.exists(direct):
                cand = [direct]
        if cand:
            preds_path = sorted(cand)[-1]
            print(f"  {task}: {preds_path}")
            return load_preds_score_table(preds_path, task)

    return None


def load_model_and_data(args):
    """
    モデルまたは予測データからスコアを作る準備と払戻の読み込み

    Args:
        args: コマンドライン引数

    Returns:
        tuple: (keiba_ai, test_data, return_processor, score_tables)
            - keiba_ai: KeibaAIモデル（pickle使用時のみ、それ以外はNone）
            - test_data: テストデータ（pickle使用時のみ、それ以外はNone）
            - return_processor: 払戻処理オブジェクト
            - score_tables: タスク名をキーとした予測スコアテーブルの辞書

    Note:
        1. 優先順位: preds_*.parquet → pickle モデル
        2. preds使用時はtrain.pyの成果物を直接利用
        3. pickle使用時は旧形式のモデルを読み込んで予測
        4. 払戻データはJRA-DB形式（n_harai.parquet）を使用
        5. モデルパス検索順: models/lightgbm/ → models/（後方互換）
        6. 複数タスク（win, top3, rank）に対応
    """
    print("モデル/予測とデータの読み込みを開始します...")

    keiba_ai = None
    test_data = None

    # タスクリストを取得（task_listがあればそれを、なければtasksから分割、それもなければ["win"]）
    if hasattr(args, "task_list") and args.task_list:
        task_list = args.task_list
    elif hasattr(args, "tasks") and args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_list = ["win"]
    score_tables: Dict[str, pd.DataFrame] = {}

    # 0) TEST_REGENERATE_PREDS=True の場合は、前処理済みデータから予測を再生成
    regenerate_preds = getattr(config, "TEST_REGENERATE_PREDS", False)
    test_input_file = getattr(config, "TEST_INPUT_FILE", "")

    if regenerate_preds:
        if not test_input_file:
            print(
                "\n警告: TEST_REGENERATE_PREDS=true ですが TEST_INPUT_FILE が未設定です"
            )
            print("既存のpreds_*.parquetを使用します")
        elif not os.path.exists(test_input_file):
            print(f"\n警告: TEST_INPUT_FILE が見つかりません: {test_input_file}")
            print("既存のpreds_*.parquetを使用します")
        else:
            print("\nTEST_REGENERATE_PREDS=true: 予測を再生成します")
            print(f"  入力ファイル: {test_input_file}")
            test_years = getattr(config, "TEST_YEARS", [2024])
            model_dir = os.path.join(config.MODEL_SAVE_DIR, "lightgbm")
            score_tables = generate_predictions_from_model(
                input_file=test_input_file,
                test_years=test_years,
                task_list=task_list,
                model_dir=model_dir,
            )
            if score_tables:
                print(f"\n予測生成完了: {list(score_tables.keys())}")
            else:
                print("\n予測生成に失敗しました")

    if not score_tables:
        # 1) 各タスクの preds からスコアを読み込む
        print("\n既存の予測結果ファイルを検索中...")
        for task in task_list:
            score_table = load_preds_for_task(
                task, args.preds_dir, config.MODEL_SAVE_DIR
            )
            if score_table is not None:
                score_tables[task] = score_table

        if score_tables:
            print(f"\n読み込み完了: {list(score_tables.keys())}")
        else:
            print("\n予測結果ファイルが見つかりません")

    # 2) 旧 pickle モデルでスコアを作る経路（後方互換）
    if not score_tables:
        if not args.model_path:
            # 旧モデル自動探索
            model_base_dir = config.MODEL_SAVE_DIR
            available_dates = sorted(
                [
                    d
                    for d in os.listdir(model_base_dir)
                    if os.path.isdir(os.path.join(model_base_dir, d))
                ],
                reverse=True,
            )
            if not available_dates:
                raise ValueError(
                    "preds も pickle モデルも見つかりません。--preds-dir または --model-path を指定してください。"
                )
            latest_date_dir = os.path.join(model_base_dir, available_dates[0])
            available_models = [
                f for f in os.listdir(latest_date_dir) if f.endswith(".pickle")
            ]
            if not available_models:
                raise ValueError(
                    "pickle モデルが見つかりません。--preds-dir を指定するのが簡単です。"
                )
            model_path = os.path.join(latest_date_dir, available_models[0])
        else:
            model_path = args.model_path
        if not os.path.exists(model_path):
            raise ValueError(f"指定されたモデルパスが存在しません: {model_path}")

        print(f"pickle モデルをロード中: {model_path}")
        keiba_ai = training.KeibaAIFactory.load(model_path)

        # テストデータ
        if args.test_data_path:
            if not os.path.exists(args.test_data_path):
                raise ValueError(
                    f"指定されたテストデータパスが存在しません: {args.test_data_path}"
                )
            print(f"外部テストデータをロード中: {args.test_data_path}")
            test_data = pd.read_pickle(args.test_data_path)
        else:
            print("モデル内部のテストデータを使用します")
            test_data = keiba_ai.datasets.test_data

        print(f"テストデータサイズ: {test_data.shape[0]}行 × {test_data.shape[1]}列")

    # 払戻金情報のロード（JRA-DB Parquet 既定）
    print("\n払戻金情報をロード中...")
    # 既定: data/DB/n_harai.parquet を内部で自動選択
    return_processor = preprocessing.ReturnProcessor(
        filepath=os.path.join(DB_DIR_STR, "n_harai.parquet")
    )

    return keiba_ai, test_data, return_processor, score_tables

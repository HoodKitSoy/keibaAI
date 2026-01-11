"""
予測実行モジュール

学習済みモデルを使って予測を行います。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import pandas as pd

from modules.core import RACE_KEY_COLS, RECORD_KEY_COLS


def load_feature_info(
    models_dir: Path, task: str
) -> tuple[list[str] | None, list[str] | None]:
    """
    特徴量情報を読み込む

    Args:
        models_dir: モデルディレクトリ
        task: タスク名

    Returns:
        tuple[list[str] | None, list[str] | None]: (特徴量名リスト, カテゴリ列名リスト)

    Note:
        検索順序: models_dir/lightgbm/ → models_dir/
    """
    # 文字列の場合はPathに変換
    if isinstance(models_dir, str):
        models_dir = Path(models_dir)

    # models_dir/lightgbm/ を優先
    lightgbm_dir = models_dir / "lightgbm"
    if lightgbm_dir.exists():
        info_path = lightgbm_dir / f"feature_info_{task}.json"
        if info_path.exists():
            return _load_feature_info_from_path(info_path)

    # models_dir 直下をフォールバック
    info_path = models_dir / f"feature_info_{task}.json"
    if not info_path.exists():
        logging.warning(f"特徴量情報ファイルが見つかりません: {info_path}")
        return None, None

    return _load_feature_info_from_path(info_path)


def _load_feature_info_from_path(
    info_path: Path,
) -> tuple[list[str] | None, list[str] | None]:
    """
    パスから特徴量情報を読み込む（内部用）
    """
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        feature_names = info.get("feature_names", [])
        categorical_features = info.get("categorical_features", [])
        logging.info(f"特徴量情報読み込み: {info_path}")
        return feature_names, categorical_features
    except Exception as e:
        logging.error(f"特徴量情報読み込みエラー: {e}")
        return None, None


def load_model(
    models_dir: Path, task: str
) -> tuple[lgb.Booster | None, list[str] | None]:
    """
    学習済みモデルを読み込む

    Args:
        models_dir: モデルディレクトリ
        task: タスク名（win, top3, rank）

    Returns:
        tuple[lgb.Booster | None, list[str] | None]:
            (読み込まれたモデル, モデルの特徴量名リスト)
            ファイルがない場合は(None, None)

    Note:
        検索順序: models_dir/lightgbm/ → models_dir/
    """
    # 文字列の場合はPathに変換
    if isinstance(models_dir, str):
        models_dir = Path(models_dir)

    # models_dir/lightgbm/ を優先
    lightgbm_dir = models_dir / "lightgbm"
    if lightgbm_dir.exists():
        model_path = lightgbm_dir / f"lgbm_{task}.model"
        if model_path.exists():
            return _load_model_from_path(model_path)

    # models_dir 直下をフォールバック
    model_path = models_dir / f"lgbm_{task}.model"
    if not model_path.exists():
        logging.warning(f"モデルファイルが見つかりません: {model_path}")
        return None, None

    return _load_model_from_path(model_path)


def _load_model_from_path(
    model_path: Path,
) -> tuple[lgb.Booster | None, list[str] | None]:
    """
    パスからモデルを読み込む（内部用）
    """
    try:
        model = lgb.Booster(model_file=str(model_path))
        feature_names = model.feature_name()
        logging.info(f"モデル読み込み成功: {model_path}")
        logging.info(f"  特徴量数: {len(feature_names)}")
        return model, feature_names
    except Exception as e:
        logging.error(f"モデル読み込みエラー ({model_path}): {e}")
        return None, None


def predict_with_models(
    df: pd.DataFrame, models_dir: Path, tasks: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    複数タスクの予測を実行

    Args:
        df: 予測対象データ
        models_dir: モデルディレクトリ
        tasks: タスクリスト

    Returns:
        Dict[str, pd.DataFrame]: タスク名をキーとした予測結果のDataFrameの辞書
    """
    results = {}

    for task in tasks:
        logging.info(f"\n--- 予測開始: {task} ---")

        # 特徴量情報とモデル読み込み
        feature_names, categorical_features = load_feature_info(models_dir, task)
        model, model_features = load_model(models_dir, task)

        if model is None or model_features is None:
            logging.warning(f"タスク {task} をスキップします")
            continue

        # 特徴量情報がない場合はモデルから取得した特徴量名を使用
        if feature_names is None:
            logging.warning("特徴量情報ファイルがないため、モデルから特徴量名を使用")
            feature_names = model_features
            categorical_features = []

        # 予測実行
        try:
            # モデルの特徴量に合わせてデータを準備
            missing_features = [f for f in feature_names if f not in df.columns]
            if missing_features:
                logging.warning(
                    f"不足している特徴量（0埋め）: {len(missing_features)}個"
                )
                if len(missing_features) <= 5:
                    logging.warning(f"  {missing_features}")
                for feat in missing_features:
                    df[feat] = 0

            # モデルが期待する順序で特徴量を選択（copy()で警告回避）
            X = df[feature_names].copy()

            # 学習時と同じ型変換処理を適用
            # coerce_numeric_or_categorical を使用
            from modules.training import coerce_numeric_or_categorical

            X_converted, detected_categorical = coerce_numeric_or_categorical(
                X, feature_names
            )

            # 学習時のカテゴリ列情報を使用（feature_info_*.jsonから取得したもの）
            # 検出されたカテゴリ列ではなく、学習時のカテゴリ列を使う
            X = X_converted

            # 学習時のカテゴリカル列を正しくcategory型に変換
            # （LightGBMはカテゴリ型の列名リストが一致している必要がある）
            for col in categorical_features:
                if col in X.columns:
                    X[col] = X[col].astype("category")

            # 学習時にカテゴリカルでなかった列がcategory型になっていたら数値に戻す
            for col in X.columns:
                if col not in categorical_features and X[col].dtype.name == "category":
                    X[col] = pd.to_numeric(X[col].cat.codes, errors="coerce").fillna(0)

            logging.info(f"  予測用特徴量: {X.shape[1]} 列")
            logging.info(f"  学習時カテゴリ列: {len(categorical_features)} 列")
            logging.info(f"  検出カテゴリ列: {len(detected_categorical)} 列")

            # 予測
            if task in ["win", "top3"]:
                # 二値分類
                preds = model.predict(X)
                pred_df = pd.DataFrame(
                    {
                        **{k: df[k] for k in RECORD_KEY_COLS if k in df.columns},
                        "proba": preds,
                        "pred_label": (preds >= 0.5).astype(int),
                    }
                )
            else:
                # 多クラス分類（rank）
                preds = model.predict(X)
                num_class = preds.shape[1]

                pred_dict = {**{k: df[k] for k in RECORD_KEY_COLS if k in df.columns}}

                # 各クラスの確率
                for i in range(num_class):
                    pred_dict[f"p{i + 1}"] = preds[:, i]

                # 予測ラベル（最も確率の高いクラス + 1）
                pred_dict["pred_label"] = preds.argmax(axis=1) + 1

                pred_df = pd.DataFrame(pred_dict)

            logging.info(f"予測完了: {len(pred_df)} レコード")
            results[task] = pred_df

        except Exception as e:
            logging.error(f"予測エラー ({task}): {e}")
            import traceback

            traceback.print_exc()

    return results


def format_predictions_for_display(
    predictions: Dict[str, pd.DataFrame], date_str: str
) -> pd.DataFrame:
    """
    予測結果を表示用に整形

    Args:
        predictions: タスク別の予測結果
        date_str: 日付文字列

    Returns:
        pd.DataFrame: 表示用に整形されたDataFrame
    """
    if not predictions:
        return pd.DataFrame()

    # winタスクの結果をベースにする
    if "win" in predictions:
        base = predictions["win"].copy()
        base = base.rename(columns={"proba": "win_proba", "pred_label": "win_pred"})

        # top3を追加
        if "top3" in predictions:
            top3 = predictions["top3"][RECORD_KEY_COLS + ["proba", "pred_label"]]
            top3 = top3.rename(
                columns={"proba": "top3_proba", "pred_label": "top3_pred"}
            )
            base = base.merge(top3, on=RECORD_KEY_COLS, how="left")

        # rankを追加
        if "rank" in predictions:
            rank = predictions["rank"][RECORD_KEY_COLS + ["pred_label"]]
            rank = rank.rename(columns={"pred_label": "rank_pred"})
            base = base.merge(rank, on=RECORD_KEY_COLS, how="left")

        # ソート（レース順、win確率降順）
        sort_cols = RACE_KEY_COLS.copy()
        if "win_proba" in base.columns:
            base = base.sort_values(
                sort_cols + ["win_proba"], ascending=[True] * len(sort_cols) + [False]
            )

        return base

    # winがない場合は最初のタスクをベースにする
    first_task = list(predictions.keys())[0]
    return predictions[first_task]


def save_predictions(
    predictions: Dict[str, pd.DataFrame], output_dir: Path, date_str: str
) -> None:
    """
    予測結果を保存

    Args:
        predictions: タスク別の予測結果
        output_dir: 出力ディレクトリ
        date_str: 日付文字列
    """
    # 文字列の場合はPathに変換
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for task, pred_df in predictions.items():
        # Parquet形式で保存
        parquet_path = output_dir / f"prediction_{date_str}_{task}.parquet"
        pred_df.to_parquet(parquet_path, index=False)
        logging.info(f"保存: {parquet_path}")

        # CSV形式でも保存（確認用）
        csv_path = output_dir / f"prediction_{date_str}_{task}.csv"
        pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logging.info(f"保存: {csv_path}")

    # 統合版も保存
    display_df = format_predictions_for_display(predictions, date_str)
    if len(display_df) > 0:
        combined_path = output_dir / f"prediction_{date_str}_combined.csv"
        display_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
        logging.info(f"保存（統合版）: {combined_path}")

    # サマリー情報をJSON形式で保存
    summary = {
        "date": date_str,
        "tasks": list(predictions.keys()),
        "record_counts": {task: len(df) for task, df in predictions.items()},
    }

    summary_path = output_dir / f"prediction_{date_str}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"保存（サマリー）: {summary_path}")

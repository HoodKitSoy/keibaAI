"""
データ処理モジュール（予測用）

当日データを前処理して予測用のフォーマットに変換します。
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd

from modules.core import FINAL_COLUMN_ORDER, RACE_KEY_COLS


def preprocess_today_data(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    当日データを前処理

    preprocessing.pyと同じロジックで結合・整形を行います。

    Args:
        dfs: テーブル名をキーとしたDataFrameの辞書

    Returns:
        pd.DataFrame: 前処理済みデータ
    """
    # s_uma_raceをベースにする
    base = dfs["s_uma_race"].copy()

    # s_raceと結合
    s_race = dfs["s_race"].copy()
    base = base.merge(s_race, on=RACE_KEY_COLS, how="left", suffixes=("", "_race"))

    # s_umaと結合
    s_uma = dfs["s_uma"].copy()
    if "KettoNum" in base.columns and "KettoNum" in s_uma.columns:
        base = base.merge(s_uma, on="KettoNum", how="left", suffixes=("", "_uma"))

    logging.info(f"主要結合完了: {base.shape}")

    # 調教データを付与
    base = attach_training_data(base, dfs["n_hanro"], dfs["n_chip"])

    return base


def attach_training_data(
    base: pd.DataFrame, hanro: pd.DataFrame, chip: pd.DataFrame
) -> pd.DataFrame:
    """
    調教データを付与

    preprocessing.pyと同じロジックでレース直前の調教データを結合します。

    Args:
        base: ベースとなるDataFrame
        hanro: 坂路調教データ
        chip: ウッドチップ調教データ

    Returns:
        pd.DataFrame: 調教データが付与されたDataFrame
    """
    # RaceDateを生成
    if "Year" in base.columns and "MonthDay" in base.columns:
        base["RaceDate"] = base["Year"].astype(str) + base["MonthDay"].astype(
            str
        ).str.zfill(4)
        base["RaceDate"] = pd.to_numeric(base["RaceDate"], errors="coerce")

    # 坂路調教
    if "KettoNum" in base.columns and len(hanro) > 0:
        hanro_copy = hanro.copy()
        if "ChokyoDate" in hanro_copy.columns:
            hanro_copy["ChokyoDate"] = pd.to_numeric(
                hanro_copy["ChokyoDate"], errors="coerce"
            )

        # レース直前の調教を取得
        base = _attach_latest_training(base, hanro_copy, suffix="")
        logging.info(f"坂路調教結合後: {base.shape}")

    # ウッドチップ調教
    if "KettoNum" in base.columns and len(chip) > 0:
        chip_copy = chip.copy()
        if "ChokyoDate" in chip_copy.columns:
            chip_copy["ChokyoDate"] = pd.to_numeric(
                chip_copy["ChokyoDate"], errors="coerce"
            )

        # レース直前の調教を取得
        base = _attach_latest_training(base, chip_copy, suffix="_chip")
        logging.info(f"ウッドチップ調教結合後: {base.shape}")

    return base


def _attach_latest_training(
    base: pd.DataFrame, training: pd.DataFrame, suffix: str = ""
) -> pd.DataFrame:
    """
    レース直前の最新調教データを結合

    Args:
        base: ベースDataFrame
        training: 調教データ
        suffix: 列名のサフィックス（ウッドチップ用は"_chip"）

    Returns:
        pd.DataFrame: 調教データが結合されたDataFrame
    """
    if "RaceDate" not in base.columns or "KettoNum" not in base.columns:
        return base

    if "KettoNum" not in training.columns or "ChokyoDate" not in training.columns:
        return base

    # 馬ごとにグループ化して処理
    result_rows = []

    for _, row in base.iterrows():
        ketto_num = row.get("KettoNum")
        race_date = row.get("RaceDate")

        if pd.isna(ketto_num) or pd.isna(race_date):
            result_rows.append(row)
            continue

        # 該当馬の調教データを抽出
        mask = (training["KettoNum"] == ketto_num) & (
            training["ChokyoDate"] < race_date
        )
        horse_training = training[mask]

        if len(horse_training) == 0:
            result_rows.append(row)
            continue

        # 最新の調教データを取得
        latest = horse_training.sort_values("ChokyoDate", ascending=False).iloc[0]

        # 結合
        row_dict = row.to_dict()
        for col in latest.index:
            if col not in ["KettoNum", "ChokyoDate"]:
                new_col = f"{col}{suffix}" if suffix else col
                row_dict[new_col] = latest[col]

        result_rows.append(row_dict)

    result = pd.DataFrame(result_rows)
    return result


def align_to_training_format(
    df: pd.DataFrame, add_dummy_target: bool = True
) -> Tuple[pd.DataFrame, list]:
    """
    学習時のフォーマットに合わせる

    Args:
        df: 前処理済みDataFrame
        add_dummy_target: ダミーのターゲット列を追加するか

    Returns:
        Tuple[pd.DataFrame, list]: 整形済みDataFrameと欠損列のリスト
    """
    # FINAL_COLUMN_ORDERに合わせる
    missing_cols = []
    for col in FINAL_COLUMN_ORDER:
        if col not in df.columns:
            if col == "KakuteiJyuni" and add_dummy_target:
                # ダミーのターゲット列（予測時は不要だが、特徴量選択で必要）
                df[col] = 0
            else:
                df[col] = None  # 欠損として扱う
                missing_cols.append(col)

    # 列順を整える
    available_cols = [c for c in FINAL_COLUMN_ORDER if c in df.columns]
    df = df[available_cols]

    # データ型を統一（数値列は数値に、その他は文字列に）
    for col in df.columns:
        if col in RACE_KEY_COLS or col in ["Umaban", "KettoNum"]:
            continue  # キー列はそのまま

        # 数値変換を試みる
        try:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if numeric_series.notna().sum() / len(df) > 0.7:  # 70%以上が数値化成功
                df[col] = numeric_series.fillna(0)
            else:
                df[col] = df[col].astype(str).fillna("0")
        except Exception:
            df[col] = df[col].astype(str).fillna("0")

    if missing_cols:
        logging.warning(f"欠損列: {missing_cols}")

    return df, missing_cols

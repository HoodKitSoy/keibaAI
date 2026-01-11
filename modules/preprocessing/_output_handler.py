"""
出力処理モジュール

データの整形、ソート、欠損値処理、およびファイル出力を行う機能を提供します。
"""

import datetime
import logging
import os

import pandas as pd

from ._config import FINAL_COLUMN_ORDER, TMP_DIR


def reorder_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    列の並び替え、欠損値埋め、ソートを実行

    Args:
        df (pd.DataFrame): 対象のデータフレーム

    Returns:
        pd.DataFrame: 整形されたデータフレーム

    Note:
        - 不足列は "0" で埋める
        - 欠損値: 文字列は "0"、数値は 0
        - ソートキー: Year, MonthDay, JyoCD, RaceNum, Umaban
    """
    # 不足列をゼロ埋めで追加（型は文字列扱い）
    for col in FINAL_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = "0"

    # 指定順に並び替え（明示コピーでSettingWithCopyWarning回避）
    df = df[FINAL_COLUMN_ORDER].copy()

    # 欠損埋め：文字列は "0"、数値は 0
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    num_cols = [c for c in df.columns if c not in obj_cols]
    df.loc[:, obj_cols] = df[obj_cols].fillna("0")
    df.loc[:, num_cols] = df[num_cols].fillna(0)

    # ソート
    sort_keys = ["Year", "MonthDay", "JyoCD", "RaceNum", "Umaban"]
    for k in sort_keys:
        if k in df.columns:
            df.loc[:, k] = (
                df[k].astype(str).str.zfill(4 if k in ["Year", "MonthDay"] else 2)
            )
    df = df.sort_values([k for k in sort_keys if k in df.columns]).reset_index(
        drop=True
    )
    return df


def save_outputs(df: pd.DataFrame, out_basename: str = "merged_for_model") -> None:
    """
    CSVとParquet形式でデータを保存

    Args:
        df (pd.DataFrame): 保存するデータフレーム
        out_basename (str): 出力ファイル名のベース名（拡張子なし）

    Note:
        - PermissionError時はタイムスタンプ付きファイル名にフォールバック
        - 保存先: TMP_DIR (data/tmp)
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    csv_path = os.path.join(TMP_DIR, f"{out_basename}.csv")
    pq_path = os.path.join(TMP_DIR, f"{out_basename}.parquet")

    # CSV 保存（PermissionError時のフォールバック）
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_csv = os.path.join(TMP_DIR, f"{out_basename}_{ts}.csv")
        logging.warning(
            "CSV を開いている可能性があります。フォールバック保存: %s", alt_csv
        )
        df.to_csv(alt_csv, index=False, encoding="utf-8-sig")
        csv_path = alt_csv

    # Parquet 保存（同様のフォールバック）
    try:
        df.to_parquet(pq_path, index=False)
    except PermissionError:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_pq = os.path.join(TMP_DIR, f"{out_basename}_{ts}.parquet")
        logging.warning(
            "Parquet を開いている可能性があります。フォールバック保存: %s", alt_pq
        )
        df.to_parquet(alt_pq, index=False)
        pq_path = alt_pq

    logging.info("保存完了: %s, %s", csv_path, pq_path)

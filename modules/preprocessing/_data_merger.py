"""
データ結合モジュール

複数のテーブルを結合し、フィルタリングを行う機能を提供します。
"""

import logging
from typing import Dict, List

import pandas as pd

from ._config import TARGET_JYO_CDS, TARGET_YEARS


def _zfill_str(s: pd.Series, width: int) -> pd.Series:
    """
    文字列をゼロ埋めする

    Args:
        s (pd.Series): 対象のシリーズ
        width (int): ゼロ埋め後の桁数

    Returns:
        pd.Series: ゼロ埋めされたシリーズ
    """
    return s.astype(str).str.zfill(width)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    キー列の型・ゼロ埋めを標準化

    存在しない列はそのままスキップします。

    Args:
        df (pd.DataFrame): 対象のデータフレーム

    Returns:
        pd.DataFrame: 標準化されたデータフレーム
    """
    for col, w in [
        ("Year", 4),
        ("MonthDay", 4),
        ("JyoCD", 2),
        ("Kaiji", 2),
        ("Nichiji", 2),
        ("RaceNum", 2),
    ]:
        if col in df.columns:
            df[col] = _zfill_str(df[col], w)
    if "Umaban" in df.columns:
        df["Umaban"] = _zfill_str(df["Umaban"], 2)
    if "Wakuban" in df.columns:
        df["Wakuban"] = _zfill_str(df["Wakuban"], 1)
    if "KettoNum" in df.columns:
        df["KettoNum"] = df["KettoNum"].astype(str).str.strip()
    return df


def filter_frames(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    対象年・競馬場・DataKubun除外のフィルタを適用

    Args:
        dfs (Dict[str, pd.DataFrame]): テーブル名をキーとしたデータフレームの辞書

    Returns:
        Dict[str, pd.DataFrame]: フィルタ適用後のデータフレームの辞書

    Note:
        - DataKubun の除外は n_uma_race と n_race のみに適用
        - 年・場の絞り込みも n_uma_race と n_race のみに適用
    """

    def _filter(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = normalize_keys(df)
        # DataKubun の除外は n_uma_race と n_race のみに適用
        if name in ("n_uma_race", "n_race") and "DataKubun" in df.columns:
            df = df[~df["DataKubun"].astype(str).isin(["0", "9"])]
        # 年・場の絞り込みは n_uma_race と n_race のみに適用
        if name in ("n_uma_race", "n_race"):
            if "Year" in df.columns:
                df = df[df["Year"].isin(TARGET_YEARS)]
            if "JyoCD" in df.columns:
                df = df[df["JyoCD"].isin(TARGET_JYO_CDS)]
        logging.info("%s after filter: %s rows", name, len(df))
        return df

    return {name: _filter(df, name) for name, df in dfs.items()}


def log_merge_quality(
    left: pd.DataFrame, right: pd.DataFrame, keys: List[str], right_name: str
) -> None:
    """
    結合品質の検証ログを出力

    - 右側キーの一意性（重複数と上位例）
    - 左側に対する未マッチ件数と割合
    - 年別（Year列があれば）未マッチ分布

    Args:
        left (pd.DataFrame): 左側（基準）のデータフレーム
        right (pd.DataFrame): 右側のデータフレーム
        keys (List[str]): 結合キー
        right_name (str): 右側テーブルの名前（ログ用）
    """
    # 右側キー一意性
    for k in keys:
        if k not in right.columns:
            logging.warning("%s: 右側にキー列 %s が存在しません。", right_name, k)
    if all(k in right.columns for k in keys):
        dup_counts = right.groupby(keys).size().reset_index(name="cnt")
        dup_rows = dup_counts[dup_counts["cnt"] > 1]
        dup_n = int(dup_rows["cnt"].sum()) if not dup_rows.empty else 0
        logging.info(
            "%s: キー重複件数（行数合計）=%s, 重複キー例=%s",
            right_name,
            dup_n,
            dup_rows.head(3).to_dict(orient="records"),
        )
    else:
        logging.info("%s: キー重複確認をスキップ（キー欠落）。", right_name)

    # 左側未マッチ件数
    if all(k in left.columns for k in keys) and all(k in right.columns for k in keys):
        probe_left = left[
            keys + (["Year"] if "Year" not in keys and "Year" in left.columns else [])
        ].drop_duplicates()
        probe_right = right[keys].drop_duplicates()
        matched = probe_left.merge(probe_right, how="left", on=keys, indicator=True)
        left_only = matched[matched["_merge"] == "left_only"]
        miss_n = len(left_only)
        total = len(probe_left)
        miss_pct = (miss_n / total * 100) if total else 0.0
        logging.info(
            "%s: 左側未マッチ=%s/%s (%.2f%%)", right_name, miss_n, total, miss_pct
        )
        if "Year" in probe_left.columns:
            by_year = (
                left_only.assign(Year=left_only.get("Year", pd.Series(dtype=str)))
                .groupby("Year")
                .size()
                .sort_index()
                .to_dict()
            )
            logging.info("%s: 年別未マッチ分布=%s", right_name, by_year)
    else:
        logging.info("%s: 左側未マッチ確認をスキップ（キー欠落）。", right_name)


def merge_core(
    df_ur: pd.DataFrame, df_r: pd.DataFrame, df_u: pd.DataFrame
) -> pd.DataFrame:
    """
    UMA_RACEを基準にRACEとUMAを結合

    Args:
        df_ur (pd.DataFrame): UMA_RACEテーブル（基準）
        df_r (pd.DataFrame): RACEテーブル
        df_u (pd.DataFrame): UMAテーブル

    Returns:
        pd.DataFrame: 結合されたデータフレーム

    Note:
        複合キー（Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum）で
        RACEを結合し、その後KettoNumでUMAを結合します。
    """
    keys = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
    for col in keys + ["KettoNum"]:
        for d in (df_ur, df_r, df_u):
            if col in d.columns:
                d[col] = d[col].astype(str)

    # 結合品質ログ（RACE）
    log_merge_quality(df_ur, df_r, keys=keys, right_name="n_race")

    # RACE 結合
    missing_race_before = len(df_ur)
    merged = df_ur.merge(df_r, how="left", on=keys, suffixes=("", "_r"))
    # 追加列（RACE由来）に基づく未マッチ検出
    race_added_cols = [c for c in merged.columns if c not in df_ur.columns]
    if race_added_cols:
        unmatched = merged[race_added_cols].isna().all(axis=1).sum()
        logging.info(
            "RACE 結合（left）: %s -> %s, 未マッチ（推定）=%s",
            missing_race_before,
            len(merged),
            unmatched,
        )
    else:
        logging.info("RACE 結合（left）: %s -> %s", missing_race_before, len(merged))

    # UMA 結合
    log_merge_quality(merged, df_u, keys=["KettoNum"], right_name="n_uma")
    if "KettoNum" in df_u.columns:
        merged = merged.merge(df_u, how="left", on="KettoNum", suffixes=("", "_u"))
    else:
        logging.warning("n_uma に KettoNum が存在しません。UMA 結合をスキップします。")
    return merged

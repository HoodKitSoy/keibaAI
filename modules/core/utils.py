"""
共通ユーティリティモジュール

レースIDの生成、レース単位のグルーピングなど、
プロジェクト全体で使用するユーティリティ関数を提供します。
"""

from __future__ import annotations

import pandas as pd

from .constants import RACE_KEY_COLS


def zfill_series(s: pd.Series, width: int) -> pd.Series:
    """
    Seriesの値を指定桁数でゼロ埋めします。

    Args:
        s: 対象のSeries
        width: ゼロ埋め桁数

    Returns:
        pd.Series: ゼロ埋めされたSeries
    """
    return s.astype(str).str.zfill(width)


def build_race_id(df: pd.DataFrame) -> pd.Series:
    """
    レースキー列からrace_idを生成します。

    race_idフォーマット: YYYYMMDD-JJKKNN-RR
    例: 20241123-010212-11

    Args:
        df: レースキー列を含むデータフレーム

    Returns:
        pd.Series: race_id文字列のSeries

    Example:
        >>> df = pd.DataFrame({
        ...     "Year": [2024], "MonthDay": [1123],
        ...     "JyoCD": [1], "Kaiji": [2], "Nichiji": [12], "RaceNum": [11]
        ... })
        >>> build_race_id(df)
        0    20241123-010212-11
        dtype: object
    """
    if not set(RACE_KEY_COLS).issubset(df.columns):
        # フォールバック: indexをrace_idとみなす
        return df.index.to_series().astype(str)

    year = zfill_series(df["Year"], 4)
    month_day = zfill_series(df["MonthDay"], 4)
    jyo_cd = zfill_series(df["JyoCD"], 2)
    kaiji = zfill_series(df["Kaiji"], 2)
    nichiji = zfill_series(df["Nichiji"], 2)
    race_num = zfill_series(df["RaceNum"], 2)

    return year + month_day + "-" + jyo_cd + kaiji + nichiji + "-" + race_num


def iter_race_groups(score_table: pd.DataFrame):
    """
    スコアテーブルをレース単位で分割するジェネレータ。

    レースキー列が存在する場合はそれでグルーピングし、
    存在しない場合はindexをrace_idとみなしてグルーピングします。

    Args:
        score_table: スコアテーブル（レースキー列または race_id を含む）

    Yields:
        tuple[str, pd.DataFrame]: (race_id, グループのデータフレーム)

    Example:
        >>> for race_id, race_df in iter_race_groups(score_table):
        ...     print(f"Race {race_id}: {len(race_df)} horses")
    """
    if set(RACE_KEY_COLS).issubset(score_table.columns):
        # 6列キーでグルーピング
        for keys, df_race in score_table.groupby(
            RACE_KEY_COLS, sort=False, dropna=False
        ):
            # keysはタプル（Year, MonthDay, ...）
            if not isinstance(keys, tuple):
                # groupbyが単一キーになった場合の対応
                keys_df = df_race
            else:
                # race_idを構築
                keys_df = pd.DataFrame([dict(zip(RACE_KEY_COLS, keys))])
            race_id = build_race_id(keys_df).iloc[0]
            yield str(race_id), df_race
    else:
        # indexベースでグルーピング
        for idx, df_race in score_table.groupby(level=0, sort=False):
            yield str(idx), df_race


def add_race_id_column(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    データフレームにrace_id列を追加します。

    Args:
        df: レースキー列を含むデータフレーム
        inplace: Trueの場合は元のデータフレームを変更

    Returns:
        pd.DataFrame: race_id列が追加されたデータフレーム
    """
    if not inplace:
        df = df.copy()

    df["race_id"] = build_race_id(df)
    return df


def validate_race_keys(df: pd.DataFrame, raise_error: bool = True) -> bool:
    """
    データフレームが必要なレースキー列を持っているか検証します。

    Args:
        df: 検証対象のデータフレーム
        raise_error: Trueの場合、不足があれば例外を発生させる

    Returns:
        bool: すべてのキー列が存在すればTrue

    Raises:
        ValueError: raise_error=Trueで必須列が不足している場合
    """
    missing_cols = [col for col in RACE_KEY_COLS if col not in df.columns]

    if missing_cols:
        if raise_error:
            raise ValueError(
                f"必須のレースキー列が不足しています: {', '.join(missing_cols)}"
            )
        return False

    return True


def normalize_race_keys(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    レースキー列を正規化します（ゼロ埋め）。

    Args:
        df: レースキー列を含むデータフレーム
        inplace: Trueの場合は元のデータフレームを変更

    Returns:
        pd.DataFrame: レースキー列が正規化されたデータフレーム
    """
    if not inplace:
        df = df.copy()

    if "Year" in df.columns:
        df["Year"] = zfill_series(df["Year"], 4)
    if "MonthDay" in df.columns:
        df["MonthDay"] = zfill_series(df["MonthDay"], 4)
    if "JyoCD" in df.columns:
        df["JyoCD"] = zfill_series(df["JyoCD"], 2)
    if "Kaiji" in df.columns:
        df["Kaiji"] = zfill_series(df["Kaiji"], 2)
    if "Nichiji" in df.columns:
        df["Nichiji"] = zfill_series(df["Nichiji"], 2)
    if "RaceNum" in df.columns:
        df["RaceNum"] = zfill_series(df["RaceNum"], 2)
    if "Umaban" in df.columns:
        df["Umaban"] = zfill_series(df["Umaban"], 2)

    return df

"""
調教データ処理モジュール

坂路およびウッドチップ調教データをレース直前の最新データとして
結合する機能を提供します。
"""

import logging
from typing import List

import pandas as pd


def ensure_race_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    YearとMonthDayからRaceDate (int: YYYYMMDD) を作成
    
    Args:
        df (pd.DataFrame): 対象のデータフレーム
        
    Returns:
        pd.DataFrame: RaceDate列が追加されたデータフレーム
        
    Raises:
        KeyError: YearまたはMonthDay列が存在しない場合
    """
    if ("Year" not in df.columns) or ("MonthDay" not in df.columns):
        raise KeyError("Year または MonthDay が存在しません。RaceDate を作れません。")
    y = df["Year"].astype(str).str.zfill(4)
    md = df["MonthDay"].astype(str).str.zfill(4)
    df["RaceDate"] = (y + md).astype(int)
    return df


def latest_training_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_date_col: str,
    by: str = "KettoNum",
    left_date_col: str = "RaceDate",
) -> pd.DataFrame:
    """
    merge_asofで直前（< left_date_col）1件の調教を結合
    
    Args:
        left (pd.DataFrame): 左側（基準）のデータフレーム
        right (pd.DataFrame): 右側（調教データ）のデータフレーム
        right_date_col (str): 右側の日付列名
        by (str): グループ化キー（デフォルト: "KettoNum"）
        left_date_col (str): 左側の日付列名（デフォルト: "RaceDate"）
        
    Returns:
        pd.DataFrame: 調教データが結合されたデータフレーム
        
    Note:
        - 厳密に "< RaceDate" を満たすよう、同日(==)を除外
        - merge_asof要件: onキーが昇順にソートされている必要がある
    """
    if by not in left.columns or by not in right.columns:
        logging.warning("by=%s が左右どちらかに存在しません。調教結合をスキップ。", by)
        return left
    if left_date_col not in left.columns or right_date_col not in right.columns:
        logging.warning(
            "日付列が不足しています（%s / %s）。調教結合をスキップ。",
            left_date_col,
            right_date_col,
        )
        return left

    left_keys = left[[by, left_date_col]].copy()
    r = right[
        [by, right_date_col]
        + [c for c in right.columns if c not in [by, right_date_col]]
    ].copy()

    # 型・ソート
    left_keys[left_date_col] = pd.to_numeric(
        left_keys[left_date_col], errors="coerce"
    ).astype("Int64")
    r[right_date_col] = pd.to_numeric(r[right_date_col], errors="coerce").astype(
        "Int64"
    )
    # merge_asof要件: onキーが昇順にソート
    left_keys = left_keys.dropna(subset=[left_date_col]).sort_values(
        [left_date_col, by]
    )
    r = r.dropna(subset=[right_date_col]).sort_values([right_date_col, by])

    merged = pd.merge_asof(
        left_keys,
        r,
        left_on=left_date_col,
        right_on=right_date_col,
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    # 厳密に "< RaceDate" を満たすよう、同日(==)を除外
    mask_ok = merged[right_date_col].notna() & (
        merged[right_date_col] < merged[left_date_col]
    )
    merged = merged[mask_ok]

    # 左本体と結合（by + left_date_col で一意に復元）
    join_keys = [by, left_date_col]
    out = left.merge(merged, how="left", on=join_keys)
    return out


def log_training_coverage(df: pd.DataFrame, label: str, prefer_cols: List[str]) -> None:
    """
    調教結合後のカバレッジ（非NA率）をログ出力
    
    Args:
        df (pd.DataFrame): 対象のデータフレーム
        label (str): ログ用のラベル
        prefer_cols (List[str]): 代表として使用する列の候補リスト
        
    Note:
        prefer_colsのうち存在する最初の列を代表として用いる
    """
    col = next((c for c in prefer_cols if c in df.columns), None)
    if not col:
        logging.info("%s: カバレッジ計測対象列が存在しません。", label)
        return
    total = len(df)
    non_na = df[col].notna().sum()
    pct = (non_na / total * 100.0) if total else 0.0
    logging.info("%s: 代表列 '%s' 非NA=%s/%s (%.2f%%)", label, col, non_na, total, pct)


def attach_hanro_and_chip(
    df: pd.DataFrame, hanro: pd.DataFrame, chip: pd.DataFrame
) -> pd.DataFrame:
    """
    坂路とウッドチップの直前最新調教データを結合
    
    Args:
        df (pd.DataFrame): ベースとなるデータフレーム
        hanro (pd.DataFrame): 坂路調教データ
        chip (pd.DataFrame): ウッドチップ調教データ
        
    Returns:
        pd.DataFrame: 調教データが結合されたデータフレーム
        
    Note:
        - ウッドチップの列には "_chip" サフィックスを付与
        - レース日前の直近1件を結合（同日は除外）
    """
    # 想定カラム名
    hanro_date_col = "ChokyoDate"
    chip_date_col = "ChokyoDate"

    # まず RaceDate を用意
    df = ensure_race_date(df)

    # 坂路: 必要列を残す（存在しない列は無視）
    hanro_cols = [
        "KettoNum",
        hanro_date_col,
        "TresenKubun",
        "HaronTime4",
        "LapTime4",
        "HaronTime3",
        "LapTime3",
        "HaronTime2",
        "LapTime2",
        "LapTime1",
    ]
    hanro = hanro[[c for c in hanro_cols if c in hanro.columns]].copy()
    if hanro_date_col in hanro.columns:
        hanro[hanro_date_col] = pd.to_numeric(
            hanro[hanro_date_col], errors="coerce"
        ).astype("Int64")
    df = latest_training_asof(df, hanro, right_date_col=hanro_date_col)
    log_training_coverage(df, label="hanro", prefer_cols=["HaronTime4", "LapTime1"])

    # CHIP: 必要列抽出 -> _chip リネーム
    chip_cols = [
        "KettoNum",
        chip_date_col,
        "TresenKubun",
        "Course",
        "BabaAround",
        "HaronTime10",
        "LapTime10",
        "HaronTime9",
        "LapTime9",
        "HaronTime8",
        "LapTime8",
        "HaronTime7",
        "LapTime7",
        "HaronTime6",
        "LapTime6",
        "HaronTime5",
        "LapTime5",
        "HaronTime4",
        "LapTime4",
        "HaronTime3",
        "LapTime3",
        "HaronTime2",
        "LapTime2",
        "LapTime1",
    ]
    chip = chip[[c for c in chip_cols if c in chip.columns]].copy()
    if len(chip.columns) > 0:
        rename_map = {
            c: f"{c}_chip" for c in chip.columns if c not in ["KettoNum", chip_date_col]
        }
        chip = chip.rename(columns=rename_map)
        # 日付列は元名のままで asof に使う
        df = latest_training_asof(
            df,
            chip,
            right_date_col=chip_date_col,
            by="KettoNum",
            left_date_col="RaceDate",
        )
        log_training_coverage(
            df, label="chip", prefer_cols=["HaronTime10_chip", "LapTime1_chip"]
        )
    else:
        logging.warning("n_chip 必須列が見つからず、結合をスキップします。")

    return df

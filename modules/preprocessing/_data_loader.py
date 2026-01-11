"""
データ読み込みモジュール（前処理用、後方互換性のため維持）

注意: このモジュールの大部分の機能は modules.core に統合されました。
新しいコードでは modules.core を直接使用してください。
"""

import glob
import logging
import os
from typing import Dict, Iterable

import pandas as pd

from modules.core import find_parquet_file, load_db_tables, read_parquet


# 後方互換性のための関数（modules.coreの関数を呼び出す）
def _find_parquet_path(db_dir: str, keywords: list[str]) -> str | None:
    """後方互換性のため維持"""
    result = find_parquet_file(db_dir, keywords)
    return str(result) if result else None


def read_parquet_or_fail(path: str) -> pd.DataFrame:
    """後方互換性のため維持"""
    return read_parquet(path)


def load_sources(db_dir: str) -> Dict[str, pd.DataFrame]:
    """
    必要テーブルを読み込む（学習用）

    ファイル名の揺れに対して緩く探索し、必要な5つのテーブルを読み込みます。

    Args:
        db_dir (str): データベースディレクトリのパス

    Returns:
        Dict[str, pd.DataFrame]: テーブル名をキーとしたデータフレームの辞書

    Raises:
        FileNotFoundError: 必要なテーブルが見つからない場合

    Note:
        必要なテーブル: n_uma_race, n_race, n_uma, n_hanro, n_chip
        この関数は modules.core.load_db_tables を使用しています。
    """
    expected = {
        "n_uma_race": ["n_uma_race", "uma_race"],
        "n_race": ["n_race", "race"],
        "n_uma": ["n_uma", "uma"],
        "n_hanro": ["n_hanro", "hanro"],
        "n_chip": ["n_chip", "chip"],
    }

    return load_db_tables(db_dir, expected)


def _find_parquet_path_strict_prefix(
    db_dir: str, prefix: str, keywords: Iterable[str], exact_match: bool = False
) -> str | None:
    """
    db_dir 配下からparquetを検索し、ファイル名がプレフィックスで始まり、指定キーワードをすべて含むファイルを返す

    Args:
        db_dir (str): 検索対象のディレクトリパス
        prefix (str): ファイル名の先頭に必要なプレフィックス（例: "s_", "n_"）
        keywords (Iterable[str]): 検索キーワード（大文字小文字無視）
        exact_match (bool): Trueの場合、プレフィックス+キーワード+.parquetの完全一致のみ

    Returns:
        str | None: 見つかったファイルパス、見つからない場合はNone
    """
    candidates = glob.glob(os.path.join(db_dir, "**", "*.parquet"), recursive=True)
    kw = [k.lower() for k in keywords]
    prefix_lower = prefix.lower()

    for p in candidates:
        name = os.path.basename(p).lower()
        # 拡張子を除いたファイル名
        name_without_ext = name.replace(".parquet", "")

        if exact_match:
            # 完全一致: prefix + keyword + .parquet
            expected_name = prefix_lower + "".join(kw)
            if name_without_ext == expected_name:
                return p
        else:
            # プレフィックスチェック + キーワードチェック
            if name.startswith(prefix_lower) and all(k in name for k in kw):
                return p
    return None


def load_today_sources(db_dir: str, target_date: str) -> Dict[str, pd.DataFrame]:
    """
    当日予想用テーブルを読み込む

    s_テーブル（s_race, s_uma_race, s_uma）と調教データ（n_hanro, n_chip）を読み込み、
    指定日付でフィルタリングします。

    Args:
        db_dir (str): データベースディレクトリのパス
        target_date (str): 対象日付 (YYYYMMDD形式)

    Returns:
        Dict[str, pd.DataFrame]: テーブル名をキーとしたデータフレームの辞書

    Raises:
        FileNotFoundError: 必要なテーブルが見つからない場合
        ValueError: 指定日付のデータが存在しない場合

    Note:
        必要なテーブル: s_uma_race, s_race, s_uma, n_hanro, n_chip
        s_テーブルは指定日付でフィルタリング、n_テーブル（調教）はそのまま使用
    """
    expected = {
        "s_uma_race": ["uma_race"],
        "s_race": ["race"],
        "s_uma": ["uma"],
        "n_hanro": ["hanro"],
        "n_chip": ["chip"],
    }

    paths: Dict[str, str] = {}
    for key, kws in expected.items():
        p = None
        # プレフィックスを取得（s_ or n_）
        prefix = key.split("_")[0] + "_"

        for kw in kws:
            # まず完全一致で検索（s_uma など短い名前用）
            p = _find_parquet_path_strict_prefix(db_dir, prefix, [kw], exact_match=True)
            if not p:
                # 完全一致がなければ部分一致で検索
                p = _find_parquet_path_strict_prefix(
                    db_dir, prefix, [kw], exact_match=False
                )
            if p:
                break

        # 最後の手段：直下の {key}.parquet
        if not p:
            cand = os.path.join(db_dir, f"{key}.parquet")
            if os.path.exists(cand):
                p = cand
        if not p:
            raise FileNotFoundError(
                f"{key} の Parquet が見つかりませんでした: {db_dir}"
            )
        paths[key] = p

    logging.info("当日予想用データ読み込み: %s", paths)

    # データフレーム読み込み
    dfs = {k: read_parquet_or_fail(v) for k, v in paths.items()}

    # s_テーブルを日付でフィルタリング
    year = target_date[:4]
    month_day = target_date[4:]

    for key in ["s_uma_race", "s_race", "s_uma"]:
        if key not in dfs:
            continue

        df = dfs[key]

        # Year, MonthDay列でフィルタ
        if "Year" in df.columns and "MonthDay" in df.columns:
            mask = (df["Year"].astype(str) == year) & (
                df["MonthDay"].astype(str) == month_day
            )
            dfs[key] = df[mask].copy()
            logging.info("%s を日付フィルタ: %s件 → %s件", key, len(df), len(dfs[key]))

            # データが空の場合はエラー（利用可能な日付を表示）
            if dfs[key].empty:
                # 利用可能な日付を取得
                available_dates = df.apply(
                    lambda row: f"{row['Year']}{row['MonthDay']}", axis=1
                ).unique()
                available_dates_sorted = sorted(available_dates)

                error_msg = (
                    f"{key} に {target_date} のデータが存在しません。\n"
                    f"利用可能な日付: {', '.join(available_dates_sorted[:10])}"
                )
                if len(available_dates_sorted) > 10:
                    error_msg += f" ...他{len(available_dates_sorted) - 10}件"

                raise ValueError(error_msg)

    return dfs

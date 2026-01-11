"""
ファイルユーティリティモジュール

モデルや各種データの安全な保存機能を提供します。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

import lightgbm as lgb
import pandas as pd


def ensure_dir(path: str) -> None:
    """
    ディレクトリを確実に作成する

    指定されたパスのディレクトリが存在しない場合は作成します。
    既に存在する場合は何もしません。

    Args:
        path: 作成するディレクトリのパス
    """
    os.makedirs(path, exist_ok=True)


def _with_ts(path: str) -> str:
    """
    ファイルパスにタイムスタンプを付与する

    ファイル保存時のPermissionError対策として、
    タイムスタンプを含むフォールバックパスを生成します。

    Args:
        path: 元のファイルパス

    Returns:
        str: タイムスタンプを含むファイルパス
    """
    base, ext = os.path.splitext(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}{ext}"


def safe_save_model(gbm: lgb.Booster, path: str) -> str:
    """
    LightGBMモデルを安全に保存する

    PermissionError発生時は、タイムスタンプ付きの
    フォールバックパスで保存します。

    Args:
        gbm: LightGBMのBoosterオブジェクト
        path: 保存先パス

    Returns:
        str: 実際に保存されたパス
    """
    try:
        gbm.save_model(path)
        return path
    except PermissionError:
        alt = _with_ts(path)
        gbm.save_model(alt)
        print(f"[warn] PermissionError: フォールバック保存 {alt}")
        return alt


def safe_save_json(obj: Dict, path: str) -> str:
    """
    JSONファイルを安全に保存する

    PermissionError発生時は、タイムスタンプ付きの
    フォールバックパスで保存します。

    Args:
        obj: 保存する辞書オブジェクト
        path: 保存先パス

    Returns:
        str: 実際に保存されたパス
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path
    except PermissionError:
        alt = _with_ts(path)
        with open(alt, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"[warn] PermissionError: フォールバック保存 {alt}")
        return alt


def safe_save_df(df: pd.DataFrame, path: str, fmt: str) -> str:
    """
    DataFrameを安全に保存する

    CSV形式またはParquet形式で保存します。
    PermissionError発生時は、タイムスタンプ付きの
    フォールバックパスで保存します。

    Args:
        df: 保存するDataFrame
        path: 保存先パス
        fmt: ファイル形式 ('csv' または 'parquet')

    Returns:
        str: 実際に保存されたパス

    Raises:
        ValueError: fmtが'csv'または'parquet'以外の場合
    """
    try:
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError("fmt must be 'csv' or 'parquet'")
        return path
    except PermissionError:
        alt = _with_ts(path)
        if fmt == "csv":
            df.to_csv(alt, index=False)
        else:
            df.to_parquet(alt, index=False)
        print(f"[warn] PermissionError: フォールバック保存 {alt}")
        return alt

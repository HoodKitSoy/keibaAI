"""
データローダーモジュール

前処理済みデータの読み込み機能を提供します。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def load_preprocessed_data(preprocessed_file: str | None, date: str) -> pd.DataFrame:
    """
    前処理済みファイルを読み込む

    Args:
        preprocessed_file: 前処理済みファイルのパス（Noneの場合は自動検索）
        date: 対象日付（YYYYMMDD形式）

    Returns:
        pd.DataFrame: 前処理済みデータ

    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    if preprocessed_file:
        path = Path(preprocessed_file)
    else:
        # デフォルトパス: ./data/tmp/today_preprocessed_{date}.parquet
        path = Path("./data/tmp") / f"today_preprocessed_{date}.parquet"

        # Parquetが無ければCSVを探す
        if not path.exists():
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                path = csv_path

    if not path.exists():
        raise FileNotFoundError(
            f"前処理済みファイルが見つかりません: {path}\n"
            f"先に以下を実行してください:\n"
            f"  python preprocessing.py --today --date {date}"
        )

    logging.info(f"前処理済みデータ読み込み: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path, low_memory=False)

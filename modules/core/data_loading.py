"""
共通データローダーモジュール

Parquetファイルの検索・読み込み機能を一元化します。
ファイル名の揺れに対応した柔軟な検索機能を提供します。
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .config import get_preprocessed_path


def find_parquet_file(
    db_dir: str | Path, keywords: Iterable[str], recursive: bool = True
) -> Path | None:
    """
    指定されたディレクトリからParquetファイルを検索します。

    ファイル名の揺れ（大文字小文字、アンダースコアの有無など）に対応した
    柔軟な検索を行います。

    Args:
        db_dir: 検索対象のディレクトリパス
        keywords: 検索キーワード（複数指定時はすべてを含むファイルを検索）
        recursive: サブディレクトリも検索する場合はTrue

    Returns:
        Path | None: 見つかったファイルパス、見つからない場合はNone

    Example:
        >>> find_parquet_file("./data/DB", ["n_uma_race"])
        Path('./data/DB/n_uma_race.parquet')
    """
    db_dir = Path(db_dir)
    keywords_lower = [k.lower() for k in keywords]

    # 検索パターンを構築
    if recursive:
        pattern = str(db_dir / "**" / "*.parquet")
        candidates = glob.glob(pattern, recursive=True)
    else:
        pattern = str(db_dir / "*.parquet")
        candidates = glob.glob(pattern)

    # キーワードマッチング
    for candidate in candidates:
        name = Path(candidate).stem.lower()
        if all(kw in name for kw in keywords_lower):
            return Path(candidate)

    # 追加の柔軟な検索: アンダースコア除外
    for candidate in candidates:
        name = Path(candidate).stem.lower().replace("_", "")
        keywords_no_underscore = [kw.replace("_", "") for kw in keywords_lower]
        if all(kw in name for kw in keywords_no_underscore):
            return Path(candidate)

    return None


def read_parquet(path: str | Path) -> pd.DataFrame:
    """
    Parquetファイルを読み込みます。

    Args:
        path: Parquetファイルのパス

    Returns:
        pd.DataFrame: 読み込んだデータフレーム

    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquetファイルが見つかりません: {path}")
    return pd.read_parquet(path)


def read_csv_or_parquet(path: str | Path) -> pd.DataFrame:
    """
    CSVまたはParquetファイルを拡張子に応じて読み込みます。

    拡張子が不明な場合は、Parquet読み込みを試み、
    失敗した場合はCSVとして読み込みます。

    Args:
        path: 読み込むファイルのパス

    Returns:
        pd.DataFrame: 読み込んだデータフレーム

    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        # Parquet優先で試行
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)


def load_preprocessed(path: str | Path | None = None) -> pd.DataFrame:
    """
    前処理済みデータを読み込みます。

    パスが指定されていない場合は、デフォルトパスから読み込みます。
    Parquetファイルが見つからない場合は、CSVファイルを試します。

    Args:
        path: 読み込むファイルのパス（Noneの場合はデフォルトパス）

    Returns:
        pd.DataFrame: 読み込んだデータフレーム

    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    if path is None:
        # デフォルトパスを試行
        default_pq = get_preprocessed_path(date=None, as_string=False)
        default_csv = Path(str(default_pq).replace(".parquet", ".csv"))

        if default_pq.exists():
            path = default_pq
        elif default_csv.exists():
            path = default_csv
        else:
            raise FileNotFoundError(
                f"前処理済みファイルが見つかりません: {default_pq} または {default_csv}"
            )
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {path}")

    return read_csv_or_parquet(path)


def load_db_tables(
    db_dir: str | Path, table_names: Dict[str, list[str]]
) -> Dict[str, pd.DataFrame]:
    """
    複数のDBテーブルを一括で読み込みます。

    ファイル名の揺れに対応した検索を行います。

    Args:
        db_dir: データベースディレクトリのパス
        table_names: テーブル名をキー、検索キーワードのリストを値とする辞書

    Returns:
        Dict[str, pd.DataFrame]: テーブル名をキーとしたデータフレームの辞書

    Raises:
        FileNotFoundError: 必須テーブルが見つからない場合

    Example:
        >>> tables = load_db_tables(
        ...     "./data/DB",
        ...     {
        ...         "n_uma_race": ["n_uma_race", "uma_race"],
        ...         "n_race": ["n_race", "race"],
        ...     }
        ... )
    """
    db_dir = Path(db_dir)
    result = {}
    missing_tables = []

    for table_name, keywords in table_names.items():
        found_path = None
        # キーワードを順番に試す
        for keyword in keywords:
            found_path = find_parquet_file(db_dir, [keyword])
            if found_path:
                break

        if found_path:
            logging.info(f"テーブル {table_name} を読み込み: {found_path}")
            result[table_name] = read_parquet(found_path)
        else:
            missing_tables.append(table_name)

    if missing_tables:
        raise FileNotFoundError(
            f"以下のテーブルが見つかりません: {', '.join(missing_tables)}\n"
            f"検索ディレクトリ: {db_dir}"
        )

    return result


def filter_by_date_range(
    df: pd.DataFrame, year_col: str = "Year", years: set | None = None
) -> pd.DataFrame:
    """
    年でデータをフィルタリングします。

    Args:
        df: フィルタリング対象のデータフレーム
        year_col: 年の列名
        years: 対象年のセット（Noneの場合はフィルタリングなし）

    Returns:
        pd.DataFrame: フィルタリング後のデータフレーム
    """
    if years is None or year_col not in df.columns:
        return df

    # 年を文字列に変換して比較
    year_str = df[year_col].astype(str).str.zfill(4)
    return df[year_str.isin(years)].copy()


def filter_by_jyo_cd(
    df: pd.DataFrame, jyo_col: str = "JyoCD", jyo_cds: set | None = None
) -> pd.DataFrame:
    """
    競馬場コードでデータをフィルタリングします。

    Args:
        df: フィルタリング対象のデータフレーム
        jyo_col: 競馬場コードの列名
        jyo_cds: 対象競馬場コードのセット（Noneの場合はフィルタリングなし）

    Returns:
        pd.DataFrame: フィルタリング後のデータフレーム
    """
    if jyo_cds is None or jyo_col not in df.columns:
        return df

    # 競馬場コードを文字列に変換して比較
    jyo_str = df[jyo_col].astype(str).str.zfill(2)
    return df[jyo_str.isin(jyo_cds)].copy()


def exclude_data_kubun(
    df: pd.DataFrame, kubun_col: str = "DataKubun", exclude_values: set | None = None
) -> pd.DataFrame:
    """
    DataKubunでデータを除外します（中止・削除レコードの除外）。

    Args:
        df: フィルタリング対象のデータフレーム
        kubun_col: DataKubun列名
        exclude_values: 除外する値のセット（Noneの場合は除外なし）

    Returns:
        pd.DataFrame: フィルタリング後のデータフレーム
    """
    if exclude_values is None or kubun_col not in df.columns:
        return df

    # DataKubunを文字列に変換して比較
    kubun_str = df[kubun_col].astype(str)
    return df[~kubun_str.isin(exclude_values)].copy()

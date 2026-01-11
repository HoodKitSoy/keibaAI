"""
特徴量エンジニアリングモジュール

特徴量の選択、型変換、データ分割機能を提供します。
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from modules.core import (
    EXCLUDE_FEATURE_COLS,
    FINAL_COLUMN_ORDER,
    RACE_KEY_COLS,
    TARGET_COL,
)


def coerce_numeric_or_categorical(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    数値列とカテゴリ列を適切に変換する

    各列について以下の処理を行います：
    1. 既に数値型の場合: 欠損値を0で埋める
    2. 文字列型の場合: 数値化を試み、70%以上成功すれば数値列として扱う
    3. 数値化に失敗した場合: カテゴリ列として扱い、欠損値を"0"で埋める

    Args:
        df: 変換対象のDataFrame
        feature_cols: 変換対象の列名リスト

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - 変換後のDataFrame
            - カテゴリ列の列名リスト
    """
    categorical_cols: List[str] = []
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.fillna(0)
            continue
        # 数値化トライ
        sn = pd.to_numeric(s, errors="coerce")
        # 7割以上数値に変換できるなら数値とみなす
        if sn.notna().mean() >= 0.7:
            out[col] = sn.fillna(0)
        else:
            # カテゴリ列
            out[col] = s.fillna("0").astype("category")
            categorical_cols.append(col)
    return out, categorical_cols


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    特徴量列を選択する

    FINAL_COLUMN_ORDERから存在する列のみを抽出し、
    特徴量として使用する列を決定します。

    特徴量からは以下を除外します：
    - レースキー列（Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum）
    - ターゲット列（KakuteiJyuni）
    - KettoNum（データリーク回避）
    - Odds, Ninki（予測時には未知の情報）
    - BaTaijyu, ZogenFugo, ZogenSa（馬体重関連、予測時には未知）
    - DMTime, DMGosaP, DMGosaM, DMJyuni（マイニング予想、予測時には未知）
    - TenkoCD, SibaBabaCD, DirtBabaCD（天候・馬場状態、予測時には未知）

    Umabanは特徴量として使用します。

    Args:
        df: 元のDataFrame

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - 選択された列を含むDataFrame
            - 特徴量列の列名リスト
    """
    # FINAL_COLUMN_ORDER から、存在する列のみを採用
    cols = [c for c in FINAL_COLUMN_ORDER if c in df.columns]
    if TARGET_COL not in cols and TARGET_COL in df.columns:
        cols.append(TARGET_COL)
    sub = df[cols].copy()

    # 特徴量: FINAL_COLUMN_ORDER からレースキーとターゲットを除外（Umaban は使用可）
    # 予測時に未知の情報となる列を除外（データリーク回避）
    # 除外列はEXCLUDE_FEATURE_COLSで一元管理
    drop_cols = set(RACE_KEY_COLS + [TARGET_COL] + EXCLUDE_FEATURE_COLS)
    feature_cols = [c for c in cols if c not in drop_cols]
    return sub, feature_cols


def split_by_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    データを年単位で訓練/検証/テストに分割する

    分割基準：
    - 訓練データ: 2016年～2022年
    - 検証データ: 2023年
    - テストデータ: 2024年

    Args:
        df: 分割対象のDataFrame（Year列が必要）

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - 訓練データ
            - 検証データ
            - テストデータ
    """
    yy = pd.to_numeric(df["Year"], errors="coerce")
    train = df[yy.between(2016, 2022)].copy()
    valid = df[yy == 2023].copy()
    test = df[yy == 2024].copy()
    return train, valid, test

# modules/policies/_score_policy.py

from abc import ABCMeta, abstractmethod
from typing import Callable

import pandas as pd

_SCORE = "score"

RACE_KEY_COLS = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]


def _build_race_id(df: pd.DataFrame) -> pd.Index:
    """
    JRA-DBの複合キーからレースID文字列を生成する。
    形式: YYYYMMDD-JJKKNN-RR （JJ:場, KK:回, NN:日, RR:レース）
    """

    def z(s, w):
        return df[s].astype(str).str.zfill(w)

    if not set(RACE_KEY_COLS).issubset(df.columns):
        # キー列がない場合は既存インデックスを維持
        return df.index
    y = z("Year", 4)
    md = z("MonthDay", 4)
    j = z("JyoCD", 2)
    ka = z("Kaiji", 2)
    ni = z("Nichiji", 2)
    rn = z("RaceNum", 2)
    return (y + md + "-" + j + ka + ni + "-" + rn).astype(str)


def _calc(model, X: pd.DataFrame) -> pd.DataFrame:
    # JRA-DBでは馬番列は `Umaban`
    if "Umaban" not in X.columns:
        raise KeyError(
            "`Umaban` 列が見つかりません。前処理出力に基づく特徴量を入力してください。"
        )

    score_table = X[["Umaban"]].copy()
    # レースID（index）をJRAキーから生成（可能な場合）
    new_index = _build_race_id(X)
    score_table.index = new_index

    # 予測確率
    pred = model.predict_proba(X)[:, 1]
    score_table[_SCORE] = pred
    return score_table


def _apply_scaler(
    score: pd.Series, scaler: Callable[[pd.Series], pd.Series]
) -> pd.Series:
    return score.groupby(level=0, group_keys=False).apply(scaler)


def _scaler_relative_proba(x: pd.Series) -> pd.Series:
    return x / (x.sum() + 1e-15)


class AbstractScorePolicy(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def calc(model, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RelativeProbaScorePolicy(AbstractScorePolicy):
    """
    レース内での相対確率化
    """

    @staticmethod
    def calc(model, X: pd.DataFrame) -> pd.DataFrame:
        score_table = _calc(model, X)
        score_table[_SCORE] = _apply_scaler(score_table[_SCORE], _scaler_relative_proba)
        return score_table

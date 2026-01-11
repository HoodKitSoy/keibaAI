# modules/policies/_race_selective_policy.py

import pandas as pd

from ._bet_policy import AbstractBetPolicy
from ._hybrid_bet_policy import (
    ExpectedValueBetPolicy,
    HybridBetPolicy,
    TieredCoverageBetPolicy,
)


class RaceSelectivePolicy(AbstractBetPolicy):
    """
    レース選別機能を持つ馬券戦略。
    指定された条件を満たすレースのみを選択して馬券を購入する。
    基本戦略には他の既存戦略（デフォルトではHybridBetPolicy）を利用する。

    特徴:
    - 上位馬のスコア差が大きいレースを選別
    - トップ馬の勝率が特に高いレースを選別
    - 信頼性の高いレースに的を絞ることで、回収率の向上を図る
    """

    def __init__(self):
        # 基本となる馬券戦略
        self.base_policy = HybridBetPolicy()

        # レース選別パラメータ（デフォルト値）
        self.min_top_score = 0.35  # トップ馬の最低スコア
        self.min_score_gap = 0.10  # 1位と2位のスコア差の最低値
        self.use_or_condition = (
            True  # True: いずれかの条件を満たせばOK、False: 両方の条件を満たす必要あり
        )

    @staticmethod
    def judge(
        score_table: pd.DataFrame,
        min_top_score: float = 0.35,
        min_score_gap: float = 0.10,
        use_or_condition: bool = True,
        base_policy_name: str = "HybridBetPolicy",
        **kwargs,
    ) -> dict:
        """
        レース選別条件に基づいて、賭けるレースと馬を選択する

        Parameters
        ----------
        score_table : pd.DataFrame
            レースごとの馬番とスコア（勝率）
        min_top_score : float, optional
            トップ馬の最低スコア（デフォルト: 0.35）
        min_score_gap : float, optional
            1位と2位のスコア差の最低値（デフォルト: 0.10）
        use_or_condition : bool, optional
            True: いずれかの条件を満たせばOK、False: 両方の条件を満たす必要あり（デフォルト: True）
        base_policy_name : str, optional
            基本となる馬券戦略の名前（デフォルト: "HybridBetPolicy"）
        **kwargs : dict
            基本戦略に渡す追加パラメータ

        Returns
        -------
        dict
            レースごとの選択された馬券情報
        """
        # 基本戦略の選択
        if base_policy_name == "HybridBetPolicy":
            base_policy_results = HybridBetPolicy.judge(score_table, **kwargs)
        elif base_policy_name == "ExpectedValueBetPolicy":
            base_policy_results = ExpectedValueBetPolicy.judge(score_table, **kwargs)
        elif base_policy_name == "TieredCoverageBetPolicy":
            base_policy_results = TieredCoverageBetPolicy.judge(score_table, **kwargs)
        else:
            # デフォルトはHybridBetPolicy
            base_policy_results = HybridBetPolicy.judge(score_table, **kwargs)

        # 結果を格納する辞書
        bet_dict = {}

        # レースごとに選別条件を適用
        from modules.core import iter_race_groups

        for race_id, df_r in iter_race_groups(score_table):
            # スコアの降順でソート
            df_sorted = df_r.sort_values("score", ascending=False)

            # レース選別のための条件チェック
            top_horse_score = df_sorted["score"].iloc[0] if len(df_sorted) > 0 else 0

            score_gap = 0
            if len(df_sorted) >= 2:
                score_gap = df_sorted["score"].iloc[0] - df_sorted["score"].iloc[1]
            # 選別条件のチェック
            condition1 = top_horse_score >= min_top_score
            condition2 = score_gap >= min_score_gap

            select_race = False
            if use_or_condition:
                # いずれかの条件を満たせばOK
                select_race = condition1 or condition2
            else:
                # 両方の条件を満たす必要あり
                select_race = condition1 and condition2

            # 条件を満たすレースのみを選択
            if select_race and race_id in base_policy_results:
                bet_dict[race_id] = base_policy_results[race_id]

        return bet_dict


class RaceSelectiveHybridPolicy(RaceSelectivePolicy):
    """
    HybridBetPolicyをベースに、レース選別機能を追加した戦略。
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        """
        HybridBetPolicyをベースとしたレース選別戦略
        """
        return RaceSelectivePolicy.judge(
            score_table, base_policy_name="HybridBetPolicy", **kwargs
        )


class RaceSelectiveExpectedValueBetPolicy(RaceSelectivePolicy):
    """
    ExpectedValueBetPolicyをベースに、レース選別機能を追加した戦略。
    オッズと勝率から期待値を計算し、期待値が1.0以上の馬券を選ぶ戦略に、
    レース選別を組み合わせたもの。
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        """
        ExpectedValueBetPolicyをベースとしたレース選別戦略
        """
        return RaceSelectivePolicy.judge(
            score_table, base_policy_name="ExpectedValueBetPolicy", **kwargs
        )


class RaceSelectiveTieredCoverageBetPolicy(RaceSelectivePolicy):
    """
    TieredCoverageBetPolicyをベースに、レース選別機能を追加した戦略。
    累積的中率とTierに基づいて馬券を選ぶ戦略に、レース選別を組み合わせたもの。
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        """
        TieredCoverageBetPolicyをベースとしたレース選別戦略
        """
        return RaceSelectivePolicy.judge(
            score_table, base_policy_name="TieredCoverageBetPolicy", **kwargs
        )

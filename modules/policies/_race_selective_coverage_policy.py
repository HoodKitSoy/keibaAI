# modules/policies/_race_selective_coverage_policy.py

import pandas as pd

from ._bet_policy import AbstractBetPolicy, BetPolicyCoverageBase


class RaceSelectiveCoveragePolicy(AbstractBetPolicy):
    """
    レース選別機能を持つカバレッジベース戦略。
    指定された条件を満たすレースのみを選択して馬券を購入する。
    基本戦略にはBetPolicyCoverageBaseを継承した各種戦略を利用する。

    特徴:
    - 上位馬のスコア差が大きいレースを選別
    - トップ馬の勝率が特に高いレースを選別
    - BetPolicyCoverageの高い的中率と、レース選別の組み合わせ
    """

    def __init__(self):
        # 基本となる馬券戦略（デフォルトはBetPolicyCoverage25）
        self.base_coverage = 0.25

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
        coverage_threshold: float = 0.25,
        max_selections: int = 5,
        **kwargs,
    ) -> dict:
        """
        レース選別条件とカバレッジに基づいて、賭けるレースと馬を選択する

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
        coverage_threshold : float, optional
            累積的中率の閾値（デフォルト: 0.25）
        max_selections : int, optional
            1レースあたりの最大選択頭数（デフォルト: 5）
        **kwargs : dict
            その他のパラメータ

        Returns
        -------
        dict
            レースごとの選択された馬券情報
        """
        # 基本戦略の選択（デフォルトはcoverage_threshold=0.25のBetPolicyCoverageBase）
        base_policy_results = BetPolicyCoverageBase.judge(
            score_table,
            coverage_threshold=coverage_threshold,
            max_selections=max_selections,
        )

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


class RaceSelectiveCoverage25Policy(RaceSelectiveCoveragePolicy):
    """
    BetPolicyCoverage25をベースに、レース選別機能を追加した戦略。
    累積的中率25%を目標とする戦略に、レース選別を組み合わせたもの。
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        """
        BetPolicyCoverage25をベースとしたレース選別戦略
        """
        return RaceSelectiveCoveragePolicy.judge(
            score_table, coverage_threshold=0.25, max_selections=5, **kwargs
        )


class RaceSelectiveCoverage50Policy(RaceSelectiveCoveragePolicy):
    """
    BetPolicyCoverage50をベースに、レース選別機能を追加した戦略。
    累積的中率50%を目標とする戦略に、レース選別を組み合わせたもの。
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        """
        BetPolicyCoverage50をベースとしたレース選別戦略
        """
        return RaceSelectiveCoveragePolicy.judge(
            score_table, coverage_threshold=0.50, max_selections=5, **kwargs
        )


class RaceSelectiveCoverage75Policy(RaceSelectiveCoveragePolicy):
    """
    BetPolicyCoverage75をベースに、レース選別機能を追加した戦略。
    累積的中率75%を目標とする戦略に、レース選別を組み合わせたもの。
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        """
        BetPolicyCoverage75をベースとしたレース選別戦略
        """
        return RaceSelectiveCoveragePolicy.judge(
            score_table, coverage_threshold=0.75, max_selections=5, **kwargs
        )

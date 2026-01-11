# modules/policies/_hybrid_bet_policy.py

from typing import Dict, Optional

import pandas as pd

from ._bet_policy import AbstractBetPolicy


class HybridBetPolicy(AbstractBetPolicy):
    """
    勝率と期待値を組み合わせた高度な馬券選択戦略。
    1. 期待値（勝率 × オッズ）が1.0以上の馬券のみを対象とする
    2. 勝率のカテゴリに応じて投資強度を変える
    3. 馬券種類別にカスタマイズされた閾値を適用する
    """

    def __init__(self):
        # 馬券種類ごとの閾値設定
        self.bet_type_thresholds = {
            "tansho": {"min_win_rate": 0.25, "min_exp_value": 1.2},
            "fukusho": {"min_win_rate": 0.25, "min_exp_value": 1.2},
            "wakuren": {"min_win_rate": 0.20, "min_exp_value": 1.15},
            "umaren": {"min_win_rate": 0.20, "min_exp_value": 1.15},
            "umatan": {"min_win_rate": 0.20, "min_exp_value": 1.15},
            "wide": {"min_win_rate": 0.30, "min_exp_value": 1.1},
            "sanrenpuku": {"min_win_rate": 0.15, "min_exp_value": 1.3},
            "sanrentan": {"min_win_rate": 0.15, "min_exp_value": 1.3},
        }
        # 勝率カテゴリごとの投資倍率
        self.tier_multipliers = {
            "tier1": 3.0,  # 40%以上の勝率
            "tier2": 2.0,  # 30-40%の勝率
            "tier3": 1.0,  # 20-30%の勝率
        }
        # 1レースあたりの選択上限数
        self.max_selections_per_race = 2
        # 基本ベット金額（実際のシミュレーションでは使われず、相対比率として機能）
        self.base_bet_amount = 100

    @staticmethod
    def judge(
        score_table: pd.DataFrame, odds_table: Optional[Dict] = None, **kwargs
    ) -> dict:
        """
        スコアテーブルとオッズテーブルから、勝率と期待値に基づいて馬券を選択する

        Parameters
        ----------
        score_table : pd.DataFrame
            レースごとの馬番とスコア（勝率）
        odds_table : dict, optional
            レースごと、馬券種類ごとのオッズ情報。指定がない場合はデフォルトオッズを使用
        **kwargs : dict
            その他のパラメータ

        Returns
        -------
        dict
            レースごとの選択された馬券情報
        """
        # HybridBetPolicyインスタンスを作成（静的メソッドから設定を参照するため）
        policy = HybridBetPolicy()

        bet_dict = {}

        # レースごとに処理
        from modules.core import iter_race_groups

        for race_id, df_r in iter_race_groups(score_table):
            # スコアの降順でソート
            df_sorted = df_r.sort_values("score", ascending=False)

            # 各馬券種類ごとに候補を選択
            bet_dict[race_id] = {}

            # オッズ情報の取得（提供されていない場合はデフォルト値を使用）
            race_odds = odds_table.get(race_id, {}) if odds_table else {}

            # 各馬券種類ごとに処理
            for bet_type in policy.bet_type_thresholds.keys():
                # 馬券種類ごとの閾値を取得
                thresholds = policy.bet_type_thresholds[bet_type]
                min_win_rate = thresholds["min_win_rate"]
                min_exp_value = thresholds["min_exp_value"]

                # 選択候補のリスト
                candidates = []

                # 単勝と複勝の場合は単純に上位馬を選択
                if bet_type in ["tansho", "fukusho"]:
                    for idx, row in df_sorted.iterrows():
                        # JRA-DB 前処理に合わせて 'Umaban' を参照（安全にget使用）
                        try:
                            umaban = int(row.get("Umaban", 0))
                            if umaban <= 0:
                                continue
                        except (ValueError, TypeError):
                            continue
                        win_rate = float(row.get("score", 0))
                        if win_rate <= 0:
                            continue

                        # オッズの取得（なければ逆数近似）
                        if bet_type in race_odds and umaban in race_odds[bet_type]:
                            odds = race_odds[bet_type][umaban]
                        else:
                            # オッズが提供されていない場合は勝率の逆数を近似として使用
                            odds = (
                                min(99.9, max(1.0, 1.0 / win_rate))
                                if win_rate > 0
                                else 99.9
                            )

                        # 期待値の計算
                        exp_value = win_rate * odds

                        # 閾値に基づいてフィルタリング
                        if win_rate >= min_win_rate and exp_value >= min_exp_value:
                            # Tierの決定
                            if win_rate >= 0.4:
                                tier = "tier1"
                            elif win_rate >= 0.3:
                                tier = "tier2"
                            elif win_rate >= 0.2:
                                tier = "tier3"
                            else:
                                tier = None

                            if tier:
                                # 投資額の計算
                                bet_amount = (
                                    policy.base_bet_amount
                                    * policy.tier_multipliers.get(tier, 1.0)
                                )

                                candidates.append(
                                    {
                                        "umaban": umaban,
                                        "win_rate": win_rate,
                                        "odds": odds,
                                        "exp_value": exp_value,
                                        "tier": tier,
                                        "bet_amount": bet_amount,
                                    }
                                )

                    # 上位馬を選択（最大selections_per_race頭まで）
                    top_candidates = sorted(
                        candidates, key=lambda x: x["exp_value"], reverse=True
                    )[: policy.max_selections_per_race]

                    if top_candidates:
                        bet_dict[race_id][bet_type] = [
                            c["umaban"] for c in top_candidates
                        ]

                # 組み合わせ馬券（馬連、馬単、ワイド、三連複、三連単、枠連）
                else:
                    # 勝率上位の馬を候補として抽出
                    top_horses = []
                    top_wakuban = []  # 枠番リスト（枠連用）
                    has_wakuban = "Wakuban" in df_sorted.columns
                    for idx, row in df_sorted.iterrows():
                        try:
                            umaban = int(row.get("Umaban", 0))
                            if umaban <= 0:
                                continue
                        except (ValueError, TypeError):
                            continue
                        win_rate = row.get("score", 0)

                        # 最低閾値を満たす馬だけを候補に
                        if win_rate >= min_win_rate:
                            top_horses.append(umaban)
                            # 枠番も収集
                            if has_wakuban:
                                try:
                                    waku = int(row.get("Wakuban", 0))
                                    if waku > 0 and waku not in top_wakuban:
                                        top_wakuban.append(waku)
                                except (ValueError, TypeError):
                                    pass

                    # 上位3〜5頭に絞る（馬券種類による）
                    if bet_type == "wakuren":
                        # 枠連は枠番を使用
                        top_wakuban = top_wakuban[:5]
                        if len(top_wakuban) >= 2:
                            bet_dict[race_id][bet_type] = top_wakuban
                    elif bet_type in ["umaren", "umatan", "wide"]:
                        top_horses = top_horses[:5]  # 2頭必要な馬券は上位5頭から選択
                        if len(top_horses) >= 2:
                            bet_dict[race_id][bet_type] = top_horses
                    else:  # 三連系
                        top_horses = top_horses[:6]  # 3頭必要な馬券は上位6頭から選択
                        if len(top_horses) >= 3:
                            bet_dict[race_id][bet_type] = top_horses

        return bet_dict


class ExpectedValueBetPolicy(AbstractBetPolicy):
    """
    期待値（勝率 × オッズ）に基づいて馬券を選択する戦略。
    期待値が1.0以上の馬券のみを対象とする。
    """

    def __init__(self):
        # 馬券種類ごとの閾値設定
        self.bet_type_thresholds = {
            "tansho": {"min_exp_value": 1.0},
            "fukusho": {"min_exp_value": 1.0},
            "wakuren": {"min_exp_value": 1.0},
            "umaren": {"min_exp_value": 1.0},
            "umatan": {"min_exp_value": 1.0},
            "wide": {"min_exp_value": 1.0},
            "sanrenpuku": {"min_exp_value": 1.0},
            "sanrentan": {"min_exp_value": 1.0},
        }
        # 1レースあたりの選択上限数
        self.max_selections_per_race = 3

    @staticmethod
    def judge(
        score_table: pd.DataFrame, odds_table: Optional[Dict] = None, **kwargs
    ) -> dict:
        """
        スコアテーブルとオッズテーブルから、期待値に基づいて馬券を選択する

        Parameters
        ----------
        score_table : pd.DataFrame
            レースごとの馬番とスコア（勝率）
        odds_table : dict, optional
            レースごと、馬券種類ごとのオッズ情報。指定がない場合はデフォルトオッズを使用
        **kwargs : dict
            その他のパラメータ

        Returns
        -------
        dict
            レースごとの選択された馬券情報
        """
        # ExpectedValueBetPolicyインスタンスを作成
        policy = ExpectedValueBetPolicy()

        bet_dict = {}

        # レースごとに処理
        from modules.core import iter_race_groups

        for race_id, df_r in iter_race_groups(score_table):
            # スコアの降順でソート
            df_sorted = df_r.sort_values("score", ascending=False)

            # 各馬券種類ごとに候補を選択
            bet_dict[race_id] = {}

            # オッズ情報の取得（提供されていない場合はデフォルト値を使用）
            race_odds = odds_table.get(race_id, {}) if odds_table else {}

            # 各馬券種類ごとに処理
            for bet_type in policy.bet_type_thresholds.keys():
                # 馬券種類ごとの閾値を取得
                min_exp_value = policy.bet_type_thresholds[bet_type]["min_exp_value"]

                # 選択候補のリスト
                candidates = []

                # 単勝と複勝の場合は単純に上位馬を選択
                if bet_type in ["tansho", "fukusho"]:
                    for idx, row in df_sorted.iterrows():
                        try:
                            umaban = int(row.get("Umaban", 0))
                            if umaban <= 0:
                                continue
                        except (ValueError, TypeError):
                            continue
                        win_rate = float(row.get("score", 0))
                        if win_rate <= 0:
                            continue

                        # オッズの取得（なければ逆数近似）
                        if bet_type in race_odds and umaban in race_odds[bet_type]:
                            odds = race_odds[bet_type][umaban]
                        else:
                            # オッズが提供されていない場合は勝率の逆数を近似として使用
                            odds = (
                                min(99.9, max(1.0, 1.0 / win_rate))
                                if win_rate > 0
                                else 99.9
                            )

                        # 期待値の計算
                        exp_value = win_rate * odds

                        # 閾値に基づいてフィルタリング
                        if exp_value >= min_exp_value:
                            candidates.append(
                                {
                                    "umaban": umaban,
                                    "win_rate": win_rate,
                                    "odds": odds,
                                    "exp_value": exp_value,
                                }
                            )

                    # 上位馬を選択（最大selections_per_race頭まで）
                    top_candidates = sorted(
                        candidates, key=lambda x: x["exp_value"], reverse=True
                    )[: policy.max_selections_per_race]

                    if top_candidates:
                        bet_dict[race_id][bet_type] = [
                            c["umaban"] for c in top_candidates
                        ]

                # 組み合わせ馬券（馬連、馬単、ワイド、三連複、三連単、枠連）
                else:
                    # 勝率上位の馬をそのまま候補として採用
                    top_horses = []
                    top_wakuban = []
                    has_wakuban = "Wakuban" in df_sorted.columns
                    for idx, row in df_sorted.head(5).iterrows():
                        try:
                            umaban = int(row.get("Umaban", 0))
                            if umaban <= 0:
                                continue
                            top_horses.append(umaban)
                            if has_wakuban:
                                waku = int(row.get("Wakuban", 0))
                                if waku > 0 and waku not in top_wakuban:
                                    top_wakuban.append(waku)
                        except (ValueError, TypeError):
                            continue

                    if bet_type == "wakuren":
                        # 枠連は枠番を使用
                        if len(top_wakuban) >= 2:
                            bet_dict[race_id][bet_type] = top_wakuban
                    elif len(top_horses) >= 2:  # 最低2頭必要
                        bet_dict[race_id][bet_type] = top_horses

        return bet_dict


class TieredCoverageBetPolicy(AbstractBetPolicy):
    """
    カバレッジとTierを組み合わせた戦略。
    カバレッジ（累積的中率）が閾値を超えるまで馬を選び、
    さらにTier（勝率カテゴリ）に応じて賭け金を調整する。
    """

    def __init__(self):
        # カバレッジ閾値
        self.coverage_threshold = 0.35
        # 最大選択頭数
        self.max_selections = 5
        # Tier区分（勝率範囲）
        self.tier_ranges = {
            "tier1": (0.4, 1.0),  # 40%以上
            "tier2": (0.3, 0.4),  # 30-40%
            "tier3": (0.2, 0.3),  # 20-30%
        }

    @staticmethod
    def judge(
        score_table: pd.DataFrame,
        coverage_threshold: float = 0.35,
        max_selections: int = 5,
        **kwargs,
    ) -> dict:
        """
        スコアテーブルから、累積的中率が閾値を超えるまで馬を選択する

        Parameters
        ----------
        score_table : pd.DataFrame
            レースごとの馬番とスコア（勝率）
        coverage_threshold : float, optional
            累積的中率の閾値（デフォルト: 0.35）
        max_selections : int, optional
            最大選択頭数（デフォルト: 5）
        **kwargs : dict
            その他のパラメータ

        Returns
        -------
        dict
            レースごとの選択された馬券情報
        """
        policy = TieredCoverageBetPolicy()
        policy.coverage_threshold = coverage_threshold
        policy.max_selections = max_selections

        bet_dict = {}
        from modules.core import iter_race_groups

        # Wakuban列の存在チェック（事前確認）
        has_wakuban = "Wakuban" in score_table.columns

        for race_id, df_r in iter_race_groups(score_table):
            df_sorted = df_r.sort_values("score", ascending=False)
            selected = []
            selected_wakuban = []  # 枠連用（枠番リスト、1〜8）
            tiers = {}  # 馬番ごとのTier情報
            cumulative = 0.0

            # 累積的中率は，1 - ∏(1 - p_i) として計算
            for idx, row in df_sorted.iterrows():
                try:
                    umaban = int(row.get("Umaban", 0))
                    if umaban <= 0:
                        continue
                except (ValueError, TypeError):
                    continue
                score = float(row.get("score", 0))

                # Tierの決定
                if score >= 0.4:
                    tiers[umaban] = "tier1"
                elif score >= 0.3:
                    tiers[umaban] = "tier2"
                elif score >= 0.2:
                    tiers[umaban] = "tier3"
                else:
                    tiers[umaban] = "tier4"

                selected.append(umaban)

                # 枠番も収集（Wakuban列がある場合のみ）
                if has_wakuban:
                    try:
                        waku = int(row.get("Wakuban", 0))
                        if waku > 0 and waku not in selected_wakuban:
                            selected_wakuban.append(waku)
                    except (ValueError, TypeError):
                        pass

                prod = 1.0
                # 選択された馬のscoreを用いて累積確率を計算
                for s in df_sorted[df_sorted["Umaban"].isin(selected)]["score"]:
                    prod *= 1 - s
                cumulative = 1.0 - prod

                if (
                    cumulative >= policy.coverage_threshold
                    or len(selected) >= policy.max_selections
                ):
                    break

            # 上限を設定
            selected = selected[: policy.max_selections]
            selected_wakuban = selected_wakuban[: policy.max_selections]

            # Tier情報を付加
            selected_with_tiers = {
                umaban: tiers.get(umaban, "tier4") for umaban in selected
            }

            # 各券種に対して同一の候補リストを出力
            # 枠連は枠番（1〜8）を使用、枠番がない場合は馬番でフォールバック
            bet_dict[race_id] = {
                "tansho": selected[:],
                "fukusho": selected[:],
                "wakuren": selected_wakuban[:] if selected_wakuban else selected[:],
                "umaren": selected[:],
                "umatan": selected[:],
                "wide": selected[:],
                "sanrenpuku": selected[:],
                "sanrentan": selected[:],
                # Tier情報も保存（実際のシミュレーションでは使用されない）
                "_tiers": selected_with_tiers,
            }

        return bet_dict

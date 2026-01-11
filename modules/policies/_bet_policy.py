# modules/policies/_bet_policy.py

from abc import ABCMeta, abstractstaticmethod

import pandas as pd


class AbstractBetPolicy(metaclass=ABCMeta):
    @abstractstaticmethod
    def judge(score_table, **params):
        pass


class BetPolicyCoverageBase(AbstractBetPolicy):
    """
    勝率が高い順に並べ，累積的中率が指定された閾値を超えるまで馬を選択する．
    選ばれた候補馬リストを，単勝だけでなく複勝，枠連，馬連，馬単，ワイド，
    三連複，三連単の各券種の候補として出力する．
    """

    @staticmethod
    def judge(
        score_table: pd.DataFrame,
        coverage_threshold: float = 0.5,
        max_selections: int = 5,
        **kwargs,
    ) -> dict:
        bet_dict = {}
        from modules.core import iter_race_groups

        # Wakuban列の存在チェック（事前確認）
        has_wakuban = "Wakuban" in score_table.columns

        for race_id, df_r in iter_race_groups(score_table):
            df_sorted = df_r.sort_values("score", ascending=False)
            selected = []
            selected_wakuban = []  # 枠連用（枠番リスト、1〜8）
            cumulative = 0.0
            # 累積的中率は，1 - ∏(1 - p_i) として計算
            for idx, row in df_sorted.iterrows():
                # JRA-DB 前処理に合わせて 'Umaban' を参照
                try:
                    umaban = int(row.get("Umaban", 0))
                    if umaban <= 0:
                        continue
                    selected.append(umaban)
                except (ValueError, TypeError):
                    continue

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
                if cumulative >= coverage_threshold or len(selected) >= max_selections:
                    break
            # 上限を設定（デフォルトは5件）
            selected = selected[:max_selections]
            selected_wakuban = selected_wakuban[:max_selections]
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
            }
        return bet_dict


class BetPolicyCoverage25(BetPolicyCoverageBase):
    """
    勝率が高い順に並べ，累積的中率が0.25を超えるまで馬を選択する．
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        return BetPolicyCoverageBase.judge(
            score_table, coverage_threshold=0.25, max_selections=5, **kwargs
        )


class BetPolicyCoverage50(BetPolicyCoverageBase):
    """
    勝率が高い順に並べ，累積的中率が0.5を超えるまで馬を選択する．
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        return BetPolicyCoverageBase.judge(
            score_table, coverage_threshold=0.5, max_selections=5, **kwargs
        )


class BetPolicyCoverage75(BetPolicyCoverageBase):
    """
    勝率が高い順に並べ，累積的中率が0.75を超えるまで馬を選択する．
    """

    @staticmethod
    def judge(score_table: pd.DataFrame, **kwargs) -> dict:
        return BetPolicyCoverageBase.judge(
            score_table, coverage_threshold=0.75, max_selections=5, **kwargs
        )

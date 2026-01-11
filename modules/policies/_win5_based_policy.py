"""
WIN5ベースポリシー

WIN5の予測結果（各レースの勝率上位馬）を使用して、
単勝、複勝、馬連などの他の券種で馬券を購入するポリシー。

WIN5では5レースすべての1着馬を当てる必要があるため、
各レースで選択された馬は1着期待が高い馬である。
この馬を使って他の券種でも馬券を購入することで、
WIN5不的中時のリスクヘッジを行う。

提供されるポリシー:
- WIN5BasedMultiPolicy: すべての馬券種を購入するポリシー
- WIN5BasedHedgePolicy: WIN5本命馬以外で購入するポリシー

使用方法:
- win5_selectionsパラメータを渡すと、WIN5予測で選択された馬を使用
- win5_selectionsがない場合は、top_nで独自に馬を選択（後方互換性）
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from modules.policies._bet_policy import BetPolicyCoverageBase
from modules.policies._utils import iter_race_groups


class WIN5BasedMultiPolicy(BetPolicyCoverageBase):
    """
    WIN5選択馬による全券種購入ポリシー

    WIN5で選択された馬（勝率上位）を軸として、
    すべての馬券種を購入するポリシー。

    - 1頭: 単勝、複勝
    - 2頭: 単勝、複勝、馬連、馬単、ワイド、枠連
    - 3頭以上: 全券種（三連複、三連単も含む）

    Attributes:
        top_n: 各レースで選択する馬の数（デフォルト: 3）
        min_proba: 最小確率閾値（デフォルト: 0.0 = 閾値なし）
    """

    def __init__(
        self,
        top_n: int = 3,
        min_proba: float = 0.0,  # WIN5対象レースは全て選択するため閾値なし
        **kwargs: Any,
    ) -> None:
        """
        初期化

        Args:
            top_n: 各レースで選択する馬の数
            min_proba: 最小確率閾値（デフォルト: 0.0 = 閾値なし）
            **kwargs: 親クラスへのキーワード引数
        """
        super().__init__(**kwargs)
        self.top_n = top_n
        self.min_proba = min_proba

    def judge(
        self,
        score_table: pd.DataFrame,
        odds_table: pd.DataFrame = None,
        win5_selections: Optional[Dict[str, List[int]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        WIN5選択馬で全券種を購入

        Args:
            score_table: 予測スコアテーブル
            odds_table: オッズテーブル（未使用）
            win5_selections: WIN5予測で選択された馬 {race_id: [馬番リスト]}
                           指定された場合はこの馬を使用し、独自選択は行わない
                           辞書の順序（挿入順）がレースの発走順序を表す
            **kwargs: その他のキーワード引数

        Returns:
            Dict[str, Dict[str, List[int]]]: {race_id: {券種: [馬番リスト]}}
            ※win5_selectionsが指定された場合、返り値もその順序を維持
        """
        bet_dict: Dict[str, Dict[str, List[int]]] = {}

        # win5_selectionsが渡された場合はその順序でレースを処理
        if win5_selections:
            for race_id, selected_umaban in win5_selections.items():
                # 該当レースのデータを取得
                race_df = score_table[score_table["race_id"] == race_id]
                if race_df.empty:
                    continue

                # 枠番を取得
                selected_wakuban: List[int] = []
                for umaban in selected_umaban:
                    row = race_df[race_df["Umaban"].astype(int) == umaban]
                    if not row.empty:
                        wakuban = int(row.iloc[0].get("Wakuban", 0))
                        if wakuban > 0 and wakuban not in selected_wakuban:
                            selected_wakuban.append(wakuban)

                # 選択された馬の数に応じて券種を決定
                if selected_umaban:
                    n_horses = len(selected_umaban)
                    n_waku = len(selected_wakuban)

                    bet_dict[race_id] = {
                        # 単勝・複勝は1頭以上で購入
                        "tansho": selected_umaban[:],
                        "fukusho": selected_umaban[:],
                        # 2頭以上で馬連・馬単・ワイドを購入
                        "umaren": selected_umaban[:] if n_horses >= 2 else [],
                        "umatan": selected_umaban[:] if n_horses >= 2 else [],
                        "wide": selected_umaban[:] if n_horses >= 2 else [],
                        # 3頭以上で三連複・三連単を購入
                        "sanrenpuku": selected_umaban[:] if n_horses >= 3 else [],
                        "sanrentan": selected_umaban[:] if n_horses >= 3 else [],
                        # 枠連は2枠以上で購入
                        "wakuren": selected_wakuban[:] if n_waku >= 2 else [],
                    }
            return bet_dict

        # win5_selectionsがない場合は従来のiter_race_groupsで処理（後方互換性）
        for race_id, race_df in iter_race_groups(score_table):
            # 独自に選択
            selected_umaban: List[int] = []
            selected_wakuban: List[int] = []

            race_sorted = race_df.sort_values("proba", ascending=False)

            for i, (_, row) in enumerate(race_sorted.iterrows()):
                if i >= self.top_n:
                    break

                proba = float(row.get("proba", 0))
                if proba < self.min_proba:
                    continue

                try:
                    umaban = int(row.get("Umaban", 0))
                    if umaban > 0 and umaban not in selected_umaban:
                        selected_umaban.append(umaban)
                        wakuban = int(row.get("Wakuban", 0))
                        if wakuban > 0 and wakuban not in selected_wakuban:
                            selected_wakuban.append(wakuban)
                except (ValueError, TypeError):
                    continue

            # 選択された馬の数に応じて券種を決定
            if selected_umaban:
                n_horses = len(selected_umaban)
                n_waku = len(selected_wakuban)

                bet_dict[race_id] = {
                    # 単勝・複勝は1頭以上で購入
                    "tansho": selected_umaban[:],
                    "fukusho": selected_umaban[:],
                    # 2頭以上で馬連・馬単・ワイドを購入
                    "umaren": selected_umaban[:] if n_horses >= 2 else [],
                    "umatan": selected_umaban[:] if n_horses >= 2 else [],
                    "wide": selected_umaban[:] if n_horses >= 2 else [],
                    # 3頭以上で三連複・三連単を購入
                    "sanrenpuku": selected_umaban[:] if n_horses >= 3 else [],
                    "sanrentan": selected_umaban[:] if n_horses >= 3 else [],
                    # 枠連は2枠以上で購入
                    "wakuren": selected_wakuban[:] if n_waku >= 2 else [],
                }

        return bet_dict


class WIN5BasedHedgePolicy(BetPolicyCoverageBase):
    """
    WIN5ヘッジポリシー

    WIN5の5レースそれぞれで、1着候補の単勝と、
    2-3着候補の複勝を購入することでWIN5のリスクをヘッジする。

    WIN5が的中すれば大きな配当、不的中でも単勝・複勝で回収を狙う。

    Attributes:
        tansho_top_n: 単勝で購入する馬の数（デフォルト: 1）
        fukusho_top_n: 複勝で購入する馬の数（デフォルト: 3）
        min_proba_tansho: 単勝の最小確率（デフォルト: 0.0 = 閾値なし）
        min_proba_fukusho: 複勝の最小確率（デフォルト: 0.0 = 閾値なし）
    """

    def __init__(
        self,
        tansho_top_n: int = 1,
        fukusho_top_n: int = 3,
        min_proba_tansho: float = 0.0,  # WIN5対象レースは全て選択するため閾値なし
        min_proba_fukusho: float = 0.0,  # WIN5対象レースは全て選択するため閾値なし
        min_proba: float = None,  # 互換性のために追加（使用しない）
        **kwargs: Any,
    ) -> None:
        """
        初期化

        Args:
            tansho_top_n: 単勝で購入する馬の数
            fukusho_top_n: 複勝で購入する馬の数
            min_proba_tansho: 単勝の最小確率（デフォルト: 0.0 = 閾値なし）
            min_proba_fukusho: 複勝の最小確率（デフォルト: 0.0 = 閾値なし）
            min_proba: 互換性のためのパラメータ（min_proba_tanshoに適用）
            **kwargs: その他のキーワード引数（無視）
        """
        # min_probaが指定された場合はmin_proba_tanshoに適用
        if min_proba is not None:
            min_proba_tansho = min_proba
        self.tansho_top_n = tansho_top_n
        self.fukusho_top_n = fukusho_top_n
        self.min_proba_tansho = min_proba_tansho
        self.min_proba_fukusho = min_proba_fukusho

    def judge(
        self,
        score_table: pd.DataFrame,
        odds_table: pd.DataFrame = None,
        win5_selections: Optional[Dict[str, List[int]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        WIN5ヘッジ戦略で馬券選択

        Args:
            score_table: 予測スコアテーブル
            odds_table: オッズテーブル（未使用）
            win5_selections: WIN5予測で選択された馬 {race_id: [馬番リスト]}
                           指定された場合はこの馬を使用し、独自選択は行わない
            **kwargs: その他のキーワード引数

        Returns:
            Dict[str, Dict[str, List[int]]]: {race_id: {券種: [馬番リスト]}}
        """
        bet_dict: Dict[str, Dict[str, List[int]]] = {}

        for race_id, race_df in iter_race_groups(score_table):
            # WIN5予測結果が渡された場合はそれを使用
            if win5_selections and race_id in win5_selections:
                all_umaban = win5_selections[race_id]
                # 単勝は1着候補（先頭）、複勝は全選択馬
                tansho_umaban = all_umaban[: self.tansho_top_n]
                fukusho_umaban = all_umaban[: self.fukusho_top_n]
                # 枠番を取得
                all_wakuban: List[int] = []
                for umaban in all_umaban:
                    row = race_df[race_df["Umaban"].astype(int) == umaban]
                    if not row.empty:
                        wakuban = int(row.iloc[0].get("Wakuban", 0))
                        if wakuban > 0 and wakuban not in all_wakuban:
                            all_wakuban.append(wakuban)
            else:
                # WIN5予測結果がない場合は独自に選択（後方互換性）
                tansho_umaban: List[int] = []
                fukusho_umaban: List[int] = []
                all_wakuban: List[int] = []

                race_sorted = race_df.sort_values("proba", ascending=False)

                for i, (_, row) in enumerate(race_sorted.iterrows()):
                    proba = float(row.get("proba", 0))

                    try:
                        umaban = int(row.get("Umaban", 0))
                        if umaban <= 0:
                            continue

                        wakuban = int(row.get("Wakuban", 0))

                        # 単勝候補
                        if (
                            i < self.tansho_top_n
                            and proba >= self.min_proba_tansho
                            and umaban not in tansho_umaban
                        ):
                            tansho_umaban.append(umaban)

                        # 複勝候補
                        if (
                            i < self.fukusho_top_n
                            and proba >= self.min_proba_fukusho
                            and umaban not in fukusho_umaban
                        ):
                            fukusho_umaban.append(umaban)
                            if wakuban > 0 and wakuban not in all_wakuban:
                                all_wakuban.append(wakuban)

                    except (ValueError, TypeError):
                        continue

                all_umaban = list(set(tansho_umaban + fukusho_umaban))

            # 選択された馬の数に応じて券種を決定
            if all_umaban:
                n_horses = len(all_umaban)
                n_waku = len(all_wakuban)

                bet_dict[race_id] = {
                    # 単勝は1着候補、複勝は上位候補
                    "tansho": tansho_umaban[:] if tansho_umaban else all_umaban[:1],
                    "fukusho": fukusho_umaban[:] if fukusho_umaban else all_umaban[:],
                    # 2頭以上で馬連・馬単・ワイドを購入
                    "umaren": all_umaban[:] if n_horses >= 2 else [],
                    "umatan": all_umaban[:] if n_horses >= 2 else [],
                    "wide": all_umaban[:] if n_horses >= 2 else [],
                    # 3頭以上で三連複・三連単を購入
                    "sanrenpuku": all_umaban[:] if n_horses >= 3 else [],
                    "sanrentan": all_umaban[:] if n_horses >= 3 else [],
                    # 枠連は2枠以上で購入
                    "wakuren": all_wakuban[:] if n_waku >= 2 else [],
                }

        return bet_dict

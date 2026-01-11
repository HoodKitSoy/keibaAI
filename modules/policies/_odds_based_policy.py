"""
オッズベースポリシー

JRA-DBから取得したオッズに基づいて馬券選択を行うポリシー。
予測確率とオッズから期待値を計算し、閾値以上の馬を選択する。
"""

from itertools import combinations, permutations
from typing import Any, Dict, List, Tuple

import pandas as pd

from modules.policies._bet_policy import BetPolicyCoverageBase
from modules.policies._utils import iter_race_groups
from modules.preprocessing._odds_loader import OddsLoader


class OddsBasedExpectedValuePolicy(BetPolicyCoverageBase):
    """
    オッズベースの期待値ポリシー

    DBから取得した実際のオッズを使用して期待値を計算し、
    期待値が閾値以上の馬を選択する。

    Attributes:
        ev_threshold: 期待値閾値（デフォルト: 1.0）
        max_selections: 最大選択数（デフォルト: 5）
    """

    def __init__(
        self, ev_threshold: float = 1.0, max_selections: int = 5, **kwargs: Any
    ) -> None:
        """
        初期化

        Args:
            ev_threshold: 期待値閾値
            max_selections: 最大選択数
            **kwargs: 親クラスへのキーワード引数
        """
        super().__init__(**kwargs)
        self.ev_threshold = ev_threshold
        self.max_selections = max_selections

    def judge(
        self, score_table: pd.DataFrame, odds_table: pd.DataFrame = None, **kwargs: Any
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        オッズベースの期待値で馬券選択

        Args:
            score_table: 予測スコアテーブル（proba列必須、TanOdds列推奨）
            odds_table: オッズテーブル（未使用、score_tableにTanOddsがあれば使用）
            **kwargs: その他のキーワード引数

        Returns:
            Dict[str, Dict[str, List[int]]]: {race_id: {券種: [馬番リスト]}}
        """
        bet_dict: Dict[str, Dict[str, List[int]]] = {}

        # オッズ列がなければスコアから推定
        has_odds = "TanOdds" in score_table.columns

        for race_id, race_df in iter_race_groups(score_table):
            selected_umaban: List[int] = []
            selected_wakuban: List[int] = []

            # スコアでソート
            race_sorted = race_df.sort_values("proba", ascending=False)

            for _, row in race_sorted.iterrows():
                proba = float(row.get("proba", 0))
                if proba <= 0:
                    continue

                # オッズを取得（なければ確率から推定）
                if has_odds:
                    odds = float(row.get("TanOdds", 0))
                    if odds <= 0:
                        odds = 1.0 / proba if proba > 0 else 100.0
                else:
                    odds = 1.0 / proba if proba > 0 else 100.0

                # 期待値を計算
                ev = proba * odds

                # 期待値が閾値以上なら選択
                if ev >= self.ev_threshold:
                    try:
                        umaban = int(row.get("Umaban", 0))
                        if umaban > 0 and umaban not in selected_umaban:
                            selected_umaban.append(umaban)
                            # 枠番も追加
                            wakuban = int(row.get("Wakuban", 0))
                            if wakuban > 0 and wakuban not in selected_wakuban:
                                selected_wakuban.append(wakuban)
                    except (ValueError, TypeError):
                        continue

                if len(selected_umaban) >= self.max_selections:
                    break

            # 馬券辞書に追加
            if selected_umaban:
                bet_dict[race_id] = {
                    "tansho": selected_umaban[:],
                    "fukusho": selected_umaban[:],
                    "umaren": selected_umaban[:],
                    "umatan": selected_umaban[:],
                    "wide": selected_umaban[:],
                    "sanrenpuku": selected_umaban[:],
                    "sanrentan": selected_umaban[:],
                    "wakuren": selected_wakuban[:],
                }

        return bet_dict


class OddsValueTieredPolicy(BetPolicyCoverageBase):
    """
    オッズ層別期待値ポリシー

    オッズの範囲（穴馬/中穴/本命）に応じて異なる期待値閾値を適用する。
    穴馬はより高い期待値を要求し、本命はより低い期待値でも許容する。

    Attributes:
        honmei_odds_max: 本命馬のオッズ上限（デフォルト: 5.0）
        chuuana_odds_max: 中穴馬のオッズ上限（デフォルト: 20.0）
        honmei_ev_threshold: 本命馬の期待値閾値（デフォルト: 0.8）
        chuuana_ev_threshold: 中穴馬の期待値閾値（デフォルト: 1.0）
        anauma_ev_threshold: 穴馬の期待値閾値（デフォルト: 1.2）
        max_selections: 最大選択数（デフォルト: 5）
    """

    def __init__(
        self,
        honmei_odds_max: float = 5.0,
        chuuana_odds_max: float = 20.0,
        honmei_ev_threshold: float = 0.8,
        chuuana_ev_threshold: float = 1.0,
        anauma_ev_threshold: float = 1.2,
        max_selections: int = 5,
        **kwargs: Any,
    ) -> None:
        """
        初期化

        Args:
            honmei_odds_max: 本命馬のオッズ上限
            chuuana_odds_max: 中穴馬のオッズ上限
            honmei_ev_threshold: 本命馬の期待値閾値
            chuuana_ev_threshold: 中穴馬の期待値閾値
            anauma_ev_threshold: 穴馬の期待値閾値
            max_selections: 最大選択数
            **kwargs: 親クラスへのキーワード引数
        """
        super().__init__(**kwargs)
        self.honmei_odds_max = honmei_odds_max
        self.chuuana_odds_max = chuuana_odds_max
        self.honmei_ev_threshold = honmei_ev_threshold
        self.chuuana_ev_threshold = chuuana_ev_threshold
        self.anauma_ev_threshold = anauma_ev_threshold
        self.max_selections = max_selections

    def _get_ev_threshold(self, odds: float) -> float:
        """
        オッズに応じた期待値閾値を取得

        Args:
            odds: 単勝オッズ

        Returns:
            float: 期待値閾値
        """
        if odds <= self.honmei_odds_max:
            return self.honmei_ev_threshold
        elif odds <= self.chuuana_odds_max:
            return self.chuuana_ev_threshold
        else:
            return self.anauma_ev_threshold

    def judge(
        self, score_table: pd.DataFrame, odds_table: pd.DataFrame = None, **kwargs: Any
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        オッズ層別の期待値で馬券選択

        Args:
            score_table: 予測スコアテーブル
            odds_table: オッズテーブル（未使用）
            **kwargs: その他のキーワード引数

        Returns:
            Dict[str, Dict[str, List[int]]]: {race_id: {券種: [馬番リスト]}}
        """
        bet_dict: Dict[str, Dict[str, List[int]]] = {}
        has_odds = "TanOdds" in score_table.columns

        for race_id, race_df in iter_race_groups(score_table):
            selected_umaban: List[int] = []
            selected_wakuban: List[int] = []

            race_sorted = race_df.sort_values("proba", ascending=False)

            for _, row in race_sorted.iterrows():
                proba = float(row.get("proba", 0))
                if proba <= 0:
                    continue

                if has_odds:
                    odds = float(row.get("TanOdds", 0))
                    if odds <= 0:
                        odds = 1.0 / proba if proba > 0 else 100.0
                else:
                    odds = 1.0 / proba if proba > 0 else 100.0

                ev = proba * odds
                ev_threshold = self._get_ev_threshold(odds)

                if ev >= ev_threshold:
                    try:
                        umaban = int(row.get("Umaban", 0))
                        if umaban > 0 and umaban not in selected_umaban:
                            selected_umaban.append(umaban)
                            wakuban = int(row.get("Wakuban", 0))
                            if wakuban > 0 and wakuban not in selected_wakuban:
                                selected_wakuban.append(wakuban)
                    except (ValueError, TypeError):
                        continue

                if len(selected_umaban) >= self.max_selections:
                    break

            if selected_umaban:
                bet_dict[race_id] = {
                    "tansho": selected_umaban[:],
                    "fukusho": selected_umaban[:],
                    "umaren": selected_umaban[:],
                    "umatan": selected_umaban[:],
                    "wide": selected_umaban[:],
                    "sanrenpuku": selected_umaban[:],
                    "sanrentan": selected_umaban[:],
                    "wakuren": selected_wakuban[:],
                }

        return bet_dict


class OddsRangeFilterPolicy(BetPolicyCoverageBase):
    """
    オッズ範囲フィルタポリシー

    指定したオッズ範囲内の馬のみを選択対象とする。
    WIN5では中穴狙いが有効なケースがあるため、オッズ範囲で絞り込む。

    Attributes:
        odds_min: 最小オッズ（デフォルト: 3.0）
        odds_max: 最大オッズ（デフォルト: 30.0）
        min_proba: 最小確率（デフォルト: 0.05）
        max_selections: 最大選択数（デフォルト: 3）
    """

    def __init__(
        self,
        odds_min: float = 3.0,
        odds_max: float = 30.0,
        min_proba: float = 0.05,
        max_selections: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        初期化

        Args:
            odds_min: 最小オッズ
            odds_max: 最大オッズ
            min_proba: 最小確率
            max_selections: 最大選択数
            **kwargs: 親クラスへのキーワード引数
        """
        super().__init__(**kwargs)
        self.odds_min = odds_min
        self.odds_max = odds_max
        self.min_proba = min_proba
        self.max_selections = max_selections

    def judge(
        self, score_table: pd.DataFrame, odds_table: pd.DataFrame = None, **kwargs: Any
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        オッズ範囲でフィルタして馬券選択

        Args:
            score_table: 予測スコアテーブル
            odds_table: オッズテーブル（未使用）
            **kwargs: その他のキーワード引数

        Returns:
            Dict[str, Dict[str, List[int]]]: {race_id: {券種: [馬番リスト]}}
        """
        bet_dict: Dict[str, Dict[str, List[int]]] = {}
        has_odds = "TanOdds" in score_table.columns

        for race_id, race_df in iter_race_groups(score_table):
            selected_umaban: List[int] = []
            selected_wakuban: List[int] = []

            race_sorted = race_df.sort_values("proba", ascending=False)

            for _, row in race_sorted.iterrows():
                proba = float(row.get("proba", 0))
                if proba < self.min_proba:
                    continue

                if has_odds:
                    odds = float(row.get("TanOdds", 0))
                else:
                    odds = 1.0 / proba if proba > 0 else 100.0

                # オッズ範囲でフィルタ
                if odds < self.odds_min or odds > self.odds_max:
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

                if len(selected_umaban) >= self.max_selections:
                    break

            if selected_umaban:
                bet_dict[race_id] = {
                    "tansho": selected_umaban[:],
                    "fukusho": selected_umaban[:],
                    "umaren": selected_umaban[:],
                    "umatan": selected_umaban[:],
                    "wide": selected_umaban[:],
                    "sanrenpuku": selected_umaban[:],
                    "sanrentan": selected_umaban[:],
                    "wakuren": selected_wakuban[:],
                }

        return bet_dict


class MultiOddsExpectedValuePolicy(BetPolicyCoverageBase):
    """
    マルチオッズ期待値ポリシー（並列処理最適化版）

    各券種別のJRA-DBオッズテーブルを使用して、券種ごとに期待値を計算し、
    期待値が閾値以上の組み合わせのみを選択する。

    並列処理とバッチプリロードにより高速化されています。

    Attributes:
        ev_threshold: 期待値閾値（デフォルト: 1.0）
        max_selections: 最大選択馬数（デフォルト: 5）
        db_dir: オッズデータディレクトリ
        use_today: 当日データを使用するか
        n_workers: 並列ワーカー数
    """

    def __init__(
        self,
        ev_threshold: float = 1.0,
        max_selections: int = 5,
        db_dir: str = None,
        use_today: bool = False,
        n_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        初期化

        Args:
            ev_threshold: 期待値閾値
            max_selections: 最大選択馬数
            db_dir: オッズデータディレクトリ（Noneの場合はデフォルト）
            use_today: 当日データ（s_プレフィックス）を使用するか
            n_workers: 並列ワーカー数（デフォルト: 4）
            **kwargs: 親クラスへのキーワード引数
        """
        super().__init__(**kwargs)
        self.ev_threshold = ev_threshold
        self.max_selections = max_selections
        self.use_today = use_today
        self.n_workers = n_workers
        self.odds_loader = OddsLoader(db_dir=db_dir)
        # 事前にインデックスを構築
        self._preload_odds_indices()

    def _preload_odds_indices(self) -> None:
        """
        全オッズテーブルのインデックスを事前構築して高速化
        """
        # 各テーブルのインデックスを事前に構築
        for table_name in [
            "tanpuku",
            "waku",
            "umaren",
            "wide",
            "umatan",
            "sanren",
            "sanrentan",
        ]:
            self.odds_loader._get_indexed_table(table_name, self.use_today)

    def _parse_race_id(self, race_id: str) -> Tuple[str, str, str, str, str, str]:
        """
        race_idから6列キーを抽出

        Args:
            race_id: "YYYYMMDD-JJKKNN-RR"形式のレースID

        Returns:
            Tuple[str, ...]: (Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum)
        """
        parts = race_id.split("-")
        if len(parts) >= 3:
            date_part = parts[0]  # YYYYMMDD
            jkn_part = parts[1]  # JJKKNN (JyoCD 2桁 + Kaiji 2桁 + Nichiji 2桁)
            race_part = parts[2]  # RR (RaceNum 2桁)
            year = date_part[:4]
            monthday = date_part[4:8]
            jyo_cd = jkn_part[:2]
            kaiji = jkn_part[2:4]
            nichiji = jkn_part[4:6]
            race_num = race_part[:2]
            return (year, monthday, jyo_cd, kaiji, nichiji, race_num)
        return ("", "", "", "", "", "")

    def _calc_combination_prob(
        self, proba_dict: Dict[int, float], horses: Tuple[int, ...]
    ) -> float:
        """
        組み合わせの確率を計算（独立性を仮定）

        Args:
            proba_dict: {馬番: 確率} の辞書
            horses: 馬番のタプル

        Returns:
            float: 組み合わせの確率
        """
        prob = 1.0
        for h in horses:
            prob *= proba_dict.get(h, 0.0)
        return prob

    def _process_single_race(
        self,
        race_id: str,
        race_df: pd.DataFrame,
    ) -> Tuple[str, Dict[str, List]]:
        """
        単一レースの処理（並列処理用）

        Args:
            race_id: レースID
            race_df: レースデータフレーム

        Returns:
            Tuple[str, Dict[str, List]]: (race_id, 券種別選択結果)
        """
        # レースキーを解析
        year, monthday, jyo_cd, kaiji, nichiji, race_num = self._parse_race_id(race_id)
        if not year:
            return (race_id, {})

        # 馬ごとの確率と馬番/枠番を取得
        proba_dict: Dict[int, float] = {}
        wakuban_dict: Dict[int, int] = {}
        race_sorted = race_df.sort_values("proba", ascending=False)

        for _, row in race_sorted.iterrows():
            try:
                umaban = int(row.get("Umaban", 0))
                proba = float(row.get("proba", 0))
                wakuban = int(row.get("Wakuban", 0))
                if umaban > 0 and proba > 0:
                    proba_dict[umaban] = proba
                    wakuban_dict[umaban] = wakuban
            except (ValueError, TypeError):
                continue

        # 上位N頭を選択候補とする
        top_horses = list(proba_dict.keys())[: self.max_selections]
        top_wakuban = list(
            set(
                wakuban_dict.get(h, 0) for h in top_horses if wakuban_dict.get(h, 0) > 0
            )
        )

        if not top_horses:
            return (race_id, {})

        # 各券種のオッズを取得（インデックス化済みなので高速）
        all_odds = self.odds_loader.get_race_odds_all(
            year, monthday, jyo_cd, kaiji, nichiji, race_num, self.use_today
        )

        result: Dict[str, List] = {
            "tansho": [],
            "fukusho": [],
            "wakuren": [],
            "umaren": [],
            "umatan": [],
            "wide": [],
            "sanrenpuku": [],
            "sanrentan": [],
        }

        # 単勝: 期待値計算
        tansho_odds = all_odds.get("tansho", {})
        for h in top_horses:
            odds = tansho_odds.get(h, 0)
            proba = proba_dict.get(h, 0)
            if odds > 0 and proba > 0:
                ev = proba * odds
                if ev >= self.ev_threshold:
                    result["tansho"].append(h)

        # 複勝: 期待値計算（最低オッズを使用）
        fukusho_odds = all_odds.get("fukusho", {})
        for h in top_horses:
            odds_tuple = fukusho_odds.get(h, (0, 0))
            odds_low = odds_tuple[0] if isinstance(odds_tuple, tuple) else odds_tuple
            proba = proba_dict.get(h, 0)
            proba_fukusho = min(proba * 3, 1.0)
            if odds_low > 0 and proba_fukusho > 0:
                ev = proba_fukusho * odds_low
                if ev >= self.ev_threshold:
                    result["fukusho"].append(h)

        # 枠連: 期待値計算
        wakuren_odds = all_odds.get("wakuren", {})
        for w1, w2 in combinations(top_wakuban, 2):
            key = (min(w1, w2), max(w1, w2))
            odds = wakuren_odds.get(key, 0)
            if odds > 0:
                horses_in_w1 = [
                    h for h, w in wakuban_dict.items() if w == w1 and h in top_horses
                ]
                horses_in_w2 = [
                    h for h, w in wakuban_dict.items() if w == w2 and h in top_horses
                ]
                prob_w1 = sum(proba_dict.get(h, 0) for h in horses_in_w1)
                prob_w2 = sum(proba_dict.get(h, 0) for h in horses_in_w2)
                prob = prob_w1 * prob_w2
                if prob > 0:
                    ev = prob * odds
                    if ev >= self.ev_threshold:
                        result["wakuren"].append(key)

        # 馬連: 期待値計算
        umaren_odds = all_odds.get("umaren", {})
        for h1, h2 in combinations(top_horses, 2):
            key = (min(h1, h2), max(h1, h2))
            odds = umaren_odds.get(key, 0)
            if odds > 0:
                prob = self._calc_combination_prob(proba_dict, (h1, h2))
                ev = prob * odds
                if ev >= self.ev_threshold:
                    result["umaren"].append(key)

        # ワイド: 期待値計算（最低オッズを使用）
        wide_odds = all_odds.get("wide", {})
        for h1, h2 in combinations(top_horses, 2):
            key = (min(h1, h2), max(h1, h2))
            odds_tuple = wide_odds.get(key, (0, 0))
            odds_low = odds_tuple[0] if isinstance(odds_tuple, tuple) else odds_tuple
            if odds_low > 0:
                prob = min(self._calc_combination_prob(proba_dict, (h1, h2)) * 9, 1.0)
                ev = prob * odds_low
                if ev >= self.ev_threshold:
                    result["wide"].append(key)

        # 馬単: 期待値計算
        umatan_odds = all_odds.get("umatan", {})
        for h1, h2 in permutations(top_horses, 2):
            key = (h1, h2)
            odds = umatan_odds.get(key, 0)
            if odds > 0:
                prob = self._calc_combination_prob(proba_dict, (h1, h2))
                ev = prob * odds
                if ev >= self.ev_threshold:
                    result["umatan"].append(key)

        # 三連複: 期待値計算
        sanren_odds = all_odds.get("sanrenpuku", {})
        for h1, h2, h3 in combinations(top_horses, 3):
            key = tuple(sorted([h1, h2, h3]))
            odds = sanren_odds.get(key, 0)
            if odds > 0:
                prob = self._calc_combination_prob(proba_dict, (h1, h2, h3))
                ev = prob * odds
                if ev >= self.ev_threshold:
                    result["sanrenpuku"].append(key)

        # 三連単: 期待値計算
        sanrentan_odds = all_odds.get("sanrentan", {})
        for h1, h2, h3 in permutations(top_horses, 3):
            key = (h1, h2, h3)
            odds = sanrentan_odds.get(key, 0)
            if odds > 0:
                prob = self._calc_combination_prob(proba_dict, (h1, h2, h3))
                ev = prob * odds
                if ev >= self.ev_threshold:
                    result["sanrentan"].append(key)

        return (race_id, result)

    def judge(
        self, score_table: pd.DataFrame, odds_table: pd.DataFrame = None, **kwargs: Any
    ) -> Dict[str, Dict[str, List]]:
        """
        マルチオッズベースの期待値で馬券選択（並列処理版）

        各券種のオッズテーブルから実際のオッズを取得し、
        予測確率との積で期待値を計算する。

        Args:
            score_table: 予測スコアテーブル（proba列必須）
            odds_table: オッズテーブル（未使用）
            **kwargs: その他のキーワード引数

        Returns:
            Dict[str, Dict[str, List]]: {race_id: {券種: [馬番/組み合わせリスト]}}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        bet_dict: Dict[str, Dict[str, List]] = {}

        # レースデータを事前に収集
        race_data_list = list(iter_race_groups(score_table))

        # 並列処理で各レースを処理
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._process_single_race, race_id, race_df): race_id
                for race_id, race_df in race_data_list
            }

            for future in as_completed(futures):
                try:
                    race_id, result = future.result()
                    if any(result.values()):
                        bet_dict[race_id] = result
                except Exception:
                    continue

        return bet_dict

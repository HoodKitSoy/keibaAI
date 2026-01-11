from typing import Dict, List, Tuple


class Win5Tickets:
    """
    WIN5（重勝式）の払戻計算ヘルパー。

    - 1日ごとに対象5レースの1着馬番（正解）を全て当てると的中。
    - 払戻は1口あたりの金額（通常100円）として与えられる。
    - 実装では amount を1口金額（円）として扱い、購入点数×amount を賭け金とする。
    """

    def __init__(self, answer_map: Dict[str, Tuple[List[int], int]]):
        """
        Parameters
        ----------
        answer_map : dict
            date_key -> (winners5, payout) を格納する辞書
            - winners5: [w1, w2, w3, w4, w5]
            - payout:   100円あたりの払戻金（円）
        """
        self.answer_map = answer_map

    def bet_win5(self, date_key: str, candidates: List[List[int]], amount: float = 100.0):
        """
        WIN5 を購入し、賭け金と払戻金を返す。

        Parameters
        ----------
        date_key : str
            YYYYMMDD の日付キー
        candidates : List[List[int]]
            5レースそれぞれの購入候補馬番のリスト（5要素のリスト、各要素は馬番のリスト）
            例: [[1,3],[5],[4,7],[1],[13]]
        amount : float, default 100.0
            1点あたりの購入金額（円）

        Returns
        -------
        n_bets : int
            購入点数（候補の直積の総数）
        bet_amount : float
            総賭け金（円）
        return_amount : float
            払戻金（円）
        """
        if date_key not in self.answer_map:
            return 0, 0.0, 0.0

        # 5リストの直積点数 = それぞれの候補数の積
        if len(candidates) != 5:
            return 0, 0.0, 0.0
        sizes = [len(lst) for lst in candidates]
        if any(s == 0 for s in sizes):
            return 0, 0.0, 0.0

        n_bets = 1
        for s in sizes:
            n_bets *= s
        bet_amount = n_bets * amount

        winners, payout = self.answer_map[date_key]
        if len(winners) != 5:
            return n_bets, bet_amount, 0.0

        # 的中判定：各レースで候補の中に勝ち馬番が含まれているか
        hit = all(winners[i] in candidates[i] for i in range(5))
        return_amount = (payout * amount / 100.0) if hit else 0.0
        return n_bets, bet_amount, return_amount

"""
WIN5（重勝式）の払戻計算ヘルパー

WIN5は総取り方式（パリミュチュエル方式）のため：
- 総プール = PayJyushosiki × TekichuHyo + シミュレーションの賭け金
- 的中した場合の払戻金 = 総プール ÷ (TekichuHyo + シミュレーションの的中数)
"""

from typing import Any, Dict, List, Tuple


class Win5Tickets:
    """
    WIN5（重勝式）の払戻計算ヘルパー。

    - 1日ごとに対象5レースの1着馬番（正解）を全て当てると的中。
    - WIN5は総取り方式のため、賭金込みのプールを的中者で分配。
    - 実装では amount を1口金額（円）として扱い、購入点数×amount を賭け金とする。
    """

    def __init__(self, answer_map: Dict[str, Tuple[List[int], int, int]]):
        """
        Parameters
        ----------
        answer_map : dict
            date_key -> (winners5, total_payout, tekichu_hyo) を格納する辞書
            - winners5: [w1, w2, w3, w4, w5] - 各レースの1着馬番
            - total_payout: 1票あたりの払戻金（円）（JRA公式データ）
            - tekichu_hyo: 的中票数
        """
        self.answer_map = answer_map

    def bet_win5(
        self, date_key: str, candidates: List[List[int]], amount: float = 100.0
    ) -> Tuple[int, float, float]:
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
            払戻金（円）- プール分配方式で計算
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

        winners, payout_per_ticket, tekichu_hyo = self.answer_map[date_key]
        if len(winners) != 5:
            return n_bets, bet_amount, 0.0

        # 的中判定：各レースで候補の中に勝ち馬番が含まれているか
        hit = all(winners[i] in candidates[i] for i in range(5))

        if hit:
            # WIN5は総取り方式（パリミュチュエル方式）
            # 総プール = 1票あたり払戻金 × 元の的中票数 + 自分の賭け金
            # 注: JRAデータのpayout_per_ticketは既に1票あたりの払戻金
            #     total_pool = payout_per_ticket * tekichu_hyo がその日の総額
            if tekichu_hyo > 0:
                total_pool = payout_per_ticket * tekichu_hyo + bet_amount
                # 自分の的中票数（的中した組み合わせは1つのみ）
                my_hit_count = 1  # 正しい組み合わせは1つだけ
                new_total_tickets = tekichu_hyo + my_hit_count
                # 1票あたりの新しい払戻金
                new_payout_per_ticket = total_pool / new_total_tickets
                return_amount = new_payout_per_ticket * (amount / 100.0)
            else:
                # 的中者なしの場合（キャリーオーバー等）
                # 自分だけが的中者になる
                total_pool = bet_amount  # プールは自分の賭け金のみ（実際はキャリーオーバー額もあるが不明）
                return_amount = total_pool  # 全額が自分のもの
        else:
            return_amount = 0.0

        return n_bets, bet_amount, return_amount

    def bet_win5_detailed(
        self, date_key: str, candidates: List[List[int]], amount: float = 100.0
    ) -> Dict[str, Any]:
        """
        WIN5 を購入し、詳細な結果を返す。

        Parameters
        ----------
        date_key : str
            YYYYMMDD の日付キー
        candidates : List[List[int]]
            5レースそれぞれの購入候補馬番のリスト
        amount : float, default 100.0
            1点あたりの購入金額（円）

        Returns
        -------
        Dict: 詳細な結果
            - date: 日付
            - n_bets: 購入点数
            - bet_amount: 総賭け金
            - return_amount: 払戻金
            - hit: 的中したか
            - winners: 正解馬番リスト
            - candidates: 購入候補馬番リスト
            - total_payout: 元の総払戻プール
            - tekichu_hyo: 元の的中票数
            - payout_per_ticket: 計算後の1票あたり払戻金
        """
        result = {
            "date": date_key,
            "n_bets": 0,
            "bet_amount": 0.0,
            "return_amount": 0.0,
            "hit": False,
            "winners": [],
            "candidates": candidates,
            "total_payout": 0,
            "tekichu_hyo": 0,
            "payout_per_ticket": 0.0,
        }

        if date_key not in self.answer_map:
            return result

        # 5リストの直積点数
        if len(candidates) != 5:
            return result
        sizes = [len(lst) for lst in candidates]
        if any(s == 0 for s in sizes):
            return result

        n_bets = 1
        for s in sizes:
            n_bets *= s
        bet_amount = n_bets * amount

        winners, payout_per_ticket, tekichu_hyo = self.answer_map[date_key]

        result["n_bets"] = n_bets
        result["bet_amount"] = bet_amount
        result["winners"] = winners
        result["tekichu_hyo"] = tekichu_hyo

        # 元の総払戻プール
        if tekichu_hyo > 0:
            original_pool = payout_per_ticket * tekichu_hyo
        else:
            original_pool = 0
        result["total_payout"] = original_pool

        if len(winners) != 5:
            return result

        # 的中判定
        hit = all(winners[i] in candidates[i] for i in range(5))
        result["hit"] = hit

        if hit:
            if tekichu_hyo > 0:
                # 総プール = 元のプール + 自分の賭け金
                total_pool = original_pool + bet_amount
                # 自分の的中票数（正しい組み合わせは1つ）
                my_hit_count = 1
                new_total_tickets = tekichu_hyo + my_hit_count
                # 新しい1票あたり払戻金
                new_payout_per_ticket = total_pool / new_total_tickets
                result["payout_per_ticket"] = new_payout_per_ticket
                result["return_amount"] = new_payout_per_ticket * (amount / 100.0)
            else:
                # 的中者なしの場合、自分だけが的中者
                result["payout_per_ticket"] = bet_amount
                result["return_amount"] = bet_amount

        return result

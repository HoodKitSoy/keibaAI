# modules/simulation/_betting_tickets.py

from itertools import permutations

import pandas as pd
from scipy.special import comb

from modules.preprocessing import ReturnProcessor


class BettingTickets:
    """
    馬券の買い方と，賭けた時のリターンを計算する．
    """

    def __init__(self, returnProcessor: ReturnProcessor) -> None:
        self.__returnTables = returnProcessor.preprocessed_data
        self.__returnTablesTansho = self.__returnTables["tansho"]
        self.__returnTablesFukusho = self.__returnTables["fukusho"]
        self.__returnTablesWakuren = self.__returnTables.get("wakuren", pd.DataFrame())
        self.__returnTablesUmaren = self.__returnTables["umaren"]
        self.__returnTablesUmatan = self.__returnTables["umatan"]
        self.__returnTablesWide = self.__returnTables["wide"]
        self.__returnTablesSanrenpuku = self.__returnTables["sanrenpuku"]
        self.__returnTablesSanrentan = self.__returnTables["sanrentan"]

    def bet_tansho(self, race_id: str, umaban: list, amount: float):
        n_bets = len(umaban)
        if n_bets == 0:
            return 0, 0, 0
        bet_amount = n_bets * amount

        # レースIDが払戻テーブルに存在するか確認
        if race_id not in self.__returnTablesTansho.index:
            print(
                f"警告: レースID '{race_id}' は払戻テーブルに存在しません。スキップします。"
            )
            return n_bets, bet_amount, 0  # 的中なしとして処理

        table_1R = self.__returnTablesTansho.loc[race_id]
        # loc の結果が Series の場合と DataFrame の場合に対応
        if isinstance(table_1R, pd.Series):
            wins = [int(table_1R["win"])]
            pays = [int(table_1R["return"])]
        else:
            wins = table_1R["win"].astype(int).tolist()
            pays = table_1R["return"].astype(int).tolist()
        hit = any(w in umaban for w in wins)
        pay = pays[0] if pays else 0
        return_amount = (pay * amount / 100) if hit else 0
        return n_bets, bet_amount, return_amount

    def bet_fukusho(self, race_id: str, umaban: list, amount: float):
        n_bets = len(umaban)
        if n_bets == 0:
            return 0, 0, 0
        bet_amount = n_bets * amount

        # レースIDが払戻テーブルに存在するか確認
        if race_id not in self.__returnTablesFukusho.index:
            print(
                f"警告: レースID '{race_id}' は払戻テーブル(複勝)に存在しません。スキップします。"
            )
            return n_bets, bet_amount, 0  # 的中なしとして処理

        table_1R = self.__returnTablesFukusho.loc[race_id]
        if isinstance(table_1R, pd.Series):
            wins = [
                int(table_1R.get("win_0", 0)),
                int(table_1R.get("win_1", 0)),
                int(table_1R.get("win_2", 0)),
            ]
            pays = [
                int(table_1R.get("return_0", 0)),
                int(table_1R.get("return_1", 0)),
                int(table_1R.get("return_2", 0)),
            ]
            return_amount = sum(
                (w in umaban) * p * amount / 100 for w, p in zip(wins, pays)
            )
        else:
            hits = table_1R[["win_0", "win_1", "win_2"]].isin(umaban)
            return_amount = float(
                (
                    hits.values
                    * table_1R[["return_0", "return_1", "return_2"]].values
                    * amount
                    / 100
                ).sum()
            )
        return n_bets, bet_amount, return_amount

    def bet_wakuren_box(self, race_id: str, wakuban: list, amount: float):
        """
        枠連BOX馬券のリターンを計算する

        Args:
            race_id: レースID
            wakuban: 枠番リスト（1〜8）
            amount: 1点あたりの金額

        Returns:
            Tuple[int, float, float]: (点数, 賭け金額, 払戻金額)
        """
        n_bets = comb(len(wakuban), 2)
        if n_bets == 0:
            return 0, 0, 0
        bet_amount = n_bets * amount

        # レースIDが払戻テーブルに存在するか確認
        if (
            self.__returnTablesWakuren.empty
            or race_id not in self.__returnTablesWakuren.index
        ):
            # 枠連データがない場合は静かにスキップ
            return n_bets, bet_amount, 0

        table_1R = self.__returnTablesWakuren.loc[race_id]
        if isinstance(table_1R, pd.Series):
            pair = {int(table_1R["win_0"]), int(table_1R["win_1"])}
            hit = pair.issubset(set(wakuban))
            pay = int(table_1R["return"])
            return_amount = (pay * amount / 100) if hit else 0
        else:
            # 複数候補があっても先頭のみ
            row0 = table_1R.iloc[0]
            pair = {int(row0["win_0"]), int(row0["win_1"])}
            hit = pair.issubset(set(wakuban))
            pay = int(row0["return"])
            return_amount = (pay * amount / 100) if hit else 0
        return n_bets, bet_amount, return_amount

    def bet_umaren_box(self, race_id: str, umaban: list, amount: float):
        n_bets = comb(len(umaban), 2)
        if n_bets == 0:
            return 0, 0, 0
        bet_amount = n_bets * amount

        # レースIDが払戻テーブルに存在するか確認
        if race_id not in self.__returnTablesUmaren.index:
            print(
                f"警告: レースID '{race_id}' は払戻テーブル(馬連)に存在しません。スキップします。"
            )
            return n_bets, bet_amount, 0  # 的中なしとして処理

        table_1R = self.__returnTablesUmaren.loc[race_id]
        if isinstance(table_1R, pd.Series):
            pair = {int(table_1R["win_0"]), int(table_1R["win_1"])}
            hit = pair.issubset(set(umaban))
            pay = int(table_1R["return"])
            return_amount = (pay * amount / 100) if hit else 0
        else:
            # 複数候補があっても先頭のみ（設計方針に合わせ簡略）
            row0 = table_1R.iloc[0]
            pair = {int(row0["win_0"]), int(row0["win_1"])}
            hit = pair.issubset(set(umaban))
            pay = int(row0["return"])
            return_amount = (pay * amount / 100) if hit else 0
        return n_bets, bet_amount, return_amount

    def _bet_umatan(self, race_id: str, pair: list, amount: float):
        if len(pair) != 2:
            return 0, 0, 0

        # レースIDが払戻テーブルに存在するか確認
        if race_id not in self.__returnTablesUmatan.index:
            print(
                f"警告: レースID '{race_id}' は払戻テーブル(馬単)に存在しません。スキップします。"
            )
            return 1, amount, 0  # 的中なしとして処理

        table_1R = self.__returnTablesUmatan.loc[race_id]
        if isinstance(table_1R, pd.Series):
            hit = (
                int(table_1R["win_0"]) == pair[0] and int(table_1R["win_1"]) == pair[1]
            )
            pay = int(table_1R["return"])
            return_amount = (pay * amount / 100) if hit else 0
        else:
            # いずれか一致で的中
            mask = (table_1R["win_0"].astype(int) == pair[0]) & (
                table_1R["win_1"].astype(int) == pair[1]
            )
            if mask.any():
                pay = int(table_1R.loc[mask, "return"].iloc[0])
                return_amount = pay * amount / 100
            else:
                return_amount = 0
        return 1, amount, return_amount

    def bet_umatan_box(self, race_id: str, umaban: list, amount: float):
        n_bets = 0
        bet_amount = 0
        return_amount = 0
        for pair in permutations(umaban, 2):
            nb, ba, ra = self._bet_umatan(race_id, list(pair), amount)
            n_bets += nb
            bet_amount += ba
            return_amount += ra
        return n_bets, bet_amount, return_amount

    def bet_wide_box(self, race_id: str, umaban: list, amount: float):
        n_bets = comb(len(umaban), 2)
        if n_bets == 0:
            return 0, 0, 0
        bet_amount = n_bets * amount

        # レースIDが払戻テーブルに存在するか確認
        if race_id not in self.__returnTablesWide.index:
            print(
                f"警告: レースID '{race_id}' は払戻テーブル(ワイド)に存在しません。スキップします。"
            )
            return n_bets, bet_amount, 0  # 的中なしとして処理

        table_1R = self.__returnTablesWide.loc[race_id]
        if isinstance(table_1R, pd.Series):
            hit = int(table_1R["win_0"]) in umaban and int(table_1R["win_1"]) in umaban
            pay = int(table_1R["return"])
            return_amount = (pay * amount / 100) if hit else 0
        else:
            hits = table_1R["win_0"].astype(int).isin(umaban) & table_1R[
                "win_1"
            ].astype(int).isin(umaban)
            return_amount = float(
                (hits.astype(int) * table_1R["return"].astype(int) * amount / 100).sum()
            )
        return n_bets, bet_amount, return_amount

    def bet_sanrenpuku_box(self, race_id: str, umaban: list, amount: float):
        n_bets = comb(len(umaban), 3)
        if n_bets == 0:
            return 0, 0, 0
        bet_amount = n_bets * amount

        # レースIDが払戻テーブルに存在するか確認
        if race_id not in self.__returnTablesSanrenpuku.index:
            print(
                f"警告: レースID '{race_id}' は払戻テーブル(三連複)に存在しません。スキップします。"
            )
            return n_bets, bet_amount, 0  # 的中なしとして処理

        table_1R = self.__returnTablesSanrenpuku.loc[race_id]
        if isinstance(table_1R, pd.Series):
            target = {
                int(table_1R["win_0"]),
                int(table_1R["win_1"]),
                int(table_1R["win_2"]),
            }
            hit = target.issubset(set(umaban))
            pay = int(table_1R["return"])
            return_amount = (pay * amount / 100) if hit else 0
        else:
            row0 = table_1R.iloc[0]
            target = {int(row0["win_0"]), int(row0["win_1"]), int(row0["win_2"])}
            hit = target.issubset(set(umaban))
            pay = int(row0["return"])
            return_amount = (pay * amount / 100) if hit else 0
        return n_bets, bet_amount, return_amount

    def _bet_sanrentan(self, race_id: str, trio: list, amount: float):
        if len(trio) != 3:
            return 0, 0, 0

        # レースIDが払戻テーブルに存在するか確認
        if race_id not in self.__returnTablesSanrentan.index:
            print(
                f"警告: レースID '{race_id}' は払戻テーブル(三連単)に存在しません。スキップします。"
            )
            return 1, amount, 0  # 的中なしとして処理

        table_1R = self.__returnTablesSanrentan.loc[race_id]
        if isinstance(table_1R, pd.Series):
            hit = (
                int(table_1R["win_0"]) == trio[0]
                and int(table_1R["win_1"]) == trio[1]
                and int(table_1R["win_2"]) == trio[2]
            )
            pay = int(table_1R["return"])
            return_amount = (pay * amount / 100) if hit else 0
        else:
            mask = (
                (table_1R["win_0"].astype(int) == trio[0])
                & (table_1R["win_1"].astype(int) == trio[1])
                & (table_1R["win_2"].astype(int) == trio[2])
            )
            if mask.any():
                pay = int(table_1R.loc[mask, "return"].iloc[0])
                return_amount = pay * amount / 100
            else:
                return_amount = 0
        return 1, amount, return_amount

    def bet_sanrentan_box(self, race_id: str, umaban: list, amount: float):
        n_bets = 0
        bet_amount = 0
        return_amount = 0
        for trio in permutations(umaban, 3):
            nb, ba, ra = self._bet_sanrentan(race_id, list(trio), amount)
            n_bets += nb
            bet_amount += ba
            return_amount += ra
        return n_bets, bet_amount, return_amount

    def others(self, race_id: str, umaban: list, amount: float):
        pass

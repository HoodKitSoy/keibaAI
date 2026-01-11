# modules/simulation/_simulator.py

from collections import defaultdict
from itertools import combinations, permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from modules.preprocessing import ReturnProcessor

from ._betting_tickets import BettingTickets


class Simulator:
    """
    賭けた馬券を元に，成績を記録していくクラス．
    """

    SUPPORTED_BET_TYPES = {
        "tansho",
        "fukusho",
        "wakuren",
        "umaren",
        "umatan",
        "wide",
        "sanrenpuku",
        "sanrentan",
    }

    BET_TYPE_ALIAS = {
        "coverage50_tansho": "tansho",
    }

    def __init__(
        self, return_processor: ReturnProcessor, bet_unit: float = 100.0
    ) -> None:
        self.betting_tickets = BettingTickets(return_processor)
        self.bet_unit = float(bet_unit)

    def calc_returns_per_race(
        self, actions: dict, return_details: bool = False
    ) -> pd.DataFrame:
        returns_per_race_dict = {}
        bet_detail_rows: List[Dict[str, object]] = []
        official_rows: List[Dict[str, object]] = []

        for race_id, race_actions in actions.items():
            n_bets_race = 0
            bet_amount_race = 0.0
            return_amount_race = 0.0

            official_outcomes = (
                self._get_official_outcomes(race_id) if return_details else None
            )

            for bet_type, selected_horses in race_actions.items():
                normalized_type = self._normalize_bet_type(bet_type)
                if normalized_type is None:
                    continue

                horses = self._to_int_list(selected_horses)
                if len(horses) == 0:
                    continue

                nb, ba, ra = self._execute_bet(normalized_type, race_id, horses)

                n_bets_race += nb
                bet_amount_race += ba
                return_amount_race += ra

                if return_details and official_outcomes is not None:
                    combos = self._expand_combinations(normalized_type, horses)
                    bet_detail_rows.extend(
                        self._build_bet_detail_rows(
                            race_id,
                            normalized_type,
                            combos,
                            official_outcomes.get(normalized_type, []),
                        )
                    )

            returns_per_race_dict[race_id] = {
                "n_bets": int(n_bets_race),
                "bet_amount": float(bet_amount_race),
                "return_amount": float(return_amount_race),
                "hit_or_not": int(return_amount_race > 0),
            }

            if return_details and official_outcomes is not None:
                official_rows.extend(
                    self._build_official_rows(race_id, official_outcomes)
                )

        returns_df = pd.DataFrame.from_dict(returns_per_race_dict, orient="index")
        if returns_df.empty:
            returns_df = pd.DataFrame(
                columns=["n_bets", "bet_amount", "return_amount", "hit_or_not"]
            )

        if return_details:
            bet_details_df = (
                pd.DataFrame(bet_detail_rows)
                if bet_detail_rows
                else pd.DataFrame(
                    columns=[
                        "race_id",
                        "bet_type",
                        "bet_combination",
                        "bet_amount",
                        "payout_amount",
                        "hit",
                        "payout_per100",
                    ]
                )
            )

            official_df = (
                pd.DataFrame(official_rows)
                if official_rows
                else pd.DataFrame(
                    columns=[
                        "race_id",
                        "bet_type",
                        "official_combination",
                        "payout_per100",
                    ]
                )
            )

            return returns_df, bet_details_df, official_df

        return returns_df

    def calc_returns(self, actions: dict) -> dict:
        returns_dict = {}
        if len(actions) == 0:
            returns_dict["n_bets"] = 0
            returns_dict["n_races"] = 0
            returns_dict["n_hits"] = 0
            returns_dict["total_bet_amount"] = 0
            returns_dict["return_rate"] = 0
            returns_dict["std"] = 0
            return returns_dict

        returns_per_race = self.calc_returns_per_race(actions)
        returns_dict["n_bets"] = returns_per_race["n_bets"].sum()
        returns_dict["n_races"] = returns_per_race.index.nunique()
        returns_dict["n_hits"] = returns_per_race["hit_or_not"].sum()
        returns_dict["total_bet_amount"] = returns_per_race["bet_amount"].sum()

        if returns_dict["total_bet_amount"] == 0:
            returns_dict["return_rate"] = 0
            returns_dict["std"] = 0
        else:
            total_return = returns_per_race["return_amount"].sum()
            returns_dict["return_rate"] = (
                total_return / returns_dict["total_bet_amount"]
            )
            returns_dict["std"] = (
                returns_per_race["return_amount"].std()
                * np.sqrt(returns_dict["n_races"])
                / returns_dict["total_bet_amount"]
            )

        return returns_dict

    # ---- internal helpers -------------------------------------------------

    def _normalize_bet_type(self, bet_type: str) -> Optional[str]:
        if not bet_type or bet_type.startswith("_"):
            return None
        if bet_type in self.SUPPORTED_BET_TYPES:
            return bet_type
        if bet_type in self.BET_TYPE_ALIAS:
            return self.BET_TYPE_ALIAS[bet_type]
        return None

    @staticmethod
    def _to_int_list(values: Sequence) -> List[int]:
        result: List[int] = []
        for v in values:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                result.append(iv)
        return result

    def _execute_bet(
        self, bet_type: str, race_id: str, horses: List[int]
    ) -> Tuple[int, float, float]:
        bet_unit = self.bet_unit
        if bet_type == "tansho":
            return self.betting_tickets.bet_tansho(race_id, horses, bet_unit)
        if bet_type == "fukusho":
            return self.betting_tickets.bet_fukusho(race_id, horses, bet_unit)
        if bet_type == "wakuren":
            return self.betting_tickets.bet_wakuren_box(race_id, horses, bet_unit)
        if bet_type == "umaren":
            return self.betting_tickets.bet_umaren_box(race_id, horses, bet_unit)
        if bet_type == "umatan":
            return self.betting_tickets.bet_umatan_box(race_id, horses, bet_unit)
        if bet_type == "wide":
            return self.betting_tickets.bet_wide_box(race_id, horses, bet_unit)
        if bet_type == "sanrenpuku":
            return self.betting_tickets.bet_sanrenpuku_box(race_id, horses, bet_unit)
        if bet_type == "sanrentan":
            return self.betting_tickets.bet_sanrentan_box(race_id, horses, bet_unit)
        return 0, 0.0, 0.0

    def _expand_combinations(
        self, bet_type: str, horses: Sequence[int]
    ) -> List[Tuple[int, ...]]:
        if bet_type in {"tansho", "fukusho"}:
            return [(h,) for h in horses]
        if bet_type in {"wakuren", "umaren", "wide"}:
            return [tuple(sorted(pair)) for pair in combinations(horses, 2)]
        if bet_type == "umatan":
            return [tuple(order) for order in permutations(horses, 2)]
        if bet_type == "sanrenpuku":
            return [tuple(sorted(trio)) for trio in combinations(horses, 3)]
        if bet_type == "sanrentan":
            return [tuple(order) for order in permutations(horses, 3)]
        return []

    @staticmethod
    def _format_combination(combo: Tuple[int, ...]) -> str:
        return "-".join(f"{v:02d}" for v in combo)

    def _build_bet_detail_rows(
        self,
        race_id: str,
        bet_type: str,
        combos: Iterable[Tuple[int, ...]],
        official_outcomes: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        outcome_map: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for outcome in official_outcomes:
            combo = outcome.get("normalized_combination")
            payout = outcome.get("payout", 0)
            if isinstance(combo, tuple) and payout:
                outcome_map[combo].append(int(payout))

        rows: List[Dict[str, object]] = []
        for combo in combos:
            payouts = outcome_map.get(combo, [])
            payout_per100 = payouts[0] if payouts else 0
            hit = int(bool(payouts))
            payout_amount = (payout_per100 * self.bet_unit / 100.0) if hit else 0.0
            rows.append(
                {
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "bet_combination": self._format_combination(combo),
                    "bet_amount": float(self.bet_unit),
                    "payout_amount": float(payout_amount),
                    "hit": hit,
                    "payout_per100": int(payout_per100),
                }
            )
        return rows

    def _build_official_rows(
        self, race_id: str, official_outcomes: Dict[str, List[Dict[str, object]]]
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for bet_type, outcomes in official_outcomes.items():
            for outcome in outcomes:
                combo = outcome.get("normalized_combination")
                payout = outcome.get("payout", 0)
                if not isinstance(combo, tuple) or not payout:
                    continue
                rows.append(
                    {
                        "race_id": race_id,
                        "bet_type": bet_type,
                        "official_combination": self._format_combination(combo),
                        "payout_per100": int(payout),
                    }
                )
        return rows

    def _get_official_outcomes(
        self, race_id: str
    ) -> Dict[str, List[Dict[str, object]]]:
        tables = {
            "tansho": self.betting_tickets._BettingTickets__returnTablesTansho,
            "fukusho": self.betting_tickets._BettingTickets__returnTablesFukusho,
            "wakuren": self.betting_tickets._BettingTickets__returnTablesWakuren,
            "umaren": self.betting_tickets._BettingTickets__returnTablesUmaren,
            "umatan": self.betting_tickets._BettingTickets__returnTablesUmatan,
            "wide": self.betting_tickets._BettingTickets__returnTablesWide,
            "sanrenpuku": self.betting_tickets._BettingTickets__returnTablesSanrenpuku,
            "sanrentan": self.betting_tickets._BettingTickets__returnTablesSanrentan,
        }

        outcomes: Dict[str, List[Dict[str, object]]] = {k: [] for k in tables}

        for bet_type, table in tables.items():
            if race_id not in table.index:
                continue
            row = table.loc[race_id]

            if bet_type == "tansho":
                records = self._extract_tansho(row)
            elif bet_type == "fukusho":
                records = self._extract_fukusho(row)
            elif bet_type == "wakuren":
                records = self._extract_wakuren(row)
            elif bet_type == "umaren":
                records = self._extract_umaren(row)
            elif bet_type == "umatan":
                records = self._extract_umatan(row)
            elif bet_type == "wide":
                records = self._extract_wide(row)
            elif bet_type == "sanrenpuku":
                records = self._extract_sanrenpuku(row)
            else:  # sanrentan
                records = self._extract_sanrentan(row)

            outcomes[bet_type].extend(records)

        return outcomes

    # ---- extraction helpers for official outcomes ------------------------

    @staticmethod
    def _ensure_iterable(row: object) -> Iterable[pd.Series]:
        if isinstance(row, pd.Series):
            return [row]
        if isinstance(row, pd.DataFrame):
            return [row.iloc[i] for i in range(len(row))]
        return []

    def _extract_tansho(self, row: object) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for series in self._ensure_iterable(row):
            win = int(series.get("win", 0))
            ret = int(series.get("return", 0))
            if win > 0 and ret > 0:
                results.append(
                    {
                        "normalized_combination": (win,),
                        "payout": ret,
                    }
                )
        return results

    def _extract_fukusho(self, row: object) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        series = row if isinstance(row, pd.Series) else row.iloc[0]
        wins = [series.get("win_0", 0), series.get("win_1", 0), series.get("win_2", 0)]
        payouts = [
            series.get("return_0", 0),
            series.get("return_1", 0),
            series.get("return_2", 0),
        ]
        for w, pay in zip(wins, payouts):
            win = int(w) if pd.notna(w) else 0
            ret = int(pay) if pd.notna(pay) else 0
            if win > 0 and ret > 0:
                results.append(
                    {
                        "normalized_combination": (win,),
                        "payout": ret,
                    }
                )
        return results

    def _extract_umaren(self, row: object) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for series in self._ensure_iterable(row):
            a = int(series.get("win_0", 0))
            b = int(series.get("win_1", 0))
            ret = int(series.get("return", 0))
            if a > 0 and b > 0 and ret > 0:
                combo = tuple(sorted((a, b)))
                results.append({"normalized_combination": combo, "payout": ret})
        return results

    def _extract_wakuren(self, row: object) -> List[Dict[str, object]]:
        """枠連の正解データを抽出（win_0, win_1は枠番1-8）"""
        results: List[Dict[str, object]] = []
        for series in self._ensure_iterable(row):
            a = int(series.get("win_0", 0))
            b = int(series.get("win_1", 0))
            ret = int(series.get("return", 0))
            if a > 0 and b > 0 and ret > 0:
                combo = tuple(sorted((a, b)))
                results.append({"normalized_combination": combo, "payout": ret})
        return results

    def _extract_umatan(self, row: object) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for series in self._ensure_iterable(row):
            a = int(series.get("win_0", 0))
            b = int(series.get("win_1", 0))
            ret = int(series.get("return", 0))
            if a > 0 and b > 0 and ret > 0:
                results.append(
                    {
                        "normalized_combination": (a, b),
                        "payout": ret,
                    }
                )
        return results

    def _extract_wide(self, row: object) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        if isinstance(row, pd.Series):
            row = pd.DataFrame([row])
        for _, series in row.iterrows():
            a = int(series.get("win_0", 0))
            b = int(series.get("win_1", 0))
            ret = int(series.get("return", 0))
            if a > 0 and b > 0 and ret > 0:
                combo = tuple(sorted((a, b)))
                results.append({"normalized_combination": combo, "payout": ret})
        return results

    def _extract_sanrenpuku(self, row: object) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for series in self._ensure_iterable(row):
            a = int(series.get("win_0", 0))
            b = int(series.get("win_1", 0))
            c = int(series.get("win_2", 0))
            ret = int(series.get("return", 0))
            if a > 0 and b > 0 and c > 0 and ret > 0:
                combo = tuple(sorted((a, b, c)))
                results.append({"normalized_combination": combo, "payout": ret})
        return results

    def _extract_sanrentan(self, row: object) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for series in self._ensure_iterable(row):
            a = int(series.get("win_0", 0))
            b = int(series.get("win_1", 0))
            c = int(series.get("win_2", 0))
            ret = int(series.get("return", 0))
            if a > 0 and b > 0 and c > 0 and ret > 0:
                results.append(
                    {
                        "normalized_combination": (a, b, c),
                        "payout": ret,
                    }
                )
        return results

"""
戦略評価モジュール

戦略の評価とWIN5シミュレーション機能を提供します。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from modules.preprocessing._win5_processor import Win5Processor
from modules.simulation import Simulator
from modules.simulation._win5_tickets import Win5Tickets

from ._output_manager import get_strategy_output_dir
from ._result_saver import save_strategy_details


def evaluate_strategies(
    actions: Dict, return_processor, args, odds_table, strategies: Dict, task: str = ""
) -> tuple:
    """
    戦略を評価し、結果を返す

    Args:
        actions: 各戦略の行動辞書
        return_processor: 払戻データプロセッサ
        args: コマンドライン引数
        odds_table: オッズテーブル
        strategies: 戦略インスタンスの辞書
        task: タスク名（win/top3/rank）

    Returns:
        tuple: (evaluation_results, comparison, bet_type_stats)
            - evaluation_results: 戦略ごとの評価結果
            - comparison: 戦略比較リスト
            - bet_type_stats: 券種別統計リスト
    """
    simulator = Simulator(return_processor)
    evaluation_results = {}
    bet_type_stats = []  # 券種別統計を収集

    for strategy_name, strategy_actions in actions.items():
        print(f"\n[評価] {strategy_name}")

        # 詳細データを取得（return_details=True）
        result = simulator.calc_returns_per_race(strategy_actions, return_details=True)

        if isinstance(result, tuple) and len(result) == 3:
            returns_df, bet_details_df, official_df = result
        else:
            returns_df = result
            bet_details_df = pd.DataFrame()
            official_df = pd.DataFrame()

        if not returns_df.empty:
            total_bets = int(returns_df["n_bets"].sum())
            total_bet_amount = float(returns_df["bet_amount"].sum())
            total_return = float(returns_df["return_amount"].sum())
            recovery_rate = (
                (total_return / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
            )

            evaluation_results[strategy_name] = {
                "total_bets": total_bets,
                "total_bet_amount": total_bet_amount,
                "total_return": total_return,
                "recovery_rate": recovery_rate,
            }

            print(f"  総賭け数: {total_bets}")
            print(f"  総投資額: {total_bet_amount:.0f}円")
            print(f"  総払戻額: {total_return:.0f}円")
            print(f"  回収率: {recovery_rate:.2f}%")

            # 戦略別の詳細ファイルを保存（タスク別ディレクトリ）
            strategy_dir = Path(get_strategy_output_dir(args, strategy_name, task))
            save_strategy_details(
                strategy_name, returns_df, bet_details_df, official_df, strategy_dir
            )

            # 券種別統計を収集
            if not bet_details_df.empty and "bet_type" in bet_details_df.columns:
                for bet_type in bet_details_df["bet_type"].unique():
                    bt_df = bet_details_df[bet_details_df["bet_type"] == bet_type]
                    bt_bets = len(bt_df)
                    bt_amount = float(bt_df["bet_amount"].sum())
                    bt_payout = float(bt_df["payout_amount"].sum())
                    bt_hits = int(bt_df["hit"].sum())
                    bt_recovery = (
                        (bt_payout / bt_amount * 100) if bt_amount > 0 else 0.0
                    )
                    bt_hit_rate = (bt_hits / bt_bets * 100) if bt_bets > 0 else 0.0

                    bet_type_stats.append(
                        {
                            "strategy": strategy_name,
                            "bet_type": bet_type,
                            "total_bets": bt_bets,
                            "bet_amount": bt_amount,
                            "payout_amount": bt_payout,
                            "hits": bt_hits,
                            "hit_rate": round(bt_hit_rate, 2),
                            "recovery_rate": round(bt_recovery, 2),
                        }
                    )

    # 比較テーブルの作成
    comparison = []
    for strategy_name, results in evaluation_results.items():
        comparison.append(
            {
                "strategy": strategy_name,
                "total_bets": results["total_bets"],
                "bet_amount": results["total_bet_amount"],
                "return_amount": results["total_return"],
                "recovery_rate": results["recovery_rate"],
            }
        )

    return evaluation_results, comparison, bet_type_stats


def run_win5_simulation(
    args, score_table: pd.DataFrame, threshold: float, win5_processor: Win5Processor
) -> Dict:
    """
    WIN5シミュレーションを実行

    Args:
        args: コマンドライン引数
        score_table: スコアテーブル（proba列を含む）
        threshold: 選択閾値（この値以上の確率の馬を候補にする）
        win5_processor: WIN5データプロセッサ

    Returns:
        Dict: シミュレーション結果
    """
    print(f"\n[WIN5] threshold={threshold:.3f} でシミュレーション実行")

    if not win5_processor.loaded:
        print("  [WARN] WIN5データが読み込まれていません")
        return {"executed": False, "threshold": threshold}

    # WIN5チケットヘルパーを作成
    win5_tickets = Win5Tickets(win5_processor.answer)

    # スコアテーブルに日付キーを生成
    score_table = score_table.copy()
    score_table["date_key"] = score_table["Year"].astype(str).str.zfill(
        4
    ) + score_table["MonthDay"].astype(str).str.zfill(4)

    # WIN5対象日を取得
    win5_dates = win5_processor.get_dates()
    available_dates = set(score_table["date_key"].unique()) & set(win5_dates)

    if not available_dates:
        print("  [INFO] WIN5対象日がありません")
        return {"executed": False, "threshold": threshold}

    # 日ごとにシミュレーション
    results = []
    total_bets = 0
    total_bet_amount = 0.0
    total_return = 0.0
    total_hits = 0

    for date_key in sorted(available_dates):
        day_info = win5_processor.get_day_info(date_key)
        if day_info is None:
            continue

        races, winners, payout = day_info

        # 5レースそれぞれの候補馬を決定
        candidates = []
        for race_key in races:
            # レースにマッチするデータを取得
            mask = (
                (score_table["Year"].astype(str).str.zfill(4) == race_key["Year"])
                & (
                    score_table["MonthDay"].astype(str).str.zfill(4)
                    == race_key["MonthDay"]
                )
                & (score_table["JyoCD"].astype(str).str.zfill(2) == race_key["JyoCD"])
                & (
                    score_table["RaceNum"].astype(str).str.zfill(2)
                    == race_key["RaceNum"]
                )
            )
            race_data = score_table[mask]

            if race_data.empty:
                # レースデータがない場合は空リスト
                candidates.append([])
            else:
                # 閾値以上の確率の馬を候補に
                if "proba" in race_data.columns:
                    selected = race_data[race_data["proba"] >= threshold]["Umaban"]
                    # 候補がいない場合はTop1を候補に
                    if selected.empty:
                        selected = race_data.nlargest(1, "proba")["Umaban"]
                    candidates.append([int(u) for u in selected.tolist()])
                else:
                    # proba列がない場合はTop1を候補に
                    candidates.append([int(race_data.iloc[0]["Umaban"])])

        # 5レース全てに候補がある場合のみ購入
        if all(len(c) > 0 for c in candidates):
            n_bets, bet_amount, return_amount = win5_tickets.bet_win5(
                date_key, candidates
            )
            hit = return_amount > 0

            results.append(
                {
                    "date": date_key,
                    "n_bets": n_bets,
                    "bet_amount": bet_amount,
                    "return_amount": return_amount,
                    "hit": hit,
                    "winners": winners,
                    "payout": payout,
                    "candidates": candidates,
                }
            )

            total_bets += n_bets
            total_bet_amount += bet_amount
            total_return += return_amount
            if hit:
                total_hits += 1

    # 結果を返す
    return_rate = (total_return / total_bet_amount * 100) if total_bet_amount > 0 else 0
    print(
        f"  実行日数: {len(results)}, 総点数: {total_bets}, "
        f"総貭け金: {total_bet_amount:,.0f}円, 総払戻: {total_return:,.0f}円, "
        f"回収率: {return_rate:.2f}%, 的中: {total_hits}回"
    )

    return {
        "executed": True,
        "threshold": threshold,
        "days": len(results),
        "total_bets": total_bets,
        "total_bet_amount": total_bet_amount,
        "total_return": total_return,
        "return_rate": return_rate,
        "hits": total_hits,
        "details": results,
    }

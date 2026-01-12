"""
結果保存モジュール

戦略別の詳細結果と比較結果の保存機能を提供します。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def save_strategy_details(
    strategy_name: str,
    returns_df: pd.DataFrame,
    bet_details_df: pd.DataFrame,
    official_df: pd.DataFrame,
    strategy_dir: Path,
) -> None:
    """
    戦略別の詳細結果を保存

    Args:
        strategy_name: 戦略名
        returns_df: レース別結果
        bet_details_df: 賭けの詳細
        official_df: 公式結果
        strategy_dir: 出力ディレクトリ
    """
    strategy_dir.mkdir(parents=True, exist_ok=True)

    # race_results.csv - レース別結果
    if not returns_df.empty:
        race_results_path = strategy_dir / "race_results.csv"
        returns_df.to_csv(race_results_path, encoding="utf-8-sig")
        print(f"    保存: {race_results_path}")

    # bet_details.csv - 賭けの詳細
    if not bet_details_df.empty:
        bet_details_path = strategy_dir / "bet_details.csv"
        bet_details_df.to_csv(bet_details_path, index=False, encoding="utf-8-sig")
        print(f"    保存: {bet_details_path}")

    # official_results.csv - 公式結果
    if not official_df.empty:
        official_path = strategy_dir / "official_results.csv"
        official_df.to_csv(official_path, index=False, encoding="utf-8-sig")
        print(f"    保存: {official_path}")

    # bet_type_analysis.csv - 券種別分析
    if not bet_details_df.empty and "bet_type" in bet_details_df.columns:
        bet_type_analysis = (
            bet_details_df.groupby("bet_type")
            .agg(
                {
                    "bet_amount": "sum",
                    "payout_amount": "sum",
                    "hit": ["sum", "count"],
                }
            )
            .reset_index()
        )
        bet_type_analysis.columns = [
            "bet_type",
            "total_bet",
            "total_payout",
            "hits",
            "total_bets",
        ]
        bet_type_analysis["hit_rate"] = (
            bet_type_analysis["hits"] / bet_type_analysis["total_bets"] * 100
        )
        bet_type_analysis["recovery_rate"] = (
            bet_type_analysis["total_payout"] / bet_type_analysis["total_bet"] * 100
        )
        bet_type_path = strategy_dir / "bet_type_analysis.csv"
        bet_type_analysis.to_csv(bet_type_path, index=False, encoding="utf-8-sig")
        print(f"    保存: {bet_type_path}")

    # bet_vs_official.csv - 賭けと公式結果の比較
    if not bet_details_df.empty and not official_df.empty:
        # マージして比較
        merged = bet_details_df.merge(
            official_df,
            on=["race_id", "bet_type"],
            how="left",
            suffixes=("_bet", "_official"),
        )
        if not merged.empty:
            bet_vs_official_path = strategy_dir / "bet_vs_official.csv"
            merged.to_csv(bet_vs_official_path, index=False, encoding="utf-8-sig")
            print(f"    保存: {bet_vs_official_path}")

    # summary.txt - サマリー
    summary_path = strategy_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"===== {strategy_name} 評価サマリー =====\n\n")

        if not returns_df.empty:
            total_races = len(returns_df)
            total_bets = int(returns_df["n_bets"].sum())
            total_bet_amount = float(returns_df["bet_amount"].sum())
            total_return = float(returns_df["return_amount"].sum())
            total_hits = int(returns_df["hit_or_not"].sum())
            recovery_rate = (
                (total_return / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
            )
            hit_rate = (total_hits / total_races * 100) if total_races > 0 else 0.0

            f.write(f"総レース数: {total_races}\n")
            f.write(f"総賭け数: {total_bets}\n")
            f.write(f"総投資額: {total_bet_amount:,.0f}円\n")
            f.write(f"総払戻額: {total_return:,.0f}円\n")
            f.write(f"純利益: {total_return - total_bet_amount:,.0f}円\n")
            f.write(f"回収率: {recovery_rate:.2f}%\n")
            f.write(f"的中レース数: {total_hits}\n")
            f.write(f"的中率: {hit_rate:.2f}%\n")

        if not bet_details_df.empty and "bet_type" in bet_details_df.columns:
            f.write("\n--- 券種別内訳 ---\n")
            for bet_type in bet_details_df["bet_type"].unique():
                bt_df = bet_details_df[bet_details_df["bet_type"] == bet_type]
                bt_bets = len(bt_df)
                bt_amount = bt_df["bet_amount"].sum()
                bt_payout = bt_df["payout_amount"].sum()
                bt_hits = bt_df["hit"].sum()
                bt_recovery = (bt_payout / bt_amount * 100) if bt_amount > 0 else 0.0
                f.write(
                    f"  {bet_type}: {bt_bets}点 / 投資{bt_amount:,.0f}円 / "
                    f"払戻{bt_payout:,.0f}円 / 的中{bt_hits}件 / 回収率{bt_recovery:.1f}%\n"
                )

    print(f"    保存: {summary_path}")


def save_comparison_results(comparison: List[Dict], output_dir) -> None:
    """
    比較結果を保存

    Args:
        comparison: 比較結果のリスト
        output_dir: 出力ディレクトリ（文字列またはPath）
    """
    # 文字列の場合はPathに変換
    output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir

    # ディレクトリ作成
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV形式で保存
    df_comparison = pd.DataFrame(comparison)
    csv_path = output_path / "strategy_comparison.csv"
    df_comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n比較結果をCSVで保存: {csv_path}")

    # JSON形式でも保存
    json_path = output_path / "strategy_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"比較結果をJSONで保存: {json_path}")


def save_comparison_by_bet_type(
    bet_type_stats: List[Dict], output_dir, task: str = "win"
) -> None:
    """
    馬券種別の回収率を保存

    Args:
        bet_type_stats: 券種別統計のリスト
        output_dir: 出力ディレクトリ（文字列またはPath）
        task: タスク名（win, top3, rank）
    """
    output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    if not bet_type_stats:
        print(f"券種別統計データがありません（タスク: {task}）")
        return

    # DataFrameに変換
    df_bet_type = pd.DataFrame(bet_type_stats)

    # CSV形式で保存
    csv_path = output_path / f"bet_type_comparison_{task}.csv"
    df_bet_type.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n券種別回収率をCSVで保存: {csv_path}")

    # JSON形式でも保存
    json_path = output_path / f"bet_type_comparison_{task}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bet_type_stats, f, ensure_ascii=False, indent=2)
    print(f"券種別回収率をJSONで保存: {json_path}")


def save_win5_results(
    win5_results: List[Dict], output_dir: Path, task: str = "win"
) -> None:
    """
    WIN5シミュレーション結果を保存

    Args:
        win5_results: WIN5シミュレーション結果のリスト
        output_dir: 出力ディレクトリ
        task: タスク名（win, top3, rank）- ファイル名に使用
    """
    win5_dir = output_dir / "win5"
    win5_dir.mkdir(parents=True, exist_ok=True)

    # サマリーCSV（タスク名を含むファイル名）
    summary_records = []
    for r in win5_results:
        summary_records.append(
            {
                "task": task,
                "threshold": r["threshold"],
                "days": r["days"],
                "total_bets": r["total_bets"],
                "total_bet_amount": r["total_bet_amount"],
                "total_return": r["total_return"],
                "return_rate": r["return_rate"],
                "hits": r["hits"],
            }
        )
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(
        win5_dir / f"win5_summary_{task}.csv", index=False, encoding="utf-8-sig"
    )

    # 詳細JSON（タスク名を含むファイル名）
    with open(win5_dir / f"win5_results_{task}.json", "w", encoding="utf-8") as f:
        # detailsのcandidatesをシリアライズ可能に変換
        serializable_results = []
        for r in win5_results:
            r_copy = r.copy()
            if "details" in r_copy:
                for d in r_copy["details"]:
                    if "candidates" in d:
                        d["candidates"] = [list(c) for c in d["candidates"]]
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"\nWIN5シミュレーション完了 (タスク: {task}): {len(win5_results)}件")
    print(f"WIN5結果保存先: {win5_dir}")


def save_win5_details(
    threshold: float,
    daily_results: List[Dict],
    output_dir: Path,
    task: str = "win",
) -> None:
    """
    WIN5シミュレーション結果をポリシーと同じ形式で保存

    Args:
        threshold: 使用した閾値
        daily_results: 日別の詳細結果リスト（bet_win5_detailedの戻り値）
        output_dir: 出力ディレクトリ
        task: タスク名
    """
    # 閾値をディレクトリ名用にフォーマット
    th_str = f"{threshold:.3f}".rstrip("0").rstrip(".")
    strategy_name = f"WIN5_th{th_str}"
    strategy_dir = output_dir / "strategies" / task / strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    # race_results.csv - 日別結果（レース単位ではなく日単位）
    race_records = []
    for r in daily_results:
        race_records.append(
            {
                "date": r["date"],
                "n_bets": r["n_bets"],
                "bet_amount": r["bet_amount"],
                "return_amount": r["return_amount"],
                "hit_or_not": 1 if r["hit"] else 0,
                "winners": "-".join(str(w).zfill(2) for w in r["winners"])
                if r["winners"]
                else "",
                "candidates_count": [len(c) for c in r["candidates"]]
                if r["candidates"]
                else [],
                "total_payout": r["total_payout"],
                "tekichu_hyo": r["tekichu_hyo"],
                "payout_per_ticket": r["payout_per_ticket"],
            }
        )
    df_race = pd.DataFrame(race_records)
    if not df_race.empty:
        df_race.to_csv(
            strategy_dir / "race_results.csv", index=False, encoding="utf-8-sig"
        )
        print(f"    保存: {strategy_dir / 'race_results.csv'}")

    # bet_details.csv - 各日の賭け詳細
    bet_records = []
    for r in daily_results:
        bet_records.append(
            {
                "date": r["date"],
                "bet_type": "win5",
                "n_bets": r["n_bets"],
                "bet_amount": r["bet_amount"],
                "payout_amount": r["return_amount"],
                "hit": 1 if r["hit"] else 0,
                "candidates_r1": ",".join(str(x) for x in r["candidates"][0])
                if r["candidates"] and len(r["candidates"]) > 0
                else "",
                "candidates_r2": ",".join(str(x) for x in r["candidates"][1])
                if r["candidates"] and len(r["candidates"]) > 1
                else "",
                "candidates_r3": ",".join(str(x) for x in r["candidates"][2])
                if r["candidates"] and len(r["candidates"]) > 2
                else "",
                "candidates_r4": ",".join(str(x) for x in r["candidates"][3])
                if r["candidates"] and len(r["candidates"]) > 3
                else "",
                "candidates_r5": ",".join(str(x) for x in r["candidates"][4])
                if r["candidates"] and len(r["candidates"]) > 4
                else "",
            }
        )
    df_bet = pd.DataFrame(bet_records)
    if not df_bet.empty:
        df_bet.to_csv(
            strategy_dir / "bet_details.csv", index=False, encoding="utf-8-sig"
        )
        print(f"    保存: {strategy_dir / 'bet_details.csv'}")

    # official_results.csv - 公式結果
    official_records = []
    for r in daily_results:
        official_records.append(
            {
                "date": r["date"],
                "bet_type": "win5",
                "winners": "-".join(str(w).zfill(2) for w in r["winners"])
                if r["winners"]
                else "",
                "winner_r1": r["winners"][0]
                if r["winners"] and len(r["winners"]) > 0
                else 0,
                "winner_r2": r["winners"][1]
                if r["winners"] and len(r["winners"]) > 1
                else 0,
                "winner_r3": r["winners"][2]
                if r["winners"] and len(r["winners"]) > 2
                else 0,
                "winner_r4": r["winners"][3]
                if r["winners"] and len(r["winners"]) > 3
                else 0,
                "winner_r5": r["winners"][4]
                if r["winners"] and len(r["winners"]) > 4
                else 0,
                "total_payout": r["total_payout"],
                "tekichu_hyo": r["tekichu_hyo"],
                "payout_per_ticket": r["payout_per_ticket"],
            }
        )
    df_official = pd.DataFrame(official_records)
    if not df_official.empty:
        df_official.to_csv(
            strategy_dir / "official_results.csv", index=False, encoding="utf-8-sig"
        )
        print(f"    保存: {strategy_dir / 'official_results.csv'}")

    # bet_vs_official.csv - 賭けと公式結果の比較
    if not df_bet.empty and not df_official.empty:
        merged = df_bet.merge(
            df_official,
            on=["date", "bet_type"],
            how="left",
            suffixes=("_bet", "_official"),
        )
        if not merged.empty:
            merged.to_csv(
                strategy_dir / "bet_vs_official.csv", index=False, encoding="utf-8-sig"
            )
            print(f"    保存: {strategy_dir / 'bet_vs_official.csv'}")

    # summary.txt - サマリー
    summary_path = strategy_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"===== {strategy_name} 評価サマリー =====\n\n")

        total_days = len(daily_results)
        total_bets = sum(r["n_bets"] for r in daily_results)
        total_bet_amount = sum(r["bet_amount"] for r in daily_results)
        total_return = sum(r["return_amount"] for r in daily_results)
        total_hits = sum(1 for r in daily_results if r["hit"])
        recovery_rate = (
            (total_return / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
        )
        hit_rate = (total_hits / total_days * 100) if total_days > 0 else 0.0

        f.write(f"閾値: {threshold}\n")
        f.write(f"対象日数: {total_days}日\n")
        f.write(f"総購入点数: {total_bets:,}点\n")
        f.write(f"総投資額: {total_bet_amount:,.0f}円\n")
        f.write(f"総払戻額: {total_return:,.0f}円\n")
        f.write(f"純利益: {total_return - total_bet_amount:,.0f}円\n")
        f.write(f"回収率: {recovery_rate:.2f}%\n")
        f.write(f"的中日数: {total_hits}日\n")
        f.write(f"的中率: {hit_rate:.2f}%\n")

        # 的中日の詳細
        hit_days = [r for r in daily_results if r["hit"]]
        if hit_days:
            f.write("\n--- 的中日詳細 ---\n")
            for r in hit_days:
                winners_str = (
                    "-".join(str(w).zfill(2) for w in r["winners"])
                    if r["winners"]
                    else "N/A"
                )
                f.write(
                    f"  {r['date']}: 投資{r['bet_amount']:,.0f}円 / "
                    f"払戻{r['return_amount']:,.0f}円 / "
                    f"正解{winners_str} / "
                    f"1票払戻{r['payout_per_ticket']:,.0f}円\n"
                )

    print(f"    保存: {summary_path}")

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

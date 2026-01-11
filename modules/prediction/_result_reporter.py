"""
結果レポートモジュール

予測結果のレポート生成と表示機能を提供します。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from modules.core.constants import get_jyo_name


def save_policy_report(
    policy_name: str, rec_df: pd.DataFrame, output_dir: Path
) -> None:
    """
    ポリシー推奨の詳細レポートを保存

    Args:
        policy_name: ポリシー名
        rec_df: 推奨結果のDataFrame
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 空のDataFrameまたはrace_id列がない場合はスキップ
    if rec_df.empty or "race_id" not in rec_df.columns:
        logging.warning(f"  {policy_name}: 推奨馬券なし（スキップ）")
        return

    # CSV保存（test.pyのbet_details.csvと同じ形式）
    output_file = output_dir / f"{policy_name}_bet_details.csv"
    rec_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logging.info(f"  {policy_name}: {output_file}")

    # 詳細テキストレポート保存
    report_file = output_dir / f"{policy_name}_detail.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"ポリシー推奨馬券詳細: {policy_name}\n")
        f.write(f"{'=' * 80}\n\n")

        # 全体サマリー
        unique_races = rec_df["race_id"].nunique()
        total_bets = len(rec_df)
        f.write(f"推奨レース数: {unique_races} レース\n")
        f.write(f"推奨組み合わせ総数: {total_bets} 点\n")
        f.write(f"購入金額（100円/点）: {total_bets * 100:,}円\n\n")

        # 券種別集計
        bet_type_summary = rec_df.groupby("bet_type").size().sort_index()
        f.write("【券種別内訳】\n")
        for bet_type, count in bet_type_summary.items():
            f.write(f"  {bet_type}: {count}点\n")
        f.write("\n")

        # レースごとの詳細（DataFrameの入力順を維持 = WIN5等の正しい発走順序を保持）
        f.write("【レースごとの推奨馬券】\n\n")
        # race_idの出現順を維持（重複除去しつつ順序を保持）
        seen = set()
        sorted_race_ids = []
        for race_id in rec_df["race_id"]:
            if race_id not in seen:
                seen.add(race_id)
                sorted_race_ids.append(race_id)
        for race_id in sorted_race_ids:
            race_bets = rec_df[rec_df["race_id"] == race_id]
            first_row = race_bets.iloc[0]
            jyo_name = get_jyo_name(first_row["JyoCD"])
            f.write(
                f"■ {first_row['Year']}/{first_row['MonthDay']} {jyo_name} {first_row['RaceNum']}R\n"
            )

            # 券種ごとに整理
            for bet_type in race_bets["bet_type"].unique():
                type_bets = race_bets[race_bets["bet_type"] == bet_type]
                combinations = type_bets["bet_combination"].tolist()
                f.write(
                    f"  {bet_type} ({len(combinations)}点): {', '.join(combinations)}\n"
                )
            f.write("\n")

    logging.info(f"  {policy_name}詳細: {report_file}")


def save_win5_report(win5_result: Dict, output_dir: Path) -> None:
    """
    WIN5推奨の詳細レポートを保存

    Args:
        win5_result: WIN5予測結果
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # WIN5詳細レポート保存
    report_file = output_dir / "win5_detail.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 80}\n")
        f.write("WIN5予測詳細\n")
        f.write(f"{'=' * 80}\n\n")

        # 対象レース情報
        if "races" in win5_result:
            f.write("【対象レース】\n")
            for i, race in enumerate(win5_result["races"], 1):
                race_key = race.get("race_key", {})
                # 辞書形式に対応
                if isinstance(race_key, dict):
                    year = race_key.get("Year", "")
                    month_day = race_key.get("MonthDay", "")
                    jyo_cd = race_key.get("JyoCD", "")
                    race_num = race_key.get("RaceNum", "")
                else:
                    # タプル/リスト形式（後方互換性）
                    year = race_key[0] if len(race_key) > 0 else ""
                    month_day = race_key[1] if len(race_key) > 1 else ""
                    jyo_cd = race_key[2] if len(race_key) > 2 else ""
                    race_num = race_key[3] if len(race_key) > 3 else ""
                jyo_name = get_jyo_name(jyo_cd)
                f.write(f"  {i}レース目: {year}/{month_day} {jyo_name} {race_num}R\n")
            f.write("\n")

        # 各レースの推奨馬
        if "selections" in win5_result:
            f.write("【レースごとの推奨馬番】\n")
            for i, selection in enumerate(win5_result["selections"], 1):
                # selectionsはList[List[int]]形式
                if isinstance(selection, list):
                    horses = selection
                else:
                    horses = selection.get("horses", [])
                f.write(f"  {i}レース目: {', '.join(f'{h}番' for h in horses)}\n")
            f.write("\n")

        # 組み合わせ総数
        f.write("【組み合わせ総数】\n")
        f.write(f"  {win5_result.get('total_combinations', 0)} 点\n\n")

        # 購入金額（100円 × 組み合わせ数）
        cost = win5_result.get("total_combinations", 0) * 100
        f.write("【購入金額（100円/点）】\n")
        f.write(f"  {cost:,}円\n\n")

    logging.info(f"  WIN5詳細: {report_file}")


def print_prediction_summary(
    predictions: Dict[str, pd.DataFrame],
    recommendations: Dict[str, pd.DataFrame] | None = None,
    win5_result: Dict | None = None,
    output_dir: str = "",
) -> None:
    """
    予測結果のサマリーを表示

    Args:
        predictions: タスク別予測結果
        recommendations: ポリシー別推奨結果
        win5_result: WIN5予測結果
        output_dir: 出力ディレクトリパス
    """
    print("\n" + "=" * 60)
    print("予測完了サマリー")
    print("=" * 60)

    print("\n【予測結果】")
    for task, pred_df in predictions.items():
        print(f"  {task}: {len(pred_df)} レコード")

        # 上位3頭を表示（win確率順）
        if task == "win" and "proba" in pred_df.columns:
            top3 = pred_df.nlargest(3, "proba")
            print("    上位3頭:")
            for idx, row in top3.iterrows():
                race_info = f"{row.get('Year', '?')}/{row.get('MonthDay', '?')}"
                race_num = row.get("RaceNum", "?")
                umaban = row.get("Umaban", "?")
                proba = row.get("proba", 0)
                print(f"      {race_info} {race_num}R {umaban}番: {proba:.1%}")

    if recommendations:
        print("\n【ポリシー推奨馬券】")

        # タスク別の構造 {task: {policy: df}} か従来構造 {policy: df} かを判定
        first_value = next(iter(recommendations.values()), None)
        is_task_based = isinstance(first_value, dict)

        if is_task_based:
            # タスク別構造: {task: {policy: df}}
            for task, task_recommendations in recommendations.items():
                print(f"\n  ▼ タスク: {task}")
                for policy_name, rec_df in task_recommendations.items():
                    print(f"\n    ◆ {policy_name}")
                    if rec_df.empty:
                        print("       推奨なし")
                        continue
                    unique_races = rec_df["race_id"].nunique()
                    total_bets = len(rec_df)
                    print(f"       推奨レース数: {unique_races} レース")
                    print(f"       推奨組み合わせ総数: {total_bets} 点")
                    print(f"       購入金額: {total_bets * 100:,}円（100円/点）")

                    # 券種別の集計
                    bet_type_summary = rec_df.groupby("bet_type").size()
                    if not bet_type_summary.empty:
                        print("       券種別内訳:")
                        for bet_type, count in bet_type_summary.items():
                            print(f"         - {bet_type}: {count}点")
        else:
            # 従来構造: {policy: df}
            for policy_name, rec_df in recommendations.items():
                print(f"\n  ◆ {policy_name}")
                if rec_df.empty:
                    print("     推奨なし")
                    continue
                unique_races = rec_df["race_id"].nunique()
                total_bets = len(rec_df)
                print(f"     推奨レース数: {unique_races} レース")
                print(f"     推奨組み合わせ総数: {total_bets} 点")
                print(f"     購入金額: {total_bets * 100:,}円（100円/点）")

                # 券種別の集計
                bet_type_summary = rec_df.groupby("bet_type").size()
                if not bet_type_summary.empty:
                    print("     券種別内訳:")
                    for bet_type, count in bet_type_summary.items():
                        print(f"       - {bet_type}: {count}点")

    if win5_result:
        print("\n【WIN5予測】")
        if "races" in win5_result and win5_result["races"]:
            print("  対象レース:")
            for i, race in enumerate(win5_result["races"], 1):
                race_key = race.get("race_key", {})
                # 辞書形式に対応
                if isinstance(race_key, dict):
                    year = race_key.get("Year", "")
                    month_day = race_key.get("MonthDay", "")
                    jyo_cd = race_key.get("JyoCD", "")
                    race_num = race_key.get("RaceNum", "")
                else:
                    # タプル/リスト形式（後方互換性）
                    year = race_key[0] if len(race_key) > 0 else ""
                    month_day = race_key[1] if len(race_key) > 1 else ""
                    jyo_cd = race_key[2] if len(race_key) > 2 else ""
                    race_num = race_key[3] if len(race_key) > 3 else ""
                print(f"    {i}. {year}/{month_day} 競馬場{jyo_cd} {race_num}R")

        if "selections" in win5_result:
            print("\n  推奨馬番:")
            for i, selection in enumerate(win5_result["selections"], 1):
                # selectionsはList[List[int]]形式
                if isinstance(selection, list):
                    horses = selection
                else:
                    horses = selection.get("horses", [])
                print(f"    {i}レース目: {', '.join(f'{h}番' for h in horses)}")

        total_combinations = win5_result.get("total_combinations", 0)
        print(f"\n  組み合わせ総数: {total_combinations} 点")
        print(f"  購入金額: {total_combinations * 100:,}円（100円/点）")

    print(f"\n出力先: {output_dir}")

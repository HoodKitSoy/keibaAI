"""
ポリシー推奨モジュール

予測結果に基づいて、各ポリシー（賭け戦略）による推奨馬券を生成します。
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List

import pandas as pd

from modules.core import RACE_KEY_COLS
from modules.core import build_race_id as build_race_id_series


def load_policy_class(policy_name: str) -> type:
    """
    ポリシークラスを動的にロード

    Args:
        policy_name: ポリシークラス名

    Returns:
        type: ポリシークラス

    Raises:
        ImportError: ポリシークラスが見つからない場合
    """
    # まずmodules.policiesから直接インポート試行
    try:
        module = importlib.import_module("modules.policies")
        if hasattr(module, policy_name):
            return getattr(module, policy_name)
    except ImportError:
        pass

    # 個別モジュールから試行
    module_map = {
        "BetPolicyCoverage25": "_bet_policy",
        "BetPolicyCoverage30": "_bet_policy",
        "BetPolicyCoverage50": "_bet_policy",
        "BetPolicyCoverage75": "_bet_policy",
        "HybridBetPolicy": "_hybrid_bet_policy",
        "ExpectedValueBetPolicy": "_hybrid_bet_policy",
        "TieredCoverageBetPolicy": "_hybrid_bet_policy",
        "RaceSelectivePolicy": "_race_selective_policy",
        "RaceSelectiveHybridPolicy": "_race_selective_policy",
        "RaceSelectiveExpectedValueBetPolicy": "_race_selective_policy",
        "RaceSelectiveTieredCoverageBetPolicy": "_race_selective_policy",
        "RaceSelectiveCoveragePolicy": "_race_selective_coverage_policy",
        "RaceSelectiveCoverage25Policy": "_race_selective_coverage_policy",
        "RaceSelectiveCoverage50Policy": "_race_selective_coverage_policy",
        "RaceSelectiveCoverage75Policy": "_race_selective_coverage_policy",
    }

    module_name = module_map.get(policy_name, "_bet_policy")
    try:
        module = importlib.import_module(f"modules.policies.{module_name}")
        if hasattr(module, policy_name):
            return getattr(module, policy_name)
    except ImportError:
        pass

    raise ImportError(f"ポリシークラス '{policy_name}' が見つかりません")


def apply_policy_to_predictions(
    predictions: pd.DataFrame,
    policy_name: str,
    task: str = "win",
    win5_selections: Dict[str, List[int]] = None,
) -> Dict[str, Any]:
    """
    予測結果にポリシーを適用して推奨馬券を生成

    Args:
        predictions: 予測結果（DataFrame）
        policy_name: ポリシー名
        task: タスク（win, top3, rank）
        win5_selections: WIN5予測で選択された馬 {race_id: [馬番リスト]}
                        WIN5Basedポリシーで使用

    Returns:
        Dict[str, Any]: レースIDをキーとした推奨馬券情報
    """
    logging.info(f"ポリシー '{policy_name}' を適用中...")

    # ポリシークラスをロード
    try:
        policy_class = load_policy_class(policy_name)
        policy = policy_class()
    except ImportError as e:
        logging.error(f"ポリシーのロードに失敗: {e}")
        return {}

    # スコアテーブルの準備
    score_table = predictions.copy()

    # タスクに応じてスコア列を設定
    if task == "win" and "proba" in score_table.columns:
        score_table["score"] = score_table["proba"]
    elif task == "top3" and "proba" in score_table.columns:
        score_table["score"] = score_table["proba"]
    elif task == "rank" and "p1" in score_table.columns:
        # rankタスクの場合は1着確率を使用
        score_table["score"] = score_table["p1"]
    else:
        logging.error(f"タスク '{task}' に対応するスコア列が見つかりません")
        return {}

    # race_idを生成または確認
    if "race_id" not in score_table.columns:
        if all(col in score_table.columns for col in RACE_KEY_COLS):
            score_table["race_id"] = build_race_id_series(score_table)
        else:
            logging.error("race_idまたはレースキー列が見つかりません")
            return {}

    # ポリシーを適用
    try:
        # WIN5Basedポリシーの場合はwin5_selectionsを渡す
        if policy_name.startswith("WIN5Based") and win5_selections:
            logging.info(f"  WIN5選択馬を使用: {len(win5_selections)} レース")
            actions = policy.judge(score_table, win5_selections=win5_selections)
        else:
            actions = policy.judge(score_table)
        logging.info(f"ポリシー適用完了: {len(actions)} レース")
        return actions
    except Exception as e:
        logging.error(f"ポリシー適用中にエラー: {e}")
        return {}


def format_recommendations(
    actions: Dict[str, Any], predictions: pd.DataFrame
) -> pd.DataFrame:
    """
    推奨馬券情報をDataFrame形式に整形
    test.pyのbet_details.csvと同じ形式で出力

    Args:
        actions: レースIDをキーとした推奨馬券情報
        predictions: 元の予測結果

    Returns:
        pd.DataFrame: 整形された推奨馬券情報（各組み合わせを個別行で出力）
    """
    from itertools import combinations, permutations

    records = []

    for race_id, action in actions.items():
        # 該当レースの予測情報を取得
        race_preds = predictions[predictions["race_id"] == race_id]

        if race_preds.empty:
            continue

        # レース情報を取得
        first_row = race_preds.iloc[0]
        race_info = {
            "race_id": race_id,
            "Year": first_row.get("Year", ""),
            "MonthDay": first_row.get("MonthDay", ""),
            "JyoCD": first_row.get("JyoCD", ""),
            "RaceNum": first_row.get("RaceNum", ""),
        }

        # 券種ごとの推奨馬番を取得し、組み合わせを展開
        for ticket_type, umaban_list in action.items():
            if not isinstance(umaban_list, list) or not umaban_list:
                continue

            # 券種に応じて組み合わせを生成
            if ticket_type in ["tansho", "fukusho"]:
                # 単勝・複勝: 各馬を個別に出力
                for umaban in umaban_list:
                    record = race_info.copy()
                    record["bet_type"] = ticket_type
                    record["bet_combination"] = str(int(umaban)).zfill(2)
                    records.append(record)

            elif ticket_type == "umaren":
                # 馬連: 2頭の組み合わせ（順不同）
                for combo in combinations(umaban_list, 2):
                    record = race_info.copy()
                    record["bet_type"] = ticket_type
                    # test.pyと同じ形式: "馬番1-馬番2" (昇順)
                    record["bet_combination"] = (
                        f"{int(combo[0]):02d}-{int(combo[1]):02d}"
                    )
                    records.append(record)

            elif ticket_type == "umatan":
                # 馬単: 2頭の順列（順序あり）
                for perm in permutations(umaban_list, 2):
                    record = race_info.copy()
                    record["bet_type"] = ticket_type
                    record["bet_combination"] = f"{int(perm[0]):02d}-{int(perm[1]):02d}"
                    records.append(record)

            elif ticket_type == "wide":
                # ワイド: 2頭の組み合わせ（順不同）
                for combo in combinations(umaban_list, 2):
                    record = race_info.copy()
                    record["bet_type"] = ticket_type
                    record["bet_combination"] = (
                        f"{int(combo[0]):02d}-{int(combo[1]):02d}"
                    )
                    records.append(record)

            elif ticket_type == "sanrenpuku":
                # 三連複: 3頭の組み合わせ（順不同）
                for combo in combinations(umaban_list, 3):
                    record = race_info.copy()
                    record["bet_type"] = ticket_type
                    record["bet_combination"] = (
                        f"{int(combo[0]):02d}-{int(combo[1]):02d}-{int(combo[2]):02d}"
                    )
                    records.append(record)

            elif ticket_type == "sanrentan":
                # 三連単: 3頭の順列（順序あり）
                for perm in permutations(umaban_list, 3):
                    record = race_info.copy()
                    record["bet_type"] = ticket_type
                    record["bet_combination"] = (
                        f"{int(perm[0]):02d}-{int(perm[1]):02d}-{int(perm[2]):02d}"
                    )
                    records.append(record)

            elif ticket_type == "wakuren":
                # 枠連: 2枠の組み合わせ（順不同）
                for combo in combinations(umaban_list, 2):
                    record = race_info.copy()
                    record["bet_type"] = ticket_type
                    # 枠番の組み合わせ（昇順）
                    sorted_combo = sorted(combo)
                    record["bet_combination"] = (
                        f"{int(sorted_combo[0]):02d}-{int(sorted_combo[1]):02d}"
                    )
                    records.append(record)

    return pd.DataFrame(records)


def _filter_win5_races(df: pd.DataFrame, win5_race_keys) -> pd.DataFrame:
    """
    WIN5対象レースのみをフィルタリング

    Args:
        df: 予測結果DataFrame
        win5_race_keys: WIN5対象レースキー（辞書のリストまたはタプルのセット）

    Returns:
        pd.DataFrame: フィルタリングされたDataFrame
    """
    if not win5_race_keys:
        return df

    filtered_dfs = []

    # タプルのセット/リストの場合
    if isinstance(win5_race_keys, (set, frozenset)):
        win5_race_keys = list(win5_race_keys)

    for race_key in win5_race_keys:
        # タプル形式: (Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum)
        if isinstance(race_key, tuple):
            year = str(race_key[0]).zfill(4)
            monthday = str(race_key[1]).zfill(4)
            jyo_cd = str(race_key[2]).zfill(2)
            race_num = str(race_key[5]).zfill(2)  # 6番目がRaceNum
        # 辞書形式
        elif isinstance(race_key, dict):
            year = str(race_key.get("Year", "")).zfill(4)
            monthday = str(race_key.get("MonthDay", "")).zfill(4)
            jyo_cd = str(race_key.get("JyoCD", "")).zfill(2)
            race_num = str(race_key.get("RaceNum", "")).zfill(2)
        else:
            continue

        # 各列をzfillして比較
        mask = (
            (df["Year"].astype(str).str.zfill(4) == year)
            & (df["MonthDay"].astype(str).str.zfill(4) == monthday)
            & (df["JyoCD"].astype(str).str.zfill(2) == jyo_cd)
            & (df["RaceNum"].astype(str).str.zfill(2) == race_num)
        )
        matched = df[mask]
        if len(matched) > 0:
            filtered_dfs.append(matched)

    if filtered_dfs:
        return pd.concat(filtered_dfs, ignore_index=True)
    return df.iloc[:0]  # 空のDataFrame


def generate_policy_recommendations(
    predictions: Dict[str, pd.DataFrame],
    policy_list: List[str],
    primary_task: str = "win",
    win5_race_keys: List[Dict[str, str]] = None,
    win5_result: Dict[str, Any] = None,
    win5base_threshold: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """
    各ポリシーによる推奨馬券を生成

    Args:
        predictions: タスクごとの予測結果
        policy_list: ポリシー名のリスト
        primary_task: 主に使用するタスク
        win5_race_keys: WIN5対象レースキーのリスト（WIN5Basedポリシー用）
        win5_result: WIN5予測結果（generate_win5_predictionsの戻り値）
                     WIN5Basedポリシーはこの選択馬を使用
        win5base_threshold: WIN5Basedポリシー用の閾値（WIN5予測とは別）

    Returns:
        Dict[str, pd.DataFrame]: ポリシー名をキーとした推奨馬券DataFrame
    """
    recommendations = {}

    # 主タスクの予測結果を取得
    if primary_task not in predictions:
        logging.error(f"タスク '{primary_task}' の予測結果が見つかりません")
        return recommendations

    pred_df = predictions[primary_task]

    # race_idがない場合は生成
    if "race_id" not in pred_df.columns:
        if all(col in pred_df.columns for col in RACE_KEY_COLS):
            pred_df = pred_df.copy()
            pred_df["race_id"] = build_race_id_series(pred_df)

    # WIN5Basedポリシー用にWIN5対象レースをフィルタリング
    win5_pred_df = None
    # WIN5レースの発走順序を保持するためにrace_id順序リストを作成
    win5_race_id_order = []
    if win5_race_keys:
        for race_key in win5_race_keys:
            # タプル形式: (Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum)
            if isinstance(race_key, tuple):
                year = str(race_key[0]).zfill(4)
                monthday = str(race_key[1]).zfill(4)
                jyo_cd = str(race_key[2]).zfill(2)
                kaiji = str(race_key[3]).zfill(2)
                nichiji = str(race_key[4]).zfill(2)
                race_num = str(race_key[5]).zfill(2)
            # 辞書形式
            elif isinstance(race_key, dict):
                year = str(race_key.get("Year", "")).zfill(4)
                monthday = str(race_key.get("MonthDay", "")).zfill(4)
                jyo_cd = str(race_key.get("JyoCD", "")).zfill(2)
                kaiji = str(race_key.get("Kaiji", "")).zfill(2)
                nichiji = str(race_key.get("Nichiji", "")).zfill(2)
                race_num = str(race_key.get("RaceNum", "")).zfill(2)
            else:
                continue
            race_id = f"{year}{monthday}-{jyo_cd}{kaiji}{nichiji}-{race_num}"
            win5_race_id_order.append(race_id)

        win5_pred_df = _filter_win5_races(pred_df, win5_race_keys)
        if len(win5_pred_df) > 0:
            logging.info(
                f"WIN5対象レース: {len(win5_pred_df['race_id'].unique())} レース"
            )
            logging.info(f"WIN5発走順序: {win5_race_id_order}")

    # WIN5Basedポリシー用に独自の選択馬を生成（win5base_thresholdを使用）
    # WIN5予測結果とは別の閾値で馬を選択する
    # 順序付き辞書としてWIN5発走順を維持
    win5base_selections = None
    if win5_pred_df is not None and len(win5_pred_df) > 0:
        win5base_selections = {}
        # win5_race_id_orderの順序で処理して発走順を維持
        for race_id in win5_race_id_order:
            race_df = win5_pred_df[win5_pred_df["race_id"] == race_id]
            if race_df.empty:
                continue
            if "proba" in race_df.columns:
                race_sorted = race_df.sort_values("proba", ascending=False)
                # 閾値以上の馬をすべて選択（上限なし）
                selected = race_sorted[race_sorted["proba"] >= win5base_threshold]
                if selected.empty:
                    # 閾値未満でも上位1頭は選択
                    selected = race_sorted.head(1)
                umaban_list = selected["Umaban"].astype(int).tolist()
                if umaban_list:
                    win5base_selections[race_id] = umaban_list
        if win5base_selections:
            logging.info(
                f"WIN5Basedポリシー用選択馬（閾値{win5base_threshold:.0%}）: "
                f"{len(win5base_selections)} レース"
            )
            for rid, umaban_list in win5base_selections.items():
                logging.info(f"  {rid}: {umaban_list}")

    for policy_name in policy_list:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"ポリシー: {policy_name}")
        logging.info(f"{'=' * 60}")

        # WIN5Basedポリシーの場合はWIN5対象レースのみを使用
        if policy_name.startswith("WIN5Based") and win5_pred_df is not None:
            target_df = win5_pred_df
            logging.info("  → WIN5対象レースのみを使用")
        else:
            target_df = pred_df

        # ポリシーを適用（WIN5Basedの場合はwin5base_selectionsを渡す）
        actions = apply_policy_to_predictions(
            target_df, policy_name, primary_task, win5_selections=win5base_selections
        )

        if not actions:
            logging.warning(
                f"ポリシー '{policy_name}' で推奨馬券が生成されませんでした"
            )
            continue

        # 推奨情報を整形（target_dfを使用）
        recommendation_df = format_recommendations(actions, target_df)
        recommendations[policy_name] = recommendation_df

        # 券種別の集計
        if not recommendation_df.empty:
            bet_type_counts = recommendation_df.groupby("bet_type").size()
            logging.info(f"推奨レース数: {len(recommendation_df['race_id'].unique())}")
            logging.info(f"推奨組み合わせ総数: {len(recommendation_df)}点")
            logging.info("券種別内訳:")
            for bet_type, count in bet_type_counts.items():
                logging.info(f"  {bet_type}: {count}点")

    return recommendations

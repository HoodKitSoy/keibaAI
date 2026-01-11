"""
WIN5予測モジュール

予測結果からWIN5の推奨組み合わせを生成します。
WIN5対象レースはJRA-DBのs_jyusyosiki_headから取得します。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from modules.core import RACE_KEY_COLS


def identify_win5_races(predictions: pd.DataFrame) -> List[Tuple[str, ...]]:
    """
    WIN5対象の5レースを特定

    JRA-DBのs_jyusyosiki_head（WIN5ヘッダ）から当日のWIN5設定を取得します。

    Args:
        predictions: 予測結果（DataFrame）

    Returns:
        List[Tuple[str, ...]]: WIN5対象レース情報のリスト（各要素は(Year, MonthDay, JyoCD, RaceNum)）
    """
    if not all(col in predictions.columns for col in RACE_KEY_COLS):
        logging.warning("レースキー列が見つかりません")
        return []

    # 予測データから日付を取得
    if predictions.empty:
        logging.warning("予測データが空です")
        return []

    first_row = predictions.iloc[0]
    year = str(first_row.get("Year", "")).zfill(4)
    month_day = str(first_row.get("MonthDay", "")).zfill(4)
    date_key = f"{year}{month_day}"

    logging.info(f"WIN5対象レース検索: {date_key}")

    # s_jyusyosiki_headからWIN5設定を読み込む
    db_dir = Path("./data/DB")
    candidate_names = [
        "s_jyusyosiki_head.parquet",
        "s_jyushoshiki_head.parquet",
        "s_jyusyosiki_head.parquet",
    ]

    head_path = None
    for name in candidate_names:
        p = db_dir / name
        if p.exists():
            head_path = p
            break

    if head_path is None:
        logging.warning(f"WIN5ヘッダファイルが見つかりません: {db_dir}")
        return []

    try:
        head_df = pd.read_parquet(head_path)
        logging.info(f"WIN5ヘッダ読み込み: {head_path.name} ({len(head_df)} レコード)")

        # データ区分の除外（中止/削除）
        if "DataKubun" in head_df.columns:
            head_df = head_df[~head_df["DataKubun"].astype(str).isin(["0", "9"])]

        # 対象日付のWIN5設定を検索
        target_rows = head_df[
            (head_df["Year"].astype(str).str.zfill(4) == year)
            & (head_df["MonthDay"].astype(str).str.zfill(4) == month_day)
        ]

        if target_rows.empty:
            logging.info(f"{date_key}: WIN5設定なし")
            return []

        # 最初の行から5レースの情報を取得
        row = target_rows.iloc[0]
        win5_races = []

        for i in range(1, 6):
            race_info = (
                str(row.get("Year", "0")).zfill(4),
                str(row.get("MonthDay", "0")).zfill(4),
                str(row.get(f"JyoCD{i}", "0")).zfill(2),
                str(row.get(f"RaceNum{i}", "0")).zfill(2),
            )
            win5_races.append(race_info)

        logging.info(f"WIN5対象レース検出: {len(win5_races)} レース")
        for i, (y, md, jyo, rn) in enumerate(win5_races, 1):
            logging.info(f"  {i}レース目: {y}/{md} 競馬場{jyo} {rn}R")

        return win5_races

    except Exception as e:
        logging.error(f"WIN5ヘッダ読み込みエラー: {e}")
        return []


def generate_win5_predictions(
    predictions: pd.DataFrame,
    threshold: float = 0.1,
    win5_race_keys: list = None,
) -> Dict[str, any]:
    """
    WIN5の予測組み合わせを生成

    Args:
        predictions: 予測結果（DataFrame、winタスク）
        threshold: 閾値（この値以上の馬をすべて選択）
        win5_race_keys: WIN5対象レースのキーリスト（Win5Processorから取得）
                        各要素は辞書 {"Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"}

    Returns:
        Dict[str, any]: WIN5予測情報
    """
    logging.info("WIN5予測の生成")
    logging.info(f"  閾値: {threshold:.2%}（閾値以上の馬をすべて選択）")

    # WIN5対象レースを取得
    if win5_race_keys:
        # Win5Processorから取得したレース情報を使用
        win5_races = win5_race_keys
        logging.info(f"  WIN5Processorからレース情報を取得: {len(win5_races)} レース")
    else:
        # フォールバック: 従来の方法でレースを特定
        win5_races_legacy = identify_win5_races(predictions)
        if not win5_races_legacy or len(win5_races_legacy) != 5:
            logging.warning("WIN5に必要な5レースが見つかりませんでした")
            return {}
        # 従来形式(tuple)を辞書形式に変換
        win5_races = []
        for y, md, jyo, rn in win5_races_legacy:
            win5_races.append(
                {
                    "Year": y,
                    "MonthDay": md,
                    "JyoCD": jyo,
                    "Kaiji": "00",
                    "Nichiji": "00",
                    "RaceNum": rn,
                }
            )

    if not win5_races or len(win5_races) != 5:
        logging.warning("WIN5に必要な5レースが見つかりませんでした")
        return {}

    # 各レースの上位馬を取得
    win5_selections = []
    win5_info = {
        "races": [],
        "selections": [],
        "total_combinations": 1,
    }

    for i, race_key in enumerate(win5_races, 1):
        # 辞書形式のレースキー
        if isinstance(race_key, dict):
            year = str(race_key.get("Year", "")).zfill(4)
            month_day = str(race_key.get("MonthDay", "")).zfill(4)
            jyo_cd = str(race_key.get("JyoCD", "")).zfill(2)
            kaiji = str(race_key.get("Kaiji", "")).zfill(2)
            nichiji = str(race_key.get("Nichiji", "")).zfill(2)
            race_num = str(race_key.get("RaceNum", "")).zfill(2)
        else:
            # タプル形式（後方互換性）
            year, month_day, jyo_cd, race_num = race_key
            kaiji, nichiji = "00", "00"

        logging.info(f"  WIN5レース{i}: {year}/{month_day} 場{jyo_cd} {race_num}R")

        # 該当レースの予測を抽出（zfillで比較を統一）
        race_mask = (
            (predictions["Year"].astype(str).str.zfill(4) == year)
            & (predictions["MonthDay"].astype(str).str.zfill(4) == month_day)
            & (predictions["JyoCD"].astype(str).str.zfill(2) == jyo_cd)
            & (predictions["RaceNum"].astype(str).str.zfill(2) == race_num)
        )
        race_pred = predictions[race_mask].copy()

        if race_pred.empty:
            logging.warning(
                f"    レース {year}{month_day}-{jyo_cd}-{race_num} の予測が見つかりません"
            )
            # 予測データの内容をデバッグ出力
            logging.debug(f"    予測データのJyoCD値: {predictions['JyoCD'].unique()}")
            logging.debug(
                f"    予測データのRaceNum値: {predictions['RaceNum'].unique()}"
            )
            continue

        # 確率でソート
        if "proba" in race_pred.columns:
            race_pred = race_pred.sort_values("proba", ascending=False)

            # 閾値以上の馬をすべて選択（上限なし）
            selected = race_pred[race_pred["proba"] >= threshold]

            if selected.empty:
                # 閾値未満でも上位1頭は選択
                selected = race_pred.head(1)
                logging.info(f"    閾値({threshold:.2%})以上の馬なし → 上位1頭を選択")

            umaban_list = selected["Umaban"].astype(int).tolist()
            proba_list = selected["proba"].tolist()

            win5_selections.append(umaban_list)

            # race_keyを辞書形式で保存
            race_key_dict = {
                "Year": year,
                "MonthDay": month_day,
                "JyoCD": jyo_cd,
                "Kaiji": kaiji,
                "Nichiji": nichiji,
                "RaceNum": race_num,
            }
            # race_idは YYYYMMDD-JJKKNN-RR 形式で統一（他モジュールと一致させる）
            race_id = f"{year}{month_day}-{jyo_cd}{kaiji}{nichiji}-{race_num}"
            win5_info["races"].append(
                {
                    "race_key": race_key_dict,
                    "race_id": race_id,
                    "selected_umaban": umaban_list,
                    "probabilities": proba_list,
                }
            )
            win5_info["total_combinations"] *= len(umaban_list)

            logging.info(
                f"    選択馬番: {umaban_list} (確率: {[f'{p:.1%}' for p in proba_list]})"
            )

    if len(win5_selections) == 5:
        win5_info["selections"] = win5_selections
        logging.info(f"\nWIN5組み合わせ総数: {win5_info['total_combinations']} 点")
        return win5_info
    else:
        logging.warning("WIN5に必要な5レース分の予測が揃いませんでした")
        return {}


def format_win5_recommendations(win5_info: Dict[str, any]) -> pd.DataFrame:
    """
    WIN5推奨情報をDataFrame形式に整形

    Args:
        win5_info: WIN5予測情報

    Returns:
        pd.DataFrame: 整形されたWIN5推奨情報
    """
    if not win5_info or "races" not in win5_info:
        return pd.DataFrame()

    records = []
    for race_info in win5_info["races"]:
        race_key = race_info["race_key"]
        # 辞書形式のrace_keyに対応
        if isinstance(race_key, dict):
            year = race_key.get("Year", "")
            month_day = race_key.get("MonthDay", "")
            jyo_cd = race_key.get("JyoCD", "")
            race_num = race_key.get("RaceNum", "")
        else:
            # タプル形式（後方互換性）
            year, month_day, jyo_cd, race_num = race_key

        records.append(
            {
                "Year": year,
                "MonthDay": month_day,
                "JyoCD": jyo_cd,
                "RaceNum": race_num,
                "race_id": race_info["race_id"],
                "selected_umaban": ",".join(
                    str(u) for u in race_info["selected_umaban"]
                ),
                "num_selections": len(race_info["selected_umaban"]),
                "probabilities": ",".join(
                    f"{p:.3f}" for p in race_info["probabilities"]
                ),
            }
        )

    df = pd.DataFrame(records)
    df["total_combinations"] = win5_info["total_combinations"]
    return df

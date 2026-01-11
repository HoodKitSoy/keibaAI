"""
ユーティリティ関数モジュール

競馬予測モデル評価に必要な各種ユーティリティ関数を提供します。
"""

import os
import re
from typing import List

import pandas as pd


def find_latest_preds_dir(base_dir: str) -> str:
    """
    最新の予測結果ディレクトリを検索

    Args:
        base_dir (str): 基準ディレクトリパス

    Returns:
        str: 最新のディレクトリパス（見つからない場合は空文字列）

    Note:
        日付形式のディレクトリ名を想定し、降順ソートで最新を取得
    """
    dates = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not dates:
        return ""
    latest = sorted(dates, reverse=True)[0]
    return os.path.join(base_dir, latest)


def load_preds_score_table(preds_path: str, task: str) -> pd.DataFrame:
    """
    予測結果ファイルからスコアテーブルを読み込み

    Args:
        preds_path (str): 予測結果ファイルのパス
        task (str): タスク種別（win, top3, rank）

    Returns:
        pd.DataFrame: スコアテーブル（index=race_id, columns=[Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban, Wakuban, score, proba]）

    Raises:
        ValueError: 必要な列が見つからない場合

    Note:
        - win/top3: proba 列を score にリネーム（probaも保持）
        - rank: p1 列（1位確率）を score にリネーム
        - race_id は Year+MonthDay-JyoCD+Kaiji+Nichiji-RaceNum 形式
        - Wakuban は枠連計算に必要（存在する場合のみ含める）
    """
    df = pd.read_parquet(preds_path)

    # 基本キー列
    base_cols = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "Umaban"]
    # Wakubanが存在すれば追加（枠連計算用）
    if "Wakuban" in df.columns:
        base_cols.append("Wakuban")

    # train.py の保存物はキー列 + proba or p1..pN
    if task in ("win", "top3"):
        if "proba" not in df.columns:
            raise ValueError(f"{preds_path} に 'proba' 列が見つかりません")
        score_table = df[base_cols + ["proba"]].copy()
        score_table["score"] = score_table["proba"]
    else:
        # rank: 1位確率に近い指標として p1 を採用（将来: 専用ポリシー）
        p1 = [c for c in df.columns if c.startswith("p1") or c == "p1"]
        if not p1:
            # 先頭クラスを p1 とみなす
            proba_cols = [c for c in df.columns if c.startswith("p")]
            if not proba_cols:
                raise ValueError(f"{preds_path} に p* 列が見つかりません")
            first = sorted(proba_cols)[0]
            pick = first
        else:
            pick = p1[0]
        score_table = df[base_cols + [pick]].copy()
        score_table["score"] = score_table[pick]
        score_table["proba"] = score_table[pick]  # WIN5用にprobaも設定

    # race_id を index に
    y = score_table["Year"].astype(str).str.zfill(4)
    md = score_table["MonthDay"].astype(str).str.zfill(4)
    j = score_table["JyoCD"].astype(str).str.zfill(2)
    ka = score_table["Kaiji"].astype(str).str.zfill(2)
    ni = score_table["Nichiji"].astype(str).str.zfill(2)
    rn = score_table["RaceNum"].astype(str).str.zfill(2)
    score_table.index = y + md + "-" + j + ka + ni + "-" + rn
    # 全ての必要な列を保持して返す
    return score_table


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    値を指定範囲内にクランプ

    Args:
        value (float): 対象の値
        minimum (float): 最小値
        maximum (float): 最大値

    Returns:
        float: クランプされた値
    """
    return max(minimum, min(maximum, value))


def format_threshold_value(value: float) -> str:
    """
    閾値を文字列形式に変換（末尾のゼロと小数点を除去）

    Args:
        value (float): 閾値

    Returns:
        str: フォーマットされた文字列

    Example:
        >>> format_threshold_value(0.300)
        '0.3'
    """
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_threshold_dir_name(value: float) -> str:
    """
    閾値をディレクトリ名形式に変換

    Args:
        value (float): 閾値

    Returns:
        str: ディレクトリ名（例: 'threshold_0p3'）
    """
    return f"threshold_{format_threshold_value(value).replace('.', 'p')}"


def build_win5_thresholds(args) -> List[float]:
    """
    WIN5シミュレーション用の閾値リストを構築

    Args:
        args: コマンドライン引数

    Returns:
        List[float]: 閾値のリスト

    Note:
        - win5フラグがFalseの場合は空リストを返す
        - min/max/step から閾値範囲を生成
        - 無効な値は自動補正される
    """
    if not getattr(args, "win5", False):
        return []

    min_th = (
        args.win5_threshold_min
        if args.win5_threshold_min is not None
        else args.win5_threshold
    )
    max_th = (
        args.win5_threshold_max
        if args.win5_threshold_max is not None
        else args.win5_threshold
    )

    min_th = clamp(min_th)
    max_th = clamp(max_th)

    if min_th > max_th:
        min_th, max_th = max_th, min_th

    step = args.win5_threshold_step if args.win5_threshold_step is not None else 0.0
    if step <= 0:
        # デフォルトの0.01刻みにフォールバック
        step = 0.01 if max_th != min_th else 0.01
        print(
            f"[WARN] 無効なWIN5ステップ幅が指定されたため 0.01 にフォールバックします (step={args.win5_threshold_step})"
        )

    thresholds: List[float] = []
    current = min_th
    while current <= max_th + 1e-9:
        thresholds.append(round(clamp(current), 4))
        current += step

    if thresholds and thresholds[-1] < max_th - 1e-9:
        thresholds.append(round(max_th, 4))

    thresholds = sorted({round(t, 4) for t in thresholds})
    if not thresholds:
        thresholds = [round(clamp(args.win5_threshold), 4)]

    return thresholds


def build_win5_candidates(score_table: pd.DataFrame, threshold: float) -> dict:
    """
    WIN5の候補馬をレースごとに構築

    Args:
        score_table (pd.DataFrame): スコアテーブル
        threshold (float): 候補選択の閾値

    Returns:
        dict: {race_id: [umaban_list]}

    Note:
        - スコアが閾値以上の馬を候補とする
        - 候補が0頭の場合はトップ1頭を自動選択
    """
    candidate_map = {}
    if score_table is None or score_table.empty:
        return candidate_map

    min_score = clamp(threshold)
    for race_id, df_race in score_table.groupby(level=0):
        df_sorted = df_race.sort_values("score", ascending=False)
        selected = df_sorted[df_sorted["score"] >= min_score - 1e-12]
        if selected.empty:
            selected = df_sorted.head(1)
        candidates = selected["Umaban"].dropna().apply(lambda x: int(float(x))).tolist()
        candidate_map[race_id] = candidates

    return candidate_map


def build_race_id_from_key(rk: dict) -> str:
    """
    レースキー辞書からrace_idを構築

    Args:
        rk (dict): レースキー辞書（Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum）

    Returns:
        str: race_id（例: '20240105-010102-11'）
    """
    y = str(rk.get("Year", 0)).zfill(4)
    md = str(rk.get("MonthDay", 0)).zfill(4)
    j = str(rk.get("JyoCD", 0)).zfill(2)
    ka = str(rk.get("Kaiji", 0)).zfill(2)
    ni = str(rk.get("Nichiji", 0)).zfill(2)
    rn = str(rk.get("RaceNum", 0)).zfill(2)
    return f"{y}{md}-{j}{ka}{ni}-{rn}"


def sanitize_name_for_path(name: str) -> str:
    """
    ファイルパスに使用できるように文字列をサニタイズ

    Args:
        name (str): 元の文字列

    Returns:
        str: サニタイズされた文字列
    """
    return re.sub(r"[<>:\"/\\|?*]", "_", name)

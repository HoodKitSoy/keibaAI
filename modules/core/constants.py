"""
共通定数モジュール

プロジェクト全体で使用する定数を一元管理します。
RACE_KEY_COLS、FINAL_COLUMN_ORDERなどの重要な定数を提供します。
"""

from typing import List

# レースキー列（レースを一意に識別する6列）
RACE_KEY_COLS: List[str] = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]

# レコードキー列（レース+馬番+枠番で出走馬を一意に識別）
# Wakubanは枠連計算に必要なため追加
RECORD_KEY_COLS: List[str] = RACE_KEY_COLS + ["Umaban", "Wakuban"]

# ターゲット列（学習時の目的変数）
TARGET_COL: str = "KakuteiJyuni"

# 特徴量から除外する列（予測時に未知の情報となる列）
# データリーク回避のため学習・予測の両方で除外
EXCLUDE_FEATURE_COLS: List[str] = [
    "KettoNum",  # 血統番号（ユニークID、個体識別）
    "Odds",  # 単勝オッズ（発走前は変動）
    "Ninki",  # 人気順（発走前は変動）
    "BaTaijyu",  # 馬体重（当日計量）
    "ZogenFugo",  # 増減符号（当日計量）
    "ZogenSa",  # 増減差（当日計量）
    "DMTime",  # マイニング予想タイム（当日確定）
    "DMGosaP",  # マイニング予想誤差+（当日確定）
    "DMGosaM",  # マイニング予想誤差-（当日確定）
    "DMJyuni",  # マイニング予想順位（当日確定）
    "TenkoCD",  # 天候コード（当日変動）
    "SibaBabaCD",  # 芝馬場状態（当日変動）
    "DirtBabaCD",  # ダート馬場状態（当日変動）
]

# 最終出力の列順（AGENTS.md 5.1 より）
# 全83項目: レース識別(6) + 馬の出走情報(25) + レース条件(8) + 血統(15) + 坂路調教(8) + ウッドチップ調教(21)
FINAL_COLUMN_ORDER: List[str] = [
    # レース識別情報 (6項目)
    "Year",
    "MonthDay",
    "JyoCD",
    "Kaiji",
    "Nichiji",
    "RaceNum",
    # 馬の出走情報 (25項目)
    "Wakuban",
    "Umaban",
    "KettoNum",
    "UmaKigoCD",
    "SexCD",
    "HinsyuCD",
    "KeiroCD",
    "Barei",
    "TozaiCD",
    "ChokyosiCode",
    "BanusiCode",
    "Futan",
    "Blinker",
    "KisyuCode",
    "MinaraiCD",
    "BaTaijyu",
    "ZogenFugo",
    "ZogenSa",
    "KakuteiJyuni",
    "Odds",
    "Ninki",
    "DMTime",
    "DMGosaP",
    "DMGosaM",
    "DMJyuni",
    # レース条件情報 (8項目)
    "YoubiCD",
    "GradeCD",
    "Kyori",
    "TrackCD",
    "SyussoTosu",
    "TenkoCD",
    "SibaBabaCD",
    "DirtBabaCD",
    # 血統情報 (15項目)
    "Ketto3InfoHansyokuNum1",
    "Ketto3InfoHansyokuNum2",
    "Ketto3InfoHansyokuNum3",
    "Ketto3InfoHansyokuNum4",
    "Ketto3InfoHansyokuNum5",
    "Ketto3InfoHansyokuNum6",
    "Ketto3InfoHansyokuNum7",
    "Ketto3InfoHansyokuNum8",
    "Ketto3InfoHansyokuNum9",
    "Ketto3InfoHansyokuNum10",
    "Ketto3InfoHansyokuNum11",
    "Ketto3InfoHansyokuNum12",
    "Ketto3InfoHansyokuNum13",
    "Ketto3InfoHansyokuNum14",
    "BreederCode",
    # 坂路調教データ (8項目)
    "TresenKubun",
    "HaronTime4",
    "LapTime4",
    "HaronTime3",
    "LapTime3",
    "HaronTime2",
    "LapTime2",
    "LapTime1",
    # ウッドチップ調教データ (21項目) - 重複回避のため _chip サフィックス
    "TresenKubun_chip",
    "Course_chip",
    "BabaAround_chip",
    "HaronTime10_chip",
    "LapTime10_chip",
    "HaronTime9_chip",
    "LapTime9_chip",
    "HaronTime8_chip",
    "LapTime8_chip",
    "HaronTime7_chip",
    "LapTime7_chip",
    "HaronTime6_chip",
    "LapTime6_chip",
    "HaronTime5_chip",
    "LapTime5_chip",
    "HaronTime4_chip",
    "LapTime4_chip",
    "HaronTime3_chip",
    "LapTime3_chip",
    "HaronTime2_chip",
    "LapTime2_chip",
    "LapTime1_chip",
]

# データ絞り込み条件（preprocessing用）
# 注意: TARGET_YEARS と TARGET_JYO_CDS は config.py から動的に読み込まれます
# ここはフォールバック値として残しています
import config as _root_config

TARGET_YEARS = {f"{y:04d}" for y in _root_config.PREPROCESSING_TARGET_YEARS}
TARGET_JYO_CDS = set(_root_config.PREPROCESSING_TARGET_JYO_CDS)

# データ区分の除外条件
EXCLUDE_DATA_KUBUNS = {"0", "9"}  # 0:削除、9:中止

# rankタスクのデフォルトクラス数（動的に上書き可能）
NUM_CLASSES_RANK = 18

# =====================================
# 競馬場コードマッピング
# =====================================
JYO_CODE_TO_NAME: dict[str, str] = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


def get_jyo_name(jyo_cd: str | int) -> str:
    """
    競馬場コードから競馬場名を取得

    Args:
        jyo_cd: 競馬場コード（文字列または整数）

    Returns:
        str: 競馬場名（不明な場合はコードをそのまま返す）

    Examples:
        >>> get_jyo_name("07")
        '中京'
        >>> get_jyo_name(5)
        '東京'
    """
    # 数値の場合は2桁のゼロ埋め文字列に変換
    if isinstance(jyo_cd, int):
        jyo_cd = f"{jyo_cd:02d}"
    else:
        jyo_cd = str(jyo_cd).zfill(2)
    return JYO_CODE_TO_NAME.get(jyo_cd, jyo_cd)

"""
自動競馬予想システム

30分ごとに以下の処理を繰り返す自動化スクリプトです。

処理フロー:
    1. preparing.pyの機能で全DBテーブルをエクスポート（s_テーブル + n_テーブル）
    2. preprocessing.pyで当日のレース前処理を実施
    3. s_raceのHassoTimeから未発走レースを判定
    4. 全レースをpredict.pyで予想実行
    5. 予想結果をメール送信（未発走レースのみフィルタリング）
    6. 30分待機して繰り返し

使用方法:
    python auto_predict.py

環境変数（.env）:
    # メール設定
    AUTO_PREDICT_EMAIL_TO: 送信先メールアドレス
    AUTO_PREDICT_EMAIL_FROM: 送信元メールアドレス
    AUTO_PREDICT_EMAIL_PASSWORD: SMTPパスワード
    AUTO_PREDICT_SMTP_HOST: SMTPサーバー（デフォルト: smtp.gmail.com）
    AUTO_PREDICT_SMTP_PORT: SMTPポート（デフォルト: 587）
    AUTO_PREDICT_INTERVAL: 実行間隔（分、デフォルト: 30）

    # テスト/デバッグ設定
    AUTO_PREDICT_TEST_DATE: テスト用日付（YYYYMMDD形式、省略時は今日）
    AUTO_PREDICT_TEST_TIME: テスト用時刻（HH:MM形式、省略時は現在時刻）
    AUTO_PREDICT_ONCE: 1回だけ実行してループしない（true/false）
    AUTO_PREDICT_SKIP_EXPORT: DBエクスポートをスキップ（true/false）
    AUTO_PREDICT_SKIP_PREPROCESSING: 前処理をスキップ（true/false）
    AUTO_PREDICT_NO_EMAIL: メール送信をスキップ（true/false）
    AUTO_PREDICT_DRY_RUN: 全処理をスキップして流れのみ確認（true/false）
"""

from __future__ import annotations

import logging
import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

import config
from modules.preprocessing._win5_processor import Win5Processor

# .env読み込み
load_dotenv()

# PyTorch警告を抑制
os.environ["PYTORCH_ALLOC_CONF"] = ""
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

# =====================================
# 環境変数設定
# =====================================
# メール設定
EMAIL_TO = os.getenv("AUTO_PREDICT_EMAIL_TO", "yosito72golf@gmail.com")
EMAIL_FROM = os.getenv("AUTO_PREDICT_EMAIL_FROM", "")
EMAIL_PASSWORD = os.getenv("AUTO_PREDICT_EMAIL_PASSWORD", "")
SMTP_HOST = os.getenv("AUTO_PREDICT_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("AUTO_PREDICT_SMTP_PORT", "587"))
INTERVAL = int(os.getenv("AUTO_PREDICT_INTERVAL", "30"))  # 分（WIN5モードでは無視）

# WIN5タイミング設定（分）
# 最初のレースの30分前に1回目の予測、2回目以降は各レースの15分前に予測
WIN5_FIRST_RACE_BEFORE = int(
    os.getenv("AUTO_PREDICT_WIN5_FIRST_RACE_BEFORE", "30")
)  # 開始：最初のレースの何分前
WIN5_NOTIFY_BEFORE = int(
    os.getenv("AUTO_PREDICT_WIN5_NOTIFY_BEFORE", "15")
)  # 通知：2-5番目レースの何分前

# 予測設定
PREDICT_TASKS = os.getenv("AUTO_PREDICT_TASKS", "win,top3,rank").split(",")
USE_POLICY_CONFIG = (
    os.getenv("AUTO_PREDICT_USE_POLICY_CONFIG", "true").lower() == "true"
)
POLICY_CONFIG_PATH = os.getenv("AUTO_PREDICT_POLICY_CONFIG_PATH", "policy_config.json")
WIN5_ENABLED = os.getenv("AUTO_PREDICT_WIN5_ENABLED", "true").lower() == "true"
WIN5_THRESHOLD = float(os.getenv("AUTO_PREDICT_WIN5_THRESHOLD", "0.1"))

# 出力ディレクトリ（predict.pyと分離）
AUTO_PREDICT_OUTPUT_DIR = os.getenv("AUTO_PREDICT_OUTPUT_DIR", "predictions_auto")

# エクスポート対象DBテーブル
DB_TABLES = os.getenv(
    "AUTO_PREDICT_DB_TABLES",
    "n_uma_race,n_race,n_uma,n_hanro,n_chip,n_harai,n_jyusyosiki_head,n_jyusyosiki,"
    "s_uma_race,s_race,s_uma,s_jyusyosiki_head,s_jyusyosiki",
).split(",")

# テスト/デバッグ設定（.envから読み込み）
_TEST_DATE_STR = os.getenv("AUTO_PREDICT_TEST_DATE", "")  # YYYYMMDD形式
_TEST_TIME_STR = os.getenv("AUTO_PREDICT_TEST_TIME", "")  # HH:MM形式
_ONCE = os.getenv("AUTO_PREDICT_ONCE", "false").lower() == "true"
_SKIP_EXPORT = os.getenv("AUTO_PREDICT_SKIP_EXPORT", "false").lower() == "true"
_SKIP_PREPROCESSING = (
    os.getenv("AUTO_PREDICT_SKIP_PREPROCESSING", "false").lower() == "true"
)
_NO_EMAIL = os.getenv("AUTO_PREDICT_NO_EMAIL", "false").lower() == "true"
_DRY_RUN = os.getenv("AUTO_PREDICT_DRY_RUN", "false").lower() == "true"

# テスト日付/時刻を解析
_TEST_DATE: Optional[str] = _TEST_DATE_STR if _TEST_DATE_STR else None
_TEST_TIME: Optional[datetime] = None
if _TEST_TIME_STR:
    try:
        target_date = (
            _TEST_DATE_STR if _TEST_DATE_STR else datetime.now().strftime("%Y%m%d")
        )
        time_parts = _TEST_TIME_STR.split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1]) if len(time_parts) > 1 else 0
        _TEST_TIME = datetime.strptime(target_date, "%Y%m%d").replace(
            hour=hour, minute=minute, second=0
        )
    except (ValueError, IndexError) as e:
        print(f"警告: AUTO_PREDICT_TEST_TIME の形式が不正です（HH:MM形式で指定）: {e}")

# 競馬場コード → 競馬場名マッピング
JYO_CODE_TO_NAME = {
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

# 競馬場名 → コードの逆マッピング
JYO_NAME_TO_CODE = {v: k for k, v in JYO_CODE_TO_NAME.items()}


# =====================================
# テスト用ヘルパー関数
# =====================================

# テストモード用：スクリプト開始時刻（経過時間計算用）
_SCRIPT_START_REAL_TIME: Optional[datetime] = None


def get_current_datetime() -> datetime:
    """
    現在の日時を取得（テストモード時はオーバーライド値を返す）

    テストモードでは、スクリプト開始からの実際の経過時間を
    テスト開始時刻に加算して返します。これにより、待機ループが
    正常に動作します。

    Returns:
        datetime: 現在の日時（テストモード時はテスト用時刻+経過時間）
    """
    global _SCRIPT_START_REAL_TIME

    if _TEST_TIME is not None:
        # テストモード: 実際の経過時間をテスト開始時刻に加算
        if _SCRIPT_START_REAL_TIME is None:
            _SCRIPT_START_REAL_TIME = datetime.now()
        elapsed = datetime.now() - _SCRIPT_START_REAL_TIME
        return _TEST_TIME + elapsed
    return datetime.now()


def get_target_date() -> str:
    """
    対象日付を取得（テストモード時はオーバーライド値を返す）

    Returns:
        str: 対象日付（YYYYMMDD形式）
    """
    if _TEST_DATE is not None:
        return _TEST_DATE
    return datetime.now().strftime("%Y%m%d")


def filter_unstarted_races_from_detail(content: str, unstarted_race_keys: set) -> str:
    """
    detail.txtの内容から未発走レースのみを抽出する

    Args:
        content: detail.txtの全内容
        unstarted_race_keys: 未発走レースのキーセット（JyoCD, RaceNum）のタプル

    Returns:
        str: 未発走レースの予想のみを含む内容（元の順序を維持 = WIN5発走順序）
    """
    import re

    lines = content.split("\n")
    race_sections = []  # [(jyo_cd, race_num, [lines]), ...]
    current_race_lines = []
    current_race_key = None
    is_unstarted_race = False
    in_race_section = False
    header_lines = []
    header_ended = False

    for line in lines:
        # レースセクションの開始を検出（■ YYYY/MMDD 競馬場名 RaceNumR）
        race_match = re.match(r"■\s*\d{4}/\d{4}\s+(\S+)\s+(\d+)R", line)
        if race_match:
            # 前のレースセクションを保存（未発走の場合）
            if in_race_section and is_unstarted_race and current_race_lines:
                race_sections.append((current_race_key, current_race_lines[:]))

            # 新しいレースセクション開始
            header_ended = True
            in_race_section = True
            current_race_lines = [line]

            jyo_name = race_match.group(1)
            race_num = race_match.group(2).zfill(2)
            jyo_cd = JYO_NAME_TO_CODE.get(jyo_name, "")
            current_race_key = (jyo_cd, race_num)

            # 未発走レースかどうかをチェック
            is_unstarted_race = current_race_key in unstarted_race_keys
        elif in_race_section:
            # レースセクション内の行を収集
            current_race_lines.append(line)
        elif not header_ended:
            # ヘッダー部分（レースセクション開始前）を収集
            header_lines.append(line)

    # 最後のレースセクションを保存
    if in_race_section and is_unstarted_race and current_race_lines:
        race_sections.append((current_race_key, current_race_lines[:]))

    # ※ソートは行わず、元のdetail.txtの順序を維持する
    # WIN5Basedポリシーの場合、元のdetail.txtがWIN5発走順序で出力されているため

    # セクションを結合
    filtered_lines = []
    for _, section_lines in race_sections:
        filtered_lines.extend(section_lines)

    # ヘッダーがある場合は結合（推奨レース数などを再計算）
    if header_lines and filtered_lines:
        # ヘッダーの推奨レース数を更新
        race_count = len(race_sections)
        updated_header = []
        for line in header_lines:
            if "推奨レース数:" in line:
                updated_header.append(
                    f"推奨レース数: {race_count} レース（未発走のみ）"
                )
            else:
                updated_header.append(line)
        return "\n".join(updated_header + [""] + filtered_lines)
    elif filtered_lines:
        return "\n".join(filtered_lines)
    else:
        return "未発走レースの推奨馬券はありません"


# =====================================
# ロギング設定
# =====================================
def setup_logging():
    """ロギング設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"auto_predict_{datetime.now().strftime('%Y%m%d')}.log"

    # Windows環境での文字化け対策: UTF-8エンコーディングを明示的に設定
    import io

    stdout_handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    )
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            stdout_handler,
        ],
    )


# =====================================
# DBエクスポート機能
# =====================================
def export_all_tables() -> bool:
    """
    preparing.pyの機能を使用して全DBテーブル（s_テーブル + n_テーブル）をエクスポート

    Returns:
        bool: 成功時True、失敗時False
    """
    try:
        logging.info("=" * 60)
        logging.info("DBエクスポート開始（必要テーブルのみ）")
        logging.info("=" * 60)

        # preparing.pyをバッチモードで実行（GUI不要）
        # ここでは直接DBエクスポート機能を呼び出す
        import queue

        from modules.preparing._config_manager import ConfigManager
        from modules.preparing._database_exporter import DatabaseExporter

        # 設定読み込み
        cfg = ConfigManager.load_config()

        # GUIキュー（ダミー）
        gui_queue = queue.Queue()

        # エクスポーター初期化
        exporter = DatabaseExporter(cfg, gui_queue)

        # 接続確認
        conn = exporter.get_connection()
        if not conn:
            logging.error("DB接続失敗")
            return False

        # .envから読み込んだテーブルリストを使用
        logging.info(f"エクスポート対象: {len(DB_TABLES)} テーブル")
        for table in DB_TABLES:
            logging.info(f"  - {table}")

        # エクスポート実行
        success = exporter.export_tables(DB_TABLES)

        if success:
            logging.info(f"エクスポート完了: {len(DB_TABLES)} テーブル")
        else:
            logging.error("エクスポート失敗")

        return success

    except Exception as e:
        logging.exception(f"DBエクスポートエラー: {e}")
        return False


# =====================================
# レース情報取得
# =====================================
def get_today_races() -> List[Dict]:
    """
    s_raceから対象日のレース情報を取得

    テストモード時は_TEST_DATEの日付のレースを取得します。

    Returns:
        List[Dict]: レース情報のリスト（Year, MonthDay, JyoCD, RaceNum, HassoTime等）
    """
    try:
        s_race_path = Path(config.PREPROCESSING_DB_DIR) / "s_race.parquet"

        if not s_race_path.exists():
            logging.error(f"s_race.parquetが見つかりません: {s_race_path}")
            return []

        df = pd.read_parquet(s_race_path)

        # 対象日付（テストモード時はオーバーライド値）
        target_date = get_target_date()
        target_year = target_date[:4]
        target_monthday = target_date[4:]

        logging.info(f"対象日付: {target_year}/{target_monthday}")

        # 対象日のレースを抽出
        mask = (df["Year"] == target_year) & (df["MonthDay"] == target_monthday)
        today_df = df[mask].copy()

        if today_df.empty:
            logging.warning(f"対象日（{target_date}）のレースが見つかりません")
            return []

        # HassoTimeをdatetime型に変換
        today_df["HassoTime"] = pd.to_datetime(
            today_df["Year"]
            + today_df["MonthDay"]
            + today_df["HassoTime"].str.zfill(4),
            format="%Y%m%d%H%M",
            errors="coerce",
        )

        # レース情報をリストに変換
        races = []
        for _, row in today_df.iterrows():
            races.append(
                {
                    "Year": row["Year"],
                    "MonthDay": row["MonthDay"],
                    "JyoCD": row["JyoCD"],
                    "Kaiji": row["Kaiji"],
                    "Nichiji": row["Nichiji"],
                    "RaceNum": row["RaceNum"],
                    "HassoTime": row["HassoTime"],
                    "RaceName": row.get("RaceNameShort", ""),
                }
            )

        # 発走時刻でソート
        races.sort(key=lambda x: x["HassoTime"])

        logging.info(f"本日のレース: {len(races)} レース")
        for race in races:
            # RaceNumを整数に変換（文字列の場合があるため）
            race_num = (
                int(race["RaceNum"])
                if isinstance(race["RaceNum"], str)
                else race["RaceNum"]
            )
            jyo_cd = str(race.get("JyoCD", "")).zfill(2)
            jyo_name = JYO_CODE_TO_NAME.get(jyo_cd, f"{jyo_cd}場")
            logging.info(
                f"  {race['HassoTime'].strftime('%H:%M')} - "
                f"{jyo_name} R{race_num:02d} {race['RaceName']}"
            )

        return races

    except Exception as e:
        logging.exception(f"レース情報取得エラー: {e}")
        return []


def get_unstarted_races(races: List[Dict]) -> List[Dict]:
    """
    未発走のレースのみをフィルタリング

    テストモード時は_TEST_TIMEの時刻を基準に判定します。

    Args:
        races: 全レース情報のリスト

    Returns:
        List[Dict]: 未発走レース情報のリスト
    """
    current_time = get_current_datetime()
    unstarted = [race for race in races if race["HassoTime"] > current_time]

    logging.info(f"判定基準時刻: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"未発走レース: {len(unstarted)} / {len(races)} レース")
    for race in unstarted:
        # RaceNumを整数に変換（文字列の場合があるため）
        race_num = (
            int(race["RaceNum"])
            if isinstance(race["RaceNum"], str)
            else race["RaceNum"]
        )
        jyo_cd = str(race.get("JyoCD", "")).zfill(2)
        jyo_name = JYO_CODE_TO_NAME.get(jyo_cd, f"{jyo_cd}場")
        logging.info(
            f"  {race['HassoTime'].strftime('%H:%M')} - "
            f"{jyo_name} R{race_num:02d} {race['RaceName']}"
        )

    return unstarted


def get_win5_race_times() -> List[Tuple[Dict[str, str], datetime]]:
    """
    WIN5対象レースの発走時刻を取得

    Returns:
        List[Tuple[Dict, datetime]]: (レースキー辞書, 発走時刻) のリスト（発走時刻順）
    """
    try:
        target_date = get_target_date()

        # Win5Processorで当日のWIN5スケジュールを取得
        win5_processor = Win5Processor(db_dir=Path(config.PREPROCESSING_DB_DIR))
        if not win5_processor.load_today(target_date):
            logging.warning(f"WIN5データが見つかりません: {target_date}")
            return []

        # s_race.parquetを読み込み
        s_race_path = Path(config.PREPROCESSING_DB_DIR) / "s_race.parquet"
        if not s_race_path.exists():
            logging.error(f"s_race.parquetが見つかりません: {s_race_path}")
            return []

        race_df = pd.read_parquet(s_race_path)

        # WIN5対象レースの発走時刻を取得
        race_times = win5_processor.get_win5_race_times(target_date, race_df)

        if not race_times:
            logging.warning(f"WIN5対象レースが見つかりません: {target_date}")
            return []

        # pd.Timestampをdatetimeに変換
        result = []
        for race, hasso_ts in race_times:
            result.append((race, hasso_ts.to_pydatetime()))

        logging.info(f"WIN5対象レース: {len(result)} レース")
        for i, (race, hasso_dt) in enumerate(result, 1):
            jyo_cd = race.get("JyoCD", "00")
            jyo_name = JYO_CODE_TO_NAME.get(jyo_cd, f"{jyo_cd}場")
            race_num = int(race.get("RaceNum", "0"))
            logging.info(
                f"  WIN5-{i}: {hasso_dt.strftime('%H:%M')} - {jyo_name} R{race_num:02d}"
            )

        return result

    except Exception as e:
        logging.exception(f"WIN5レース時刻取得エラー: {e}")
        return []


def get_next_notification_time(
    win5_races: List[Tuple[Dict[str, str], datetime]],
) -> Optional[Tuple[datetime, int]]:
    """
    次に通知すべき時刻とレース番号を取得

    最初のレースは30分前に既に出力済みのため、2番目以降のレース（15分前）のみ通知対象

    Args:
        win5_races: WIN5対象レースのリスト（発走時刻順）

    Returns:
        Optional[Tuple[datetime, int]]: (通知時刻, レース番号2-5) または None（全て終了）
    """
    current_time = get_current_datetime()

    # 2番目以降のレースのみ通知対象（i=2から開始）
    for i, (race, hasso_dt) in enumerate(win5_races, 1):
        if i == 1:
            # 最初のレースは30分前に既に出力済み
            continue
        notify_time = hasso_dt - timedelta(minutes=WIN5_NOTIFY_BEFORE)
        if current_time < notify_time:
            return (notify_time, i)

    return None


def should_start_processing(win5_races: List[Tuple[Dict[str, str], datetime]]) -> bool:
    """
    処理開始タイミングかどうかを判定

    最初のレースの30分前から処理を開始

    Args:
        win5_races: WIN5対象レースのリスト（発走時刻順）

    Returns:
        bool: 処理開始すべきならTrue
    """
    if not win5_races:
        return False

    current_time = get_current_datetime()
    first_race_time = win5_races[0][1]
    start_time = first_race_time - timedelta(minutes=WIN5_FIRST_RACE_BEFORE)

    return current_time >= start_time


def is_all_races_finished(win5_races: List[Tuple[Dict[str, str], datetime]]) -> bool:
    """
    全てのWIN5レースが発走済みかどうかを判定

    Args:
        win5_races: WIN5対象レースのリスト（発走時刻順）

    Returns:
        bool: 全レース発走済みならTrue
    """
    if not win5_races:
        return True

    current_time = get_current_datetime()
    last_race_time = win5_races[-1][1]

    return current_time > last_race_time


# =====================================
# 前処理実行
# =====================================
def run_preprocessing(target_date: str) -> bool:
    """
    preprocessing.pyを実行して当日のレース前処理を作成

    Args:
        target_date: 対象日付（YYYYMMDD形式）

    Returns:
        bool: 成功時True、失敗時False
    """
    try:
        logging.info("=" * 60)
        logging.info(f"前処理実行（当日モード）: {target_date}")
        logging.info("=" * 60)

        # 環境変数を設定してpreprocessing.pyを実行（コマンドライン引数は廃止済み）
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"  # 子プロセスの出力をUTF-8に強制
        env["PREPROCESSING_MODE"] = "date"
        env["PREPROCESSING_TARGET_DATE"] = target_date

        cmd = [sys.executable, "preprocessing.py"]
        logging.info(f"実行コマンド: {' '.join(cmd)}")
        logging.info("  PREPROCESSING_MODE: date")
        logging.info(f"  PREPROCESSING_TARGET_DATE: {target_date}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # ignoreからreplaceに変更
            env=env,
        )

        if result.returncode == 0:
            logging.info("前処理成功")
            return True
        else:
            logging.error(f"前処理失敗: {result.stderr}")
            return False

    except Exception as e:
        logging.exception(f"前処理エラー: {e}")
        return False


# =====================================
# 予想実行
# =====================================
def run_prediction(target_date: str) -> Optional[Path]:
    """
    predict.pyを実行して全レースの予想を生成

    Args:
        target_date: 対象日付（YYYYMMDD形式）

    Returns:
        Optional[Path]: 予想結果ディレクトリのパス（成功時）、None（失敗時）
    """
    try:
        logging.info("=" * 60)
        logging.info(f"予想実行（全レース）: {target_date}")
        logging.info("=" * 60)

        # 環境変数を設定してpredict.pyを実行（コマンドライン引数は廃止済み）
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"  # 子プロセスの出力をUTF-8に強制
        env["PREDICT_MODE"] = "date"
        env["PREDICT_TARGET_DATE"] = target_date
        env["PREDICT_OUTPUT_DIR"] = AUTO_PREDICT_OUTPUT_DIR
        env["PREDICT_TASKS"] = ",".join(PREDICT_TASKS)
        env["PREDICT_USE_POLICY_CONFIG"] = "true" if USE_POLICY_CONFIG else "false"
        env["PREDICT_POLICY_CONFIG_PATH"] = POLICY_CONFIG_PATH
        env["PREDICT_WIN5_ENABLED"] = "true" if WIN5_ENABLED else "false"
        env["PREDICT_WIN5_THRESHOLD"] = str(WIN5_THRESHOLD)
        env["PREDICT_VERBOSE"] = "false"

        cmd = [sys.executable, "predict.py"]
        logging.info(f"実行コマンド: {' '.join(cmd)}")
        logging.info("  PREDICT_MODE: date")
        logging.info(f"  PREDICT_TARGET_DATE: {target_date}")
        logging.info(f"  PREDICT_OUTPUT_DIR: {AUTO_PREDICT_OUTPUT_DIR}")
        logging.info(f"  PREDICT_TASKS: {','.join(PREDICT_TASKS)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # ignoreからreplaceに変更（読めない文字は置換）
            env=env,
        )

        # PyTorchの警告は無視してreturncodeで判定
        if result.returncode == 0:
            logging.info("予想成功")
            output_dir = Path(AUTO_PREDICT_OUTPUT_DIR) / target_date
            return output_dir
        else:
            # stderrから警告を除いたエラーメッセージを抽出
            error_lines = []
            for line in result.stderr.split("\n"):
                # PyTorch/CUDA警告は無視
                if "PYTORCH" in line or "Warning:" in line or "AllocatorConfig" in line:
                    continue
                if line.strip():
                    error_lines.append(line)

            if error_lines:
                logging.error(f"予想失敗: {' '.join(error_lines)}")
            else:
                # 警告のみでエラーがない場合は成功扱い
                logging.warning(f"警告あり（無視）: {result.stderr[:200]}")
                output_dir = Path(AUTO_PREDICT_OUTPUT_DIR) / target_date
                return output_dir
            return None

    except Exception as e:
        logging.exception(f"予想エラー: {e}")
        return None


# =====================================
# メール送信
# =====================================
def send_email(
    subject: str, body: str, attachments: Optional[List[Path]] = None
) -> bool:
    """
    メール送信

    Args:
        subject: 件名
        body: 本文
        attachments: 添付ファイルのリスト

    Returns:
        bool: 成功時True、失敗時False
    """
    try:
        if not EMAIL_FROM or not EMAIL_PASSWORD:
            logging.error("メール設定が不完全です（EMAIL_FROM, EMAIL_PASSWORD）")
            return False

        # メッセージ作成
        msg = MIMEMultipart()
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg["Subject"] = subject

        # 本文
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # 添付ファイル（TODO: 実装予定）
        # if attachments:
        #     for file_path in attachments:
        #         # 添付ファイル処理

        # SMTP送信
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)

        logging.info(f"メール送信成功: {EMAIL_TO}")
        return True

    except Exception as e:
        logging.exception(f"メール送信エラー: {e}")
        return False


def format_prediction_email(
    target_date: str, output_dir: Path, races: List[Dict]
) -> str:
    """
    予想結果をメール本文用にフォーマット（未発走レースのみを含む）

    Args:
        target_date: 対象日付
        output_dir: 予想結果ディレクトリ（全レースの予測が保存されている）
        races: 未発走レース情報リスト

    Returns:
        str: メール本文（未発走レースの予想のみ含む）
    """
    body_lines = [
        "=" * 60,
        "競馬AI自動予想システム",
        "=" * 60,
        f"予想日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
        f"対象日: {target_date[:4]}年{target_date[4:6]}月{target_date[6:]}日",
        "",
        "=" * 60,
        f"未発走レース一覧（{len(races)}レース）",
        "=" * 60,
    ]

    # 未発走レースのJyoCD+RaceNumセットを作成（フィルタリング用）
    unstarted_race_keys = set()
    for race in races:
        jyo_cd = str(race.get("JyoCD", "")).zfill(2)
        race_num = str(race.get("RaceNum", "")).zfill(2)
        unstarted_race_keys.add((jyo_cd, race_num))

        race_num_int = (
            int(race["RaceNum"])
            if isinstance(race["RaceNum"], str)
            else race["RaceNum"]
        )
        jyo_name = JYO_CODE_TO_NAME.get(jyo_cd, f"{jyo_cd}場")
        body_lines.append(
            f"{race['HassoTime'].strftime('%H:%M')} - "
            f"{jyo_name} R{race_num_int:02d} {race['RaceName']}"
        )

    # --- WIN5予測結果 ---
    body_lines.extend(["", "=" * 60, "WIN5予測結果", "=" * 60])

    # win5_detail.txtを優先的に使用（フォーマット済みの詳細情報）
    win5_detail_file = output_dir / "win5_detail.txt"
    win5_csv_file = output_dir / "win5_recommendations.csv"

    if win5_detail_file.exists():
        try:
            with open(win5_detail_file, "r", encoding="utf-8") as f:
                win5_content = f.read()
            # ヘッダー行（=====）を除いて本文を追加
            lines = win5_content.strip().split("\n")
            for line in lines:
                if not line.startswith("="):
                    body_lines.append(line)
        except Exception as e:
            body_lines.append(f"WIN5詳細読み込みエラー: {e}")
    elif win5_csv_file.exists():
        try:
            win5_df = pd.read_csv(win5_csv_file)
            if len(win5_df) > 0:
                body_lines.append("")
                body_lines.append("【対象レース】")
                total_combinations = (
                    win5_df["total_combinations"].iloc[0]
                    if "total_combinations" in win5_df.columns
                    else 0
                )

                for i, row in win5_df.iterrows():
                    jyo_cd = str(row.get("JyoCD", "")).zfill(2)
                    jyo_name = JYO_CODE_TO_NAME.get(jyo_cd, f"{jyo_cd}場")
                    race_num = int(row.get("RaceNum", 0))
                    selected = row.get("selected_umaban", "")
                    body_lines.append(
                        f"  {i + 1}レース目: {jyo_name} {race_num}R → 推奨馬番: {selected}"
                    )

                body_lines.append("")
                body_lines.append(f"【組み合わせ総数】 {total_combinations} 点")
                body_lines.append(
                    f"【購入金額（100円/点）】 {total_combinations * 100:,}円"
                )
            else:
                body_lines.append("WIN5対象レースがありません")
        except Exception as e:
            body_lines.append(f"WIN5結果読み込みエラー: {e}")
    else:
        body_lines.append("WIN5予測ファイルが見つかりません")

    # --- ポリシー別予測結果（未発走レースのみフィルタリング） ---
    body_lines.extend(
        ["", "=" * 60, "ポリシー別推奨馬券（未発走レースのみ）", "=" * 60]
    )

    # タスクごとにディレクトリを探索
    tasks_found = []
    for task in PREDICT_TASKS:
        task_policy_dir = output_dir / task / "policy_recommendations"
        if task_policy_dir.exists():
            tasks_found.append(task)
            body_lines.extend(
                ["", "=" * 60, f"★★★ タスク: {task.upper()} ★★★", "=" * 60]
            )

            # 各ポリシーのdetail.txtを読み込み、未発走レースのみ抽出
            detail_files = sorted(task_policy_dir.glob("*_detail.txt"))
            if detail_files:
                for detail_file in detail_files:
                    policy_name = detail_file.stem.replace("_detail", "")
                    body_lines.extend(["", "-" * 50, f"【{policy_name}】", "-" * 50])
                    try:
                        with open(detail_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        # 未発走レースのみをフィルタリングして出力
                        filtered_content = filter_unstarted_races_from_detail(
                            content, unstarted_race_keys
                        )
                        body_lines.append(filtered_content)
                    except Exception as e:
                        body_lines.append(f"読み込みエラー: {e}")
            else:
                body_lines.append(f"  {task} のポリシー詳細ファイルが見つかりません")

    if not tasks_found:
        # 従来のディレクトリ構造（タスク別でない場合）にフォールバック
        policy_dir = output_dir / "policy_recommendations"
        if policy_dir.exists():
            body_lines.extend(["", "(従来形式のポリシー推奨)", ""])
            detail_files = sorted(policy_dir.glob("*_detail.txt"))
            if detail_files:
                for detail_file in detail_files:
                    policy_name = detail_file.stem.replace("_detail", "")
                    body_lines.extend(["", "-" * 50, f"【{policy_name}】", "-" * 50])
                    try:
                        with open(detail_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        # 未発走レースのみをフィルタリング
                        filtered_content = filter_unstarted_races_from_detail(
                            content, unstarted_race_keys
                        )
                        body_lines.append(filtered_content)
                    except Exception as e:
                        body_lines.append(f"読み込みエラー: {e}")
            else:
                body_lines.append("ポリシー詳細ファイルが見つかりません")
        else:
            body_lines.append("ポリシー推奨ディレクトリが見つかりません")

    # ファイル保存先情報を作成
    policy_paths = []
    for task in PREDICT_TASKS:
        task_policy_dir = output_dir / task / "policy_recommendations"
        if task_policy_dir.exists():
            policy_paths.append(f"  {task}: {task_policy_dir.absolute()}")

    body_lines.extend(
        [
            "",
            "=" * 60,
            "ファイル保存先",
            "=" * 60,
            f"予測結果: {output_dir.absolute()}",
            "ポリシー推奨（タスク別）:",
        ]
    )
    if policy_paths:
        body_lines.extend(policy_paths)
    else:
        body_lines.append(f"  {output_dir / 'policy_recommendations'}")

    body_lines.extend(
        [
            f"WIN5予測: {win5_csv_file}",
            "",
            "=" * 60,
            "注意事項",
            "=" * 60,
            "・この予想はAIモデルによる統計的な予測です",
            "・馬券購入は自己責任でお願いします",
            "・投資は余裕資金の範囲内で行ってください",
            "",
            "=" * 60,
        ]
    )

    return "\n".join(body_lines)


# =====================================
# メイン処理
# =====================================
def process_prediction_cycle() -> bool:
    """
    予想処理サイクルを1回実行

    グローバル変数_SKIP_EXPORT, _SKIP_PREPROCESSING, _NO_EMAIL, _DRY_RUNで
    各処理のスキップを制御できます。

    Returns:
        bool: 成功時True、失敗時False
    """
    try:
        target_date = get_target_date()

        logging.info("=" * 60)
        logging.info("予想処理サイクル開始")
        logging.info(f"対象日付: {target_date}")
        if _TEST_TIME:
            logging.info(f"テスト時刻: {_TEST_TIME.strftime('%H:%M:%S')}")
        if _DRY_RUN:
            logging.info("【DRY-RUN モード】全処理をスキップします")
        logging.info("=" * 60)

        # 1. DBエクスポート（全テーブル）
        if _DRY_RUN or _SKIP_EXPORT:
            logging.info("[スキップ] DBエクスポート")
        else:
            if not export_all_tables():
                logging.error("DBエクスポート失敗")
                return False

        # 2. 前処理実行（当日モード）
        if _DRY_RUN or _SKIP_PREPROCESSING:
            logging.info("[スキップ] 前処理")
        else:
            if not run_preprocessing(target_date):
                logging.error("前処理失敗")
                return False

        # 3. レース情報取得
        races = get_today_races()
        if not races:
            logging.warning(f"対象日（{target_date}）のレースが見つかりません")
            return False

        # 4. 未発走レースをフィルタリング（メール送信用）
        unstarted_races = get_unstarted_races(races)
        if not unstarted_races:
            logging.info("未発走レースがありません")
            return True  # エラーではないのでTrue

        # 5. 予想実行（全レースを予測、フィルタなし）
        if _DRY_RUN:
            logging.info("[スキップ] 予想実行")
            output_dir = Path(AUTO_PREDICT_OUTPUT_DIR) / target_date
        else:
            output_dir = run_prediction(target_date)
            if not output_dir:
                logging.error("予想失敗")
                return False

        # 6. メール送信（未発走レースのみをフィルタリングして送信）
        subject = (
            f"【競馬AI予想】{target_date[:4]}/{target_date[4:6]}/{target_date[6:]} "
            f"未発走レース {len(unstarted_races)} レース"
        )

        if _DRY_RUN:
            logging.info("[スキップ] メール本文生成")
            body = "(dry-run: メール本文は生成されませんでした)"
        else:
            body = format_prediction_email(target_date, output_dir, unstarted_races)

        if _DRY_RUN or _NO_EMAIL:
            logging.info("[スキップ] メール送信")
            logging.info(f"件名: {subject}")
            if not _DRY_RUN:
                # メール本文をログに出力（最初の2000文字）
                logging.info("=" * 40)
                logging.info("メール本文（先頭2000文字）:")
                logging.info("=" * 40)
                for line in body[:2000].split("\n"):
                    logging.info(line)
                if len(body) > 2000:
                    logging.info(f"... (残り {len(body) - 2000} 文字)")
        else:
            if not send_email(subject, body):
                logging.error("メール送信失敗")
                return False

        logging.info("予想処理サイクル完了")
        return True

    except Exception as e:
        logging.exception(f"予想処理サイクルエラー: {e}")
        return False


def main():
    """メイン関数"""
    setup_logging()

    logging.info("=" * 60)
    logging.info("競馬AI自動予想システム起動")
    logging.info("=" * 60)

    # テストモード情報を表示
    is_test_mode = (
        _TEST_DATE
        or _TEST_TIME
        or _ONCE
        or _SKIP_EXPORT
        or _SKIP_PREPROCESSING
        or _NO_EMAIL
        or _DRY_RUN
    )
    if is_test_mode:
        logging.info("【テストモード】")
        if _TEST_DATE:
            logging.info(f"  テスト日付: {_TEST_DATE}")
        if _TEST_TIME:
            logging.info(f"  テスト時刻: {_TEST_TIME.strftime('%H:%M:%S')}")
        if _ONCE:
            logging.info("  単発実行: 有効")
        if _SKIP_EXPORT:
            logging.info("  DBエクスポート: スキップ")
        if _SKIP_PREPROCESSING:
            logging.info("  前処理: スキップ")
        if _NO_EMAIL:
            logging.info("  メール送信: スキップ")
        if _DRY_RUN:
            logging.info("  DRY-RUN: 有効（全処理スキップ）")
        logging.info("-" * 60)

    logging.info(f"送信先: {EMAIL_TO}")
    logging.info(
        f"WIN5モード: 最初のレースの{WIN5_FIRST_RACE_BEFORE}分前に出力, 2回目以降は各レースの{WIN5_NOTIFY_BEFORE}分前に出力"
    )
    logging.info(f"予測タスク: {', '.join(PREDICT_TASKS)}")
    logging.info(f"出力ディレクトリ: {AUTO_PREDICT_OUTPUT_DIR}")
    logging.info(
        f"ポリシー設定: {POLICY_CONFIG_PATH if USE_POLICY_CONFIG else '全ポリシー使用'}"
    )
    logging.info(
        f"WIN5予測: {'有効' if WIN5_ENABLED else '無効'}"
        + (f" (閾値: {WIN5_THRESHOLD})" if WIN5_ENABLED else "")
    )
    logging.info(f"DBテーブル数: {len(DB_TABLES)}")
    logging.info("=" * 60)

    # メール設定確認（NO_EMAILまたはDRY_RUNの場合はスキップ）
    if not _NO_EMAIL and not _DRY_RUN:
        if not EMAIL_FROM or not EMAIL_PASSWORD:
            logging.error("メール設定が不完全です")
            logging.error(".envに以下を設定してください:")
            logging.error("  AUTO_PREDICT_EMAIL_FROM")
            logging.error("  AUTO_PREDICT_EMAIL_PASSWORD")
            logging.error("または AUTO_PREDICT_NO_EMAIL=true を設定してください")
            return 1

    try:
        # 単発実行モード
        if _ONCE:
            logging.info("単発実行モード: 1回だけ実行して終了します")
            success = process_prediction_cycle()
            return 0 if success else 1

        # WIN5対象レースのタイミングに合わせたループモード
        logging.info("=" * 60)
        logging.info("WIN5対象レースのタイミングで動作開始")
        logging.info("=" * 60)

        # DBエクスポート（WIN5レース情報取得のため先に実行）
        if not _SKIP_EXPORT and not _DRY_RUN:
            logging.info("WIN5レース情報取得のためDBエクスポートを実行...")
            if not export_all_tables():
                logging.error("DBエクスポート失敗")
                return 1

        # WIN5レース時刻を取得
        win5_races = get_win5_race_times()
        if not win5_races:
            logging.error("本日のWIN5対象レースが見つかりません")
            return 1

        first_race_time = win5_races[0][1]
        last_race_time = win5_races[-1][1]
        start_time = first_race_time - timedelta(minutes=WIN5_FIRST_RACE_BEFORE)

        logging.info(f"最初のレース: {first_race_time.strftime('%H:%M')}")
        logging.info(f"最後のレース: {last_race_time.strftime('%H:%M')}")
        logging.info(
            f"処理開始予定: {start_time.strftime('%H:%M')}（最初のレースの{WIN5_FIRST_RACE_BEFORE}分前）"
        )

        # 処理開始まで待機
        while True:
            now = get_current_datetime()
            if now >= start_time:
                break

            wait_seconds = min((start_time - now).total_seconds(), 300)  # 最大5分待機
            if wait_seconds > 0:
                logging.info(
                    f"処理開始まで待機中... 開始予定: {start_time.strftime('%H:%M')} "
                    f"(残り {int((start_time - now).total_seconds() / 60)} 分)"
                )
                time.sleep(wait_seconds)

        # 1回目の予想処理サイクル（最初のレースの30分前）
        logging.info("=" * 60)
        logging.info("【1回目】予想処理サイクル開始（最初のレースの30分前）")
        logging.info("=" * 60)
        process_prediction_cycle()

        # 2回目以降：各レースの15分前に通知ループ（2-5番目のレースのみ）
        notified_races = set()
        notified_races.add(1)  # 最初のレースは既に出力済み

        while True:
            now = get_current_datetime()

            # 全レース終了チェック
            if is_all_races_finished(win5_races):
                logging.info("=" * 60)
                logging.info("全WIN5レースが発走しました。システムを終了します。")
                logging.info("=" * 60)
                return 0

            # 次の通知時刻を取得
            next_notify = get_next_notification_time(win5_races)
            if next_notify is None:
                # 全レースの通知完了、最後のレース発走まで待機
                wait_until = last_race_time + timedelta(minutes=1)
                wait_seconds = max((wait_until - now).total_seconds(), 0)
                if wait_seconds > 0:
                    logging.info("全レース通知完了。最後のレース発走まで待機...")
                    time.sleep(min(wait_seconds, 60))
                continue

            notify_time, race_num = next_notify

            # 既に通知済みならスキップ
            if race_num in notified_races:
                time.sleep(30)
                continue

            # 通知時刻まで待機
            if now < notify_time:
                wait_seconds = (notify_time - now).total_seconds()
                if wait_seconds > 60:
                    logging.info(
                        f"次の通知: WIN5-{race_num} ({notify_time.strftime('%H:%M')}) - "
                        f"残り {int(wait_seconds / 60)} 分"
                    )
                    time.sleep(min(wait_seconds, 60))
                    continue
                else:
                    time.sleep(wait_seconds)

            # 通知実行（予想処理サイクル）
            logging.info("=" * 60)
            logging.info(f"WIN5-{race_num} の通知時刻になりました")
            logging.info("=" * 60)
            process_prediction_cycle()
            notified_races.add(race_num)

    except KeyboardInterrupt:
        logging.info("\n自動予想システムを停止しました")
        return 0
    except Exception:
        logging.exception("予期しないエラーが発生しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())

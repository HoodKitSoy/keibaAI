"""
JRA-VAN 公式DB (Parquet) を用いた前処理スクリプト

AGENTS.md の仕様に基づき、以下を実施する：
 1) ./data/DB/ から Parquet を読み込み
 2) フィルタ（2016-2024年、競馬場01-10、DataKubun除外）
 3) 複合キーでレース情報結合、KettoNumで馬情報結合
 4) レース日前の直近の調教（坂路とウッドチップ）を結合
 5) 列順整形・ソート・欠損埋め
 6) ./data/tmp に CSV/Parquet を出力、途中の検証ログを出力

使用方法:
    # .envファイルで設定を行い、実行
    python preprocessing.py

設定（.env）:
    PREPROCESSING_MODE: 処理モード（train=学習用, today=当日予想, date=日付指定）
    PREPROCESSING_TARGET_DATE: 対象日付（YYYYMMDD形式、MODE=date時に使用）

機能:
    - JRA-VAN DBからのデータ読み込み（n_テーブルまたはs_テーブル）
    - データ結合とフィルタリング
    - 調教データの結合
    - データ整形と出力

注：初期リリースは pandas ベース。データ量次第で polars/dask への移行を検討。
"""

import logging
import os
import sys
from datetime import date

import config as root_config
from modules.core import DB_DIR_STR as DB_DIR
from modules.core import FINAL_COLUMN_ORDER
from modules.core import LOG_PATH_STR as LOG_PATH
from modules.core import TMP_DIR_STR as TMP_DIR
from modules.preprocessing import (
    attach_hanro_and_chip,
    filter_frames,
    load_sources,
    load_today_sources,
    merge_core,
    reorder_and_fill,
    save_outputs,
)


def setup_logging() -> None:
    """
    ログ設定を初期化

    ログファイルと標準出力の両方にログを出力するよう設定します。
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    """
    メイン関数

    データ前処理パイプライン全体を実行します：
    1. データソースの読み込み（n_テーブルまたはs_テーブル）
    2. フィルタリングとデータ結合
    3. 調教データの付与
    4. データ整形と出力

    モード:
    - train: 学習用データ（2016-2024年）の前処理
    - today: 当日予想用データの前処理（今日の日付）
    - date: 指定日付の予想用データの前処理

    設定は.envファイルから読み込みます：
    - PREPROCESSING_MODE: 処理モード
    - PREPROCESSING_TARGET_DATE: 対象日付（mode=date時）

    Raises:
        ValueError: mode=dateで日付が未設定の場合
        Exception: 処理中に致命的なエラーが発生した場合
    """
    setup_logging()

    # .envから設定を取得
    mode = root_config.PREPROCESSING_MODE.lower()
    target_date_env = root_config.PREPROCESSING_TARGET_DATE

    # mode=date で日付が未指定の場合はエラー
    if mode == "date" and not target_date_env:
        raise ValueError(
            "PREPROCESSING_MODE=date の場合は PREPROCESSING_TARGET_DATE を設定してください"
        )

    try:
        if mode in ("today", "date"):
            # 当日予想用 または 日付指定予想用データの処理
            if mode == "today":
                target_date = date.today().strftime("%Y%m%d")
            else:
                target_date = target_date_env

            logging.info("予想用データ処理モード: mode=%s, 日付=%s", mode, target_date)

            # ステップ1: 読み込み（s_テーブル）
            dfs = load_today_sources(DB_DIR, target_date)
            logging.info("読み込み完了: %s", {k: v.shape for k, v in dfs.items()})

            # ステップ2: マージ（フィルタは不要 - すでに当日データのみ）
            base = merge_core(dfs["s_uma_race"], dfs["s_race"], dfs["s_uma"])
            logging.info("主要結合完了: %s", base.shape)

            # 調教（最新）付与（n_hanro, n_chipを使用）
            base = attach_hanro_and_chip(base, dfs["n_hanro"], dfs["n_chip"])
            logging.info("調教結合完了: %s", base.shape)

            # ステップ3: 整形・出力
            out = reorder_and_fill(base)

            # 簡易検証
            missing = [c for c in FINAL_COLUMN_ORDER if c not in out.columns]
            if missing:
                logging.warning("最終列不足: %s", missing)
            logging.info("出力データ shape=%s", out.shape)

            # 当日予想用として保存
            save_outputs(out, out_basename=f"today_preprocessed_{target_date}")

        else:  # mode == "train"
            # 過去データの処理（学習用）
            logging.info(
                "学習用データ処理モード: 対象年=%s",
                root_config.PREPROCESSING_TARGET_YEARS,
            )

            # ステップ1: 読み込み
            dfs = load_sources(DB_DIR)
            logging.info("読み込み完了: %s", {k: v.shape for k, v in dfs.items()})

            # ステップ2: フィルタ・マージ
            dfs = filter_frames(dfs)
            base = merge_core(
                dfs["n_uma_race"], dfs["n_race"], dfs["n_uma"]
            )  # UMA_RACE ベース
            logging.info("主要結合完了: %s", base.shape)

            # 調教（最新）付与
            base = attach_hanro_and_chip(base, dfs["n_hanro"], dfs["n_chip"])
            logging.info("調教結合完了: %s", base.shape)

            # ステップ3: 整形・出力
            out = reorder_and_fill(base)

            # 簡易検証
            missing = [c for c in FINAL_COLUMN_ORDER if c not in out.columns]
            if missing:
                logging.warning("最終列不足: %s", missing)
            logging.info("出力データ shape=%s", out.shape)

            # 対象年から動的にファイル名を生成
            years = sorted(root_config.PREPROCESSING_TARGET_YEARS)
            if len(years) >= 2:
                out_basename = f"db_preprocessed_{years[0]}_{years[-1]}"
            elif len(years) == 1:
                out_basename = f"db_preprocessed_{years[0]}"
            else:
                out_basename = "db_preprocessed"
            save_outputs(out, out_basename=out_basename)

    except Exception:
        logging.exception("前処理で致命的なエラーが発生しました。")
        raise


if __name__ == "__main__":
    main()

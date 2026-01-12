"""
当日レース予測スクリプト

preprocessing.pyで作成された前処理済みファイルを使用して、
学習済みモデルで予測を行います。

使用方法:
    # .envファイルで設定を行い、実行
    python predict.py

設定（.env）:
    PREDICT_MODE: 予測モード（today=当日、date=日付指定）
    PREDICT_TARGET_DATE: 対象日付（MODE=date時、YYYYMMDD形式）
    PREDICT_TASKS: 予測タスク（win,top3,rank）
    PREDICT_WIN5_ENABLED: WIN5予測有効化

機能:
    - 前処理済み当日データの読み込み（./data/tmp/today_preprocessed_{date}.parquet）
    - 学習済みモデルによる予測（win, top3, rank）
    - 予測結果の保存（Parquet, CSV, JSON）

出力:
    - ./predictions/prediction_{date}_{task}.parquet: タスク別予測結果
    - ./predictions/prediction_{date}_{task}.csv: タスク別予測結果（CSV）
    - ./predictions/prediction_{date}_combined.csv: 統合予測結果
    - ./predictions/prediction_{date}_summary.json: サマリー情報
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import config
from modules.prediction import (
    align_to_training_format,
    format_win5_recommendations,
    generate_policy_recommendations,
    generate_win5_predictions,
    load_preprocessed_data,
    predict_with_models,
    print_prediction_summary,
    save_policy_report,
    save_predictions,
    save_win5_report,
)
from modules.preprocessing._win5_processor import Win5Processor


def filter_unstarted_races(df: pd.DataFrame, predict_date: str) -> pd.DataFrame:
    """
    未発走レースのみをフィルタリング

    s_raceのHassoTimeを参照して、現在時刻より後のレースのみを抽出します。

    Args:
        df: 前処理済みデータ
        predict_date: 予測対象日（YYYYMMDD形式）

    Returns:
        pd.DataFrame: 未発走レースのみのデータ
    """
    # s_raceを読み込んで発走時刻を取得
    s_race_path = Path(config.PREPROCESSING_DB_DIR) / "s_race.parquet"
    if not s_race_path.exists():
        logging.warning(f"s_race.parquetが見つかりません: {s_race_path}")
        return df

    s_race_df = pd.read_parquet(s_race_path)

    # 予測対象日のレースのみ
    year = predict_date[:4]
    monthday = predict_date[4:]
    mask = (s_race_df["Year"] == year) & (s_race_df["MonthDay"] == monthday)
    today_races = s_race_df[mask].copy()

    if today_races.empty:
        logging.warning("本日のレース情報がs_raceにありません")
        return df

    # HassoTimeをdatetime型に変換
    today_races["HassoTime_dt"] = pd.to_datetime(
        today_races["Year"]
        + today_races["MonthDay"]
        + today_races["HassoTime"].str.zfill(4),
        format="%Y%m%d%H%M",
        errors="coerce",
    )

    # 現在時刻より後のレースのみを抽出
    now = datetime.now()
    unstarted = today_races[today_races["HassoTime_dt"] > now]

    if unstarted.empty:
        logging.warning("未発走レースがありません")
        return pd.DataFrame()

    # 未発走レースのキー（JyoCD, RaceNum）を取得
    unstarted_keys = set(zip(unstarted["JyoCD"], unstarted["RaceNum"]))
    logging.info(f"未発走レース数: {len(unstarted_keys)}")

    # dfから未発走レースのみをフィルタリング
    if "JyoCD" in df.columns and "RaceNum" in df.columns:
        df_filtered = df[
            df.apply(
                lambda row: (row["JyoCD"], row["RaceNum"]) in unstarted_keys, axis=1
            )
        ]
        logging.info(
            f"フィルタリング前: {len(df)} 件 → フィルタリング後: {len(df_filtered)} 件"
        )
        return df_filtered
    else:
        logging.warning("JyoCDまたはRaceNum列がデータにありません")
        return df


def setup_logging(verbose: bool = False) -> None:
    """
    ロギング設定

    Args:
        verbose: 詳細ログを出力するか
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_settings() -> SimpleNamespace:
    """
    .envから設定を読み込み、SimpleNamespaceとして返す

    WIN5は必須（設定に関わらず常に有効化）

    Returns:
        SimpleNamespace: 設定オブジェクト（argsの代替）
    """
    # ポリシー設定ファイルからポリシーリストを取得
    policy_list = []
    if config.PREDICT_USE_POLICY_CONFIG:
        try:
            with open(config.PREDICT_POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
                policy_config = json.load(f)
                policy_list = policy_config.get("enabled_policies", [])
        except Exception:
            pass

    return SimpleNamespace(
        # 予測モード
        mode=config.PREDICT_MODE,
        # 予測日付（mode=date時のみ使用）
        date=config.PREDICT_TARGET_DATE if config.PREDICT_TARGET_DATE else None,
        # タスク選択
        tasks=",".join(config.PREDICT_TASKS),
        task_list=config.PREDICT_TASKS,
        # モデルディレクトリ
        models_dir=config.PREDICT_MODELS_DIR,
        # 前処理済みファイル
        preprocessed_file=None,
        preprocessed_dir=config.PREDICT_PREPROCESSED_DIR,
        # 出力ディレクトリ
        output_dir=config.PREDICT_OUTPUT_DIR,
        # ポリシー
        policy_list=policy_list,
        use_policy_config=config.PREDICT_USE_POLICY_CONFIG,
        policy_config_path=config.PREDICT_POLICY_CONFIG_PATH,
        # WIN5（必須のため常にTrue）
        enable_win5=True,
        win5_threshold=config.PREDICT_WIN5_THRESHOLD,
        # WIN5BasedポリシーもWIN5と同じ閾値を使用
        win5base_threshold=config.PREDICT_WIN5_THRESHOLD,
        # 詳細出力
        verbose=config.PREDICT_VERBOSE,
        # 未発走レースフィルタリング
        filter_unstarted=config.PREDICT_FILTER_UNSTARTED,
        # WIN5対象レースのみ（必須のため常にTrue）
        win5_races_only=True,
    )


def main() -> int:
    """
    メイン関数

    以下の処理を実行します：
    1. 設定の読み込み
    2. 当日レースデータの読み込み
    3. データの前処理と整形
    4. 学習済みモデルによる予測
    5. WIN5予測（オプション）
    6. 予測結果の保存

    Returns:
        int: 終了コード（0=成功、1=エラー）
    """
    settings = _build_settings()
    setup_logging(settings.verbose)

    # モードに応じて予測日付を決定
    if settings.mode == "today":
        # 当日予測モード: 今日の日付を使用
        predict_date = datetime.now().strftime("%Y%m%d")
    else:  # settings.mode == "date"
        # 日付指定モード: .envのPREDICT_TARGET_DATEを使用
        if settings.date:
            predict_date = settings.date
        else:
            logging.error("日付指定モード(date)では.envのPREDICT_TARGET_DATEが必須です")
            return 1

    print("=" * 60)
    print("競馬予測AI - 当日レース予測")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"予測モード: {settings.mode}")
    print(f"予測対象日: {predict_date}")
    print(f"予測タスク: {', '.join(settings.task_list)}")
    print(f"モデルディレクトリ: {settings.models_dir}")
    print(f"出力先: {settings.output_dir}")
    if settings.policy_list:
        print(f"ポリシー: {', '.join(settings.policy_list)}")
    if settings.enable_win5:
        print("WIN5予測: 有効")
    if settings.filter_unstarted:
        print("未発走レースフィルタ: 有効")
    print("=" * 60)
    print()

    try:
        # ステップ0: WIN5プロセッサの初期化（必須）
        logging.info("=" * 60)
        logging.info("WIN5対象レースの特定（必須モード）")
        logging.info("=" * 60)

        win5_processor = Win5Processor()
        # 当日データを読み込み
        win5_processor.load_today(predict_date)
        # 過去データもロード（フォールバック）
        win5_processor.load()

        date_key = predict_date
        win5_races = win5_processor.get_win5_race_keys(date_key)
        if not win5_races:
            logging.error(f"WIN5対象レースが見つかりません: {predict_date}")
            print(f"エラー: {predict_date} のWIN5対象レースが見つかりません")
            return 1
        logging.info(f"WIN5対象レース: {len(win5_races)} レース")

        # ステップ1: 前処理済みデータの読み込み
        logging.info("=" * 60)
        logging.info("ステップ1: 前処理済みデータの読み込み")
        logging.info("=" * 60)

        preprocessed = load_preprocessed_data(
            settings.preprocessed_file,
            predict_date,
        )
        logging.info(f"データ読み込み完了: {preprocessed.shape}")

        # 未発走レースフィルタリング
        if settings.filter_unstarted:
            logging.info("\n" + "=" * 60)
            logging.info("未発走レースフィルタリング")
            logging.info("=" * 60)

            preprocessed = filter_unstarted_races(preprocessed, predict_date)
            logging.info(f"フィルタ後: {preprocessed.shape}")

            if preprocessed.empty:
                logging.warning("未発走レースがありません")
                return 0

        # WIN5対象レースのみにフィルタリング（必須）
        if settings.win5_races_only:
            logging.info("\n" + "=" * 60)
            logging.info("WIN5対象レースのみにフィルタリング")
            logging.info("=" * 60)

            original_count = len(preprocessed)
            preprocessed = win5_processor.filter_win5_races(preprocessed, date_key)
            filtered_count = len(preprocessed)
            logging.info(
                f"フィルタ: {original_count} → {filtered_count} 行（WIN5対象のみ）"
            )

            if preprocessed.empty:
                logging.warning("WIN5対象レースがありません")
                return 0

        # ステップ2: 学習フォーマットへの変換
        logging.info("\n" + "=" * 60)
        logging.info("ステップ2: 学習フォーマットへの変換")
        logging.info("=" * 60)

        aligned, missing_cols = align_to_training_format(preprocessed)
        logging.info(f"変換完了: {aligned.shape}")

        if missing_cols and settings.verbose:
            logging.debug(f"欠損列: {missing_cols}")

        # ステップ3: 予測実行
        logging.info("\n" + "=" * 60)
        logging.info("ステップ3: 予測実行")
        logging.info("=" * 60)

        predictions = predict_with_models(
            aligned, settings.models_dir, settings.task_list
        )

        if not predictions:
            logging.error("予測に失敗しました（全タスク）")
            return 1

        # ステップ4: 予測結果の保存
        logging.info("\n" + "=" * 60)
        logging.info("ステップ4: 予測結果の保存")
        logging.info("=" * 60)

        save_predictions(predictions, settings.output_dir, predict_date)

        # ステップ5: WIN5予測（ポリシー推奨生成の前に実行）
        win5_result = {}
        if settings.enable_win5 and "win" in predictions:
            logging.info("\n" + "=" * 60)
            logging.info("ステップ5: WIN5予測の生成")
            logging.info("=" * 60)
            logging.info(f"WIN5閾値: {settings.win5_threshold}")

            # Win5Processorからレース情報を取得
            win5_race_keys_list = None
            if date_key in win5_processor.schedule:
                win5_race_keys_list = win5_processor.schedule[date_key]
                logging.info(f"  WIN5対象レース情報: {len(win5_race_keys_list)} レース")

            win5_result = generate_win5_predictions(
                predictions["win"],
                threshold=settings.win5_threshold,
                win5_race_keys=win5_race_keys_list,
            )

            if win5_result:
                # WIN5推奨を保存
                output_dir = Path(settings.output_dir) / predict_date

                win5_df = format_win5_recommendations(win5_result)
                output_file = output_dir / "win5_recommendations.csv"
                output_dir.mkdir(parents=True, exist_ok=True)
                win5_df.to_csv(output_file, index=False, encoding="utf-8-sig")
                logging.info(f"  WIN5推奨: {output_file}")

                save_win5_report(win5_result, output_dir)

        # ステップ6: ポリシーによる推奨馬券生成（各タスクごと）
        # WIN5Basedポリシーは上記WIN5予測で選択された馬を使用
        recommendations = {}  # {task: {policy: df}}
        if settings.policy_list:
            logging.info("\n" + "=" * 60)
            logging.info("ステップ6: ポリシー推奨馬券の生成（各タスクごと）")
            logging.info("=" * 60)

            for task in settings.task_list:
                if task not in predictions:
                    logging.warning(
                        f"タスク '{task}' の予測結果がありません。スキップします。"
                    )
                    continue

                logging.info(f"\n--- タスク: {task} ---")
                task_recommendations = generate_policy_recommendations(
                    predictions,
                    settings.policy_list,
                    primary_task=task,
                    win5_race_keys=win5_races,  # WIN5Basedポリシー用にWIN5対象レースを渡す
                    win5_result=win5_result,  # WIN5予測結果（選択馬）を渡す
                    win5base_threshold=settings.win5base_threshold,  # WIN5Basedポリシー用の閾値
                )
                recommendations[task] = task_recommendations

                # ポリシー推奨を保存（タスク別ディレクトリ）
                output_dir = (
                    Path(settings.output_dir)
                    / predict_date
                    / task
                    / "policy_recommendations"
                )

                for policy_name, rec_df in task_recommendations.items():
                    save_policy_report(policy_name, rec_df, output_dir)

        # 結果サマリー表示
        print_prediction_summary(
            predictions, recommendations, win5_result, settings.output_dir
        )

        print(f"\n終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except FileNotFoundError as e:
        logging.error(f"ファイルが見つかりません: {e}")
        return 1
    except ValueError as e:
        logging.error(f"データエラー: {e}")
        return 1
    except Exception:
        logging.exception("予測処理中に予期しないエラーが発生しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())

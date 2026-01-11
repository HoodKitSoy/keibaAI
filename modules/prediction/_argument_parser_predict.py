"""
引数パーサーモジュール（予測用）

predict.pyのコマンドライン引数を解析します。
.envファイルの設定をデフォルト値として使用し、コマンドライン引数で上書き可能です。
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import config


def parse_predict_arguments() -> argparse.Namespace:
    """
    コマンドライン引数をパース

    Returns:
        argparse.Namespace: パースされた引数

    Note:
        - .envファイルの設定をデフォルト値として使用
        - コマンドライン引数で指定した場合はそちらを優先
        - WIN5予測はデフォルトで有効（.env: PREDICT_WIN5_ENABLED）
        - モード指定がない場合は.envのPREDICT_MODEを使用
    """
    parser = argparse.ArgumentParser(
        description="当日レース予測スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # =====================================
    # 予測モード
    # =====================================
    parser.add_argument(
        "--mode",
        type=str,
        choices=["today", "date"],
        default=config.PREDICT_MODE,
        help="予測モード: today=当日予測、date=日付指定予測（.env: PREDICT_MODE）",
    )

    # =====================================
    # 予測日付
    # =====================================
    parser.add_argument(
        "--date",
        type=str,
        default=config.PREDICT_TARGET_DATE if config.PREDICT_TARGET_DATE else None,
        help="予測対象日 (YYYYMMDD形式)。mode=dateで未指定の場合は.envのPREDICT_TARGET_DATEを使用、mode=todayでは今日の日付",
    )

    # =====================================
    # タスク選択
    # =====================================
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(config.PREDICT_TASKS),
        help="予測タスク（カンマ区切り: win,top3,rank）（.env: PREDICT_TASKS）",
    )

    # =====================================
    # モデルディレクトリ
    # =====================================
    parser.add_argument(
        "--models-dir",
        type=str,
        default=config.PREDICT_MODELS_DIR,
        help="学習済みモデルが格納されたディレクトリ（.env: PREDICT_MODELS_DIR）",
    )

    # =====================================
    # 前処理済みファイル
    # =====================================
    parser.add_argument(
        "--preprocessed-file",
        type=str,
        default=None,
        help="前処理済みファイルのパス。未指定時は ./data/tmp/today_preprocessed_{date}.parquet を使用",
    )

    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=config.PREDICT_PREPROCESSED_DIR,
        help="前処理済みファイルのディレクトリ（.env: PREDICT_PREPROCESSED_DIR）",
    )

    # =====================================
    # 出力ディレクトリ
    # =====================================
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.PREDICT_OUTPUT_DIR,
        help="予測結果の出力先ディレクトリ（.env: PREDICT_OUTPUT_DIR）",
    )

    # =====================================
    # ポリシー（賭け戦略）
    # =====================================
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="使用するポリシー（カンマ区切り）。ポリシー設定ファイル使用時は無視",
    )

    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="単一ポリシー指定（後方互換用）",
    )

    parser.add_argument(
        "--no-policies",
        action="store_true",
        help="ポリシー推奨を無効化する",
    )

    # ポリシー設定ファイルの使用
    parser.add_argument(
        "--use-policy-config",
        action="store_true",
        default=config.PREDICT_USE_POLICY_CONFIG,
        help="policy_config.jsonの設定に従ってポリシーを選択（.env: PREDICT_USE_POLICY_CONFIG=true）",
    )

    parser.add_argument(
        "--no-policy-config",
        action="store_true",
        help="ポリシー設定ファイルを使用しない（--policiesを使用）",
    )

    parser.add_argument(
        "--policy-config-path",
        type=str,
        default=config.PREDICT_POLICY_CONFIG_PATH,
        help="ポリシー設定ファイルのパス（.env: PREDICT_POLICY_CONFIG_PATH）",
    )

    # =====================================
    # WIN5予測
    # =====================================
    parser.add_argument(
        "--win5",
        action="store_true",
        default=config.PREDICT_WIN5_ENABLED,
        help="WIN5の予測を実行する（.env: PREDICT_WIN5_ENABLED）",
    )

    parser.add_argument(
        "--no-win5",
        action="store_true",
        help="WIN5予測を無効化する",
    )

    parser.add_argument(
        "--win5-threshold",
        type=float,
        default=config.PREDICT_WIN5_THRESHOLD,
        help="WIN5で候補とする最低予測確率（固定値）（.env: PREDICT_WIN5_THRESHOLD）",
    )

    # =====================================
    # 詳細出力
    # =====================================
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細なログを出力",
    )

    # =====================================
    # 未発走レースフィルタリング
    # =====================================
    parser.add_argument(
        "--filter-unstarted",
        action="store_true",
        help="未発走レースのみを予測対象とする（s_raceのHassoTimeでフィルタリング）",
    )

    args = parser.parse_args()

    # =====================================
    # 引数の後処理
    # =====================================

    # 日付の処理
    if args.date is None:
        today = date.today()
        args.date = today.strftime("%Y%m%d")

    # タスクの処理
    if args.tasks:
        args.task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        args.task_list = ["win", "top3", "rank"]

    # ポリシーの処理
    # --no-policy-configが指定されたら無効化
    if hasattr(args, "no_policy_config") and args.no_policy_config:
        args.use_policy_config = False

    if args.no_policies:
        args.policy_list = []
    elif args.use_policy_config:
        # ポリシーセレクターを使用
        from pathlib import Path as PathLib

        from modules.policies import PolicySelector

        config_path = (
            PathLib(args.policy_config_path) if args.policy_config_path else None
        )
        selector = PolicySelector(config_path)
        enabled_policies = selector.get_enabled_policies()
        args.policy_list = list(enabled_policies.keys())
        print(f"policy_config.json から {len(args.policy_list)} 個のポリシーを有効化")
    elif args.policy:
        args.policy_list = [args.policy]
    elif args.policies:
        args.policy_list = [p.strip() for p in args.policies.split(",") if p.strip()]
    else:
        # ポリシー設定がない場合はすべてのポリシーを使用
        from modules.policies import get_all_policy_classes

        args.policy_list = list(get_all_policy_classes().keys())

    # WIN5フラグの統合（--no-win5が指定されたら無効化）
    if args.no_win5:
        args.enable_win5 = False
    else:
        args.enable_win5 = args.win5

    # パスをPathオブジェクトに変換
    args.models_dir = Path(args.models_dir)
    args.preprocessed_dir = Path(args.preprocessed_dir)
    args.output_dir = Path(args.output_dir)

    # 未発走フィルタリングフラグ
    args.filter_unstarted = getattr(args, "filter_unstarted", False)

    return args

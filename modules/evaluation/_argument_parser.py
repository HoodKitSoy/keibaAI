"""
コマンドライン引数解析モジュール

競馬予測モデル評価スクリプトのコマンドライン引数を解析します。
.envファイルの設定をデフォルト値として使用し、コマンドライン引数で上書き可能です。
"""

import argparse

import config


def parse_arguments():
    """
    コマンドライン引数を解析

    Returns:
        argparse.Namespace: 解析された引数

    Note:
        - .envファイルの設定をデフォルト値として使用
        - コマンドライン引数で指定した場合はそちらを優先
        - WIN5シミュレーションはデフォルトで有効（.env: TEST_WIN5_ENABLED）
    """
    parser = argparse.ArgumentParser(
        description="競馬予測モデルの評価・シミュレーションスクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # =====================================
    # モデル/予測関連の引数
    # =====================================
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="(旧) dill pickle モデルのパス。未指定時は --preds-dir を優先",
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        default=config.TEST_PREDS_DIR,
        help="予測ファイルディレクトリ（.env: TEST_PREDS_DIR）",
    )

    # =====================================
    # タスク選択
    # =====================================
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(config.TEST_TASKS),
        help="評価タスク（カンマ区切り: win,top3,rank）（.env: TEST_TASKS）",
    )

    # =====================================
    # データ関連の引数
    # =====================================
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="",
        help="テストデータのパス（指定がない場合はモデルの内部テストデータを使用）",
    )

    # =====================================
    # 評価関連の引数
    # =====================================
    parser.add_argument(
        "--rank-threshold",
        type=int,
        default=1,
        help="二値分類時の順位閾値（1=単勝、3=複勝など）",
    )

    # =====================================
    # 出力関連の引数
    # =====================================
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.TEST_OUTPUT_DIR,
        help="結果を保存するディレクトリ（.env: TEST_OUTPUT_DIR）",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="結果をグラフで可視化する",
    )

    # =====================================
    # ポリシー（賭け戦略）関連
    # =====================================
    parser.add_argument(
        "--tiered-coverage-threshold",
        type=float,
        default=0.5,
        help="TieredCoverageBetPolicyの累積的中率閾値",
    )

    # レース選別戦略のパラメータ
    parser.add_argument(
        "--min-top-score",
        type=float,
        default=0.35,
        help="レース選別: トップ馬の最低スコア（勝率）",
    )

    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=0.10,
        help="レース選別: 1位と2位のスコア差の最低値",
    )

    parser.add_argument(
        "--use-or-condition",
        action="store_true",
        help="レース選別: 条件を「または」で結合（デフォルト: 「かつ」）",
    )

    # 使用するポリシーの選択（ポリシー設定ファイル未使用時のみ）
    parser.add_argument(
        "--policy",
        type=str,
        default="all",
        help="実行するポリシー（all=すべて、ポリシー設定ファイル使用時は無視）",
    )

    # ポリシー設定ファイルの使用
    parser.add_argument(
        "--use-policy-config",
        action="store_true",
        default=config.TEST_USE_POLICY_CONFIG,
        help="policy_config.jsonの設定に従ってポリシーを選択（.env: TEST_USE_POLICY_CONFIG=true）",
    )

    parser.add_argument(
        "--no-policy-config",
        action="store_true",
        help="ポリシー設定ファイルを使用しない（--policyを使用）",
    )

    parser.add_argument(
        "--policy-config-path",
        type=str,
        default=config.TEST_POLICY_CONFIG_PATH,
        help="ポリシー設定ファイルのパス（.env: TEST_POLICY_CONFIG_PATH）",
    )

    # グラフ表示有無
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="グラフを画面に表示する（デフォルト: 表示せず、保存のみ）",
    )

    # =====================================
    # WIN5（重勝式）関連
    # =====================================
    parser.add_argument(
        "--win5",
        action="store_true",
        default=config.TEST_WIN5_ENABLED,
        help="WIN5シミュレーションを実行する（.env: TEST_WIN5_ENABLED）",
    )
    parser.add_argument(
        "--no-win5",
        action="store_true",
        help="WIN5シミュレーションを無効化する",
    )
    parser.add_argument(
        "--win5-amount",
        type=float,
        default=config.TEST_WIN5_AMOUNT,
        help="WIN5の1点あたり金額（円）（.env: TEST_WIN5_AMOUNT）",
    )
    parser.add_argument(
        "--win5-threshold-min",
        type=float,
        default=config.TEST_WIN5_THRESHOLD_MIN,
        help="WIN5閾値範囲の最小値（.env: TEST_WIN5_THRESHOLD_MIN）",
    )
    parser.add_argument(
        "--win5-threshold-max",
        type=float,
        default=config.TEST_WIN5_THRESHOLD_MAX,
        help="WIN5閾値範囲の最大値（.env: TEST_WIN5_THRESHOLD_MAX）",
    )
    parser.add_argument(
        "--win5-threshold-step",
        type=float,
        default=config.TEST_WIN5_THRESHOLD_STEP,
        help="WIN5閾値のステップ幅（.env: TEST_WIN5_THRESHOLD_STEP）",
    )

    args = parser.parse_args()

    # =====================================
    # 引数の後処理
    # =====================================

    # タスクリストの処理
    if args.tasks:
        args.task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        args.task_list = ["win"]  # デフォルトはwinのみ

    # WIN5フラグの統合（--no-win5が指定されたら無効化）
    if args.no_win5:
        args.win5 = False

    # ポリシー設定フラグの統合（--no-policy-configが指定されたら無効化）
    if hasattr(args, "no_policy_config") and args.no_policy_config:
        args.use_policy_config = False

    return args

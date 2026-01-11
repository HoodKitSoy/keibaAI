"""
出力管理モジュール

評価結果の出力ディレクトリを管理します。
"""

import os
import re
from datetime import datetime

import config


def get_output_dir(args):
    """
    出力ディレクトリのパスを取得または作成

    Args:
        args: コマンドライン引数

    Returns:
        str: 出力ディレクトリパス

    Note:
        - 常にmodels/test/YYYYMMDD/の構造で出力
        - --output-dirが指定された場合はベースディレクトリとして使用
    """
    # ベースディレクトリを決定
    base_dir = args.output_dir if args.output_dir else config.MODEL_SAVE_DIR

    # 常に test/{date}/ 構造を追加
    output_dir = os.path.join(
        base_dir,
        "test",
        datetime.now().strftime("%Y%m%d"),
    )

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def sanitize_name_for_path(name: str) -> str:
    """
    パスに使用可能な文字列にサニタイズ

    Args:
        name (str): 元の名前

    Returns:
        str: サニタイズされた名前
    """
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", str(name))
    sanitized = sanitized.strip("_")
    return sanitized or "strategy"


def get_strategy_output_dir(args, strategy_name: str, task: str = "") -> str:
    """
    戦略別の出力ディレクトリを取得

    Args:
        args: コマンドライン引数
        strategy_name (str): 戦略名
        task (str): タスク名（win/top3/rank）。空の場合はタスクディレクトリなし

    Returns:
        str: 戦略別の出力ディレクトリパス
    """
    base_dir = get_output_dir(args)
    if task:
        strategy_dir = os.path.join(
            base_dir, "strategies", task, sanitize_name_for_path(strategy_name)
        )
    else:
        strategy_dir = os.path.join(
            base_dir, "strategies", sanitize_name_for_path(strategy_name)
        )
    os.makedirs(strategy_dir, exist_ok=True)
    return strategy_dir


def get_comparison_output_dir(args) -> str:
    """
    比較結果の出力ディレクトリを取得

    Args:
        args: コマンドライン引数

    Returns:
        str: 比較結果の出力ディレクトリパス
    """
    base_dir = get_output_dir(args)
    comparison_dir = os.path.join(base_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    return comparison_dir

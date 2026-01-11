"""
競馬予測モデル評価スクリプト

複数の馬券戦略を評価・比較し、パフォーマンスを分析します。

使用方法:
    # .envファイルで設定を行い、実行
    python test.py

設定（.env）:
    TEST_TASKS: 評価タスク（win,top3,rank）
    TEST_PREDS_DIR: 予測ファイルディレクトリ
    TEST_WIN5_ENABLED: WIN5シミュレーション有効化
    TEST_USE_POLICY_CONFIG: ポリシー設定ファイル使用
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List

import config
from modules.evaluation import (
    build_win5_thresholds,
    evaluate_strategies,
    get_comparison_output_dir,
    get_output_dir,
    load_model_and_data,
    prepare_strategies,
    run_win5_simulation,
    save_comparison_by_bet_type,
    save_comparison_results,
    save_win5_results,
)
from modules.preprocessing._win5_processor import Win5Processor


def _build_settings() -> SimpleNamespace:
    """
    .envから設定を読み込み、SimpleNamespaceとして返す

    WIN5は必須（設定に関わらず常に有効化）

    Returns:
        SimpleNamespace: 設定オブジェクト（argsの代替）
    """
    settings = SimpleNamespace(
        # モデル/予測関連
        model_path="",
        preds_dir=config.TEST_PREDS_DIR,
        # タスク選択
        tasks=",".join(config.TEST_TASKS),
        task_list=config.TEST_TASKS,
        # データ関連
        test_data_path="",
        # 評価関連
        rank_threshold=config.TEST_RANK_THRESHOLD,
        # 出力関連
        output_dir=config.TEST_OUTPUT_DIR,
        plot=False,
        show_plot=config.TEST_SHOW_PLOT,
        # ポリシー関連
        tiered_coverage_threshold=config.TEST_TIERED_COVERAGE_THRESHOLD,
        min_top_score=config.TEST_MIN_TOP_SCORE,
        min_score_gap=config.TEST_MIN_SCORE_GAP,
        use_or_condition=config.TEST_USE_OR_CONDITION,
        policy=config.TEST_POLICY,
        use_policy_config=config.TEST_USE_POLICY_CONFIG,
        policy_config_path=config.TEST_POLICY_CONFIG_PATH,
        # WIN5関連（必須のため常にTrue）
        win5=True,  # WIN5は必須
        win5_amount=config.TEST_WIN5_AMOUNT,
        win5_threshold_min=config.TEST_WIN5_THRESHOLD_MIN,
        win5_threshold_max=config.TEST_WIN5_THRESHOLD_MAX,
        win5_threshold_step=config.TEST_WIN5_THRESHOLD_STEP,
        # WIN5対象レースのみフィルタ（必須のため常にTrue）
        win5_races_only=True,
    )
    return settings


def main() -> int:
    """
    メイン関数

    以下の処理を実行します：
    1. 設定の読み込み
    2. モデルと予測データの読み込み（複数タスク対応）
    3. 各タスクに対して馬券戦略の準備と実行
    4. 戦略のパフォーマンス評価
    5. WIN5シミュレーション（オプション）
    6. 結果の保存と可視化

    Returns:
        int: 終了コード（0=成功、1=エラー）

    Note:
        - preds_*.parquet使用時はtrain.pyの成果物を直接利用
        - pickle使用時は旧形式のモデルを読み込んで評価
        - 全ての結果はmodels/test/YYYYMMDD/に保存
        - 複数タスク（win, top3, rank）に対応
    """
    # 設定の読み込み
    settings = _build_settings()

    print("===== 競馬予測モデルポリシー比較 =====")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"評価タスク: {', '.join(settings.task_list)}")
    print(f"出力先: {get_output_dir(settings)}")

    try:
        # WIN5プロセッサを初期化
        win5_processor = Win5Processor()
        win5_processor.load()
        if win5_processor.loaded:
            print(f"WIN5データ読み込み完了: {len(win5_processor.get_dates())}日分")
            print("WIN5対象レースのみを評価（WIN5必須モード）")
        else:
            print("警告: WIN5データなし - WIN5は必須のため処理を中断します")
            return 1

        # モデルとデータの読み込み（複数タスク対応）
        keiba_ai, test_data, return_processor, score_tables = load_model_and_data(
            settings
        )

        # WIN5対象レースのみにフィルタリング
        if settings.win5_races_only:
            print("\nWIN5対象レースのみにフィルタリング中...")
            filtered_score_tables = {}
            total_races_before = 0
            total_races_after = 0

            for task, df in score_tables.items():
                original_count = len(df)
                total_races_before += original_count

                # 各日付ごとにフィルタリング
                filtered_dfs = []
                if "Year" in df.columns and "MonthDay" in df.columns:
                    for (year, monthday), group_df in df.groupby(["Year", "MonthDay"]):
                        date_key = f"{str(year).zfill(4)}{str(monthday).zfill(4)}"
                        filtered = win5_processor.filter_win5_races(group_df, date_key)
                        if len(filtered) > 0:
                            filtered_dfs.append(filtered)

                if filtered_dfs:
                    import pandas as pd

                    filtered_score_tables[task] = pd.concat(
                        filtered_dfs, ignore_index=True
                    )
                else:
                    filtered_score_tables[task] = df.head(0)  # 空のDataFrame

                filtered_count = len(filtered_score_tables[task])
                total_races_after += filtered_count
                print(f"  {task}: {original_count} → {filtered_count} 行")

            score_tables = filtered_score_tables
            print(
                f"  合計: {total_races_before} → {total_races_after} 行（WIN5対象のみ）"
            )

        # 各タスクに対して評価を実行
        all_comparisons = {}
        for task in settings.task_list:
            if task not in score_tables:
                print(
                    f"\n警告: タスク '{task}' の予測結果が見つかりません。スキップします。"
                )
                continue

            print(f"\n{'=' * 60}")
            print(f"タスク: {task}")
            print("=" * 60)

            score_table_for_task = score_tables[task]
            output_dir = Path(get_comparison_output_dir(settings)) / task

            # === WIN5シミュレーション（先に実行）===
            # 5レースの1着馬組み合わせ予想（閾値スイープ）
            win5_results: List[dict] = []
            if win5_processor.loaded:
                print(f"\n--- WIN5シミュレーション (タスク: {task}) ---")
                thresholds = build_win5_thresholds(settings)
                for th in thresholds:
                    try:
                        result = run_win5_simulation(
                            settings,
                            score_table_for_task,
                            threshold=th,
                            win5_processor=win5_processor,
                        )
                        if result.get("executed"):
                            win5_results.append(result)
                    except Exception as e:
                        print(
                            f"[WARN] WIN5 シミュレーションで例外 (threshold={th}): {e}"
                        )

                # WIN5結果をタスク別ディレクトリに保存
                if win5_results:
                    save_win5_results(win5_results, output_dir, task)
                    print(
                        f"WIN5シミュレーション完了 (タスク: {task}): {len(win5_results)}件"
                    )
                    print(f"WIN5結果保存先: {output_dir / 'win5'}")

            # === 馬券戦略の評価（WIN5シミュレーションの後）===
            # WIN5ベースポリシーを含む全ポリシーを閾値スイープで評価
            # WIN5Basedポリシーは、WIN5シミュレーションと同じ閾値で選択された馬を使用
            strategies, actions, score_table, odds_table = prepare_strategies(
                keiba_ai,
                test_data,
                settings,
                score_table_for_task,
                win5_processor=win5_processor,
            )

            # 戦略の評価（詳細ファイル出力付き、タスク別ディレクトリ）
            evaluation_results, comparison, bet_type_stats = evaluate_strategies(
                actions, return_processor, settings, odds_table, strategies, task
            )

            # タスク別の結果を保存
            all_comparisons[task] = comparison

            # 結果の保存（タスク別ディレクトリ）
            save_comparison_results(comparison, output_dir)

            # 券種別回収率を保存
            if bet_type_stats:
                save_comparison_by_bet_type(bet_type_stats, output_dir, task)

        # 全タスクのサマリーを保存
        if all_comparisons:
            output_dir = Path(get_comparison_output_dir(settings))
            summary_path = output_dir / "all_tasks_summary.json"
            summary = {
                "evaluated_tasks": list(all_comparisons.keys()),
                "results": all_comparisons,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n全タスクサマリーを保存: {summary_path}")

        print("\n===== ポリシー比較が正常に完了しました =====")
        print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"結果保存先: {get_output_dir(settings)}")
        return 0

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

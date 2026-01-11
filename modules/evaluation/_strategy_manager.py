"""
戦略管理モジュール

馬券戦略の準備と実行を行います。
"""

import inspect
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from modules import policies
from modules.policies import PolicySelector
from modules.policies._utils import build_race_id_series


def generate_win5_selections_for_threshold(
    score_table: pd.DataFrame,
    threshold: float,
    win5_processor=None,
) -> Dict[str, List[int]]:
    """
    指定閾値でWIN5選択馬を生成

    Args:
        score_table: スコアテーブル（proba列を含む）
        threshold: 選択閾値（この値以上の確率の馬を候補にする）
        win5_processor: WIN5データプロセッサ（Noneの場合は空辞書を返す）

    Returns:
        Dict[str, List[int]]: {race_id: [馬番リスト]}
    """
    if win5_processor is None or not win5_processor.loaded:
        return {}

    selections = {}

    # スコアテーブルにrace_idを生成
    if "race_id" not in score_table.columns:
        score_table = score_table.copy()
        score_table["race_id"] = build_race_id_series(score_table)

    # 日付キーを生成
    score_table = score_table.copy()
    score_table["date_key"] = score_table["Year"].astype(str).str.zfill(
        4
    ) + score_table["MonthDay"].astype(str).str.zfill(4)

    # WIN5対象日を取得
    win5_dates = win5_processor.get_dates()
    available_dates = set(score_table["date_key"].unique()) & set(win5_dates)

    for date_key in sorted(available_dates):
        day_info = win5_processor.get_day_info(date_key)
        if day_info is None:
            continue

        races, winners, payout = day_info

        # 5レースそれぞれの候補馬を決定
        for race_key in races:
            # レースにマッチするデータを取得
            mask = (
                (score_table["Year"].astype(str).str.zfill(4) == race_key["Year"])
                & (
                    score_table["MonthDay"].astype(str).str.zfill(4)
                    == race_key["MonthDay"]
                )
                & (score_table["JyoCD"].astype(str).str.zfill(2) == race_key["JyoCD"])
                & (
                    score_table["RaceNum"].astype(str).str.zfill(2)
                    == race_key["RaceNum"]
                )
            )
            race_data = score_table[mask]

            if race_data.empty:
                continue

            # race_idを取得
            race_id = race_data.iloc[0]["race_id"]

            # 閾値以上の確率の馬を候補に
            if "proba" in race_data.columns:
                selected = race_data[race_data["proba"] >= threshold]["Umaban"]
                # 候補がいない場合はTop1を候補に
                if selected.empty:
                    selected = race_data.nlargest(1, "proba")["Umaban"]
                selections[race_id] = [int(u) for u in selected.tolist()]

    return selections


def simulate_odds_table(score_table):
    """
    スコアテーブルからオッズテーブルをシミュレーション

    Args:
        score_table (pd.DataFrame): レースごとの馬番とスコア（勝率）

    Returns:
        dict: レースごと、馬券種類ごとのオッズ情報

    Note:
        - 単勝オッズ: 勝率の逆数+ノイズ
        - 複勝オッズ: 単勝の約1/3程度
        - その他の券種は空のディクショナリ（将来拡張用）
    """
    odds_table = {}

    for race_id, df_r in score_table.groupby(level=0):
        odds_table[race_id] = {}

        # 単勝オッズ（勝率の逆数+ノイズ）
        tansho_odds = {}
        fukusho_odds = {}

        for idx, row in df_r.iterrows():
            umaban = row["Umaban"]
            win_rate = row["score"]

            # 勝率の逆数をベースにオッズを計算
            base_odds = 1.0 / (win_rate + 0.01)  # ゼロ除算回避

            # ノイズを加えてより現実的にする
            noise = np.random.normal(0, 0.2)
            tansho_odds[umaban] = min(99.9, max(1.1, base_odds * (1 + noise)))

            # 複勝オッズは単勝の約1/3程度
            fukusho_odds[umaban] = min(99.9, max(1.0, tansho_odds[umaban] / 3))

        odds_table[race_id]["tansho"] = tansho_odds
        odds_table[race_id]["fukusho"] = fukusho_odds

        # 他の馬券種類は今回は使用しないため、空のディクショナリを設定
        for bet_type in ["umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]:
            odds_table[race_id][bet_type] = {}

    return odds_table


def prepare_strategies(
    keiba_ai, test_data, args, score_table_from_preds=None, win5_processor=None
):
    """
    馬券戦略の準備と実行

    Args:
        keiba_ai: KeibaAIモデル（pickle使用時のみ）
        test_data: テストデータ（pickle使用時のみ）
        args: コマンドライン引数
        score_table_from_preds (pd.DataFrame): 予測スコアテーブル（preds使用時）
        win5_processor: WIN5データプロセッサ（WIN5Basedポリシーで使用）

    Returns:
        tuple: (strategies, actions, score_table, odds_table)
            - strategies (dict): 戦略インスタンスの辞書
            - actions (dict): 戦略ごとのアクションの辞書
            - score_table (pd.DataFrame): スコアテーブル
            - odds_table (dict): オッズテーブル

    Note:
        - preds使用時は直接スコアテーブルを使用
        - pickle使用時はモデルでスコアを計算
        - オッズはスコアから逆算してシミュレート
        - WIN5Basedポリシーは、指定閾値で選択された馬を使用
    """
    print("\n馬券戦略の準備を開始します...")

    # スコアテーブルの準備
    if score_table_from_preds is not None:
        print("train.py の preds_* からスコアテーブルを使用します")
        score_table = score_table_from_preds
    else:
        # 互換: pickle モデルでスコア計算
        score_policy = policies.RelativeProbaScorePolicy()
        X_for_calc = test_data.select_dtypes(exclude=["datetime64"]).copy()
        datetime_cols = test_data.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_cols:
            print(f"Datetime列を除外: {datetime_cols}")
        # 学習時に使用された特徴量のみを抽出
        if hasattr(keiba_ai, "datasets") and hasattr(keiba_ai.datasets, "X_train"):
            training_columns = keiba_ai.datasets.X_train.columns.tolist()
            common_cols = [col for col in training_columns if col in X_for_calc.columns]
            X_for_calc = X_for_calc[common_cols]
        # スコアテーブル計算
        score_table = keiba_ai.calc_score(X_for_calc, score_policy)

    # オッズテーブルのシミュレーション（実際のオッズデータがない場合）
    # 勝率の逆数をオッズとして使用（実際のオッズに近い仮想データ）
    odds_table = simulate_odds_table(score_table)

    # 戦略の準備
    strategies = {}
    actions = {}

    # 実行するポリシーのリストを決定
    selected_policy = args.policy

    # すべての利用可能なポリシーとその設定をマッピング
    policy_settings = {
        "BetPolicyCoverage25": {
            "policy_class": policies.BetPolicyCoverage25,
            "requires_odds": False,
            "init_params": {},
            "judge_params": {},
        },
        "BetPolicyCoverage50": {
            "policy_class": policies.BetPolicyCoverage50,
            "requires_odds": False,
            "init_params": {},
            "judge_params": {},
        },
        "BetPolicyCoverage75": {
            "policy_class": policies.BetPolicyCoverage75,
            "requires_odds": False,
            "init_params": {},
            "judge_params": {},
        },
        "HybridBetPolicy": {
            "policy_class": policies.HybridBetPolicy,
            "requires_odds": True,
            "init_params": {},
            "judge_params": {"odds_table": odds_table},
        },
        "ExpectedValueBetPolicy": {
            "policy_class": policies.ExpectedValueBetPolicy,
            "requires_odds": True,
            "init_params": {},
            "judge_params": {"odds_table": odds_table},
        },
        "TieredCoverageBetPolicy": {
            "policy_class": policies.TieredCoverageBetPolicy,
            "requires_odds": False,
            "init_params": {
                "coverage_threshold": args.tiered_coverage_threshold,
                "max_selections": 5,
            },
            "judge_params": {},
        },
        "RaceSelectiveHybridPolicy": {
            "policy_class": policies.RaceSelectiveHybridPolicy,
            "requires_odds": True,
            "init_params": {
                "min_top_score": args.min_top_score,
                "min_score_gap": args.min_score_gap,
                "use_or_condition": args.use_or_condition,
            },
            "judge_params": {"odds_table": odds_table},
        },
        "RaceSelectiveCoverage25Policy": {
            "policy_class": policies.RaceSelectiveCoverage25Policy,
            "requires_odds": False,
            "init_params": {
                "min_top_score": args.min_top_score,
                "min_score_gap": args.min_score_gap,
                "use_or_condition": args.use_or_condition,
            },
            "judge_params": {},
        },
        "RaceSelectiveCoverage50Policy": {
            "policy_class": policies.RaceSelectiveCoverage50Policy,
            "requires_odds": False,
            "init_params": {
                "min_top_score": args.min_top_score,
                "min_score_gap": args.min_score_gap,
                "use_or_condition": args.use_or_condition,
            },
            "judge_params": {},
        },
        "RaceSelectiveCoverage75Policy": {
            "policy_class": policies.RaceSelectiveCoverage75Policy,
            "requires_odds": False,
            "init_params": {
                "min_top_score": args.min_top_score,
                "min_score_gap": args.min_score_gap,
                "use_or_condition": args.use_or_condition,
            },
            "judge_params": {},
        },
        "RaceSelectiveExpectedValueBetPolicy": {
            "policy_class": policies.RaceSelectiveExpectedValueBetPolicy,
            "requires_odds": True,
            "init_params": {
                "min_top_score": args.min_top_score,
                "min_score_gap": args.min_score_gap,
                "use_or_condition": args.use_or_condition,
            },
            "judge_params": {"odds_table": odds_table},
        },
        "RaceSelectiveTieredCoverageBetPolicy": {
            "policy_class": policies.RaceSelectiveTieredCoverageBetPolicy,
            "requires_odds": False,
            "init_params": {
                "min_top_score": args.min_top_score,
                "min_score_gap": args.min_score_gap,
                "use_or_condition": args.use_or_condition,
                "coverage_threshold": args.tiered_coverage_threshold,
                "max_selections": 5,
            },
            "judge_params": {},
        },
        # オッズベース系ポリシー
        "OddsBasedExpectedValuePolicy": {
            "policy_class": policies.OddsBasedExpectedValuePolicy,
            "requires_odds": True,
            "init_params": {},
            "judge_params": {},
        },
        "OddsValueTieredPolicy": {
            "policy_class": policies.OddsValueTieredPolicy,
            "requires_odds": True,
            "init_params": {},
            "judge_params": {},
        },
        "OddsRangeFilterPolicy": {
            "policy_class": policies.OddsRangeFilterPolicy,
            "requires_odds": True,
            "init_params": {},
            "judge_params": {"odds_table": odds_table},
        },
        "MultiOddsExpectedValuePolicy": {
            "policy_class": policies.MultiOddsExpectedValuePolicy,
            "requires_odds": True,
            "init_params": {},
            "judge_params": {"odds_table": odds_table},
        },
        # WIN5ベース系ポリシー（2種類のみ、閾値はsettingsから取得）
        "WIN5BasedMultiPolicy": {
            "policy_class": policies.WIN5BasedMultiPolicy,
            "requires_odds": False,
            "init_params": {
                "min_proba": getattr(args, "win5_threshold_min", 0.1),
            },
            "judge_params": {},
        },
        "WIN5BasedHedgePolicy": {
            "policy_class": policies.WIN5BasedHedgePolicy,
            "requires_odds": False,
            "init_params": {
                "min_proba": getattr(args, "win5_threshold_min", 0.1),
            },
            "judge_params": {},
        },
    }

    # RLポリシーを追加（利用可能な場合）- 統合ポリシー1つのみ
    if hasattr(policies, "RL_AVAILABLE") and policies.RL_AVAILABLE:
        # 統合RLポリシー（全馬券種に対応）
        policy_settings["RLBetPolicy"] = {
            "policy_class": policies.RLBetPolicy,
            "requires_odds": False,
            "init_params": {},
            "judge_params": {},
            "use_static_judge": True,  # judge_staticを使用
        }
        # WIN5ポリシー
        if hasattr(policies, "RLWIN5Policy"):
            policy_settings["RLWIN5Policy"] = {
                "policy_class": policies.RLWIN5Policy,
                "requires_odds": False,
                "init_params": {},
                "judge_params": {},
            }

    # 実行するポリシーリストを決定
    if hasattr(args, "use_policy_config") and args.use_policy_config:
        # ポリシーセレクターを使用
        config_path = Path(args.policy_config_path) if args.policy_config_path else None
        selector = PolicySelector(config_path)
        enabled_policies = selector.get_enabled_policies()
        policy_names = [
            name for name in enabled_policies.keys() if name in policy_settings
        ]
        print(f"  policy_config.json から {len(policy_names)} 個のポリシーを有効化")
    elif selected_policy == "all":
        policy_names = list(policy_settings.keys())
    else:
        policy_names = [selected_policy]

    # WIN5ベースポリシーの閾値スイープ設定を取得
    win5_threshold_min = getattr(args, "win5_threshold_min", 0.01)
    win5_threshold_max = getattr(args, "win5_threshold_max", 0.10)
    win5_threshold_step = getattr(args, "win5_threshold_step", 0.01)

    # WIN5ベースポリシーを閾値でスイープするための設定（2種類のみ）
    win5_base_policies = [
        "WIN5BasedMultiPolicy",
        "WIN5BasedHedgePolicy",
    ]

    # 閾値リストを生成
    win5_thresholds = []
    th = win5_threshold_min
    while th <= win5_threshold_max + 1e-9:
        win5_thresholds.append(round(th, 4))
        th += win5_threshold_step

    # 選択されたポリシーを実行
    for policy_name in policy_names:
        if policy_name in policy_settings:
            # WIN5ベースポリシーの場合は閾値スイープ
            if policy_name in win5_base_policies:
                print(
                    f"\n{policy_name} 戦略の閾値スイープ実行 (閾値: {win5_threshold_min}〜{win5_threshold_max})..."
                )

                for threshold in win5_thresholds:
                    sweep_policy_name = f"{policy_name}_th{threshold:.2f}"
                    print(f"  閾値 {threshold:.2f} で実行中...")

                    # ポリシーの設定を取得
                    policy_info = policy_settings[policy_name]
                    policy_class = policy_info["policy_class"]
                    judge_params = policy_info.get("judge_params", {})

                    # 閾値パラメータを設定（全てmin_probaを使用）
                    init_params = {"min_proba": threshold}

                    try:
                        # WIN5選択馬を生成（WIN5シミュレーションと同じ閾値）
                        win5_selections = generate_win5_selections_for_threshold(
                            score_table, threshold, win5_processor
                        )
                        if win5_selections:
                            print(f"    WIN5選択馬: {len(win5_selections)} レース")

                        # インスタンスメソッドとして実行
                        policy_instance = policy_class(**init_params)
                        # WIN5選択馬をjudgeに渡す
                        judge_params_with_win5 = {
                            **judge_params,
                            "win5_selections": win5_selections,
                        }
                        policy_actions = policy_instance.judge(
                            score_table, **judge_params_with_win5
                        )

                        # 結果を保存（閾値付きの名前で）
                        strategies[sweep_policy_name] = policy_instance
                        actions[sweep_policy_name] = policy_actions
                    except Exception as e:
                        print(
                            f"    エラー: {sweep_policy_name} の実行に失敗しました: {e}"
                        )
                        import traceback

                        traceback.print_exc()
            else:
                # 通常のポリシー
                print(f"\n{policy_name} 戦略の実行...")

                # ポリシーの設定を取得
                policy_info = policy_settings[policy_name]
                policy_class = policy_info["policy_class"]
                init_params = policy_info.get("init_params", {})
                judge_params = policy_info.get("judge_params", {})
                use_static_judge = policy_info.get("use_static_judge", False)

                try:
                    # ポリシーを実行
                    if use_static_judge and hasattr(policy_class, "judge_static"):
                        # RLBetPolicy等: judge_static内でインスタンス生成するため、ここでは生成しない
                        policy_actions = policy_class.judge_static(
                            score_table, **init_params, **judge_params
                        )
                        policy_instance = policy_class.__new__(
                            policy_class
                        )  # 空インスタンス（メタ情報用）
                    else:
                        # ポリシーのjudgeメソッドがstaticmethodかどうかを確認
                        judge_method = getattr(policy_class, "judge", None)
                        is_static = isinstance(
                            inspect.getattr_static(policy_class, "judge"), staticmethod
                        )

                        if is_static:
                            # 静的メソッドの場合: クラスから直接呼び出し
                            all_params = {**init_params, **judge_params}
                            policy_actions = policy_class.judge(
                                score_table, **all_params
                            )
                            policy_instance = policy_class
                        else:
                            # インスタンスメソッドの場合: インスタンスを生成してから呼び出し
                            policy_instance = policy_class(**init_params)
                            policy_actions = policy_instance.judge(
                                score_table, **judge_params
                            )

                    # 結果を保存
                    strategies[policy_name] = policy_instance
                    actions[policy_name] = policy_actions
                except Exception as e:
                    print(f"  エラー: {policy_name} の実行に失敗しました: {e}")
                    import traceback

                    traceback.print_exc()

    # 少なくとも1つのポリシーが実行されたか確認
    if not actions:
        print(
            f"警告: ポリシー '{selected_policy}' は見つかりませんでした。デフォルトでBetPolicyCoverage25を実行します。"
        )
        actions25 = policies.BetPolicyCoverage25.judge(score_table)
        strategies["BetPolicyCoverage25"] = policies.BetPolicyCoverage25
        actions["BetPolicyCoverage25"] = actions25

    return strategies, actions, score_table, odds_table

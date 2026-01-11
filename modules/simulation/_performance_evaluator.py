# modules/simulation/_performance_evaluator.py

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from ._simulator import Simulator


class PerformanceEvaluator:
    """
    モデルと馬券戦略の性能評価を行うクラス

    バックテストの結果を分析し、競馬予測モデルと馬券戦略の性能を評価します。
    回収率、的中率、最大ドローダウン、シャープレシオなどの指標を計算し、
    結果をグラフや表で可視化します。
    """

    def __init__(self, simulator: Simulator):
        """
        Parameters
        ----------
        simulator : Simulator
            馬券購入シミュレーターのインスタンス
        """
        self.simulator = simulator

    def evaluate_strategy(self, actions: Dict, strategy_name: str = "Strategy") -> Dict:
        """
        馬券戦略の評価を行い、各種評価指標を計算します

        Parameters
        ----------
        actions : Dict
            レースIDをキーとし、券種ごとの購入候補馬番リストを値とする辞書
        strategy_name : str, optional (default="Strategy")
            評価する戦略の名前（結果表示用）

        Returns
        -------
        Dict
            評価指標をキー、値をバリューとする辞書:
            - return_rate: 回収率（%）
            - hit_rate: 的中率（%）
            - profit: 総利益（円）
            - max_drawdown: 最大ドローダウン（%）
            - sharpe_ratio: シャープレシオ
            - win_loss_ratio: 勝率（勝ちレース数/全レース数）
            - strategy_name: 戦略名
            - returns_per_race: レースごとのリターン情報DataFrame
        """
        # シミュレーション実行（明細付き）
        returns_per_race_df, bet_details, official_results = (
            self.simulator.calc_returns_per_race(actions, return_details=True)
        )
        returns_per_race = returns_per_race_df.copy()
        returns_summary = self.simulator.calc_returns(actions)

        # 評価指標の計算
        return_rate = returns_summary.get("return_rate", 0) * 100

        n_bets = returns_summary.get("n_bets", 0)
        hit_rate = (
            (returns_summary.get("n_hits", 0) / n_bets * 100) if n_bets > 0 else 0
        )

        total_returns = returns_per_race["return_amount"].sum()
        total_bets = returns_per_race["bet_amount"].sum()
        profit = total_returns - total_bets

        if len(returns_per_race) > 0:
            returns_per_race["net_return"] = (
                returns_per_race["return_amount"] - returns_per_race["bet_amount"]
            )
            returns_per_race["cumulative_return"] = returns_per_race[
                "net_return"
            ].cumsum()
            returns_per_race["cumulative_max"] = returns_per_race[
                "cumulative_return"
            ].cummax()
            returns_per_race["drawdown"] = (
                returns_per_race["cumulative_max"]
                - returns_per_race["cumulative_return"]
            )

            if returns_per_race["cumulative_max"].max() > 0:
                max_drawdown = (
                    returns_per_race["drawdown"].max()
                    / returns_per_race["cumulative_max"].max()
                ) * 100
            else:
                max_drawdown = 0
        else:
            max_drawdown = 0

        if len(returns_per_race) > 1:
            returns_per_race["roi"] = returns_per_race["net_return"] / returns_per_race[
                "bet_amount"
            ].replace(0, 1)
            avg_return = returns_per_race["roi"].mean()
            std_return = returns_per_race["roi"].std()
            sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        n_races = returns_summary.get("n_races", 0)
        if "net_return" in returns_per_race.columns:
            win_races = (returns_per_race["net_return"] > 0).sum()
        else:
            win_races = 0
        win_loss_ratio = (win_races / n_races * 100) if n_races > 0 else 0

        evaluation = {
            "return_rate": return_rate,
            "hit_rate": hit_rate,
            "profit": profit,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_loss_ratio": win_loss_ratio,
            "strategy_name": strategy_name,
            "returns_per_race": returns_per_race,
            "bet_details": bet_details,
            "official_results": official_results,
        }

        return evaluation

    def compare_strategies(self, strategies_actions: Dict[str, Dict]) -> pd.DataFrame:
        """
        複数の馬券戦略を比較評価する

        Parameters
        ----------
        strategies_actions : Dict[str, Dict]
            戦略名をキー、アクション辞書を値とする辞書

        Returns
        -------
        pd.DataFrame
            各戦略の評価指標を行、評価指標を列とするDataFrame
        """
        results = []

        for strategy_name, actions in strategies_actions.items():
            evaluation = self.evaluate_strategy(actions, strategy_name)

            # 主要な指標だけを抽出
            result = {
                "Strategy": strategy_name,
                "Return Rate (%)": round(evaluation["return_rate"], 2),
                "Hit Rate (%)": round(evaluation["hit_rate"], 2),
                "Profit (JPY)": int(evaluation["profit"]),
                "Max Drawdown (%)": round(evaluation["max_drawdown"], 2),
                "Sharpe Ratio": round(evaluation["sharpe_ratio"], 3),
                "Win/Loss Ratio (%)": round(evaluation["win_loss_ratio"], 2),
                "Races": len(evaluation["returns_per_race"]),
                "Bets": evaluation["returns_per_race"]["n_bets"].sum(),
            }

            results.append(result)

        # 結果をDataFrameに変換し、回収率でソート
        comparison_df = pd.DataFrame(results)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values(
                "Return Rate (%)", ascending=False
            )

        return comparison_df

    def plot_cumulative_returns(
        self,
        evaluations: List[Dict],
        figsize=(12, 6),
        title="Cumulative Returns Comparison",
    ):
        """
        複数戦略の累積リターンをグラフで可視化する

        Parameters
        ----------
        evaluations : List[Dict]
            evaluate_strategy()の結果のリスト
        figsize : Tuple[int, int], optional (default=(12, 6))
            図のサイズ
        title : str, optional (default="Cumulative Returns Comparison")
            グラフのタイトル

        Returns
        -------
        matplotlib.figure.Figure
            作成したグラフのFigureオブジェクト
        """
        plt.figure(figsize=figsize)

        for eval_dict in evaluations:
            returns_per_race = eval_dict["returns_per_race"]
            strategy_name = eval_dict["strategy_name"]

            # 時系列順でインデックスをソート
            if isinstance(returns_per_race.index[0], str):
                # レースIDが文字列の場合（通常は日付を含む）
                returns_per_race = returns_per_race.sort_index()

            # 累積リターンの計算（既存の値があれば使用）
            if "cumulative_return" not in returns_per_race.columns:
                returns_per_race["net_return"] = (
                    returns_per_race["return_amount"] - returns_per_race["bet_amount"]
                )
                returns_per_race["cumulative_return"] = returns_per_race[
                    "net_return"
                ].cumsum()

            # 累積リターンをプロット
            plt.plot(
                range(len(returns_per_race)),
                returns_per_race["cumulative_return"],
                label=f"{strategy_name} (RoR: {eval_dict['return_rate']:.2f}%)",
            )

        # グラフの設定
        plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
        plt.xlabel("Race Number")
        plt.ylabel("Cumulative Profit (JPY)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        return plt.gcf()

    def plot_drawdown(
        self, evaluation: Dict, figsize=(12, 6), title="Drawdown Analysis"
    ):
        """
        ドローダウン（最大値からの下落）をグラフで可視化する

        Parameters
        ----------
        evaluation : Dict
            evaluate_strategy()の結果
        figsize : Tuple[int, int], optional (default=(12, 6))
            図のサイズ
        title : str, optional (default="Drawdown Analysis")
            グラフのタイトル

        Returns
        -------
        matplotlib.figure.Figure
            作成したグラフのFigureオブジェクト
        """
        returns_per_race = evaluation["returns_per_race"].copy()
        strategy_name = evaluation["strategy_name"]

        # 時系列順でインデックスをソート
        if isinstance(returns_per_race.index[0], str):
            returns_per_race = returns_per_race.sort_index()

        # 必要な計算を実施
        if "drawdown" not in returns_per_race.columns:
            returns_per_race["net_return"] = (
                returns_per_race["return_amount"] - returns_per_race["bet_amount"]
            )
            returns_per_race["cumulative_return"] = returns_per_race[
                "net_return"
            ].cumsum()
            returns_per_race["cumulative_max"] = returns_per_race[
                "cumulative_return"
            ].cummax()
            returns_per_race["drawdown"] = (
                returns_per_race["cumulative_max"]
                - returns_per_race["cumulative_return"]
            )

            # 相対ドローダウン（最大値に対する割合）
            max_value = returns_per_race["cumulative_max"].replace(
                0, 1
            )  # ゼロ除算を防ぐ
            returns_per_race["drawdown_pct"] = (
                returns_per_race["drawdown"] / max_value * 100
            )

        plt.figure(figsize=figsize)

        # ドローダウングラフの描画
        plt.bar(
            range(len(returns_per_race)),
            returns_per_race["drawdown"] * -1,
            color="r",
            alpha=0.5,
            label="Drawdown (JPY)",
        )

        # 参照用に累積リターンも描画
        ax2 = plt.twinx()
        ax2.plot(
            range(len(returns_per_race)),
            returns_per_race["cumulative_return"],
            color="b",
            label="Cumulative Return",
        )

        # 最大ドローダウンに注釈
        max_dd_idx = returns_per_race["drawdown"].idxmax()
        max_dd = returns_per_race.loc[max_dd_idx, "drawdown"]

        plt.axhline(y=0, color="k", linestyle="-", alpha=0.2)
        plt.xlabel("Race Number")
        plt.ylabel("Drawdown (JPY)")
        ax2.set_ylabel("Cumulative Return (JPY)")
        plt.title(f"{title} - {strategy_name}\nMax Drawdown: {max_dd} JPY")

        # 凡例
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.grid(True, alpha=0.3)

        return plt.gcf()

    def create_performance_summary(self, evaluation: Dict) -> str:
        """
        戦略の性能評価結果をテキスト形式でまとめる

        Parameters
        ----------
        evaluation : Dict
            evaluate_strategy()の結果

        Returns
        -------
        str
            性能評価のテキストサマリー
        """
        strategy_name = evaluation["strategy_name"]
        returns_per_race = evaluation["returns_per_race"]

        total_races = len(returns_per_race)
        total_bets = returns_per_race["n_bets"].sum()
        total_bet_amount = returns_per_race["bet_amount"].sum()
        total_return_amount = returns_per_race["return_amount"].sum()
        profit = total_return_amount - total_bet_amount

        # 勝率計算
        returns_per_race["net_return"] = (
            returns_per_race["return_amount"] - returns_per_race["bet_amount"]
        )
        winning_races = len(returns_per_race[returns_per_race["net_return"] > 0])
        winning_rate = (winning_races / total_races * 100) if total_races > 0 else 0

        # 的中率（馬券単位）
        hit_races = len(returns_per_race[returns_per_race["hit_or_not"] > 0])
        hit_rate = (hit_races / total_races * 100) if total_races > 0 else 0

        # 回収率
        return_rate = (
            (total_return_amount / total_bet_amount * 100)
            if total_bet_amount > 0
            else 0
        )

        # 最大連敗数の計算
        if total_races > 0:
            win_loss_series = (returns_per_race["net_return"] > 0).astype(int)
            lose_indicators = (win_loss_series == 0).astype(int)

            # 連続する0（負け）の最大数をカウント
            max_consecutive_losses = 0
            current_streak = 0

            for result in lose_indicators:
                if result == 1:  # 負け
                    current_streak += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                else:  # 勝ち
                    current_streak = 0
        else:
            max_consecutive_losses = 0

        # サマリーテキストの作成
        summary = f"""
        ===== {strategy_name} の性能評価サマリー =====
        
        【基本情報】
        - 対象レース数: {total_races}
        - 購入馬券数: {total_bets}
        - 総投資額: {total_bet_amount:,} 円
        - 総払戻金: {total_return_amount:,} 円
        - 純利益: {profit:,} 円
        
        【収益性指標】
        - 回収率: {return_rate:.2f}%
        - 的中率（レース単位）: {hit_rate:.2f}%
        - 勝率（利益の出たレース）: {winning_rate:.2f}%
        - 最大ドローダウン: {evaluation["max_drawdown"]:.2f}%
        - シャープレシオ: {evaluation["sharpe_ratio"]:.3f}
        
        【リスク指標】
        - 最大連敗数: {max_consecutive_losses}回
        
        【注意事項】
        - このバックテスト結果は、過去のデータに基づくシミュレーションであり、
          将来の成績を保証するものではありません。
        - 馬券投資は自己責任で行い、余剰資金の範囲内で行いましょう。
        """

        return summary

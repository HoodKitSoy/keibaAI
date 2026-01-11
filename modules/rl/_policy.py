# modules/rl/_policy.py
"""
強化学習ポリシーモジュール

学習済みDQNエージェントを使用した馬券選択ポリシーを提供します。
すべての馬券種とWIN5に対応。

モデルパス構成:
- 強化学習モデル: models/rl/{bet_type}/best_model.pt または final_model.pt
- WIN5モデル: models/rl/win5/best_model.pt
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# 馬券種の設定
BET_TYPES = ["tansho", "fukusho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]


class RLBetPolicy:
    """
    強化学習ベースの馬券選択ポリシー（統合ポリシー）

    各馬券種ごとに学習済みDQNエージェントを読み込み、
    1つのポリシーとして全馬券種の推奨馬番を出力する。

    モデルは models/rl/{bet_type}/ ディレクトリに保存される。
    """

    # デフォルトのモデルディレクトリ（train.py --enable-rl の出力先に合わせる）
    DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "rl"

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        bet_types: Optional[List[str]] = None,
        top_k: int = 3,
    ):
        """
        初期化

        Args:
            model_dir: 学習済みモデルのディレクトリ（各馬券種のサブディレクトリを含む）
                       例: models/rl (この下に tansho/, fukusho/ ... がある)
            bet_types: 使用する券種リスト（Noneの場合はすべて）
            top_k: 選択する馬の最大数
        """
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self.bet_types = bet_types or BET_TYPES
        self.top_k = top_k

        self.agents = {}
        self._load_agents()

    def _load_agents(self) -> None:
        """券種ごとのエージェントを読み込み"""
        try:
            from ._agent import DQNAgent
        except ImportError:
            print("警告: PyTorchがインストールされていません")
            return

        for bet_type in self.bet_types:
            model_path = self.model_dir / bet_type / "best_model.pt"
            if not model_path.exists():
                # final_model.ptもチェック
                model_path = self.model_dir / bet_type / "final_model.pt"

            if model_path.exists():
                try:
                    self.agents[bet_type] = DQNAgent.from_checkpoint(model_path)
                    print(f"[{bet_type}] モデル読み込み: {model_path}")
                except Exception as e:
                    print(f"警告: [{bet_type}] モデル読み込みエラー: {e}")
            else:
                print(f"警告: [{bet_type}] モデルが見つかりません: {model_path}")

    @staticmethod
    def _build_observation(df_race: pd.DataFrame, max_horses: int = 18) -> np.ndarray:
        """レースデータから観測ベクトルを構築"""
        num_horses = len(df_race)

        scores = np.zeros(max_horses, dtype=np.float32)
        odds = np.zeros(max_horses, dtype=np.float32)

        for _, row in df_race.iterrows():
            umaban = int(row["Umaban"])
            if 1 <= umaban <= max_horses:
                scores[umaban - 1] = float(row.get("score", 0.0))
                if "Odds" in row and pd.notna(row["Odds"]):
                    try:
                        odds[umaban - 1] = float(row["Odds"])
                    except (ValueError, TypeError):
                        odds[umaban - 1] = 1.0 / (scores[umaban - 1] + 0.01)
                else:
                    odds[umaban - 1] = 1.0 / (scores[umaban - 1] + 0.01)

        num_normalized = np.array([num_horses / max_horses], dtype=np.float32)
        return np.concatenate([scores, odds, num_normalized])

    def _get_top_horses_by_q(
        self,
        df_race: pd.DataFrame,
        agent,
        bet_type: str,
        max_horses: int = 18,
    ) -> List[int]:
        """Q値上位の馬番を取得"""
        import torch

        observation = self._build_observation(df_race, max_horses)
        valid_umabans = [int(row["Umaban"]) for _, row in df_race.iterrows()]

        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor).squeeze(0).cpu().numpy()

        # 単勝/複勝の場合：action = 馬番 - 1（0始まり）
        if bet_type in ["tansho", "fukusho"]:
            valid_q_pairs = []
            for umaban in valid_umabans:
                # action index は umaban - 1（0始まり）
                action_idx = umaban - 1
                if 0 <= action_idx < len(q_values):
                    q_val = q_values[action_idx]
                    valid_q_pairs.append((umaban, q_val))

            # Q値でソート（降順）し、top_k個を返す
            valid_q_pairs.sort(key=lambda x: x[1], reverse=True)
            return [u for u, _ in valid_q_pairs[: self.top_k]]
        else:
            # 組み合わせ券種の場合はスコア上位を返す
            df_sorted = df_race.sort_values("score", ascending=False)
            return [
                int(row["Umaban"]) for _, row in df_sorted.head(self.top_k).iterrows()
            ]

    def judge(self, score_table: pd.DataFrame, **kwargs) -> Dict[str, Dict[str, List]]:
        """
        各レースで全馬券種の推奨馬番を選択

        Args:
            score_table: 予測スコアテーブル
            **kwargs: 追加パラメータ

        Returns:
            Dict: レースIDをキー、券種ごとの選択馬番リストを値とする辞書
        """
        from modules.core import iter_race_groups

        bet_dict = {}

        for race_id, df_race in iter_race_groups(score_table):
            bet_dict[race_id] = {}

            for bet_type in self.bet_types:
                if bet_type in self.agents:
                    agent = self.agents[bet_type]
                    selected = self._get_top_horses_by_q(df_race, agent, bet_type)
                else:
                    # エージェントがない場合はスコア上位を使用（フォールバック）
                    df_sorted = df_race.sort_values("score", ascending=False)
                    selected = [
                        int(row["Umaban"])
                        for _, row in df_sorted.head(self.top_k).iterrows()
                    ]

                bet_dict[race_id][bet_type] = [str(u) for u in selected]

        return bet_dict

    @classmethod
    def judge_static(cls, score_table: pd.DataFrame, **kwargs) -> Dict:
        """
        静的メソッドとして呼び出すためのラッパー（test.pyから呼び出し用）

        Args:
            score_table: 予測スコアテーブル
            **kwargs: 追加パラメータ
                - model_dir: モデルディレクトリ
                - top_k: 選択馬数（デフォルト: 3）

        Returns:
            Dict: レースIDをキー、券種ごとの選択馬番リストを値とする辞書
        """
        model_dir = kwargs.get("model_dir")
        top_k = kwargs.get("top_k", 3)
        instance = cls(model_dir=model_dir, top_k=top_k)
        return instance.judge(score_table, **kwargs)


class RLWIN5Policy:
    """
    WIN5専用の強化学習ポリシー

    5レースの勝ち馬を予測する特殊なポリシー。
    モデルは models/rl/win5/ ディレクトリに保存される。
    """

    # デフォルトのモデルパス（train.py --enable-rl の出力先に合わせる）
    DEFAULT_MODEL_PATH = (
        Path(__file__).parent.parent.parent / "models" / "rl" / "win5" / "best_model.pt"
    )

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        top_k: int = 3,
    ):
        """
        初期化

        Args:
            model_path: モデルパス
            top_k: 各レースで候補とする上位馬数
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.top_k = top_k
        self.agent = None
        self._load_agent()

    def _load_agent(self) -> None:
        """エージェントを読み込み"""
        if not self.model_path.exists():
            # final_model.ptもチェック
            alt_path = self.model_path.parent / "final_model.pt"
            if alt_path.exists():
                self.model_path = alt_path
            else:
                print(f"警告: WIN5モデルが見つかりません: {self.model_path}")
                return

        try:
            from ._agent import DQNAgent

            self.agent = DQNAgent.from_checkpoint(self.model_path)
            print(f"WIN5モデル読み込み: {self.model_path}")
        except Exception as e:
            print(f"警告: WIN5モデル読み込みエラー: {e}")

    def predict(
        self,
        race_data_list: List[pd.DataFrame],
        max_horses: int = 18,
    ) -> List[int]:
        """
        WIN5の5レース分の勝ち馬を予測

        Args:
            race_data_list: 5レース分のDataFrameリスト
            max_horses: 最大出走頭数

        Returns:
            List[int]: 5レース分の予測馬番
        """
        if self.agent is None:
            # エージェントがない場合は各レースの1番人気を返す
            return self._predict_by_score(race_data_list)

        import torch

        # 状態ベクトルを構築
        observations = []
        for df in race_data_list:
            obs = RLBetPolicy._build_observation(df, max_horses)
            observations.append(obs)

        state = np.concatenate(observations)

        # Q値から最適行動を選択
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.q_network(state_tensor).squeeze(0).cpu().numpy()

        best_action = int(np.argmax(q_values))

        # 行動を5レース分の馬番に変換
        selections = []
        for i in range(5):
            df = race_data_list[i]
            df_sorted = df.sort_values("score", ascending=False).head(self.top_k)
            top_umabans = df_sorted["Umaban"].astype(int).tolist()

            idx = best_action % self.top_k
            best_action //= self.top_k

            if idx < len(top_umabans):
                selections.append(top_umabans[idx])
            elif len(top_umabans) > 0:
                selections.append(top_umabans[0])
            else:
                selections.append(1)

        return selections

    def _predict_by_score(self, race_data_list: List[pd.DataFrame]) -> List[int]:
        """スコア上位で予測（フォールバック）"""
        selections = []
        for df in race_data_list:
            df_sorted = df.sort_values("score", ascending=False)
            if len(df_sorted) > 0:
                selections.append(int(df_sorted.iloc[0]["Umaban"]))
            else:
                selections.append(1)
        return selections

    @classmethod
    def judge(cls, score_table: pd.DataFrame, **kwargs) -> Dict:
        """
        WIN5用のjudgeメソッド（他のポリシーとのインターフェース互換）

        Note:
            WIN5は通常のレース単位ではなく日付単位で5レースを処理するため、
            通常のポリシーとは異なる処理を行う。
            このメソッドは単純にスコア上位馬を返す（WIN5シミュレーションはtest.pyで行う）
        """
        from modules.core import iter_race_groups

        instance = cls(kwargs.get("model_path"), kwargs.get("top_k", 3))
        bet_dict = {}

        for race_id, df_race in iter_race_groups(score_table):
            # WIN5は単勝のみ（各レースの勝ち馬を予測）
            df_sorted = df_race.sort_values("score", ascending=False)
            top_horses = [
                str(int(row["Umaban"]))
                for _, row in df_sorted.head(instance.top_k).iterrows()
            ]
            bet_dict[race_id] = {
                "tansho": top_horses,
                "fukusho": [],
                "umaren": [],
                "umatan": [],
                "wide": [],
                "sanrenpuku": [],
                "sanrentan": [],
            }

        return bet_dict

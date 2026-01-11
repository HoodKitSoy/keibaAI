# modules/rl/_trainer.py
"""
強化学習トレーナーモジュール

DQNエージェントの学習を管理するトレーナークラスを提供します。
すべての馬券種とWIN5に対応。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._agent import DQNAgent
from ._environment import BET_TYPES, BettingEnvironment, WIN5Environment


class RLTrainer:
    """
    強化学習トレーナー

    DQNエージェントの学習を管理し、モデルの保存・評価を行う。
    すべての馬券種に対応。
    """

    def __init__(
        self,
        train_score_table: pd.DataFrame,
        valid_score_table: Optional[pd.DataFrame] = None,
        return_table: Optional[Dict] = None,
        bet_amount: float = 100.0,
        max_horses: int = 18,
        bet_type: str = "tansho",
        model_save_dir: Union[str, Path] = "./models/rl",
        # エージェントハイパーパラメータ
        hidden_dims: List[int] = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[str] = None,
    ):
        """
        初期化

        Args:
            train_score_table: 学習用スコアテーブル
            valid_score_table: 検証用スコアテーブル
            return_table: 払戻データ
            bet_amount: 1点あたりの賭け金
            max_horses: 最大出走頭数
            bet_type: 券種（tansho, fukusho, umaren, umatan, wide, sanrenpuku, sanrentan）
            model_save_dir: モデル保存ディレクトリ
            hidden_dims: 隠れ層のユニット数
            learning_rate: 学習率
            gamma: 割引率
            epsilon_start: 初期ε
            epsilon_end: 最終ε
            epsilon_decay: ε減衰率
            buffer_size: リプレイバッファサイズ
            batch_size: バッチサイズ
            target_update_freq: ターゲット更新頻度
            device: 計算デバイス
        """
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        self.train_score_table = train_score_table
        self.valid_score_table = valid_score_table
        self.return_table = return_table or {}
        self.bet_amount = bet_amount
        self.max_horses = max_horses
        self.bet_type = bet_type
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # 券種別のサブディレクトリを作成
        self.bet_type_dir = self.model_save_dir / bet_type
        self.bet_type_dir.mkdir(parents=True, exist_ok=True)

        # 環境作成
        self.train_env = BettingEnvironment(
            score_table=train_score_table,
            return_table=return_table,
            bet_amount=bet_amount,
            max_horses=max_horses,
            bet_type=bet_type,
        )

        if valid_score_table is not None:
            self.valid_env = BettingEnvironment(
                score_table=valid_score_table,
                return_table=return_table,
                bet_amount=bet_amount,
                max_horses=max_horses,
                bet_type=bet_type,
            )
        else:
            self.valid_env = None

        # 状態・行動次元
        state_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.n

        print(f"[{bet_type}] 状態次元: {state_dim}, 行動次元: {action_dim}")

        # エージェント作成
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
        )

        # 学習履歴
        self.history = {
            "episode": [],
            "train_profit": [],
            "train_bets": [],
            "train_wins": [],
            "train_hit_rate": [],
            "valid_profit": [],
            "valid_bets": [],
            "valid_wins": [],
            "valid_hit_rate": [],
            "epsilon": [],
            "loss": [],
        }

        self.best_valid_profit = float("-inf")

    def train(
        self,
        num_episodes: int = 100,
        eval_interval: int = 10,
        save_interval: int = 20,
        verbose: bool = True,
    ) -> Dict:
        """
        学習を実行

        Args:
            num_episodes: エピソード数
            eval_interval: 評価間隔
            save_interval: 保存間隔
            verbose: 詳細出力

        Returns:
            Dict: 学習履歴
        """
        print(f"\n[{self.bet_type}] 学習開始: {num_episodes}エピソード")
        print(f"  レース数: {self.train_env.num_races}")

        for episode in range(1, num_episodes + 1):
            # 学習エピソード
            train_result = self._run_episode(self.train_env, training=True)

            # 履歴更新
            self.history["episode"].append(episode)
            self.history["train_profit"].append(train_result["profit"])
            self.history["train_bets"].append(train_result["bets"])
            self.history["train_wins"].append(train_result["wins"])
            self.history["train_hit_rate"].append(train_result["hit_rate"])
            self.history["epsilon"].append(self.agent.epsilon)
            self.history["loss"].append(train_result.get("avg_loss", 0.0))

            # 評価
            if episode % eval_interval == 0 and self.valid_env is not None:
                valid_result = self._run_episode(self.valid_env, training=False)
                self.history["valid_profit"].append(valid_result["profit"])
                self.history["valid_bets"].append(valid_result["bets"])
                self.history["valid_wins"].append(valid_result["wins"])
                self.history["valid_hit_rate"].append(valid_result["hit_rate"])

                # ベストモデル保存
                if valid_result["profit"] > self.best_valid_profit:
                    self.best_valid_profit = valid_result["profit"]
                    self.agent.save(self.bet_type_dir / "best_model.pt")

                if verbose:
                    print(
                        f"Episode {episode:4d} | "
                        f"Train: {train_result['profit']:+8.0f}円 ({train_result['hit_rate'] * 100:5.1f}%) | "
                        f"Valid: {valid_result['profit']:+8.0f}円 ({valid_result['hit_rate'] * 100:5.1f}%) | "
                        f"ε={self.agent.epsilon:.3f}"
                    )
            else:
                self.history["valid_profit"].append(None)
                self.history["valid_bets"].append(None)
                self.history["valid_wins"].append(None)
                self.history["valid_hit_rate"].append(None)

                if verbose and episode % 10 == 0:
                    print(
                        f"Episode {episode:4d} | "
                        f"Train: {train_result['profit']:+8.0f}円 ({train_result['hit_rate'] * 100:5.1f}%) | "
                        f"ε={self.agent.epsilon:.3f}"
                    )

            # 定期保存
            if episode % save_interval == 0:
                self.agent.save(self.bet_type_dir / f"checkpoint_ep{episode}.pt")

        # 最終保存
        self.agent.save(self.bet_type_dir / "final_model.pt")
        self._save_history()
        self._save_meta()

        print(f"\n[{self.bet_type}] 学習完了")
        return self.history

    def _run_episode(self, env: BettingEnvironment, training: bool = True) -> Dict:
        """1エピソードを実行"""
        state, info = env.reset()
        losses = []

        while True:
            # 有効な行動を取得
            valid_actions = info.get("valid_actions", list(range(env.action_space.n)))

            # 行動選択
            if training:
                action = self.agent.select_action_with_mask(
                    state, valid_actions, training=True
                )
            else:
                action = self.agent.select_action_with_mask(
                    state, valid_actions, training=False
                )

            # 環境ステップ
            next_state, reward, terminated, truncated, info = env.step(action)

            # 経験保存と学習
            if training:
                self.agent.store_transition(
                    state, action, reward, next_state, terminated
                )
                loss = self.agent.train()
                if loss is not None:
                    losses.append(loss)

            state = next_state

            if terminated or truncated:
                break

        return {
            "profit": info.get("episode_profit", 0.0),
            "bets": info.get("episode_bets", 0),
            "wins": info.get("episode_wins", 0),
            "hit_rate": info.get("hit_rate", 0.0),
            "avg_loss": np.mean(losses) if losses else 0.0,
        }

    def _save_history(self) -> None:
        """学習履歴を保存"""
        df = pd.DataFrame(self.history)
        df.to_csv(self.bet_type_dir / "training_history.csv", index=False)

    def _save_meta(self) -> None:
        """メタ情報を保存"""
        meta = {
            "bet_type": self.bet_type,
            "bet_amount": self.bet_amount,
            "max_horses": self.max_horses,
            "train_races": self.train_env.num_races,
            "valid_races": self.valid_env.num_races if self.valid_env else 0,
            "best_valid_profit": self.best_valid_profit,
            "final_epsilon": self.agent.epsilon,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.bet_type_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


class WIN5Trainer:
    """
    WIN5専用トレーナー

    WIN5環境でDQNエージェントを学習する。
    """

    def __init__(
        self,
        train_win5_data: List[Dict],
        valid_win5_data: Optional[List[Dict]] = None,
        bet_amount: float = 100.0,
        max_horses: int = 18,
        top_k: int = 3,
        model_save_dir: Union[str, Path] = "./models/rl",
        hidden_dims: List[int] = None,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 50,
        device: Optional[str] = None,
    ):
        """初期化"""
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.bet_amount = bet_amount
        self.max_horses = max_horses
        self.top_k = top_k
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # WIN5ディレクトリ
        self.win5_dir = self.model_save_dir / "win5"
        self.win5_dir.mkdir(parents=True, exist_ok=True)

        # 環境作成
        self.train_env = WIN5Environment(
            win5_data=train_win5_data,
            bet_amount=bet_amount,
            max_horses=max_horses,
            top_k=top_k,
        )

        if valid_win5_data is not None:
            self.valid_env = WIN5Environment(
                win5_data=valid_win5_data,
                bet_amount=bet_amount,
                max_horses=max_horses,
                top_k=top_k,
            )
        else:
            self.valid_env = None

        # 状態・行動次元
        state_dim = self.train_env.observation_space.shape[0]
        action_dim = self.train_env.action_space.n

        print(f"[WIN5] 状態次元: {state_dim}, 行動次元: {action_dim}")

        # エージェント作成
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
        )

        self.history = {
            "episode": [],
            "train_profit": [],
            "train_wins": [],
            "valid_profit": [],
            "valid_wins": [],
            "epsilon": [],
        }

        self.best_valid_profit = float("-inf")

    def train(
        self,
        num_episodes: int = 100,
        eval_interval: int = 10,
        save_interval: int = 20,
        verbose: bool = True,
    ) -> Dict:
        """学習を実行"""
        print(f"\n[WIN5] 学習開始: {num_episodes}エピソード")
        print(f"  WIN5数: {self.train_env.num_win5}")

        for episode in range(1, num_episodes + 1):
            train_result = self._run_episode(self.train_env, training=True)

            self.history["episode"].append(episode)
            self.history["train_profit"].append(train_result["profit"])
            self.history["train_wins"].append(train_result["wins"])
            self.history["epsilon"].append(self.agent.epsilon)

            if episode % eval_interval == 0 and self.valid_env is not None:
                valid_result = self._run_episode(self.valid_env, training=False)
                self.history["valid_profit"].append(valid_result["profit"])
                self.history["valid_wins"].append(valid_result["wins"])

                if valid_result["profit"] > self.best_valid_profit:
                    self.best_valid_profit = valid_result["profit"]
                    self.agent.save(self.win5_dir / "best_model.pt")

                if verbose:
                    print(
                        f"Episode {episode:4d} | "
                        f"Train: {train_result['profit']:+10.0f}円 (的中{train_result['wins']}) | "
                        f"Valid: {valid_result['profit']:+10.0f}円 (的中{valid_result['wins']}) | "
                        f"ε={self.agent.epsilon:.3f}"
                    )
            else:
                self.history["valid_profit"].append(None)
                self.history["valid_wins"].append(None)

                if verbose and episode % 10 == 0:
                    print(
                        f"Episode {episode:4d} | "
                        f"Train: {train_result['profit']:+10.0f}円 (的中{train_result['wins']}) | "
                        f"ε={self.agent.epsilon:.3f}"
                    )

            if episode % save_interval == 0:
                self.agent.save(self.win5_dir / f"checkpoint_ep{episode}.pt")

        self.agent.save(self.win5_dir / "final_model.pt")
        self._save_history()

        print("\n[WIN5] 学習完了")
        return self.history

    def _run_episode(self, env: WIN5Environment, training: bool = True) -> Dict:
        """1エピソードを実行"""
        state, info = env.reset()
        losses = []

        while True:
            action = self.agent.select_action(state, training=training)
            next_state, reward, terminated, truncated, info = env.step(action)

            if training:
                self.agent.store_transition(
                    state, action, reward, next_state, terminated
                )
                loss = self.agent.train()
                if loss is not None:
                    losses.append(loss)

            state = next_state

            if terminated or truncated:
                break

        return {
            "profit": info.get("episode_profit", 0.0),
            "wins": info.get("episode_wins", 0),
        }

    def _save_history(self) -> None:
        """学習履歴を保存"""
        df = pd.DataFrame(self.history)
        df.to_csv(self.win5_dir / "training_history.csv", index=False)


class MultiTypeRLTrainer:
    """
    複数馬券種を一括で学習するトレーナー
    """

    def __init__(
        self,
        train_score_table: pd.DataFrame,
        valid_score_table: Optional[pd.DataFrame] = None,
        return_table: Optional[Dict] = None,
        bet_types: List[str] = None,
        bet_amount: float = 100.0,
        max_horses: int = 18,
        model_save_dir: Union[str, Path] = "./models/rl",
        **agent_kwargs,
    ):
        """
        初期化

        Args:
            train_score_table: 学習用スコアテーブル
            valid_score_table: 検証用スコアテーブル
            return_table: 払戻データ
            bet_types: 学習する券種リスト（Noneの場合は全券種）
            bet_amount: 賭け金
            max_horses: 最大出走頭数
            model_save_dir: 保存ディレクトリ
            **agent_kwargs: エージェントパラメータ
        """
        if bet_types is None:
            bet_types = BET_TYPES

        self.bet_types = bet_types
        self.trainers = {}

        for bet_type in bet_types:
            print(f"\n[{bet_type}] トレーナー初期化中...")
            self.trainers[bet_type] = RLTrainer(
                train_score_table=train_score_table,
                valid_score_table=valid_score_table,
                return_table=return_table,
                bet_amount=bet_amount,
                max_horses=max_horses,
                bet_type=bet_type,
                model_save_dir=model_save_dir,
                **agent_kwargs,
            )

    def train_all(
        self,
        num_episodes: int = 100,
        eval_interval: int = 10,
        save_interval: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """全券種を学習"""
        results = {}
        for bet_type, trainer in self.trainers.items():
            print(f"\n{'=' * 60}")
            print(f"  {bet_type} の学習を開始")
            print(f"{'=' * 60}")
            results[bet_type] = trainer.train(
                num_episodes=num_episodes,
                eval_interval=eval_interval,
                save_interval=save_interval,
                verbose=verbose,
            )
        return results

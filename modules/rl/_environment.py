# modules/rl/_environment.py
"""
強化学習環境モジュール

競馬の馬券購入をシミュレートする強化学習環境を提供します。
Gymnasium準拠のインターフェース。すべての馬券種とWIN5に対応。
"""

from itertools import combinations, permutations
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# 対応する馬券種
BET_TYPES = [
    "tansho",  # 単勝
    "fukusho",  # 複勝
    "umaren",  # 馬連
    "umatan",  # 馬単
    "wide",  # ワイド
    "sanrenpuku",  # 三連複
    "sanrentan",  # 三連単
]

# 各馬券種に必要な選択馬数
BET_TYPE_HORSES = {
    "tansho": 1,
    "fukusho": 1,
    "umaren": 2,
    "umatan": 2,
    "wide": 2,
    "sanrenpuku": 3,
    "sanrentan": 3,
}


class BettingEnvironment(gym.Env):
    """
    競馬馬券購入の強化学習環境

    すべての馬券種（単勝、複勝、馬連、馬単、ワイド、三連複、三連単）に対応。

    状態 (State):
        - 各馬の予測スコア（勝率）
        - 各馬のオッズ
        - レース情報（出走頭数など）

    行動 (Action):
        - 単勝/複勝: 0=賭けない, 1~N=馬番Nに賭ける
        - 馬連/馬単/ワイド/三連複/三連単: 組み合わせインデックス
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        score_table: pd.DataFrame,
        return_table: Optional[Dict] = None,
        bet_amount: float = 100.0,
        max_horses: int = 18,
        bet_type: str = "tansho",
        render_mode: Optional[str] = None,
    ):
        """
        環境の初期化

        Args:
            score_table: 予測スコアテーブル（race_id, Umaban, score列を含む）
            return_table: 払戻データ（race_id -> {bet_type -> {組番 -> 払戻金}}）
            bet_amount: 1点あたりの賭け金
            max_horses: 最大出走頭数
            bet_type: 券種（tansho, fukusho, umaren, umatan, wide, sanrenpuku, sanrentan）
            render_mode: 描画モード
        """
        super().__init__()

        self.score_table = score_table
        self.return_table = return_table or {}
        self.bet_amount = bet_amount
        self.max_horses = max_horses
        self.bet_type = bet_type
        self.render_mode = render_mode

        # レースIDリストを作成
        self._prepare_races()

        # 行動空間を設定（券種により異なる）
        self._setup_action_space()

        # 状態空間: [スコア(18), オッズ(18), 出走頭数(1)]
        obs_dim = max_horses * 2 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 内部状態
        self.current_race_idx = 0
        self.total_profit = 0.0
        self.episode_profit = 0.0
        self.episode_bets = 0
        self.episode_wins = 0

    def _setup_action_space(self) -> None:
        """券種に応じた行動空間を設定"""
        if self.bet_type in ["tansho", "fukusho"]:
            # 0=賭けない, 1~18=馬番
            self.action_space = spaces.Discrete(self.max_horses + 1)
            self.action_to_bet = self._action_to_bet_single
        elif self.bet_type in ["umaren", "wide"]:
            # 2頭の組み合わせ（順不同）+ 賭けない
            n_combinations = len(list(combinations(range(1, self.max_horses + 1), 2)))
            self.action_space = spaces.Discrete(n_combinations + 1)
            self._build_combination_map(2, ordered=False)
            self.action_to_bet = self._action_to_bet_combination
        elif self.bet_type == "umatan":
            # 2頭の順列（順序あり）+ 賭けない
            n_permutations = len(list(permutations(range(1, self.max_horses + 1), 2)))
            self.action_space = spaces.Discrete(n_permutations + 1)
            self._build_combination_map(2, ordered=True)
            self.action_to_bet = self._action_to_bet_combination
        elif self.bet_type == "sanrenpuku":
            # 3頭の組み合わせ（順不同）+ 賭けない
            n_combinations = len(list(combinations(range(1, self.max_horses + 1), 3)))
            self.action_space = spaces.Discrete(n_combinations + 1)
            self._build_combination_map(3, ordered=False)
            self.action_to_bet = self._action_to_bet_combination
        elif self.bet_type == "sanrentan":
            # 3頭の順列（順序あり）+ 賭けない
            n_permutations = len(list(permutations(range(1, self.max_horses + 1), 3)))
            self.action_space = spaces.Discrete(n_permutations + 1)
            self._build_combination_map(3, ordered=True)
            self.action_to_bet = self._action_to_bet_combination
        else:
            raise ValueError(f"未対応の券種: {self.bet_type}")

    def _build_combination_map(self, n_horses: int, ordered: bool) -> None:
        """組み合わせ/順列のインデックスマップを構築"""
        if ordered:
            combos = list(permutations(range(1, self.max_horses + 1), n_horses))
        else:
            combos = list(combinations(range(1, self.max_horses + 1), n_horses))

        # action_idx -> (馬番タプル)
        self.idx_to_combo = {i + 1: combo for i, combo in enumerate(combos)}
        # (馬番タプル) -> action_idx
        self.combo_to_idx = {combo: i + 1 for i, combo in enumerate(combos)}

    def _action_to_bet_single(self, action: int) -> Optional[Tuple[int, ...]]:
        """単勝/複勝: 行動を馬番に変換"""
        if action == 0:
            return None
        return (action,)

    def _action_to_bet_combination(self, action: int) -> Optional[Tuple[int, ...]]:
        """組み合わせ券種: 行動を馬番タプルに変換"""
        if action == 0:
            return None
        return self.idx_to_combo.get(action)

    def _prepare_races(self) -> None:
        """レースデータを準備"""
        from modules.core import iter_race_groups

        self.races = []
        for race_id, df_race in iter_race_groups(self.score_table):
            race_data = {
                "race_id": race_id,
                "df": df_race.copy(),
                "num_horses": len(df_race),
            }
            self.races.append(race_data)

    def _get_observation(self) -> np.ndarray:
        """現在のレースから観測を作成"""
        race = self.races[self.current_race_idx]
        df = race["df"]
        num_horses = race["num_horses"]

        # スコア配列
        scores = np.zeros(self.max_horses, dtype=np.float32)
        for _, row in df.iterrows():
            umaban = int(row["Umaban"])
            if 1 <= umaban <= self.max_horses:
                scores[umaban - 1] = float(row.get("score", 0.0))

        # オッズ配列
        odds = np.zeros(self.max_horses, dtype=np.float32)
        for _, row in df.iterrows():
            umaban = int(row["Umaban"])
            if 1 <= umaban <= self.max_horses:
                if "Odds" in row and pd.notna(row["Odds"]):
                    try:
                        odds[umaban - 1] = float(row["Odds"])
                    except (ValueError, TypeError):
                        odds[umaban - 1] = 1.0 / (scores[umaban - 1] + 0.01)
                else:
                    odds[umaban - 1] = 1.0 / (scores[umaban - 1] + 0.01)

        # 出走頭数（正規化）
        num_horses_normalized = np.array(
            [num_horses / self.max_horses], dtype=np.float32
        )

        return np.concatenate([scores, odds, num_horses_normalized])

    def _get_valid_actions(self) -> List[int]:
        """現在のレースで有効な行動リストを取得"""
        race = self.races[self.current_race_idx]
        df = race["df"]
        valid_umabans = [int(row["Umaban"]) for _, row in df.iterrows()]

        valid_actions = [0]  # 賭けないは常に有効

        if self.bet_type in ["tansho", "fukusho"]:
            valid_actions.extend(valid_umabans)
        else:
            # 組み合わせ券種
            n_horses = BET_TYPE_HORSES[self.bet_type]
            ordered = self.bet_type in ["umatan", "sanrentan"]

            if ordered:
                combos = permutations(valid_umabans, n_horses)
            else:
                combos = combinations(valid_umabans, n_horses)

            for combo in combos:
                if combo in self.combo_to_idx:
                    valid_actions.append(self.combo_to_idx[combo])

        return valid_actions

    def _get_reward(self, action: int) -> Tuple[float, bool]:
        """行動に対する報酬を計算"""
        bet = self.action_to_bet(action)
        if bet is None:
            return 0.0, False

        race = self.races[self.current_race_idx]
        race_id = race["race_id"]
        df = race["df"]

        # 着順情報を取得
        if "KakuteiJyuni" not in df.columns:
            return -self.bet_amount, False

        # 着順マップを作成
        rank_map = {}
        for _, row in df.iterrows():
            try:
                jyuni = int(row["KakuteiJyuni"])
                umaban = int(row["Umaban"])
                rank_map[jyuni] = umaban
            except (ValueError, TypeError):
                continue

        hit = False
        payout = 0.0

        if self.bet_type == "tansho":
            winner = rank_map.get(1)
            if winner == bet[0]:
                hit = True
                payout = self._get_payout(race_id, "tansho", str(bet[0]))

        elif self.bet_type == "fukusho":
            top3 = [rank_map.get(i) for i in [1, 2, 3] if i in rank_map]
            if bet[0] in top3:
                hit = True
                payout = self._get_payout(race_id, "fukusho", str(bet[0]))

        elif self.bet_type == "umaren":
            top2 = set([rank_map.get(1), rank_map.get(2)])
            if set(bet) == top2:
                hit = True
                key = "-".join(sorted(str(b) for b in bet))
                payout = self._get_payout(race_id, "umaren", key)

        elif self.bet_type == "umatan":
            if bet[0] == rank_map.get(1) and bet[1] == rank_map.get(2):
                hit = True
                key = f"{bet[0]:02d}-{bet[1]:02d}"
                payout = self._get_payout(race_id, "umatan", key)

        elif self.bet_type == "wide":
            top3 = [rank_map.get(i) for i in [1, 2, 3] if i in rank_map]
            if bet[0] in top3 and bet[1] in top3:
                hit = True
                key = "-".join(sorted(str(b) for b in bet))
                payout = self._get_payout(race_id, "wide", key)

        elif self.bet_type == "sanrenpuku":
            top3 = set([rank_map.get(i) for i in [1, 2, 3] if i in rank_map])
            if set(bet) == top3:
                hit = True
                key = "-".join(sorted(str(b) for b in bet))
                payout = self._get_payout(race_id, "sanrenpuku", key)

        elif self.bet_type == "sanrentan":
            if (
                bet[0] == rank_map.get(1)
                and bet[1] == rank_map.get(2)
                and bet[2] == rank_map.get(3)
            ):
                hit = True
                key = f"{bet[0]:02d}-{bet[1]:02d}-{bet[2]:02d}"
                payout = self._get_payout(race_id, "sanrentan", key)

        if hit:
            reward = payout - self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, hit

    def _get_payout(self, race_id: str, bet_type: str, key: str) -> float:
        """払戻金を取得"""
        if race_id in self.return_table:
            race_returns = self.return_table[race_id]
            if bet_type in race_returns:
                payout = race_returns[bet_type].get(key, 0.0)
                if payout > 0:
                    return payout

        # 払戻データがない場合の推定
        default_payouts = {
            "tansho": 500,
            "fukusho": 200,
            "umaren": 1500,
            "umatan": 3000,
            "wide": 500,
            "sanrenpuku": 3000,
            "sanrentan": 15000,
        }
        return default_payouts.get(bet_type, 500)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """環境をリセット"""
        super().reset(seed=seed)

        self.current_race_idx = 0
        self.episode_profit = 0.0
        self.episode_bets = 0
        self.episode_wins = 0

        observation = self._get_observation()
        info = {
            "race_id": self.races[0]["race_id"],
            "num_horses": self.races[0]["num_horses"],
            "valid_actions": self._get_valid_actions(),
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """1ステップ実行"""
        reward, hit = self._get_reward(action)

        if action > 0:
            self.episode_bets += 1
            self.episode_profit += reward
            if hit:
                self.episode_wins += 1

        self.total_profit += reward
        self.current_race_idx += 1
        terminated = self.current_race_idx >= len(self.races)
        truncated = False

        if not terminated:
            observation = self._get_observation()
            info = {
                "race_id": self.races[self.current_race_idx]["race_id"],
                "num_horses": self.races[self.current_race_idx]["num_horses"],
                "valid_actions": self._get_valid_actions(),
                "episode_profit": self.episode_profit,
                "episode_bets": self.episode_bets,
                "episode_wins": self.episode_wins,
            }
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {
                "episode_profit": self.episode_profit,
                "episode_bets": self.episode_bets,
                "episode_wins": self.episode_wins,
                "hit_rate": self.episode_wins / max(1, self.episode_bets),
            }

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """環境の状態を描画"""
        if self.render_mode == "human":
            race = self.races[min(self.current_race_idx, len(self.races) - 1)]
            print(f"Race: {race['race_id']}, Horses: {race['num_horses']}")
            print(
                f"Profit: {self.episode_profit:.0f}, Bets: {self.episode_bets}, Wins: {self.episode_wins}"
            )
        elif self.render_mode == "ansi":
            race = self.races[min(self.current_race_idx, len(self.races) - 1)]
            return f"Race: {race['race_id']}, Profit: {self.episode_profit:.0f}"
        return None

    def close(self) -> None:
        """環境を閉じる"""
        pass

    @property
    def num_races(self) -> int:
        """レース数を取得"""
        return len(self.races)


class WIN5Environment(gym.Env):
    """
    WIN5（重勝式）専用の強化学習環境

    5つの対象レースで1着馬を予測し、全レース的中で高額配当を狙う。

    状態 (State):
        - 5レース分の各馬スコア（5 x 18 = 90次元）
        - 5レース分の各馬オッズ（5 x 18 = 90次元）
        - 5レース分の出走頭数（5次元）

    行動 (Action):
        - 各レースでどの馬を選択するか（5レース分の馬番の組み合わせ）
        - 簡略化: 各レース上位k頭から1頭ずつ選択 → 最大k^5通り
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        win5_data: List[Dict],
        bet_amount: float = 100.0,
        max_horses: int = 18,
        top_k: int = 3,
        render_mode: Optional[str] = None,
    ):
        """
        初期化

        Args:
            win5_data: WIN5データのリスト（各要素は5レース分のDataFrameと正解を含む辞書）
            bet_amount: 1点あたりの賭け金
            max_horses: 最大出走頭数
            top_k: 各レースで選択候補とする上位馬数
            render_mode: 描画モード
        """
        super().__init__()

        self.win5_data = win5_data
        self.bet_amount = bet_amount
        self.max_horses = max_horses
        self.top_k = top_k
        self.render_mode = render_mode

        # 行動空間: 各レースでtop_k馬から1頭選択 → top_k^5通り
        self.action_space = spaces.Discrete(top_k**5)

        # 状態空間: 5レース分の情報
        obs_dim = 5 * (max_horses * 2 + 1)  # 5 * 37 = 185次元
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 内部状態
        self.current_win5_idx = 0
        self.total_profit = 0.0
        self.episode_profit = 0.0
        self.episode_bets = 0
        self.episode_wins = 0

    def _get_observation(self) -> np.ndarray:
        """現在のWIN5から観測を作成"""
        win5 = self.win5_data[self.current_win5_idx]
        observations = []

        for i in range(5):
            df = win5["races"][i]["df"]
            num_horses = len(df)

            scores = np.zeros(self.max_horses, dtype=np.float32)
            odds = np.zeros(self.max_horses, dtype=np.float32)

            for _, row in df.iterrows():
                umaban = int(row["Umaban"])
                if 1 <= umaban <= self.max_horses:
                    scores[umaban - 1] = float(row.get("score", 0.0))
                    if "Odds" in row and pd.notna(row["Odds"]):
                        try:
                            odds[umaban - 1] = float(row["Odds"])
                        except (ValueError, TypeError):
                            odds[umaban - 1] = 1.0 / (scores[umaban - 1] + 0.01)
                    else:
                        odds[umaban - 1] = 1.0 / (scores[umaban - 1] + 0.01)

            num_normalized = np.array([num_horses / self.max_horses], dtype=np.float32)
            observations.append(np.concatenate([scores, odds, num_normalized]))

        return np.concatenate(observations)

    def _action_to_selections(self, action: int) -> List[int]:
        """行動を5レース分の馬番に変換"""
        win5 = self.win5_data[self.current_win5_idx]
        selections = []

        for i in range(5):
            df = win5["races"][i]["df"]
            # スコア上位k頭を取得
            df_sorted = df.sort_values("score", ascending=False).head(self.top_k)
            top_umabans = df_sorted["Umaban"].astype(int).tolist()

            # 行動からこのレースの選択インデックスを取得
            idx = action % self.top_k
            action //= self.top_k

            if idx < len(top_umabans):
                selections.append(top_umabans[idx])
            elif len(top_umabans) > 0:
                selections.append(top_umabans[0])
            else:
                selections.append(1)

        return selections

    def _get_reward(self, action: int) -> Tuple[float, bool]:
        """行動に対する報酬を計算"""
        selections = self._action_to_selections(action)
        win5 = self.win5_data[self.current_win5_idx]

        # 的中判定
        hit = True
        for i in range(5):
            correct = win5["answers"][i]
            if selections[i] != correct:
                hit = False
                break

        if hit:
            payout = win5.get("payout", 1000000)  # デフォルト100万円
            reward = payout - self.bet_amount
        else:
            reward = -self.bet_amount

        return reward, hit

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """環境をリセット"""
        super().reset(seed=seed)

        self.current_win5_idx = 0
        self.episode_profit = 0.0
        self.episode_bets = 0
        self.episode_wins = 0

        observation = self._get_observation()
        info = {"win5_idx": 0}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """1ステップ実行"""
        reward, hit = self._get_reward(action)

        self.episode_bets += 1
        self.episode_profit += reward
        if hit:
            self.episode_wins += 1

        self.total_profit += reward
        self.current_win5_idx += 1
        terminated = self.current_win5_idx >= len(self.win5_data)
        truncated = False

        if not terminated:
            observation = self._get_observation()
            info = {
                "win5_idx": self.current_win5_idx,
                "episode_profit": self.episode_profit,
                "episode_bets": self.episode_bets,
                "episode_wins": self.episode_wins,
            }
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {
                "episode_profit": self.episode_profit,
                "episode_bets": self.episode_bets,
                "episode_wins": self.episode_wins,
                "hit_rate": self.episode_wins / max(1, self.episode_bets),
            }

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """環境の状態を描画"""
        if self.render_mode == "human":
            print(f"WIN5 #{self.current_win5_idx}, Profit: {self.episode_profit:.0f}")
        return None

    def close(self) -> None:
        """環境を閉じる"""
        pass

    @property
    def num_win5(self) -> int:
        """WIN5数を取得"""
        return len(self.win5_data)

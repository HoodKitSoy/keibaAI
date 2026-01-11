# modules/rl/_agent.py
"""
強化学習エージェントモジュール

DQN（Deep Q-Network）を使用した馬券購入エージェントを提供します。
PyTorch実装。
"""

import json
import os
import random
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# RTX 50シリーズ（Blackwell）との互換性のための環境変数設定
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True

    # cuDNN設定（新しいGPUとの互換性向上）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

except ImportError:
    TORCH_AVAILABLE = False
    print(
        "警告: PyTorchがインストールされていません。pip install torch でインストールしてください。"
    )


def get_safe_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    安全にデバイスを取得する

    CUDAを優先し、初期化時にメモリをクリアする

    Args:
        preferred_device: 希望するデバイス（"cuda", "cpu", または None で自動検出）

    Returns:
        torch.device: 使用可能なデバイス
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorchがインストールされていません")

    # 明示的にCPUが指定された場合のみCPUを使用
    if preferred_device and preferred_device.lower() == "cpu":
        print("警告: CPUが明示的に指定されましたが、CUDAを使用することを推奨します")
        return torch.device("cpu")

    # CUDAが利用可能か確認
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAが利用できません。GPU環境を確認してください。\n"
            "  - NVIDIAドライバがインストールされているか確認\n"
            "  - PyTorchのCUDA版がインストールされているか確認\n"
            "  - nvidia-smi コマンドでGPUが認識されているか確認"
        )

    # CUDAの初期化とメモリクリア
    try:
        # ガベージコレクションを先に実行
        import gc

        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # CUDAコンテキストの初期化を確実に行う（ウォームアップ）
        # 新しいGPU（RTX 50シリーズなど）ではこれが重要
        warmup_tensor = torch.zeros(1, device="cuda")
        warmup_tensor = warmup_tensor + 1  # 簡単な演算でCUDAを初期化
        del warmup_tensor
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()

        return torch.device("cuda")
    except Exception as e:
        raise RuntimeError(
            f"CUDA初期化エラー: {e}\nnvidia-smiでGPU状態を確認してください"
        )


def clear_cuda_memory():
    """CUDAメモリをクリアする"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


class ReplayBuffer:
    """
    経験再生バッファ

    過去の経験（状態、行動、報酬、次状態、終了フラグ）を保存し、
    ランダムにサンプリングして学習に使用する。
    """

    def __init__(self, capacity: int = 100000):
        """
        初期化

        Args:
            capacity: バッファの最大サイズ
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        経験を追加

        Args:
            state: 現在の状態
            action: 実行した行動
            reward: 得られた報酬
            next_state: 次の状態
            done: エピソード終了フラグ
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        ランダムにサンプリング

        Args:
            batch_size: バッチサイズ

        Returns:
            Tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


if TORCH_AVAILABLE:

    class QNetwork(nn.Module):
        """
        Q値を出力するニューラルネットワーク

        入力: 状態ベクトル
        出力: 各行動のQ値
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int] = [256, 256, 128],
        ):
            """
            初期化

            Args:
                state_dim: 状態の次元数
                action_dim: 行動の数
                hidden_dims: 隠れ層のユニット数リスト
            """
            super().__init__()

            layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, action_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """順伝播"""
            return self.network(x)


class DQNAgent:
    """
    DQN（Deep Q-Network）エージェント

    Double DQNを使用してQ値を学習し、最適な馬券購入行動を選択する。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
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
            state_dim: 状態の次元数
            action_dim: 行動の数
            hidden_dims: 隠れ層のユニット数リスト
            learning_rate: 学習率
            gamma: 割引率
            epsilon_start: ε-greedy法の初期ε
            epsilon_end: ε-greedy法の最終ε
            epsilon_decay: εの減衰率
            buffer_size: リプレイバッファのサイズ
            batch_size: バッチサイズ
            target_update_freq: ターゲットネットワーク更新頻度
            device: 計算デバイス（cuda/cpu）
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorchがインストールされていません")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # デバイス設定（安全な初期化）
        self.device = get_safe_device(device)
        print(f"  使用デバイス: {self.device}")

        # GPUメモリクリア
        clear_cuda_memory()

        # ネットワーク
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # オプティマイザ
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # リプレイバッファ
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 学習ステップカウンタ
        self.train_step = 0

        # ハイパーパラメータを保存用に記録
        self.hyperparams = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dims": hidden_dims,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "target_update_freq": target_update_freq,
        }

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        行動を選択

        Args:
            state: 現在の状態
            training: 学習中か（ε-greedy法を使用するか）

        Returns:
            int: 選択した行動
        """
        if training and random.random() < self.epsilon:
            # ランダム行動
            return random.randint(0, self.action_dim - 1)
        else:
            # 最適行動
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def select_action_with_mask(
        self, state: np.ndarray, valid_actions: List[int], training: bool = True
    ) -> int:
        """
        有効な行動のみから選択

        Args:
            state: 現在の状態
            valid_actions: 有効な行動のリスト
            training: 学習中か

        Returns:
            int: 選択した行動
        """
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze(0)
                # 無効な行動のQ値を-∞にする
                mask = torch.full((self.action_dim,), float("-inf"), device=self.device)
                for a in valid_actions:
                    mask[a] = 0.0
                masked_q = q_values + mask
                return masked_q.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """経験をバッファに保存"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self) -> Optional[float]:
        """
        1ステップ学習

        Returns:
            Optional[float]: 損失値（バッファが不足している場合はNone）
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # サンプリング
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # テンソル変換
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 現在のQ値
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: 行動選択はオンラインネットワーク、評価はターゲットネットワーク
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = (
                self.target_network(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 損失計算
        loss = F.smooth_l1_loss(current_q, target_q)

        # 勾配更新
        self.optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # ターゲットネットワーク更新
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # ε減衰
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: Union[str, Path]) -> None:
        """
        モデルを保存

        Args:
            path: 保存先パス
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_step": self.train_step,
            "hyperparams": self.hyperparams,
        }
        torch.save(checkpoint, path)

        # ハイパーパラメータをJSONでも保存（numpy型をPython型に変換）
        params_path = path.with_suffix(".json")

        def convert_to_serializable(obj):
            """numpy型をPython標準型に変換"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            else:
                return obj

        serializable_params = convert_to_serializable(self.hyperparams)
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(serializable_params, f, indent=2, ensure_ascii=False)

        print(f"モデルを保存しました: {path}")

    def load(self, path: Union[str, Path]) -> None:
        """
        モデルを読み込み

        Args:
            path: 読み込み元パス
        """
        path = Path(path)
        # PyTorch 2.6以降はweights_only=Trueがデフォルトなので明示的にFalseを指定
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.train_step = checkpoint.get("train_step", 0)

        print(f"モデルを読み込みました: {path}")

    @classmethod
    def from_checkpoint(
        cls, path: Union[str, Path], device: Optional[str] = None
    ) -> "DQNAgent":
        """
        チェックポイントからエージェントを復元

        Args:
            path: チェックポイントパス
            device: 計算デバイス

        Returns:
            DQNAgent: 復元されたエージェント
        """
        path = Path(path)
        # PyTorch 2.6以降はweights_only=Trueがデフォルトなので明示的にFalseを指定
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        hyperparams = checkpoint["hyperparams"]

        agent = cls(
            state_dim=hyperparams["state_dim"],
            action_dim=hyperparams["action_dim"],
            hidden_dims=hyperparams["hidden_dims"],
            learning_rate=hyperparams["learning_rate"],
            gamma=hyperparams["gamma"],
            epsilon_start=hyperparams["epsilon_start"],
            epsilon_end=hyperparams["epsilon_end"],
            epsilon_decay=hyperparams["epsilon_decay"],
            buffer_size=hyperparams["buffer_size"],
            batch_size=hyperparams["batch_size"],
            target_update_freq=hyperparams["target_update_freq"],
            device=device,
        )

        agent.load(path)
        return agent

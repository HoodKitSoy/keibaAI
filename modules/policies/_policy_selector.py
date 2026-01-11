# modules/policies/_policy_selector.py
"""
ポリシーセレクター

test.pyやpredict.pyで使用するポリシーを設定ファイルで管理するためのモジュール。
JSON形式の設定ファイルで有効/無効を切り替えることができる。

policy_config.jsonの構造:
{
  "enabled_policies": ["ポリシー名1", "ポリシー名2", ...],
  "disabled_policies": ["無効化するポリシー名1", ...],
  "policy_settings": {
    "RLBetPolicy": {
      "model_dir": "models/rl",
      "top_k": 3,
      "bet_types": ["tansho", ...]
    }
  }
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ._bet_policy import (
    AbstractBetPolicy,
    BetPolicyCoverage25,
    BetPolicyCoverage50,
    BetPolicyCoverage75,
)
from ._hybrid_bet_policy import (
    ExpectedValueBetPolicy,
    HybridBetPolicy,
    TieredCoverageBetPolicy,
)
from ._odds_based_policy import (
    MultiOddsExpectedValuePolicy,
    OddsBasedExpectedValuePolicy,
    OddsRangeFilterPolicy,
    OddsValueTieredPolicy,
)
from ._race_selective_coverage_policy import (
    RaceSelectiveCoverage25Policy,
    RaceSelectiveCoverage50Policy,
    RaceSelectiveCoverage75Policy,
)
from ._race_selective_policy import (
    RaceSelectiveExpectedValueBetPolicy,
    RaceSelectiveHybridPolicy,
    RaceSelectiveTieredCoverageBetPolicy,
)
from ._win5_based_policy import (
    WIN5BasedHedgePolicy,
    WIN5BasedMultiPolicy,
)

# RLポリシーのインポート（オプション）
try:
    from modules.rl._policy import RLBetPolicy, RLWIN5Policy

    RL_AVAILABLE = True
except ImportError:
    RLBetPolicy = None
    RLWIN5Policy = None
    RL_AVAILABLE = False

# 利用可能な全ポリシーのレジストリ
AVAILABLE_POLICIES: Dict[str, Type[AbstractBetPolicy]] = {
    # カバレッジ系
    "BetPolicyCoverage25": BetPolicyCoverage25,
    "BetPolicyCoverage50": BetPolicyCoverage50,
    "BetPolicyCoverage75": BetPolicyCoverage75,
    # 期待値系
    "HybridBetPolicy": HybridBetPolicy,
    "ExpectedValueBetPolicy": ExpectedValueBetPolicy,
    "TieredCoverageBetPolicy": TieredCoverageBetPolicy,
    # オッズベース系
    "OddsBasedExpectedValuePolicy": OddsBasedExpectedValuePolicy,
    "OddsValueTieredPolicy": OddsValueTieredPolicy,
    "OddsRangeFilterPolicy": OddsRangeFilterPolicy,
    "MultiOddsExpectedValuePolicy": MultiOddsExpectedValuePolicy,
    # WIN5ベース系
    "WIN5BasedMultiPolicy": WIN5BasedMultiPolicy,
    "WIN5BasedHedgePolicy": WIN5BasedHedgePolicy,
    # レース選別系
    "RaceSelectiveHybridPolicy": RaceSelectiveHybridPolicy,
    "RaceSelectiveExpectedValueBetPolicy": RaceSelectiveExpectedValueBetPolicy,
    "RaceSelectiveTieredCoverageBetPolicy": RaceSelectiveTieredCoverageBetPolicy,
    "RaceSelectiveCoverage25Policy": RaceSelectiveCoverage25Policy,
    "RaceSelectiveCoverage50Policy": RaceSelectiveCoverage50Policy,
    "RaceSelectiveCoverage75Policy": RaceSelectiveCoverage75Policy,
}

# RLポリシーを追加（利用可能な場合）
if RL_AVAILABLE:
    AVAILABLE_POLICIES["RLBetPolicy"] = RLBetPolicy
    AVAILABLE_POLICIES["RLWIN5Policy"] = RLWIN5Policy

# デフォルト設定ファイルパス
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "policy_config.json"

# デフォルト設定
DEFAULT_CONFIG = {
    "enabled_policies": [
        "BetPolicyCoverage25",
        "BetPolicyCoverage50",
        "BetPolicyCoverage75",
        "HybridBetPolicy",
        "ExpectedValueBetPolicy",
        "TieredCoverageBetPolicy",
        "OddsBasedExpectedValuePolicy",
        "OddsValueTieredPolicy",
        "OddsRangeFilterPolicy",
        "WIN5BasedMultiPolicy",
        "WIN5BasedHedgePolicy",
        "RaceSelectiveHybridPolicy",
        "RaceSelectiveExpectedValueBetPolicy",
        "RaceSelectiveTieredCoverageBetPolicy",
        "RaceSelectiveCoverage25Policy",
        "RaceSelectiveCoverage50Policy",
        "RaceSelectiveCoverage75Policy",
        "RLBetPolicy",
    ],
    "disabled_policies": [
        "RLWIN5Policy",
    ],
    "policy_settings": {
        "RLBetPolicy": {
            "model_dir": "models/rl",
            "top_k": 3,
            "bet_types": [
                "tansho",
                "fukusho",
                "umaren",
                "umatan",
                "wide",
                "sanrenpuku",
                "sanrentan",
            ],
        },
        "RLWIN5Policy": {
            "model_path": "models/rl/win5/best_model.pt",
            "top_k": 3,
        },
    },
}


class PolicySelector:
    """
    ポリシー選択・管理クラス

    設定ファイルを読み込み、有効なポリシーのみを返す。
    設定ファイルがない場合はデフォルト設定を使用。
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルのパス。Noneの場合はデフォルトパスを使用
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """
        設定ファイルを読み込む

        Returns:
            Dict: 設定情報
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # デフォルト設定とマージ
                config = DEFAULT_CONFIG.copy()
                config.update(loaded)
                return config
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 設定ファイルの読み込みに失敗しました: {e}")
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    def save_config(self) -> None:
        """
        現在の設定をファイルに保存
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"設定を保存しました: {self.config_path}")
        except IOError as e:
            print(f"エラー: 設定ファイルの保存に失敗しました: {e}")

    def get_enabled_policies(self) -> Dict[str, Type[AbstractBetPolicy]]:
        """
        有効なポリシーのみを取得

        Returns:
            Dict[str, Type[AbstractBetPolicy]]: ポリシー名とクラスの辞書
        """
        enabled = set(self.config.get("enabled_policies", []))
        disabled = set(self.config.get("disabled_policies", []))

        # enabled_policiesが空の場合は全ポリシーを有効とする
        if not enabled:
            enabled = set(AVAILABLE_POLICIES.keys())

        # disabled_policiesで指定されたものを除外
        active = enabled - disabled

        return {name: cls for name, cls in AVAILABLE_POLICIES.items() if name in active}

    def get_policy_settings(self, policy_name: str) -> Dict[str, Any]:
        """
        特定のポリシーの設定を取得

        Args:
            policy_name: ポリシー名

        Returns:
            Dict[str, Any]: ポリシー設定（存在しない場合は空の辞書）
        """
        policy_settings = self.config.get("policy_settings", {})
        return policy_settings.get(policy_name, {})

    def get_all_policy_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        全ポリシーの設定を取得

        Returns:
            Dict[str, Dict[str, Any]]: ポリシー名と設定の辞書
        """
        return self.config.get("policy_settings", {})

    def get_disabled_policies(self) -> List[str]:
        """
        無効化されているポリシー名のリストを取得

        Returns:
            List[str]: 無効化されているポリシー名のリスト
        """
        return self.config.get("disabled_policies", [])

    def enable_policy(self, policy_name: str) -> bool:
        """
        ポリシーを有効化

        Args:
            policy_name: 有効化するポリシー名

        Returns:
            bool: 成功した場合True
        """
        if policy_name not in AVAILABLE_POLICIES:
            print(f"エラー: 不明なポリシー名: {policy_name}")
            return False

        # disabled_policiesから削除
        if policy_name in self.config.get("disabled_policies", []):
            self.config["disabled_policies"].remove(policy_name)

        # enabled_policiesに追加（重複しない場合のみ）
        if policy_name not in self.config.get("enabled_policies", []):
            self.config.setdefault("enabled_policies", []).append(policy_name)

        return True

    def disable_policy(self, policy_name: str) -> bool:
        """
        ポリシーを無効化

        Args:
            policy_name: 無効化するポリシー名

        Returns:
            bool: 成功した場合True
        """
        if policy_name not in AVAILABLE_POLICIES:
            print(f"エラー: 不明なポリシー名: {policy_name}")
            return False

        # enabled_policiesから削除
        if policy_name in self.config.get("enabled_policies", []):
            self.config["enabled_policies"].remove(policy_name)

        # disabled_policiesに追加（重複しない場合のみ）
        if policy_name not in self.config.get("disabled_policies", []):
            self.config.setdefault("disabled_policies", []).append(policy_name)

        return True

    def list_all_policies(self) -> None:
        """
        全ポリシーの状態を表示
        """
        enabled = self.get_enabled_policies()
        disabled = self.get_disabled_policies()

        print("\n=== ポリシー一覧 ===\n")
        print("【有効】")
        for name in sorted(enabled.keys()):
            print(f"  ✓ {name}")

        print("\n【無効】")
        for name in sorted(disabled):
            if name in AVAILABLE_POLICIES:
                print(f"  ✗ {name}")

        print(
            f"\n合計: {len(AVAILABLE_POLICIES)}個 "
            f"(有効: {len(enabled)}個, 無効: {len(disabled)}個)"
        )

    def get_policies_by_names(
        self, names: List[str]
    ) -> Dict[str, Type[AbstractBetPolicy]]:
        """
        指定した名前のポリシーを取得

        Args:
            names: 取得するポリシー名のリスト

        Returns:
            Dict[str, Type[AbstractBetPolicy]]: ポリシー名とクラスの辞書
        """
        result = {}
        for name in names:
            if name in AVAILABLE_POLICIES:
                result[name] = AVAILABLE_POLICIES[name]
            else:
                print(f"警告: 不明なポリシー名: {name}")
        return result

    @staticmethod
    def get_all_policy_names() -> List[str]:
        """
        利用可能な全ポリシー名のリストを取得

        Returns:
            List[str]: ポリシー名のリスト
        """
        return list(AVAILABLE_POLICIES.keys())


def create_default_config(path: Optional[Path] = None) -> None:
    """
    デフォルト設定ファイルを作成

    Args:
        path: 作成先のパス。Noneの場合はデフォルトパス
    """
    config_path = path or DEFAULT_CONFIG_PATH

    if config_path.exists():
        response = input(
            f"設定ファイルが既に存在します: {config_path}\n上書きしますか? (y/N): "
        )
        if response.lower() != "y":
            print("キャンセルしました。")
            return

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)

    print(f"デフォルト設定ファイルを作成しました: {config_path}")


def get_enabled_policies(
    config_path: Optional[Path] = None,
) -> Dict[str, Type[AbstractBetPolicy]]:
    """
    有効なポリシーを取得するユーティリティ関数

    Args:
        config_path: 設定ファイルのパス

    Returns:
        Dict[str, Type[AbstractBetPolicy]]: ポリシー名とクラスの辞書
    """
    selector = PolicySelector(config_path)
    return selector.get_enabled_policies()


# モジュール直接実行時のCLIインターフェース
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ポリシーセレクター管理ツール")
    subparsers = parser.add_subparsers(dest="command", help="コマンド")

    # list コマンド
    subparsers.add_parser("list", help="全ポリシーの状態を表示")

    # enable コマンド
    enable_parser = subparsers.add_parser("enable", help="ポリシーを有効化")
    enable_parser.add_argument("policy", help="有効化するポリシー名")

    # disable コマンド
    disable_parser = subparsers.add_parser("disable", help="ポリシーを無効化")
    disable_parser.add_argument("policy", help="無効化するポリシー名")

    # init コマンド
    subparsers.add_parser("init", help="デフォルト設定ファイルを作成")

    args = parser.parse_args()

    if args.command == "list":
        selector = PolicySelector()
        selector.list_all_policies()
    elif args.command == "enable":
        selector = PolicySelector()
        if selector.enable_policy(args.policy):
            selector.save_config()
            print(f"'{args.policy}' を有効化しました。")
    elif args.command == "disable":
        selector = PolicySelector()
        if selector.disable_policy(args.policy):
            selector.save_config()
            print(f"'{args.policy}' を無効化しました。")
    elif args.command == "init":
        create_default_config()
    else:
        parser.print_help()

"""
設定管理モジュール

競馬データ準備ツールの設定ファイルの読み込み、保存、
およびデフォルト設定を管理します。
"""

import json
import os
from typing import Dict

# --- 設定ファイルパス ---
CONFIG_FILE = "preparing_config.json"

# --- デフォルト設定 ---
DEFAULT_CONFIG = {
    "database": {
        "user": "mikado",
        "password": "Mikado4410",
        "host": "127.0.0.1",
        "port": "3306",
        "database": "everydb2",
        "charset": "utf8",
    },
    "directories": {"pickle_dir": "data/DB", "output_dir": "output"},
    "export": {
        "chunk_size": 200000,  # 大容量テーブル読み出し時のチャンク行数
        "max_write_workers": 4,  # チャンク書き込み用の最大スレッド数
        "compression": None,  # 'gzip' 等を指定可能(Noneで無圧縮)
        "finalize_single_file": True,  # 分割出力後に単一PKLへ統合
        "delete_parts_after_finalize": True,  # 統合成功後にパート/マニフェスト削除
        "finalize_mode": "in-memory",  # 'in-memory' or 'chunked'
        "format": "parquet",  # 'parquet' | 'pkl'
        "parquet_compression": "snappy",  # 'snappy' | 'zstd' | 'gzip' | None
    },
    "data_processing": {
        "target_years": [str(y) for y in range(2016, 2025)],
        "target_jyo_cds": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
    },
}


class ConfigManager:
    """
    設定管理クラス

    設定ファイルの読み込み、保存を行うためのユーティリティクラスです。
    設定ファイルが存在しない場合は、デフォルト設定を使用します。
    """

    @staticmethod
    def load_config() -> Dict:
        """
        設定ファイルを読み込む

        設定ファイル(CONFIG_FILE)が存在する場合は読み込み、
        デフォルト設定とマージして返します。
        ファイルが存在しない、または読み込みに失敗した場合は
        デフォルト設定を返します。

        Returns:
            Dict: 設定情報の辞書
        """
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                # デフォルト設定とマージ
                merged_config = DEFAULT_CONFIG.copy()
                for key, value in config.items():
                    if isinstance(value, dict) and key in merged_config:
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                return merged_config
            except Exception:
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()

    @staticmethod
    def save_config(config: Dict) -> None:
        """
        設定ファイルに保存

        指定された設定情報をJSONファイルとして保存します。

        Args:
            config (Dict): 保存する設定情報の辞書
        """
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"設定保存エラー: {e}")

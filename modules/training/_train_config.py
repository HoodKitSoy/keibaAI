"""
訓練設定モジュール（後方互換性のため維持）

注意: このモジュールの大部分の機能は modules.core に統合されました。
新しいコードでは modules.core を直接使用してください。
"""

from __future__ import annotations

# rankタスクのクラス数: デフォルト18（後で動的に上書き可能）
NUM_CLASSES_RANK = 18

# 注意: FINAL_COLUMN_ORDER, RACE_KEY_COLS, RECORD_KEY_COLS, TARGET_COL は
# modules.core からインポートしています。重複定義を削除しました。

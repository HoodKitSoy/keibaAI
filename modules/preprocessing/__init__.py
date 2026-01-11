"""
前処理モジュール

JRA-VAN DBからのデータ読み込み、結合、調教データ処理、
出力を行うモジュール群を提供します。

注意: 設定関連（DB_DIR等）は modules.core に統合されました。
"""

from ._data_loader import load_sources, load_today_sources
from ._data_merger import filter_frames, merge_core
from ._output_handler import reorder_and_fill, save_outputs
from ._return_processor import ReturnProcessor
from ._training_processor import attach_hanro_and_chip
from ._win5_processor import Win5Processor

__all__ = [
    "ReturnProcessor",
    "Win5Processor",
    "load_sources",
    "load_today_sources",
    "merge_core",
    "filter_frames",
    "attach_hanro_and_chip",
    "reorder_and_fill",
    "save_outputs",
]

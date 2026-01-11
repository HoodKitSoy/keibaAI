"""
データベースエクスポートモジュール

MySQLデータベースからテーブルデータをParquet形式でエクスポートする
機能を提供します。
"""

import os
import queue
from typing import Dict, List, Optional

import mysql.connector
import pandas as pd
from mysql.connector import Error

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False

# --- マーカー定義 ---
MARKER_DONE = "[済]"
MARKER_TODO = "[未]"
MARKER_ERROR = "[✗]"


class DatabaseExporter:
    """
    データベースエクスポートクラス

    MySQLデータベースからテーブルデータを読み取り、
    Parquet形式でファイルシステムに保存します。
    大容量テーブルにも対応するため、チャンク単位での
    ストリーミング処理を行います。
    """

    def __init__(self, config: Dict, gui_queue: queue.Queue):
        """
        初期化

        Args:
            config (Dict): 設定情報の辞書
            gui_queue (queue.Queue): GUIへのメッセージ送信用キュー
        """
        self.config = config
        self.gui_queue = gui_queue
        self.conn = None
        self.pickle_dir = config["directories"]["pickle_dir"]
        # エクスポート設定
        exp = config.get("export", {})
        self.chunk_size = int(exp.get("chunk_size", 200000))
        self.max_write_workers = int(exp.get("max_write_workers", 4))
        self.compression = exp.get("compression", None)
        self.finalize_single_file = bool(exp.get("finalize_single_file", True))
        self.delete_parts_after_finalize = bool(
            exp.get("delete_parts_after_finalize", True)
        )
        self.finalize_mode = str(exp.get("finalize_mode", "in-memory")).lower()
        self.output_format = str(exp.get("format", "parquet")).lower()
        self.parquet_compression = exp.get("parquet_compression", "snappy")

    def get_connection(self) -> Optional[mysql.connector.MySQLConnection]:
        """
        データベース接続を取得

        既存の接続がない、または接続が切れている場合は
        新たに接続を確立します。

        Returns:
            Optional[mysql.connector.MySQLConnection]: データベース接続オブジェクト
                接続に失敗した場合はNone
        """
        try:
            if not self.conn or not self.conn.is_connected():
                self.gui_queue.put(("log", "MySQLデータベースへの接続を開始します..."))
                self.conn = mysql.connector.connect(**self.config["database"])
                self.gui_queue.put(("log", "MySQLデータベースへの接続に成功しました"))
            return self.conn
        except Error as e:
            error_msg = f"データベース接続エラー: {e}"
            self.gui_queue.put(("log", error_msg))
            return None

    def get_tables_list(self) -> List[str]:
        """
        テーブル一覧を取得

        データベースから 'n_' または 's_' で始まるテーブル名を取得し、
        既にエクスポート済みかどうかのマーカー付きで返します。

        Returns:
            List[str]: マーカー付きテーブル名のリスト
                例: ["[済] n_uma_race", "[未] n_race", "[未] s_kisyu"]
        """
        conn = self.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor()
            # 'n_' と 's_' で始まるテーブルを取得
            cursor.execute("SHOW TABLES LIKE 'n_%'")
            n_tables = [table[0] for table in cursor.fetchall()]
            cursor.execute("SHOW TABLES LIKE 's_%'")
            s_tables = [table[0] for table in cursor.fetchall()]
            tables = sorted(n_tables + s_tables)
            cursor.close()

            existing_pickles = set()
            if os.path.exists(self.pickle_dir):
                files = os.listdir(self.pickle_dir)
                for f in files:
                    if f.endswith(".parquet"):
                        existing_pickles.add(f.replace(".parquet", ""))

            tables_with_markers = []
            for table_name in tables:
                marker = MARKER_DONE if table_name in existing_pickles else MARKER_TODO
                tables_with_markers.append(f"{marker} {table_name}")

            return tables_with_markers
        except Error as e:
            self.gui_queue.put(("log", f"テーブルリスト取得エラー: {e}"))
            return []

    def export_tables(self, tables_to_export: List[str]) -> bool:
        """
        テーブルをエクスポート

        指定されたテーブルリストをParquet形式でエクスポートします。

        Args:
            tables_to_export (List[str]): エクスポート対象のテーブル名リスト

        Returns:
            bool: 少なくとも1つのテーブルがエクスポート成功した場合True
        """
        conn = self.get_connection()
        if not conn:
            return False

        os.makedirs(self.pickle_dir, exist_ok=True)
        success_count = 0
        total_tables = len(tables_to_export)

        for i, table_name in enumerate(tables_to_export):
            try:
                self.gui_queue.put(
                    (
                        "log",
                        f"テーブル '{table_name}' のエクスポートを開始... ({i + 1}/{total_tables})",
                    )
                )

                if self._export_table_streaming(conn, table_name):
                    success_count += 1

            except Exception as e:
                self.gui_queue.put(
                    (
                        "log",
                        f"  -> エラー: '{table_name}' のエクスポート中に失敗しました: {e}",
                    )
                )

            # プログレスバー更新
            progress = int(((i + 1) / total_tables) * 100)
            self.gui_queue.put(("progress_export", progress))

        self.gui_queue.put(
            ("log", f"エクスポート完了 ({success_count}/{total_tables} 成功)")
        )
        return success_count > 0

    def _export_table_streaming(
        self, conn: mysql.connector.MySQLConnection, table_name: str
    ) -> bool:
        """
        巨大テーブルをチャンクで単一Parquetに書き出す

        大容量テーブルをメモリ効率よく処理するため、
        チャンク単位で読み込みながらParquetファイルに書き込みます。

        Args:
            conn (mysql.connector.MySQLConnection): データベース接続
            table_name (str): エクスポート対象のテーブル名

        Returns:
            bool: エクスポート成功時True、失敗時False

        Note:
            pyarrowライブラリが必要です。インストールされていない場合は
            エラーメッセージを出力してFalseを返します。
        """
        try:
            if not _HAS_PYARROW:
                self.gui_queue.put(
                    (
                        "log",
                        "  -> エラー: pyarrow が見つかりません。'pip install pyarrow' を実行してください",
                    )
                )
                return False
            # 総件数
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            total_rows = int(cur.fetchone()[0])
            cur.close()

            out_path = os.path.join(self.pickle_dir, f"{table_name}.parquet")

            if total_rows == 0:
                empty_table = pa.Table.from_pandas(pd.DataFrame())
                pq.write_table(
                    empty_table, out_path, compression=self.parquet_compression
                )
                self.gui_queue.put(("log", f"  -> 空テーブルを書き出し: {out_path}"))
                return True

            chunk_size = max(1, int(self.chunk_size))
            query = f"SELECT * FROM `{table_name}`"
            chunk_iter = pd.read_sql(query, conn, chunksize=chunk_size)

            processed_rows = 0
            writer = None
            try:
                for chunk in chunk_iter:
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            out_path, table.schema, compression=self.parquet_compression
                        )
                    writer.write_table(table)
                    processed_rows += len(chunk)
                    pct = int(processed_rows / total_rows * 100) if total_rows else 0
                    self.gui_queue.put(
                        ("log", f"  -> 進捗: {processed_rows}/{total_rows} 行 ({pct}%)")
                    )
            finally:
                if writer is not None:
                    writer.close()
            self.gui_queue.put(("log", f"  -> 完了: '{out_path}' に Parquet 保存"))
            return True
        except Exception as e:
            self.gui_queue.put(("log", f"  -> ストリーミングエクスポートに失敗: {e}"))
            return False

    def close_connection(self):
        """
        データベース接続を閉じる

        アクティブなデータベース接続がある場合はクローズします。
        """
        if self.conn and self.conn.is_connected():
            self.conn.close()
            self.gui_queue.put(("log", "MySQLの接続をクローズしました"))

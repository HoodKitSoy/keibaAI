"""
メインアプリケーションモジュール

競馬データ準備ツールのメインGUIアプリケーションクラスを提供します。
"""

import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, ttk

import mysql.connector
from mysql.connector import Error

try:
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False

from .._config_manager import ConfigManager
from .._database_exporter import DatabaseExporter
from ._tooltip import create_tooltip


class PreparingApp(tk.Tk):
    """
    メインアプリケーションクラス

    競馬データ準備ツールのGUIアプリケーションを提供します。
    データベースからのエクスポート、データビューワー、設定管理などの
    機能を統合したインターフェースです。
    """

    def __init__(self):
        """
        初期化

        GUIの構築、設定の読み込み、初期化処理を行います。
        """
        super().__init__()

        # --- 設定読み込み ---
        self.config = ConfigManager.load_config()

        # --- ウィンドウ設定 ---
        self.title("競馬データ準備ツール - Database Export & Data Processing")
        self.geometry("1200x800")
        self.minsize(800, 600)

        # --- メンバー変数 ---
        self.gui_queue = queue.Queue()
        self.db_exporter = DatabaseExporter(self.config, self.gui_queue)
        self.all_tables_with_markers = []

        # --- スタイル設定 ---
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # カスタムスタイル
        self.style.configure("Title.TLabel", font=("Arial", 12, "bold"))
        self.style.configure(
            "Success.TLabel", foreground="green", font=("Arial", 9, "bold")
        )
        self.style.configure(
            "Error.TLabel", foreground="red", font=("Arial", 9, "bold")
        )
        self.style.configure(
            "Warning.TLabel", foreground="orange", font=("Arial", 9, "bold")
        )
        self.style.configure("Info.TLabel", foreground="blue", font=("Arial", 9))

        # プログレスバーのカスタムスタイル
        self.style.configure("Success.Horizontal.TProgressbar", background="green")
        self.style.configure("Warning.Horizontal.TProgressbar", background="orange")
        self.style.configure("Error.Horizontal.TProgressbar", background="red")

        # --- GUI作成 ---
        self.create_widgets()

        # --- 初期化 ---
        self.after(100, self.process_queue)
        self.refresh_table_list(show_popup=False)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # キーボードショートカット
        self.bind("<Control-r>", lambda e: self.refresh_table_list())
        self.bind("<Control-e>", lambda e: self.export_selected_tables())
        self.bind("<Control-a>", lambda e: self.select_all_items())
        self.bind("<F5>", lambda e: self.refresh_table_list())

        # フォーカス管理
        self.focus_set()

    def create_widgets(self):
        """
        GUIウィジェットを作成

        メインフレーム、タブ、ステータスバーなど、
        アプリケーションのすべてのGUI要素を構築します。
        """
        # メインフレーム
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # タイトルとアイコン
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(
            title_frame, text="🏇 競馬データ準備ツール", style="Title.TLabel"
        )
        title_label.pack(side=tk.LEFT)

        # バージョン表示
        version_label = ttk.Label(
            title_frame, text="v2.0", font=("Arial", 8), foreground="gray"
        )
        version_label.pack(side=tk.RIGHT)

        # タブウィジェット
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        # タブ作成
        self.create_export_tab()
        self.create_settings_tab()
        # ステータス/ログフレーム
        self.create_status_frame(main_frame)

    def create_export_tab(self):
        """
        エクスポートタブの作成

        データベーステーブルのエクスポート機能を提供するタブを構築します。
        """
        export_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(export_frame, text="データベースエクスポート")

        # 説明
        desc_label = ttk.Label(
            export_frame,
            text="MySQLデータベースからテーブルをPickle/Parquetファイルとしてエクスポートします",
        )
        desc_label.pack(anchor=tk.W, pady=(0, 10))

        # コントロールフレーム
        controls_frame = ttk.Frame(export_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_export_selected = ttk.Button(
            controls_frame,
            text="📤 選択したテーブルをエクスポート",
            command=self.export_selected_tables,
        )
        self.btn_export_selected.pack(side=tk.LEFT, padx=(0, 5))
        create_tooltip(
            self.btn_export_selected,
            "リストで選択したテーブルのみをPickleファイルとしてエクスポートします",
        )

        self.btn_export_all = ttk.Button(
            controls_frame,
            text="📦 未出力の全テーブルをエクスポート",
            command=self.export_all_unexported,
        )
        self.btn_export_all.pack(side=tk.LEFT, padx=5)
        create_tooltip(
            self.btn_export_all,
            "まだエクスポートされていない全てのテーブルを一括でエクスポートします",
        )

        self.btn_refresh = ttk.Button(
            controls_frame,
            text="🔄 テーブルリストを更新",
            command=self.refresh_table_list,
        )
        self.btn_refresh.pack(side=tk.LEFT, padx=5)
        create_tooltip(
            self.btn_refresh,
            "データベースから最新のテーブル一覧を取得し、エクスポート状況を更新します",
        )

        # フィルターフレーム
        filter_frame = ttk.Frame(export_frame)
        filter_frame.pack(fill=tk.X, pady=5)
        filter_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(filter_frame, text="フィルター:").grid(row=0, column=0, padx=(0, 5))
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", self.update_listbox_filter)
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var)
        self.filter_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 5))

        self.btn_select_all = ttk.Button(
            filter_frame, text="すべて選択", command=self.select_all_items
        )
        self.btn_select_all.grid(row=0, column=2, padx=5)

        self.btn_deselect_all = ttk.Button(
            filter_frame, text="選択解除", command=self.deselect_all_items
        )
        self.btn_deselect_all.grid(row=0, column=3, padx=(5, 0))

        # テーブルリストフレーム
        list_frame = ttk.LabelFrame(export_frame, text="データベーステーブル一覧")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # リストボックスとスクロールバー
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_container, selectmode=tk.EXTENDED)
        self.listbox.grid(row=0, column=0, sticky="nsew")

        # リストボックスのコンテキストメニュー
        self.listbox_context_menu = tk.Menu(self.listbox, tearoff=0)
        self.listbox_context_menu.add_command(
            label="選択したテーブルをエクスポート", command=self.export_selected_tables
        )
        self.listbox_context_menu.add_separator()
        self.listbox_context_menu.add_command(
            label="すべて選択", command=self.select_all_items
        )
        self.listbox_context_menu.add_command(
            label="選択解除", command=self.deselect_all_items
        )
        self.listbox_context_menu.add_separator()
        self.listbox_context_menu.add_command(
            label="リスト更新", command=self.refresh_table_list
        )

        self.listbox.bind("<Button-3>", self.show_listbox_context_menu)
        self.listbox.bind("<Double-Button-1>", self.on_listbox_double_click)

        scrollbar = ttk.Scrollbar(
            list_container, orient=tk.VERTICAL, command=self.listbox.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scrollbar.set)

        # エクスポート進捗
        progress_frame = ttk.LabelFrame(export_frame, text="エクスポート進捗")
        progress_frame.pack(fill=tk.X, pady=(5, 0))

        self.progress_export = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress_export.pack(fill=tk.X, padx=5, pady=5)

    def create_settings_tab(self):
        """
        設定タブの作成

        データベース接続設定、ディレクトリ設定などを行うタブを構築します。
        """
        settings_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(settings_frame, text="設定")

        # データベース設定
        db_frame = ttk.LabelFrame(settings_frame, text="データベース設定")
        db_frame.pack(fill=tk.X, pady=(0, 10))

        db_inner = ttk.Frame(db_frame, padding="5")
        db_inner.pack(fill=tk.X)
        db_inner.grid_columnconfigure(1, weight=1)
        db_inner.grid_columnconfigure(3, weight=1)

        # データベース設定項目
        ttk.Label(db_inner, text="ホスト:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.db_host_var = tk.StringVar(value=self.config["database"]["host"])
        ttk.Entry(db_inner, textvariable=self.db_host_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(0, 10)
        )

        ttk.Label(db_inner, text="ポート:").grid(
            row=0, column=2, sticky=tk.W, padx=(0, 5)
        )
        self.db_port_var = tk.StringVar(value=self.config["database"]["port"])
        ttk.Entry(db_inner, textvariable=self.db_port_var).grid(
            row=0, column=3, sticky=tk.EW
        )

        ttk.Label(db_inner, text="ユーザー:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.db_user_var = tk.StringVar(value=self.config["database"]["user"])
        ttk.Entry(db_inner, textvariable=self.db_user_var).grid(
            row=1, column=1, sticky=tk.EW, padx=(0, 10), pady=(5, 0)
        )

        ttk.Label(db_inner, text="データベース:").grid(
            row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.db_database_var = tk.StringVar(value=self.config["database"]["database"])
        ttk.Entry(db_inner, textvariable=self.db_database_var).grid(
            row=1, column=3, sticky=tk.EW, pady=(5, 0)
        )

        ttk.Label(db_inner, text="パスワード:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.db_password_var = tk.StringVar(value=self.config["database"]["password"])
        password_entry = ttk.Entry(
            db_inner, textvariable=self.db_password_var, show="*"
        )
        password_entry.grid(row=2, column=1, sticky=tk.EW, padx=(0, 10), pady=(5, 0))

        # 接続テストボタン
        self.btn_test_connection = ttk.Button(
            db_inner, text="🔗 接続テスト", command=self.test_database_connection
        )
        self.btn_test_connection.grid(row=2, column=3, pady=(5, 0))
        create_tooltip(
            self.btn_test_connection,
            "現在の設定でデータベースに正常に接続できるかをテストします",
        )

        # ディレクトリ設定
        dir_frame = ttk.LabelFrame(settings_frame, text="ディレクトリ設定")
        dir_frame.pack(fill=tk.X, pady=(0, 10))

        dir_inner = ttk.Frame(dir_frame, padding="5")
        dir_inner.pack(fill=tk.X)
        dir_inner.grid_columnconfigure(1, weight=1)

        ttk.Label(dir_inner, text="Pickleディレクトリ:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5)
        )
        self.pickle_dir_var = tk.StringVar(
            value=self.config["directories"]["pickle_dir"]
        )
        ttk.Entry(dir_inner, textvariable=self.pickle_dir_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(0, 5)
        )
        ttk.Button(
            dir_inner,
            text="参照",
            command=lambda: self.browse_directory(self.pickle_dir_var),
        ).grid(row=0, column=2)

        ttk.Label(dir_inner, text="出力ディレクトリ:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0)
        )
        self.output_dir_var = tk.StringVar(
            value=self.config["directories"]["output_dir"]
        )
        ttk.Entry(dir_inner, textvariable=self.output_dir_var).grid(
            row=1, column=1, sticky=tk.EW, padx=(0, 5), pady=(5, 0)
        )
        ttk.Button(
            dir_inner,
            text="参照",
            command=lambda: self.browse_directory(self.output_dir_var),
        ).grid(row=1, column=2, pady=(5, 0))

        # 設定保存ボタン
        save_frame = ttk.Frame(settings_frame)
        save_frame.pack(fill=tk.X, pady=10)

        self.btn_save_settings = ttk.Button(
            save_frame, text="💾 設定を保存", command=self.save_settings
        )
        self.btn_save_settings.pack(side=tk.LEFT)
        create_tooltip(
            self.btn_save_settings,
            "現在の設定をファイルに保存し、次回起動時に適用されます",
        )

        self.btn_reset_settings = ttk.Button(
            save_frame, text="🔄 デフォルトに戻す", command=self.reset_settings
        )
        self.btn_reset_settings.pack(side=tk.LEFT, padx=(10, 0))
        create_tooltip(self.btn_reset_settings, "全ての設定をデフォルト値に戻します")

    def create_status_frame(self, parent):
        """
        ステータスフレームの作成

        Args:
            parent: 親ウィジェット
        """
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        # ログエリア
        log_frame = ttk.LabelFrame(status_frame, text="ログ")
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, state="disabled", height=8)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # ステータスバー
        status_bar = ttk.Frame(status_frame)
        status_bar.pack(fill=tk.X, pady=(5, 0))

        self.status_var = tk.StringVar(value="準備完了")
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

        # 時計表示
        self.time_var = tk.StringVar()
        time_label = ttk.Label(status_bar, textvariable=self.time_var)
        time_label.pack(side=tk.RIGHT)
        self.update_time()

    def update_time(self):
        """
        時計を更新

        1秒ごとに現在時刻を更新します。
        """
        current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.time_var.set(current_time)
        self.after(1000, self.update_time)

    def log(self, message: str, level: str = "INFO"):
        """
        ログを追加

        Args:
            message (str): ログメッセージ
            level (str): ログレベル(INFO, WARNING, ERRORなど)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"

        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, formatted_message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def update_status(self, message: str):
        """
        ステータスを更新

        Args:
            message (str): ステータスメッセージ
        """
        self.status_var.set(message)

    def process_queue(self):
        """
        GUIキューを処理

        バックグラウンドタスクからのメッセージを処理してGUIを更新します。
        """
        try:
            while True:
                msg_type, data = self.gui_queue.get_nowait()

                if msg_type == "log":
                    self.log(data)
                elif msg_type == "status":
                    self.update_status(data)
                elif msg_type == "table_list_update":
                    self.all_tables_with_markers = data
                    self.update_listbox_filter()
                elif msg_type == "progress_export":
                    self.progress_export["value"] = data
                    if data == 100:
                        self.progress_export.configure(
                            style="Success.Horizontal.TProgressbar"
                        )
                elif msg_type == "task_start":
                    self.set_buttons_state(tk.DISABLED)
                    self.progress_export["value"] = 0
                    # プログレスバーのスタイルをリセット
                    self.progress_export.configure(style="TProgressbar")
                elif msg_type == "task_done":
                    self.set_buttons_state(tk.NORMAL)
                    self.update_status("準備完了")
                    if data:  # メッセージがあれば表示
                        messagebox.showinfo("完了", data)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def set_buttons_state(self, state):
        """
        ボタンの状態を設定

        Args:
            state: ボタンの状態(tk.NORMAL または tk.DISABLED)
        """
        button_attrs = [
            "btn_export_selected",
            "btn_export_all",
            "btn_refresh",
            "btn_select_all",
            "btn_deselect_all",
            "btn_test_connection",
            "btn_save_settings",
        ]
        for name in button_attrs:
            if hasattr(self, name):
                getattr(self, name).config(state=state)

    # --- エクスポート関連メソッド ---

    def refresh_table_list(self, show_popup: bool = True):
        """
        テーブルリストを更新

        Args:
            show_popup (bool): 完了時にポップアップを表示するかどうか
        """
        self.log("テーブルリストを更新しています...")
        self.update_status("テーブルリスト更新中...")
        threading.Thread(
            target=self._worker_get_tables, args=(show_popup,), daemon=True
        ).start()

    def _worker_get_tables(self, show_popup: bool):
        """
        テーブル取得ワーカー

        Args:
            show_popup (bool): 完了時にポップアップを表示するかどうか
        """
        self.gui_queue.put(("task_start", None))
        tables = self.db_exporter.get_tables_list()
        self.gui_queue.put(("table_list_update", tables))
        self.gui_queue.put(("log", f"{len(tables)}個のテーブルを取得しました"))
        popup_message = "テーブルリストの更新が完了しました" if show_popup else None
        self.gui_queue.put(("task_done", popup_message))

    def update_listbox_filter(self, *args):
        """
        フィルターに基づいてリストボックスを更新

        Args:
            *args: トレースコールバックの引数(未使用)
        """
        filter_text = self.filter_var.get().lower()
        self.listbox.delete(0, tk.END)
        for item in self.all_tables_with_markers:
            if filter_text in item.lower():
                self.listbox.insert(tk.END, item)

    def select_all_items(self):
        """
        全選択
        """
        self.listbox.select_set(0, tk.END)

    def deselect_all_items(self):
        """
        選択解除
        """
        self.listbox.selection_clear(0, tk.END)

    def show_listbox_context_menu(self, event):
        """
        リストボックスのコンテキストメニューを表示

        Args:
            event: イベントオブジェクト
        """
        try:
            self.listbox_context_menu.post(event.x_root, event.y_root)
        except tk.TclError:
            pass

    def on_listbox_double_click(self, event):
        """
        リストボックスのダブルクリック処理

        Args:
            event: イベントオブジェクト
        """
        self.export_selected_tables()

    def export_selected_tables(self):
        """
        選択されたテーブルをエクスポート
        """
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showwarning(
                "選択なし", "エクスポートするテーブルを選択してください"
            )
            return

        targets = [self.listbox.get(i).split("] ")[1] for i in selected_indices]
        self._start_export_worker(targets)

    def export_all_unexported(self):
        """
        未出力のテーブルを全てエクスポート
        """
        targets = [
            item.split("] ")[1]
            for item in self.all_tables_with_markers
            if item.startswith("[未]")
        ]

        if not targets:
            messagebox.showinfo("対象なし", "未出力のテーブルはありません")
            return

        if messagebox.askokcancel(
            "確認", f"{len(targets)}個の未出力テーブルをすべてエクスポートしますか？"
        ):
            self._start_export_worker(targets)

    def _start_export_worker(self, targets):
        """
        エクスポートワーカーを開始

        Args:
            targets: エクスポート対象のテーブル名リスト
        """
        self.log(f"{len(targets)}個のテーブルのエクスポートを開始します")
        self.update_status(f"{len(targets)}個のテーブルをエクスポート中...")
        threading.Thread(
            target=self._worker_export_tables, args=(targets,), daemon=True
        ).start()

    def _worker_export_tables(self, targets):
        """
        エクスポートワーカー

        Args:
            targets: エクスポート対象のテーブル名リスト
        """
        self.gui_queue.put(("task_start", None))
        success = self.db_exporter.export_tables(targets)

        if success:
            # エクスポート完了後にリスト更新
            self.refresh_table_list(show_popup=False)
            self.gui_queue.put(("task_done", "エクスポート処理が完了しました"))
        else:
            self.gui_queue.put(("task_done", None))

    def open_pickle_folder(self):
        """
        データフォルダを開く
        """
        base_dir = self.config["directories"]["pickle_dir"]
        if os.path.exists(base_dir):
            os.startfile(base_dir)
        else:
            messagebox.showwarning("警告", "データフォルダが存在しません")

    # --- 設定関連メソッド ---

    def test_database_connection(self):
        """
        データベース接続テスト

        現在の設定でデータベースに接続できるかテストします。
        """
        self.update_config_from_ui()

        try:
            test_config = self.config["database"].copy()
            conn = mysql.connector.connect(**test_config)
            if conn.is_connected():
                conn.close()
                messagebox.showinfo("成功", "データベースへの接続に成功しました")
                self.log("データベース接続テスト成功")
        except Error as e:
            messagebox.showerror(
                "接続エラー", f"データベースへの接続に失敗しました:\n{e}"
            )
            self.log(f"データベース接続テスト失敗: {e}", "ERROR")

    def browse_directory(self, var: tk.StringVar):
        """
        ディレクトリ選択ダイアログ

        Args:
            var (tk.StringVar): 選択されたパスを格納する変数
        """
        from tkinter import filedialog

        directory = filedialog.askdirectory(initialdir=var.get())
        if directory:
            var.set(directory)

    def save_settings(self):
        """
        設定を保存

        現在の設定をファイルに保存します。
        """
        self.update_config_from_ui()
        ConfigManager.save_config(self.config)
        messagebox.showinfo("保存完了", "設定が保存されました")
        self.log("設定を保存しました")

    def reset_settings(self):
        """
        設定をデフォルトに戻す
        """
        if messagebox.askokcancel("確認", "設定をデフォルトに戻しますか？"):
            from .._config_manager import DEFAULT_CONFIG

            self.config = DEFAULT_CONFIG.copy()
            self.update_ui_from_config()
            self.log("設定をデフォルトに戻しました")

    def update_config_from_ui(self):
        """
        UIから設定を更新

        GUI上の入力値を内部の設定辞書に反映します。
        """
        # データベース設定
        self.config["database"]["host"] = self.db_host_var.get()
        self.config["database"]["port"] = self.db_port_var.get()
        self.config["database"]["user"] = self.db_user_var.get()
        self.config["database"]["database"] = self.db_database_var.get()
        self.config["database"]["password"] = self.db_password_var.get()

        # ディレクトリ設定
        self.config["directories"]["pickle_dir"] = self.pickle_dir_var.get()
        self.config["directories"]["output_dir"] = self.output_dir_var.get()

        # オブジェクトに設定を反映
        self.db_exporter.config = self.config
        self.db_exporter.pickle_dir = self.config["directories"]["pickle_dir"]

    def update_ui_from_config(self):
        """
        設定からUIを更新

        内部の設定辞書をGUI上の入力欄に反映します。
        """
        # データベース設定
        self.db_host_var.set(self.config["database"]["host"])
        self.db_port_var.set(self.config["database"]["port"])
        self.db_user_var.set(self.config["database"]["user"])
        self.db_database_var.set(self.config["database"]["database"])
        self.db_password_var.set(self.config["database"]["password"])

        # ディレクトリ設定
        self.pickle_dir_var.set(self.config["directories"]["pickle_dir"])
        self.output_dir_var.set(self.config["directories"]["output_dir"])

    def on_closing(self):
        """
        アプリケーション終了処理

        終了確認ダイアログを表示し、確認後にデータベース接続を閉じて終了します。
        """
        if messagebox.askokcancel("終了", "アプリケーションを終了しますか？"):
            self.db_exporter.close_connection()
            self.destroy()

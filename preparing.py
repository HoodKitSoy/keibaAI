"""
競馬データ準備ツール メインスクリプト

JRA-VAN DBからのデータエクスポートと管理を行うGUIアプリケーションです。
MySQLデータベースからテーブルデータをParquet形式でエクスポートし、
エクスポートされたデータの閲覧機能を提供します。

使用方法:
    python preparing.py

機能:
    - MySQLデータベースからのテーブルエクスポート
    - Parquetファイルのデータビューワー
    - データベース接続設定の管理
"""

from tkinter import messagebox

from modules.preparing.gui import PreparingApp


def main():
    """
    メイン関数
    
    競馬データ準備ツールのGUIアプリケーションを起動します。
    アプリケーションの起動に失敗した場合は、エラーメッセージを表示します。
    
    Raises:
        Exception: アプリケーション起動時のエラー
    """
    try:
        app = PreparingApp()
        app.mainloop()
    except Exception as e:
        print(f"アプリケーション起動エラー: {e}")
        messagebox.showerror(
            "起動エラー", f"アプリケーションの起動に失敗しました:\n{e}"
        )


if __name__ == "__main__":
    main()

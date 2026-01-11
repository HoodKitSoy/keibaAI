"""
ツールチップモジュール

GUIウィジェットにマウスオーバー時のツールチップ機能を提供します。
"""

import tkinter as tk


class ToolTip:
    """
    ツールチップ機能を提供するクラス

    tkinterウィジェットにマウスオーバー時に説明テキストを
    表示する機能を追加します。
    """

    def __init__(self, widget, text):
        """
        初期化

        Args:
            widget: ツールチップを表示するtkinterウィジェット
            text (str): 表示するツールチップテキスト
        """
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.tooltip_window = None

    def on_enter(self, event=None):
        """
        マウスが入った時の処理

        ウィジェットにマウスカーソルが入った時にツールチップを表示します。

        Args:
            event: イベントオブジェクト(未使用)
        """
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="lightyellow",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", 8),
        )
        label.pack()

    def on_leave(self, event=None):
        """
        マウスが出た時の処理

        ウィジェットからマウスカーソルが出た時にツールチップを非表示にします。

        Args:
            event: イベントオブジェクト(未使用)
        """
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


def create_tooltip(widget, text):
    """
    ツールチップを作成する便利関数

    Args:
        widget: ツールチップを表示するtkinterウィジェット
        text (str): 表示するツールチップテキスト

    Returns:
        ToolTip: 作成されたToolTipインスタンス
    """
    return ToolTip(widget, text)

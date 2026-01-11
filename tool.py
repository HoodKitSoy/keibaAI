#!/usr/bin/env python3
import os
import sys


def main():
    if len(sys.argv) < 2:
        print(
            f"Usage: {os.path.basename(sys.argv[0])} <ファイルパス1> [ファイルパス2 ...]"
        )
        return

    try:
        with open("output.txt", "w", encoding="utf-8") as outfile:
            for file_path in sys.argv[1:]:
                if not os.path.exists(file_path):
                    print(f"ファイルが見つかりません: {file_path}")
                    continue

                # フルパスをそのまま出力する
                outfile.write(file_path + "\n")

                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        content = infile.read()
                    outfile.write(content)
                except Exception as e:
                    outfile.write(f"ファイル読み込みエラー: {e}\n")

                outfile.write("\n")
        print("output.txt に出力しました。")
    except Exception as e:
        print(f"出力ファイル作成中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()

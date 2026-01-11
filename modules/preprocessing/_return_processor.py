import os
from typing import Dict, List

import pandas as pd

from modules.core import DB_DIR_STR

from ._abstract_data_processor import AbstractDataProcessor


class ReturnProcessor(AbstractDataProcessor):
    def __init__(self, filepath):
        """払戻テーブルの読み込み/前処理

        - 旧実装: pickle（スクレイピング由来）
        - 新実装: JRA-DB Parquet（data/DB/n_harai.parquet）
        """
        # 旧pickleが指定・存在し、かつ拡張子がpickle系なら従来どおり
        if (
            filepath
            and os.path.exists(filepath)
            and os.path.splitext(filepath)[1].lower() in {".pkl", ".pickle"}
        ):
            super().__init__(filepath)
            return

        # JRA-DB Parquet を優先的に探索
        db_path = filepath
        if (
            not db_path
            or not os.path.exists(db_path)
            or os.path.splitext(db_path)[1].lower() not in {".parquet", ".pq"}
        ):
            db_path = os.path.join(DB_DIR_STR, "n_harai.parquet")

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"払戻データが見つかりません。指定: {filepath!r}, 既定: {db_path!r}"
            )

        # Parquet 読み込み（JRA-DB）
        df = pd.read_parquet(db_path)
        # DataKubun=0/9 は除外（中止/削除）
        if "DataKubun" in df.columns:
            df = df[~df["DataKubun"].astype(str).isin(["0", "9"])]

        # 親クラスの隠し属性に直接設定（プロパティ互換維持）
        self._AbstractDataProcessor__raw_data = df
        self._AbstractDataProcessor__preprocessed_data = self._preprocess_jra(df)

    def _preprocess(self):
        """
        前処理
        """
        return_dict = {}
        return_dict["tansho"] = self.__tansho()
        return_dict["fukusho"] = self.__fukusho()
        return_dict["umaren"] = self.__umaren()
        return_dict["umatan"] = self.__umatan()
        return_dict["wide"] = self.__wide()
        return_dict["sanrentan"] = self.__sanrentan()
        return_dict["sanrenpuku"] = self.__sanrenpuku()
        return return_dict

    # -------- JRA-DB (HARAI) 用 前処理 --------
    def _preprocess_jra(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """HARAI仕様のParquetから、従来の返却フォーマットを構築する。
        返り値の各DataFrameは index=race_id（文字列）で統一。
        """

        def zfill(s: pd.Series, w: int) -> pd.Series:
            return s.astype(str).str.zfill(w)

        required = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"HARAIに必要なキー列が不足しています: {missing}")

        race_id = (
            zfill(df["Year"], 4)
            + zfill(df["MonthDay"], 4)
            + "-"
            + zfill(df["JyoCD"], 2)
            + zfill(df["Kaiji"], 2)
            + zfill(df["Nichiji"], 2)
            + "-"
            + zfill(df["RaceNum"], 2)
        )

        base = df.copy()
        base.index = race_id

        def _to_int(x):
            try:
                return int(str(x).strip())
            except Exception:
                return 0

        def _pair_from_kumi(s: str) -> List[int]:
            s = str(s).strip().zfill(4)
            a, b = _to_int(s[:2]), _to_int(s[2:])
            return [a, b]

        def _trio_from_kumi(s: str) -> List[int]:
            s = str(s).strip().zfill(6)
            a, b, c = _to_int(s[:2]), _to_int(s[2:4]), _to_int(s[4:])
            return [a, b, c]

        # 単勝: 同着は先頭のみ採用（互換性のため）
        tansho_rows = []
        for idx, row in base.iterrows():
            winner = 0
            pay = 0
            for i in (1, 2, 3):
                u = _to_int(row.get(f"PayTansyoUmaban{i}", 0))
                p = _to_int(str(row.get(f"PayTansyoPay{i}", 0)).replace(",", ""))
                if u > 0 and p > 0:
                    winner, pay = u, p
                    break
            tansho_rows.append({"race_id": idx, "win": winner, "return": pay})
        tansho = pd.DataFrame(tansho_rows).set_index("race_id")

        # 複勝: 最大3つまで採用
        fuku_records = []
        for idx, row in base.iterrows():
            wins, pays = [], []
            for i in (1, 2, 3, 4, 5):
                u = _to_int(row.get(f"PayFukusyoUmaban{i}", 0))
                p = _to_int(str(row.get(f"PayFukusyoPay{i}", 0)).replace(",", ""))
                if u > 0 and p > 0:
                    wins.append(u)
                    pays.append(p)
            wins = (wins + [0, 0, 0])[:3]
            pays = (pays + [0, 0, 0])[:3]
            fuku_records.append(
                {
                    "race_id": idx,
                    "win_0": wins[0],
                    "win_1": wins[1],
                    "win_2": wins[2],
                    "return_0": pays[0],
                    "return_1": pays[1],
                    "return_2": pays[2],
                }
            )
        fukusho = pd.DataFrame(fuku_records).set_index("race_id")

        # 馬連: 先頭のみ採用（同着対応は簡略化）
        umaren_rows = []
        for idx, row in base.iterrows():
            pair = _pair_from_kumi(row.get("PayUmarenKumi1", "0000"))
            pay = _to_int(str(row.get("PayUmarenPay1", 0)).replace(",", ""))
            umaren_rows.append(
                {"race_id": idx, "win_0": pair[0], "win_1": pair[1], "return": pay}
            )
        umaren = pd.DataFrame(umaren_rows).set_index("race_id")

        # 枠連: 先頭のみ採用
        # 枠連の組番は2桁（枠1枠2）で格納されている
        def _waku_pair_from_kumi(s: str) -> List[int]:
            s = str(s).strip().zfill(2)
            a, b = _to_int(s[0]), _to_int(s[1])
            return [a, b]

        wakuren_rows = []
        for idx, row in base.iterrows():
            pair = _waku_pair_from_kumi(row.get("PayWakurenKumi1", "00"))
            pay = _to_int(str(row.get("PayWakurenPay1", 0)).replace(",", ""))
            wakuren_rows.append(
                {"race_id": idx, "win_0": pair[0], "win_1": pair[1], "return": pay}
            )
        wakuren = pd.DataFrame(wakuren_rows).set_index("race_id")

        # 馬単: 先頭のみ採用
        umatan_rows = []
        for idx, row in base.iterrows():
            pair = _pair_from_kumi(row.get("PayUmatanKumi1", "0000"))
            pay = _to_int(str(row.get("PayUmatanPay1", 0)).replace(",", ""))
            umatan_rows.append(
                {"race_id": idx, "win_0": pair[0], "win_1": pair[1], "return": pay}
            )
        umatan = pd.DataFrame(umatan_rows).set_index("race_id")

        # ワイド: 最大7件まで、全件を行として展開
        wide_rows = []
        for idx, row in base.iterrows():
            for i in range(1, 8):
                pair = _pair_from_kumi(row.get(f"PayWideKumi{i}", "0000"))
                pay = _to_int(str(row.get(f"PayWidePay{i}", 0)).replace(",", ""))
                if pair[0] > 0 and pair[1] > 0 and pay > 0:
                    wide_rows.append(
                        {
                            "race_id": idx,
                            "win_0": pair[0],
                            "win_1": pair[1],
                            "return": pay,
                        }
                    )
        wide = (
            pd.DataFrame(wide_rows).set_index("race_id")
            if wide_rows
            else pd.DataFrame(columns=["win_0", "win_1", "return"]).astype(int)
        )

        # 三連複: 先頭のみ採用
        renpuku_rows = []
        for idx, row in base.iterrows():
            trio = _trio_from_kumi(row.get("PaySanrenpukuKumi1", "000000"))
            pay = _to_int(str(row.get("PaySanrenpukuPay1", 0)).replace(",", ""))
            renpuku_rows.append(
                {
                    "race_id": idx,
                    "win_0": trio[0],
                    "win_1": trio[1],
                    "win_2": trio[2],
                    "return": pay,
                }
            )
        sanrenpuku = pd.DataFrame(renpuku_rows).set_index("race_id")

        # 三連単: 先頭のみ採用
        rentan_rows = []
        for idx, row in base.iterrows():
            trio = _trio_from_kumi(row.get("PaySanrentanKumi1", "000000"))
            pay = _to_int(str(row.get("PaySanrentanPay1", 0)).replace(",", ""))
            rentan_rows.append(
                {
                    "race_id": idx,
                    "win_0": trio[0],
                    "win_1": trio[1],
                    "win_2": trio[2],
                    "return": pay,
                }
            )
        sanrentan = pd.DataFrame(rentan_rows).set_index("race_id")

        # 型整備
        for df_ in [
            tansho,
            fukusho,
            wakuren,
            umaren,
            umatan,
            wide,
            sanrenpuku,
            sanrentan,
        ]:
            for c in df_.columns:
                df_[c] = pd.to_numeric(df_[c], errors="coerce").fillna(0).astype(int)

        return {
            "tansho": tansho,
            "fukusho": fukusho,
            "wakuren": wakuren,
            "umaren": umaren,
            "umatan": umatan,
            "wide": wide,
            "sanrenpuku": sanrenpuku,
            "sanrentan": sanrentan,
        }

    def __tansho(self):
        """
        単勝
        """
        tansho = self.raw_data[self.raw_data[0] == "単勝"][[1, 2]]
        tansho.columns = ["win", "return"]

        for column in tansho.columns:
            tansho[column] = pd.to_numeric(tansho[column], errors="coerce")

        return tansho

    def __fukusho(self):
        """
        複勝
        """
        fukusho = self.raw_data[self.raw_data[0] == "複勝"][[1, 2]]
        wins = fukusho[1].str.split("br", expand=True)[[0, 1, 2]]

        wins.columns = ["win_0", "win_1", "win_2"]
        returns = fukusho[2].str.split("br", expand=True)[[0, 1, 2]]
        returns.columns = ["return_0", "return_1", "return_2"]

        df = pd.concat([wins, returns], axis=1)
        for column in df.columns:
            df[column] = df[column].str.replace(",", "")
        return df.fillna(0).astype(int)

    def __umaren(self):
        """
        馬連
        """
        umaren = self.raw_data[self.raw_data[0] == "馬連"][[1, 2]]
        wins = umaren[1].str.split("-", expand=True)[[0, 1]].add_prefix("win_")
        return_ = umaren[2].rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors="coerce"))

    def __umatan(self):
        """
        馬単
        """
        umatan = self.raw_data[self.raw_data[0] == "馬単"][[1, 2]]
        wins = umatan[1].str.split("→", expand=True)[[0, 1]].add_prefix("win_")
        return_ = umatan[2].rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors="coerce"))

    def __wide(self):
        """
        ワイド
        """
        wide = self.raw_data[self.raw_data[0] == "ワイド"][[1, 2]]
        wins = wide[1].str.split("br", expand=True)[[0, 1, 2]]
        wins = wins.stack().str.split("-", expand=True).add_prefix("win_")
        return_ = wide[2].str.split("br", expand=True)[[0, 1, 2]]
        return_ = return_.stack().rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(
            lambda x: pd.to_numeric(x.str.replace(",", ""), errors="coerce")
        )

    def __sanrentan(self):
        """
        三連単
        """
        rentan = self.raw_data[self.raw_data[0] == "三連単"][[1, 2]]
        wins = rentan[1].str.split("→", expand=True)[[0, 1, 2]].add_prefix("win_")
        return_ = rentan[2].rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors="coerce"))

    def __sanrenpuku(self):
        """
        三連複
        """
        renpuku = self.raw_data[self.raw_data[0] == "三連複"][[1, 2]]
        wins = renpuku[1].str.split("-", expand=True)[[0, 1, 2]].add_prefix("win_")
        return_ = renpuku[2].rename("return")
        df = pd.concat([wins, return_], axis=1)
        return df.apply(lambda x: pd.to_numeric(x, errors="coerce"))

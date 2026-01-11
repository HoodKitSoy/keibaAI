"""
オッズデータローダー

JRA-DB から各種オッズデータを読み込み、レースごとのオッズ情報を提供する。

対応券種:
- 単勝/複勝 (n_odds_tanpuku)
- 枠連 (n_odds_waku)
- 馬連 (n_odds_umaren)
- ワイド (n_odds_wide)
- 馬単 (n_odds_umatan)
- 三連複 (n_odds_sanren)
- 三連単 (n_odds_sanrentan)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

# レースキー列
RACE_KEY_COLS = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]


def _build_race_key(year, monthday, jyo_cd, kaiji, nichiji, race_num) -> str:
    """レースキーを構築（インデックス用）"""
    return f"{str(year).zfill(4)}-{str(monthday).zfill(4)}-{str(jyo_cd).zfill(2)}-{str(kaiji).zfill(2)}-{str(nichiji).zfill(2)}-{str(race_num).zfill(2)}"


class OddsLoader:
    """
    オッズデータローダークラス

    JRA-DBのオッズテーブルを読み込み、レースごとのオッズを提供する。
    インデックス化により高速なルックアップを実現。
    """

    DB_DIR = Path("./data/DB")

    # テーブル名とファイル名のマッピング
    ODDS_TABLES = {
        "tanpuku": "n_odds_tanpuku.parquet",
        "waku": "n_odds_waku.parquet",
        "umaren": "n_odds_umaren.parquet",
        "wide": "n_odds_wide.parquet",
        "umatan": "n_odds_umatan.parquet",
        "sanren": "n_odds_sanren.parquet",
        "sanrentan": "n_odds_sanrentan.parquet",
    }

    # 当日用テーブル（s_プレフィックス）
    TODAY_ODDS_TABLES = {
        "tanpuku": "s_odds_tanpuku.parquet",
        "waku": "s_odds_waku.parquet",
        "umaren": "s_odds_umaren.parquet",
        "wide": "s_odds_wide.parquet",
        "umatan": "s_odds_umatan.parquet",
        "sanren": "s_odds_sanren.parquet",
        "sanrentan": "s_odds_sanrentan.parquet",
    }

    def __init__(self, db_dir: Optional[Path] = None) -> None:
        """
        初期化

        Args:
            db_dir: データベースディレクトリ（デフォルト: ./data/DB）
        """
        self.db_dir = Path(db_dir) if db_dir else self.DB_DIR
        self._cache: Dict[str, pd.DataFrame] = {}
        self._indexed_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    def _load_table(
        self, table_name: str, use_today: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        オッズテーブルを読み込む

        Args:
            table_name: テーブル名（tanpuku, waku, umaren, wide, umatan, sanren, sanrentan）
            use_today: 当日データ（s_プレフィックス）を使用するか

        Returns:
            pd.DataFrame or None: オッズデータ
        """
        cache_key = f"{'today_' if use_today else ''}{table_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        tables = self.TODAY_ODDS_TABLES if use_today else self.ODDS_TABLES
        if table_name not in tables:
            return None

        file_path = self.db_dir / tables[table_name]
        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)
            # データ区分の除外（中止/削除）
            if "DataKubun" in df.columns:
                df = df[~df["DataKubun"].astype(str).isin(["0", "9"])]
            self._cache[cache_key] = df
            return df
        except Exception:
            return None

    def _get_indexed_table(
        self, table_name: str, use_today: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        インデックス化されたオッズテーブルを取得（高速ルックアップ用）

        Args:
            table_name: テーブル名
            use_today: 当日データを使用するか

        Returns:
            Dict[str, pd.DataFrame]: {race_key: レースデータ} の辞書
        """
        cache_key = f"{'today_' if use_today else ''}indexed_{table_name}"
        if cache_key in self._indexed_cache:
            return self._indexed_cache[cache_key]

        df = self._load_table(table_name, use_today)
        if df is None:
            return {}

        # レースキーを構築してグループ化
        indexed = {}
        try:
            # 各レースキー列を文字列に変換してゼロパディング
            df = df.copy()
            df["_race_key"] = (
                df["Year"].astype(str).str.zfill(4)
                + "-"
                + df["MonthDay"].astype(str).str.zfill(4)
                + "-"
                + df["JyoCD"].astype(str).str.zfill(2)
                + "-"
                + df["Kaiji"].astype(str).str.zfill(2)
                + "-"
                + df["Nichiji"].astype(str).str.zfill(2)
                + "-"
                + df["RaceNum"].astype(str).str.zfill(2)
            )
            # グループ化してインデックス辞書を作成
            for race_key, group_df in df.groupby("_race_key"):
                indexed[race_key] = group_df.drop(columns=["_race_key"])
        except Exception:
            pass

        self._indexed_cache[cache_key] = indexed
        return indexed

    def _get_race_data(
        self,
        table_name: str,
        year,
        monthday,
        jyo_cd,
        kaiji,
        nichiji,
        race_num,
        use_today: bool = False,
    ) -> pd.DataFrame:
        """
        指定レースのデータを高速に取得

        Args:
            table_name: テーブル名
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            pd.DataFrame: レースデータ（見つからない場合は空のDataFrame）
        """
        indexed = self._get_indexed_table(table_name, use_today)
        race_key = _build_race_key(year, monthday, jyo_cd, kaiji, nichiji, race_num)
        return indexed.get(race_key, pd.DataFrame())

    def get_tansho_odds(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[int, float]:
        """
        指定レースの単勝オッズを取得

        Args:
            year: 開催年
            monthday: 開催月日
            jyo_cd: 競馬場コード
            kaiji: 開催回
            nichiji: 開催日目
            race_num: レース番号
            use_today: 当日データを使用するか

        Returns:
            Dict[int, float]: {馬番: オッズ} の辞書
        """
        race_df = self._get_race_data(
            "tanpuku", year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
        )
        if race_df.empty:
            return {}

        result = {}
        for _, row in race_df.iterrows():
            try:
                umaban = int(row.get("Umaban", 0))
                odds_str = str(row.get("TanOdds", "")).strip()
                # 特殊値を除外
                if odds_str in ["", "----", "****", "0000"]:
                    continue
                odds = float(odds_str) / 10.0  # 999.9形式を実際のオッズに変換
                if umaban > 0 and odds > 0:
                    result[umaban] = odds
            except (ValueError, TypeError):
                continue

        return result

    def get_fukusho_odds(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[int, Tuple[float, float]]:
        """
        指定レースの複勝オッズを取得

        Args:
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            Dict[int, Tuple[float, float]]: {馬番: (最低オッズ, 最高オッズ)} の辞書
        """
        race_df = self._get_race_data(
            "tanpuku", year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
        )
        if race_df.empty:
            return {}

        result = {}
        for _, row in race_df.iterrows():
            try:
                umaban = int(row.get("Umaban", 0))
                odds_low_str = str(row.get("FukuOddsLow", "")).strip()
                odds_high_str = str(row.get("FukuOddsHigh", "")).strip()
                # 特殊値を除外
                if odds_low_str in ["", "----", "****", "0000"]:
                    continue
                if odds_high_str in ["", "----", "****", "0000"]:
                    continue
                odds_low = float(odds_low_str) / 10.0
                odds_high = float(odds_high_str) / 10.0
                if umaban > 0 and odds_low > 0:
                    result[umaban] = (odds_low, odds_high)
            except (ValueError, TypeError):
                continue

        return result

    def get_wakuren_odds(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[Tuple[int, int], float]:
        """
        指定レースの枠連オッズを取得

        Args:
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            Dict[Tuple[int, int], float]: {(枠1, 枠2): オッズ} の辞書
        """
        race_df = self._get_race_data(
            "waku", year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
        )
        if race_df.empty:
            return {}

        result = {}
        for _, row in race_df.iterrows():
            try:
                # Kumi列は4桁（枠1:2桁 + 枠2:2桁）
                kumi = str(row.get("Kumi", "")).strip()
                if len(kumi) < 2:
                    continue
                waku1 = int(kumi[0])
                waku2 = int(kumi[1]) if len(kumi) >= 2 else 0
                odds_str = str(row.get("Odds", "")).strip()
                if odds_str in ["", "----", "****", "0000"]:
                    continue
                odds = float(odds_str) / 10.0
                if waku1 > 0 and waku2 > 0 and odds > 0:
                    key = (min(waku1, waku2), max(waku1, waku2))
                    result[key] = odds
            except (ValueError, TypeError):
                continue

        return result

    def get_umaren_odds(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[Tuple[int, int], float]:
        """
        指定レースの馬連オッズを取得

        Args:
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            Dict[Tuple[int, int], float]: {(馬番1, 馬番2): オッズ} の辞書（順不同）
        """
        race_df = self._get_race_data(
            "umaren", year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
        )
        if race_df.empty:
            return {}

        result = {}
        for _, row in race_df.iterrows():
            try:
                # Kumi列は4桁（馬番1:2桁 + 馬番2:2桁）
                kumi = str(row.get("Kumi", "")).strip().zfill(4)
                if len(kumi) < 4:
                    continue
                uma1 = int(kumi[:2])
                uma2 = int(kumi[2:4])
                odds_str = str(row.get("Odds", "")).strip()
                if odds_str in ["", "----", "****", "0000"]:
                    continue
                odds = float(odds_str) / 10.0
                if uma1 > 0 and uma2 > 0 and odds > 0:
                    key = (min(uma1, uma2), max(uma1, uma2))
                    result[key] = odds
            except (ValueError, TypeError):
                continue

        return result

    def get_wide_odds(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        指定レースのワイドオッズを取得

        Args:
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            Dict[Tuple[int, int], Tuple[float, float]]: {(馬番1, 馬番2): (最低オッズ, 最高オッズ)} の辞書
        """
        race_df = self._get_race_data(
            "wide", year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
        )
        if race_df.empty:
            return {}

        result = {}
        for _, row in race_df.iterrows():
            try:
                kumi = str(row.get("Kumi", "")).strip().zfill(4)
                if len(kumi) < 4:
                    continue
                uma1 = int(kumi[:2])
                uma2 = int(kumi[2:4])
                odds_low_str = str(row.get("OddsLow", "")).strip()
                odds_high_str = str(row.get("OddsHigh", row.get("OddsLow", ""))).strip()
                if odds_low_str in ["", "----", "****", "0000"]:
                    continue
                odds_low = float(odds_low_str) / 10.0
                odds_high = (
                    float(odds_high_str) / 10.0
                    if odds_high_str not in ["", "----", "****", "0000"]
                    else odds_low
                )
                if uma1 > 0 and uma2 > 0 and odds_low > 0:
                    key = (min(uma1, uma2), max(uma1, uma2))
                    result[key] = (odds_low, odds_high)
            except (ValueError, TypeError):
                continue

        return result

    def get_umatan_odds(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[Tuple[int, int], float]:
        """
        指定レースの馬単オッズを取得

        Args:
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            Dict[Tuple[int, int], float]: {(1着馬番, 2着馬番): オッズ} の辞書（順序あり）
        """
        race_df = self._get_race_data(
            "umatan", year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
        )
        if race_df.empty:
            return {}

        result = {}
        for _, row in race_df.iterrows():
            try:
                kumi = str(row.get("Kumi", "")).strip().zfill(4)
                if len(kumi) < 4:
                    continue
                uma1 = int(kumi[:2])  # 1着
                uma2 = int(kumi[2:4])  # 2着
                odds_str = str(row.get("Odds", "")).strip()
                if odds_str in ["", "----", "****", "0000"]:
                    continue
                odds = float(odds_str) / 10.0
                if uma1 > 0 and uma2 > 0 and odds > 0:
                    result[(uma1, uma2)] = odds  # 順序維持
            except (ValueError, TypeError):
                continue

        return result

    def get_sanren_odds(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[Tuple[int, int, int], float]:
        """
        指定レースの三連複オッズを取得

        Args:
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            Dict[Tuple[int, int, int], float]: {(馬番1, 馬番2, 馬番3): オッズ} の辞書（順不同）
        """
        race_df = self._get_race_data(
            "sanren", year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
        )
        if race_df.empty:
            return {}

        result = {}
        for _, row in race_df.iterrows():
            try:
                # Kumi列は6桁（馬番1:2桁 + 馬番2:2桁 + 馬番3:2桁）
                kumi = str(row.get("Kumi", "")).strip().zfill(6)
                if len(kumi) < 6:
                    continue
                uma1 = int(kumi[:2])
                uma2 = int(kumi[2:4])
                uma3 = int(kumi[4:6])
                odds_str = str(row.get("Odds", "")).strip()
                if odds_str in ["", "----", "****", "0000"]:
                    continue
                odds = float(odds_str) / 10.0
                if uma1 > 0 and uma2 > 0 and uma3 > 0 and odds > 0:
                    key = tuple(sorted([uma1, uma2, uma3]))
                    result[key] = odds
            except (ValueError, TypeError):
                continue

        return result

    def get_sanrentan_odds(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[Tuple[int, int, int], float]:
        """
        指定レースの三連単オッズを取得

        Args:
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            Dict[Tuple[int, int, int], float]: {(1着, 2着, 3着): オッズ} の辞書（順序あり）
        """
        race_df = self._get_race_data(
            "sanrentan", year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
        )
        if race_df.empty:
            return {}

        result = {}
        for _, row in race_df.iterrows():
            try:
                kumi = str(row.get("Kumi", "")).strip().zfill(6)
                if len(kumi) < 6:
                    continue
                uma1 = int(kumi[:2])  # 1着
                uma2 = int(kumi[2:4])  # 2着
                uma3 = int(kumi[4:6])  # 3着
                odds_str = str(row.get("Odds", "")).strip()
                if odds_str in ["", "----", "****", "0000"]:
                    continue
                odds = float(odds_str) / 10.0
                if uma1 > 0 and uma2 > 0 and uma3 > 0 and odds > 0:
                    result[(uma1, uma2, uma3)] = odds  # 順序維持
            except (ValueError, TypeError):
                continue

        return result

    def get_race_odds_all(
        self,
        year: str,
        monthday: str,
        jyo_cd: str,
        kaiji: str,
        nichiji: str,
        race_num: str,
        use_today: bool = False,
    ) -> Dict[str, Union[Dict, pd.DataFrame]]:
        """
        指定レースの全オッズを取得

        Args:
            year, monthday, jyo_cd, kaiji, nichiji, race_num: レース識別キー
            use_today: 当日データを使用するか

        Returns:
            Dict: {券種名: オッズデータ} の辞書
        """
        return {
            "tansho": self.get_tansho_odds(
                year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
            ),
            "fukusho": self.get_fukusho_odds(
                year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
            ),
            "wakuren": self.get_wakuren_odds(
                year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
            ),
            "umaren": self.get_umaren_odds(
                year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
            ),
            "wide": self.get_wide_odds(
                year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
            ),
            "umatan": self.get_umatan_odds(
                year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
            ),
            "sanrenpuku": self.get_sanren_odds(
                year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
            ),
            "sanrentan": self.get_sanrentan_odds(
                year, monthday, jyo_cd, kaiji, nichiji, race_num, use_today
            ),
        }

    def build_tansho_odds_for_score_table(
        self, score_table: pd.DataFrame, use_today: bool = False
    ) -> pd.DataFrame:
        """
        スコアテーブルに単勝オッズを結合

        Args:
            score_table: 予測スコアテーブル
            use_today: 当日データを使用するか

        Returns:
            pd.DataFrame: オッズ列が追加されたテーブル
        """
        df = self._load_table("tanpuku", use_today)
        if df is None:
            score_table["TanOdds"] = 0.0
            return score_table

        # キー列を正規化
        for col in RACE_KEY_COLS:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.zfill(4 if col in ["Year", "MonthDay"] else 2)
                )
            if col in score_table.columns:
                score_table[col] = (
                    score_table[col]
                    .astype(str)
                    .str.zfill(4 if col in ["Year", "MonthDay"] else 2)
                )

        # Umabanも正規化
        if "Umaban" in df.columns:
            df["Umaban"] = df["Umaban"].astype(str).str.zfill(2)
        if "Umaban" in score_table.columns:
            score_table["Umaban"] = score_table["Umaban"].astype(str).str.zfill(2)

        # オッズを数値化
        df["TanOdds_num"] = pd.to_numeric(df["TanOdds"], errors="coerce") / 10.0

        # マージ
        merge_cols = RACE_KEY_COLS + ["Umaban"]
        score_table = score_table.merge(
            df[merge_cols + ["TanOdds_num"]], on=merge_cols, how="left"
        )
        score_table["TanOdds"] = score_table["TanOdds_num"].fillna(0.0)
        score_table = score_table.drop(columns=["TanOdds_num"], errors="ignore")

        return score_table

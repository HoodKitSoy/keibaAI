from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class Win5Processor:
    """
    JRA-DB の重勝式（WIN5）データを読み込み、日付ごとの対象5レースと正解組・払戻を提供する。

    入力ファイル（いずれかが存在すれば可）:
      - data/DB/n_jyusyosiki_head.parquet | n_jyushoshiki_head.parquet | n_jyusyosiki_head.parquet
      - data/DB/n_jyusyosiki.parquet      | n_jyushoshiki.parquet      | n_jyusyosiki.parquet
      - data/DB/s_jyusyosiki_head.parquet（当日用）

    生成:
      - schedule: dict[date_key -> list[dict]]  各日の対象5レースの6列キー
      - answer:   dict[date_key -> (winners: List[int], payout: int)]  勝ち馬番5つと払戻金
    """

    DB_DIR = Path("./data/DB")

    CANDIDATE_HEAD_NAMES = [
        "n_jyusyosiki_head.parquet",
        "n_jyushoshiki_head.parquet",
        "n_jyusyosiki_head.parquet",
    ]
    CANDIDATE_BODY_NAMES = [
        "n_jyusyosiki.parquet",
        "n_jyushoshiki.parquet",
        "n_jyusyosiki.parquet",
    ]
    # 当日データ用（s_プレフィックス）
    CANDIDATE_TODAY_HEAD_NAMES = [
        "s_jyusyosiki_head.parquet",
        "s_jyushoshiki_head.parquet",
    ]

    def __init__(self, db_dir: Optional[Path] = None) -> None:
        self.db_dir = Path(db_dir) if db_dir else self.DB_DIR
        self.schedule: Dict[str, List[Dict[str, str]]] = {}
        self.answer: Dict[str, Tuple[List[int], int]] = {}
        self._loaded = False

    def _find_first(self, candidates: List[str]) -> Optional[Path]:
        for name in candidates:
            p = self.db_dir / name
            if p.exists():
                return p
        return None

    @staticmethod
    def _date_key(year: str, monthday: str) -> str:
        y = str(year).zfill(4)
        md = str(monthday).zfill(4)
        return f"{y}{md}"

    def load(self) -> None:
        head_path = self._find_first(self.CANDIDATE_HEAD_NAMES)
        body_path = self._find_first(self.CANDIDATE_BODY_NAMES)
        if head_path is None or body_path is None:
            # どちらか無ければ読み込みスキップ
            self._loaded = False
            return

        head_df = pd.read_parquet(head_path)
        body_df = pd.read_parquet(body_path)

        # データ区分の除外（中止/削除）: ヘッダに DataKubun がある場合
        if "DataKubun" in head_df.columns:
            head_df = head_df[~head_df["DataKubun"].astype(str).isin(["0", "9"])]
        if "DataKubun" in body_df.columns:
            body_df = body_df[~body_df["DataKubun"].astype(str).isin(["0", "9"])]

        # スケジュール構築（各日5レースの6列キー）
        schedule: Dict[str, List[Dict[str, str]]] = {}
        for _, row in head_df.iterrows():
            date_key = self._date_key(row.get("Year", "0"), row.get("MonthDay", "0"))
            races: List[Dict[str, str]] = []
            for i in range(1, 6):
                key = {
                    "Year": str(row.get("Year", "0")).zfill(4),
                    "MonthDay": str(row.get("MonthDay", "0")).zfill(4),
                    "JyoCD": str(row.get(f"JyoCD{i}", "0")).zfill(2),
                    "Kaiji": str(row.get(f"Kaiji{i}", "0")).zfill(2),
                    "Nichiji": str(row.get(f"Nichiji{i}", "0")).zfill(2),
                    "RaceNum": str(row.get(f"RaceNum{i}", "0")).zfill(2),
                }
                races.append(key)
            schedule[date_key] = races

        # 正解と払戻の構築（Kumi: 10桁、各2桁の馬番×5）
        answer: Dict[str, Tuple[List[int], int]] = {}
        for _, row in body_df.iterrows():
            date_key = self._date_key(row.get("Year", "0"), row.get("MonthDay", "0"))
            kumi = str(row.get("Kumi", "")).strip()
            payout = int(str(row.get("PayJyushosiki", "0")).strip() or 0)
            winners: List[int] = []
            if len(kumi) >= 10:
                try:
                    winners = [int(kumi[i : i + 2]) for i in range(0, 10, 2)]
                except ValueError:
                    winners = []
            answer[date_key] = (winners, payout)

        self.schedule = schedule
        self.answer = answer
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def get_dates(self) -> List[str]:
        return sorted(set(self.schedule.keys()) & set(self.answer.keys()))

    def get_day_info(
        self, date_key: str
    ) -> Optional[Tuple[List[Dict[str, str]], List[int], int]]:
        if date_key not in self.schedule or date_key not in self.answer:
            return None
        races = self.schedule[date_key]
        winners, payout = self.answer[date_key]
        return races, winners, payout

    def load_today(self, date_key: Optional[str] = None) -> bool:
        """
        当日WIN5データ（s_プレフィックス）を読み込む

        Args:
            date_key: 対象日（YYYYMMDD形式）。Noneの場合は全日付を読み込み

        Returns:
            bool: 読み込み成功したかどうか
        """
        head_path = self._find_first(self.CANDIDATE_TODAY_HEAD_NAMES)
        if head_path is None:
            return False

        head_df = pd.read_parquet(head_path)

        # データ区分の除外（中止/削除）
        if "DataKubun" in head_df.columns:
            head_df = head_df[~head_df["DataKubun"].astype(str).isin(["0", "9"])]

        # 日付でフィルタリング
        if date_key:
            year = date_key[:4]
            monthday = date_key[4:]
            head_df = head_df[
                (head_df["Year"].astype(str) == year)
                & (head_df["MonthDay"].astype(str).str.zfill(4) == monthday)
            ]

        # スケジュールに追加
        for _, row in head_df.iterrows():
            dk = self._date_key(row.get("Year", "0"), row.get("MonthDay", "0"))
            races: List[Dict[str, str]] = []
            for i in range(1, 6):
                key = {
                    "Year": str(row.get("Year", "0")).zfill(4),
                    "MonthDay": str(row.get("MonthDay", "0")).zfill(4),
                    "JyoCD": str(row.get(f"JyoCD{i}", "0")).zfill(2),
                    "Kaiji": str(row.get(f"Kaiji{i}", "0")).zfill(2),
                    "Nichiji": str(row.get(f"Nichiji{i}", "0")).zfill(2),
                    "RaceNum": str(row.get(f"RaceNum{i}", "0")).zfill(2),
                }
                races.append(key)
            self.schedule[dk] = races

        return len(head_df) > 0

    def get_win5_race_keys(
        self, date_key: str
    ) -> List[Tuple[str, str, str, str, str, str]]:
        """
        指定日のWIN5対象レースのキーリストを取得（発走順序を維持）

        Args:
            date_key: 対象日（YYYYMMDD形式）

        Returns:
            List[Tuple]: (Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum) のリスト
                        ※WIN5の発走順序（1レース目→5レース目）を維持
        """
        if date_key not in self.schedule:
            return []

        race_keys = []
        for race in self.schedule[date_key]:
            race_key = (
                race["Year"],
                race["MonthDay"],
                race["JyoCD"],
                race["Kaiji"],
                race["Nichiji"],
                race["RaceNum"],
            )
            # 重複を避けつつ順序を維持
            if race_key not in race_keys:
                race_keys.append(race_key)
        return race_keys

    def filter_win5_races(self, df: pd.DataFrame, date_key: str) -> pd.DataFrame:
        """
        DataFrameをWIN5対象レースのみにフィルタリング

        Args:
            df: フィルタリング対象のDataFrame
            date_key: 対象日（YYYYMMDD形式）

        Returns:
            pd.DataFrame: WIN5対象レースのみのDataFrame
        """
        race_keys = self.get_win5_race_keys(date_key)
        if not race_keys:
            return pd.DataFrame()

        # 必要なカラムが存在するか確認
        required_cols = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
        if not all(col in df.columns for col in required_cols):
            return df

        # 各行のキーを作成してフィルタリング
        def is_win5_race(row):
            key = (
                str(row["Year"]).zfill(4),
                str(row["MonthDay"]).zfill(4),
                str(row["JyoCD"]).zfill(2),
                str(row["Kaiji"]).zfill(2),
                str(row["Nichiji"]).zfill(2),
                str(row["RaceNum"]).zfill(2),
            )
            return key in race_keys

        mask = df.apply(is_win5_race, axis=1)
        return df[mask]

    def get_win5_race_times(
        self, date_key: str, race_df: pd.DataFrame
    ) -> List[Tuple[Dict[str, str], pd.Timestamp]]:
        """
        WIN5対象レースの発走時刻を取得

        Args:
            date_key: 対象日（YYYYMMDD形式）
            race_df: s_race.parquetから読み込んだDataFrame（HassoTime含む）

        Returns:
            List[Tuple[Dict, Timestamp]]: (レースキー辞書, 発走時刻) のリスト（発走時刻順）
        """
        if date_key not in self.schedule:
            return []

        race_keys = self.schedule[date_key]
        result = []

        for race in race_keys:
            # レースをrace_dfから検索
            mask = (
                (race_df["Year"].astype(str) == race["Year"])
                & (race_df["MonthDay"].astype(str).str.zfill(4) == race["MonthDay"])
                & (race_df["JyoCD"].astype(str).str.zfill(2) == race["JyoCD"])
                & (race_df["Kaiji"].astype(str).str.zfill(2) == race["Kaiji"])
                & (race_df["Nichiji"].astype(str).str.zfill(2) == race["Nichiji"])
                & (race_df["RaceNum"].astype(str).str.zfill(2) == race["RaceNum"])
            )
            matched = race_df[mask]

            if not matched.empty:
                row = matched.iloc[0]
                # HassoTimeをdatetimeに変換
                hasso = str(row.get("HassoTime", "")).zfill(4)
                try:
                    hasso_dt = pd.to_datetime(
                        f"{race['Year']}{race['MonthDay']}{hasso}", format="%Y%m%d%H%M"
                    )
                    result.append((race, hasso_dt))
                except (ValueError, TypeError):
                    pass

        # 発走時刻順でソート
        result.sort(key=lambda x: x[1])
        return result

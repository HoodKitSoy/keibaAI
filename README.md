# keibaAI

JRA-VAN公式DBを使用した競馬予測AIシステム

**バージョン: 1.0**

> **注意**: このシステムは学習・研究用のプロジェクトであり、実際の投資を推奨するものではありません。

## 目次

- [概要](#概要)
- [主な特徴](#主な特徴)
- [環境構築](#環境構築)
- [使い方](#使い方)
  - [1. データ準備（preparing.py）](#1-データ準備preparingpy)
  - [2. データ前処理（preprocessing.py）](#2-データ前処理preprocessingpy)
  - [3. モデル学習（train.py）](#3-モデル学習trainpy)
  - [4. モデル評価（test.py）](#4-モデル評価testpy)
  - [5. 当日予測（predict.py）](#5-当日予測predictpy)
  - [6. 自動予想（auto_predict.py）](#6-自動予想auto_predictpy)
- [環境設定（.env）](#環境設定env)
- [ライセンス・免責事項](#ライセンス免責事項)

---

## 概要

本プロジェクトは、JRA-VAN公式データベース（Parquet形式）を使用した競馬予測AIシステムです。公式の高品質なデータソースと調教データを活用し、高精度な予測モデルの構築を目指します。

---

## 主な特徴

| 特徴 | 説明 |
|:---|:---|
| **公式データ活用** | JRA-VAN公式DBの高品質なデータを使用 |
| **調教データ統合** | 坂路・ウッドチップコースの調教タイムを特徴量として活用 |
| **複数タスク対応** | 1着予測(win)、3着内予測(top3)、着順予測(rank) |
| **WIN5予測機能** | 重勝式(WIN5)の組み合わせ生成と分析 |
| **多様な賭け戦略** | カバレッジ系、期待値系、WIN5ベースなど20種類以上のポリシー |
| **包括的な評価** | 回収率、的中率、ROIなどの詳細シミュレーション |

---

## 環境構築

### 必要な環境

- **OS**: Windows / macOS / Linux
- **Python**: 3.10以上推奨
- **MySQL**: JRA-VAN Data Lab.データベース
- **主要ライブラリ**: pandas, numpy, scikit-learn, lightgbm, optuna

### セットアップ手順

```powershell
# 1. リポジトリのクローン
git clone https://github.com/HoodKitSoy/keibaAI.git
cd keibaAI

# 2. 仮想環境の作成（推奨）
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. 依存パッケージのインストール
pip install -r requirements.txt

# 4. 環境設定ファイルの作成
copy .env.example .env  # サンプルファイルがある場合
notepad .env            # 設定を編集
```

---

## 使い方

本システムは6つのスクリプトで構成されています。基本的な流れは以下の通りです：

```
preparing.py → preprocessing.py → train.py → test.py → predict.py
```

---

### 1. データ準備（preparing.py）

JRA-VAN MySQLデータベースからParquet形式でデータをエクスポートするGUIツールです。

```powershell
python preparing.py
```

**機能:**
- データベース接続設定
- テーブル一覧表示（n_系: 主要データ、s_系: マスタデータ）
- Parquet形式でのエクスポート
- エクスポート状態の管理

**出力先:** `./data/DB/*.parquet`

---

### 2. データ前処理（preprocessing.py）

複数のParquetファイルを結合し、機械学習用の単一テーブルを生成します。

```powershell
# 学習用データの前処理
# .envで PREPROCESSING_MODE=train を設定
python preprocessing.py

# 当日予測用データの前処理
# .envで PREPROCESSING_MODE=today を設定
python preprocessing.py
```

**設定（.env）:**
| 変数名 | 説明 | デフォルト |
|:---|:---|:---|
| `PREPROCESSING_MODE` | 処理モード（train/today） | train |
| `PREPROCESSING_TARGET_DATE` | 対象日（todayモード時、YYYYMMDD形式） | 今日の日付 |
| `PREPROCESSING_TARGET_YEARS` | 対象年（カンマ区切り） | 2016,...,2024 |

**出力先:**
- 学習用: `./data/tmp/db_preprocessed_2016_2024.parquet`
- 当日用: `./data/tmp/today_preprocessed_{YYYYMMDD}.parquet`

---

### 3. モデル学習（train.py）

LightGBMによるモデル学習を実行します。Optunaによるハイパーパラメータ最適化に対応。

```powershell
python train.py
```

**設定（.env）:**
| 変数名 | 説明 | デフォルト |
|:---|:---|:---|
| `TRAIN_TASKS` | 学習タスク（win,top3,rank） | win,top3,rank |
| `LGBM_NUM_BOOST_ROUND` | ブースティングラウンド数 | 2000 |
| `LGBM_N_TRIALS` | Optuna試行回数 | 50 |

**出力先:** `./models/lightgbm/`
- `lgbm_{task}.model` - 学習済みモデル
- `best_params_{task}.json` - 最適パラメータ
- `preds_{task}.parquet` - 検証データの予測結果

---

### 4. モデル評価（test.py）

学習済みモデルの予測結果を使用して、各種賭け戦略のシミュレーションと評価を行います。

```powershell
python test.py
```

**設定（.env）:**
| 変数名 | 説明 | デフォルト |
|:---|:---|:---|
| `TEST_TASKS` | 評価タスク（win,top3,rank） | win |
| `TEST_POLICY` | 使用ポリシー（all で全ポリシー） | all |
| `TEST_WIN5_ENABLED` | WIN5シミュレーション有効化 | true |

**出力先:** `./models/{date}/policy_comparison/{task}/`

**対応ポリシー（一部）:**
- カバレッジ系: BetPolicyCoverage25, BetPolicyCoverage50, BetPolicyCoverage75
- 期待値系: HybridBetPolicy, ExpectedValueBetPolicy
- オッズ系: OddsValueTieredPolicy, OddsBasedExpectedValuePolicy
- WIN5系: WIN5BasedMultiPolicy, WIN5BasedHedgePolicy

---

### 5. 当日予測（predict.py）

当日のレースに対する予測を実行し、ポリシーによる推奨馬券を生成します。

```powershell
# 事前に preprocessing.py を PREPROCESSING_MODE=today で実行
python preprocessing.py

# 予測実行
python predict.py
```

**設定（.env）:**
| 変数名 | 説明 | デフォルト |
|:---|:---|:---|
| `PREDICT_MODE` | 予測モード（today/date） | today |
| `PREDICT_TARGET_DATE` | 対象日（dateモード時） | 今日の日付 |
| `PREDICT_TASKS` | 予測タスク | win,top3,rank |
| `PREDICT_WIN5_ENABLED` | WIN5予測有効化 | true |
| `PREDICT_WIN5_THRESHOLD` | WIN5組み合わせ生成用閾値 | 0.06 |
| `PREDICT_WIN5BASE_THRESHOLD` | WIN5Basedポリシー用閾値 | 0.10 |

**出力先:**
- `./predictions/{date}/` - 予測結果
- `./predictions/{date}/{task}/policy_recommendations/` - ポリシー別推奨

---

### 6. 自動予想（auto_predict.py）

WIN5対象レースの発走時刻に合わせて自動的に予測を実行し、メール送信するシステムです。

```powershell
python auto_predict.py
```

**設定（.env）:**
| 変数名 | 説明 | デフォルト |
|:---|:---|:---|
| `AUTO_PREDICT_EMAIL_TO` | 送信先メールアドレス | - |
| `AUTO_PREDICT_EMAIL_FROM` | 送信元メールアドレス | - |
| `AUTO_PREDICT_EMAIL_PASSWORD` | Gmailアプリパスワード | - |
| `AUTO_PREDICT_WIN5_FIRST_RACE_BEFORE` | 最初のレース何分前に出力 | 30 |

---

## 環境設定（.env）

すべてのスクリプトは`.env`ファイルから設定を読み込みます。

### .envファイルの例

```dotenv
# =====================================
# 競馬AI環境設定ファイル (.env)
# =====================================

# [共通設定]
SEED=42
DEVICE=cuda

# [preparing.py] データベース接続
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=everydb2

# [preprocessing.py] 前処理設定
PREPROCESSING_MODE=train
PREPROCESSING_TARGET_YEARS=2016,2017,2018,2019,2020,2021,2022,2023,2024

# [train.py] 学習設定
TRAIN_TASKS=win,top3,rank
LGBM_N_TRIALS=50
LGBM_NUM_BOOST_ROUND=2000

# [test.py] 評価設定
TEST_TASKS=win
TEST_WIN5_ENABLED=true

# [predict.py] 予測設定
PREDICT_TASKS=win,top3,rank
PREDICT_WIN5_ENABLED=true
PREDICT_WIN5_THRESHOLD=0.06
PREDICT_WIN5BASE_THRESHOLD=0.10

# [auto_predict.py] 自動予想設定
AUTO_PREDICT_EMAIL_TO=your_email@example.com
AUTO_PREDICT_EMAIL_FROM=sender@gmail.com
AUTO_PREDICT_EMAIL_PASSWORD=your_app_password
```

---

## ライセンス・免責事項

### 免責事項

1. **学習・研究目的**: 本システムは学習および研究を目的として開発されています。

2. **投資推奨ではない**: 本システムの予測結果は、実際の馬券購入を推奨するものではありません。

3. **リスクの理解**: 競馬を含むギャンブルには必ず損失のリスクが伴います。

4. **自己責任**: 本システムを使用した結果生じたいかなる損失についても、開発者は一切の責任を負いません。

5. **データの取り扱い**: JRA-VAN Data Lab.のデータ利用規約を遵守してください。

6. **法令遵守**: 日本国内における競馬法、賭博に関する法令を遵守してください。

7. **精度の保証なし**: 本システムの予測精度は保証されません。

**本システムの使用は、上記の免責事項を理解し、同意したものとみなします。**

---

## 開発情報

- **開発者**: HoodKitSoy
- **リポジトリ**: https://github.com/HoodKitSoy/keibaAI
- **バージョン**: 1.0
- **最終更新**: 2026年1月11日

---

**本プロジェクトは継続的に開発されています。詳細な技術仕様はAGENTS.mdを参照してください。**

"""
モデル訓練モジュール

LightGBM + Optunaによるハイパーパラメータ最適化と
モデル訓練のコアロジックを提供します。
"""

from __future__ import annotations

import inspect
import os
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)

import config
from modules.core import RECORD_KEY_COLS, SEED, TARGET_COL

from ._file_utils import ensure_dir, safe_save_df, safe_save_json, safe_save_model


def build_lgb_dataset(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    categorical_cols: List[str],
) -> Tuple[lgb.Dataset, lgb.Dataset]:
    """
    LightGBMのDatasetを構築する

    訓練データと検証データをLightGBMのDataset形式に変換します。
    カテゴリ列のインデックスを適切に設定します。

    Args:
        train_df: 訓練データのDataFrame
        valid_df: 検証データのDataFrame
        feature_cols: 特徴量の列名リスト
        y_train: 訓練データのターゲット
        y_valid: 検証データのターゲット
        categorical_cols: カテゴリ列の列名リスト

    Returns:
        Tuple[lgb.Dataset, lgb.Dataset]: 訓練用・検証用Dataset
    """
    cat_idx = [i for i, c in enumerate(feature_cols) if c in set(categorical_cols)]
    lgb_train = lgb.Dataset(
        train_df[feature_cols],
        label=y_train,
        categorical_feature=cat_idx,
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        valid_df[feature_cols],
        label=y_valid,
        categorical_feature=cat_idx,
        free_raw_data=False,
    )
    return lgb_train, lgb_valid


def suggest_params(trial: optuna.Trial, task: str, num_class: int = None) -> Dict:
    """
    Optunaでハイパーパラメータを提案する

    タスク（win/top3/rank）に応じた適切なパラメータを
    Optunaの試行から提案します。

    Args:
        trial: Optunaの試行オブジェクト
        task: タスク名（'win', 'top3', 'rank'）
        num_class: rankタスクのクラス数（rankの場合のみ必要）

    Returns:
        Dict: LightGBMのパラメータ辞書
    """
    params = {
        "objective": "binary"
        if task in ("win", "top3")
        else ("multiclass" if task == "rank" else "binary"),
        "metric": "auc" if task in ("win", "top3") else "multi_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, step=8),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "verbose": -1,
        "force_row_wise": True,
        # 動的に min_child_samples を変えたいので pre-filter を無効化
        "feature_pre_filter": False,
        "seed": SEED,
        "bagging_seed": SEED,
        "feature_fraction_seed": SEED,
    }
    if task == "rank" and num_class is not None:
        params["num_class"] = num_class
    return params


def evaluate(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    モデルの予測結果を評価する

    タスクに応じた適切な評価指標を計算します。
    - win/top3: ROC-AUC, log loss, accuracy
    - rank: accuracy, log loss, top-3 accuracy

    Args:
        task: タスク名（'win', 'top3', 'rank'）
        y_true: 真のターゲット値
        y_pred: 予測値（win/top3は確率、rankは各クラスの確率）

    Returns:
        Dict: 評価指標の辞書
    """
    if task in ("win", "top3"):
        # y_pred は確率（1クラスの確率）を想定
        proba = y_pred
        auc = (
            roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
        )
        ll = log_loss(y_true, np.vstack([1 - proba, proba]).T, labels=[0, 1])
        acc = accuracy_score(y_true, (proba >= 0.5).astype(int))
        return {"roc_auc": auc, "logloss": ll, "accuracy": acc}
    else:  # rank (multiclass)
        ll = log_loss(y_true, y_pred)
        pred_label = y_pred.argmax(axis=1)
        acc = accuracy_score(y_true, pred_label)
        # Top-3 Accuracy
        topk = 3 if y_pred.shape[1] >= 3 else y_pred.shape[1]
        top3_pred = np.argsort(-y_pred, axis=1)[:, :topk]
        top3_acc = float(np.mean([yt in top3_pred[i] for i, yt in enumerate(y_true)]))
        return {"accuracy": acc, "logloss": ll, "top3_accuracy": top3_acc}


def make_focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """
    Focal Loss関数を生成する

    2値分類用のFocal Loss近似実装を返します。
    参考: Focal Loss for Dense Object Detection (Lin et al.)

    勾配/ヘッシアンは厳密解でなくre-weight近似を使用：
    - g = (p - y) * w
    - h = p * (1 - p) * w

    Args:
        alpha: クラス重みパラメータ（デフォルト: 0.25）
        gamma: フォーカスパラメータ（デフォルト: 2.0）

    Returns:
        callable: LightGBMのfobj関数
    """

    def _focal_obj(preds: np.ndarray, dataset: lgb.Dataset):  # type: ignore
        y_true = dataset.get_label()
        # raw score -> probability
        p = 1.0 / (1.0 + np.exp(-preds))
        # 重み
        w_pos = alpha * (1 - p) ** gamma
        w_neg = (1 - alpha) * (p) ** gamma
        w = np.where(y_true == 1, w_pos, w_neg)
        grad = (p - y_true) * w
        hess = p * (1 - p) * w  # 近似
        return grad, hess

    return _focal_obj


def make_binary_accuracy_metric():
    """
    Binary accuracy カスタムメトリックを生成する

    LightGBMのカスタム評価関数として使用できる
    accuracy指標を返します。

    Returns:
        callable: LightGBMのfeval関数
    """

    def _acc(preds: np.ndarray, dataset: lgb.Dataset):  # type: ignore
        y_true = dataset.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        pred_label = (p >= 0.5).astype(int)
        acc = float((pred_label == y_true).mean())
        return "accuracy", acc, True  # higher is better

    return _acc


def train_one_task(
    task: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    out_dir: str,
    *,
    custom_loss: str = "none",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    dynamic_num_class: int | None = None,
) -> None:
    """
    1つのタスクについてモデルを訓練・評価・保存する

    以下の処理を実行します：
    1. ターゲット変数の作成とデータフィルタリング
    2. 訓練/検証/テストへのデータ分割
    3. Optunaによるハイパーパラメータ最適化
    4. ベストパラメータでの再学習（訓練+検証データ）
    5. テストデータでの評価
    6. モデル・パラメータ・重要度・メトリクス・予測結果の保存

    Args:
        task: タスク名（'win', 'top3', 'rank'）
        df: 入力データのDataFrame
        feature_cols: 特徴量の列名リスト
        categorical_cols: カテゴリ列の列名リスト
        out_dir: 出力ディレクトリ
        custom_loss: カスタム損失関数（'none' または 'focal'）
        focal_alpha: Focal Lossのalphaパラメータ
        focal_gamma: Focal Lossのgammaパラメータ
        dynamic_num_class: rankタスクのクラス数（Noneの場合は自動決定）
    """
    assert task in ("win", "top3", "rank")

    # ターゲット作成
    kj = pd.to_numeric(df[TARGET_COL], errors="coerce")
    # rankのクラス数は動的指定（NoneならデフォルトNUM_CLASSES_RANK）
    from modules.core import NUM_CLASSES_RANK

    rank_num_class = (
        dynamic_num_class if dynamic_num_class is not None else NUM_CLASSES_RANK
    )
    mask_valid = kj.between(1, rank_num_class if task == "rank" else 18)
    if task == "win":
        y = (kj == 1).astype(int)
    elif task == "top3":
        y = (kj <= 3).astype(int)
    else:  # rank
        # LightGBM の multiclass は 0..num_class-1
        y = (kj - 1).astype(int)

    df2 = df.loc[mask_valid].copy()
    y = y.loc[mask_valid].values

    # 年分割
    from ._feature_engineering import split_by_year

    tr, va, te = split_by_year(df2)
    y_tr = y[df2.index.get_indexer(tr.index)]
    y_va = y[df2.index.get_indexer(va.index)]
    y_te = y[df2.index.get_indexer(te.index)]

    # LightGBM Dataset 準備
    lgb_tr, lgb_va = build_lgb_dataset(
        tr, va, feature_cols, y_tr, y_va, categorical_cols
    )

    # LightGBM の fobj サポート可否検出（古いバージョン互換）
    try:
        _supports_fobj = "fobj" in inspect.signature(lgb.train).parameters
    except (ValueError, TypeError):
        _supports_fobj = True  # 取得失敗時は楽観視

    # Optuna で最適化
    num_class = rank_num_class if task == "rank" else None

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, task, num_class)
        use_focal = custom_loss == "focal" and task in ("win", "top3")
        fobj = (
            make_focal_loss(alpha=focal_alpha, gamma=focal_gamma) if use_focal else None
        )
        fevals = []
        if task in ("win", "top3"):
            fevals.append(make_binary_accuracy_metric())

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=0),
        ]
        # fobj がサポートされなければ除去してフォールバック
        try:
            if use_focal and _supports_fobj:
                gbm = lgb.train(
                    params,
                    lgb_tr,
                    num_boost_round=config.NUM_BOOST_ROUND,
                    valid_sets=[lgb_tr, lgb_va],
                    valid_names=["train", "valid"],
                    fobj=fobj,
                    feval=fevals if fevals else None,
                    callbacks=callbacks,
                )
            else:
                # fobj 未対応時は静かに標準 binary objective にフォールバック
                gbm = lgb.train(
                    params,
                    lgb_tr,
                    num_boost_round=config.NUM_BOOST_ROUND,
                    valid_sets=[lgb_tr, lgb_va],
                    valid_names=["train", "valid"],
                    feval=fevals if fevals else None,
                    callbacks=callbacks,
                )
        except TypeError as e:
            if "fobj" in str(e):
                print(
                    "[warn] fobj引数未対応で再トライ: focal lossを無効化して再学習します"
                )
                gbm = lgb.train(
                    params,
                    lgb_tr,
                    num_boost_round=config.NUM_BOOST_ROUND,
                    valid_sets=[lgb_tr, lgb_va],
                    valid_names=["train", "valid"],
                    feval=fevals if fevals else None,
                    callbacks=callbacks,
                )
            else:
                raise
        # 検証スコア
        if task in ("win", "top3"):
            pred_va = gbm.predict(va[feature_cols], num_iteration=gbm.best_iteration)
            score = (
                roc_auc_score(y_va, pred_va)
                if len(np.unique(y_va)) > 1
                else -log_loss(y_va, np.vstack([1 - pred_va, pred_va]).T, labels=[0, 1])
            )
            # maximize 指向のため、AUC が定義不可のケースは -logloss を代替
            return float(score)
        else:
            pred_va = gbm.predict(va[feature_cols], num_iteration=gbm.best_iteration)
            # 多クラスは logloss を最小化
            return float(-log_loss(y_va, pred_va))

    direction = (
        "maximize" if task in ("win", "top3") else "maximize"
    )  # 上のobjectiveはAUC/(-logloss)
    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=config.N_TRIALS)
    best_params = study.best_trial.params

    # ベストパラメタで再学習（train+valid）
    full = pd.concat([tr, va], axis=0)
    y_full = np.concatenate([y_tr, y_va])
    lgb_full = lgb.Dataset(
        full[feature_cols],
        label=y_full,
        categorical_feature=[
            i for i, c in enumerate(feature_cols) if c in set(categorical_cols)
        ],
        free_raw_data=False,
    )
    if task == "rank":
        best_params["objective"] = "multiclass"
        best_params["metric"] = "multi_logloss"
        best_params["num_class"] = num_class
    else:
        best_params["objective"] = "binary"
        best_params["metric"] = "auc"
        if custom_loss == "focal":
            best_params["custom_loss"] = "focal"
            best_params["focal_alpha"] = focal_alpha
            best_params["focal_gamma"] = focal_gamma
        else:
            best_params["custom_loss"] = "none"

    # シードを明示
    best_params["seed"] = SEED
    best_params["bagging_seed"] = SEED
    best_params["feature_fraction_seed"] = SEED
    fobj_final = None
    use_focal_final = custom_loss == "focal" and task in ("win", "top3")
    if use_focal_final:
        fobj_final = make_focal_loss(alpha=focal_alpha, gamma=focal_gamma)
    fevals_final = []
    if task in ("win", "top3"):
        fevals_final.append(make_binary_accuracy_metric())
    # fobj サポート状況に合わせて呼び出し
    try:
        if use_focal_final and _supports_fobj:
            gbm = lgb.train(
                best_params,
                lgb_full,
                num_boost_round=config.NUM_BOOST_ROUND,
                valid_sets=[lgb_full],
                valid_names=["train"],
                fobj=fobj_final,
                feval=fevals_final if fevals_final else None,
                callbacks=[lgb.log_evaluation(period=0)],
            )
        else:
            # 最終学習も fobj 未対応なら静かにフォールバック
            gbm = lgb.train(
                best_params,
                lgb_full,
                num_boost_round=config.NUM_BOOST_ROUND,
                valid_sets=[lgb_full],
                valid_names=["train"],
                feval=fevals_final if fevals_final else None,
                callbacks=[lgb.log_evaluation(period=0)],
            )
    except TypeError as e:
        if "fobj" in str(e):
            print("[warn] fobj未対応で再トライ(最終学習): focal無効")
            gbm = lgb.train(
                best_params,
                lgb_full,
                num_boost_round=config.NUM_BOOST_ROUND,
                valid_sets=[lgb_full],
                valid_names=["train"],
                feval=fevals_final if fevals_final else None,
                callbacks=[lgb.log_evaluation(period=0)],
            )
        else:
            raise

    # fobj 実際に利用できたかのフラグを追加
    best_params["custom_loss_effective"] = bool(use_focal_final and _supports_fobj)

    # テスト予測と評価
    if task in ("win", "top3"):
        pred_te = gbm.predict(te[feature_cols])
        metrics = evaluate(task, y_te, pred_te)
    else:
        pred_te = gbm.predict(te[feature_cols])
        metrics = evaluate(task, y_te, pred_te)
        metrics["num_class"] = num_class

    # 保存物
    ensure_dir(out_dir)
    # モデル
    model_path = os.path.join(out_dir, f"lgbm_{task}.model")
    model_path = safe_save_model(gbm, model_path)
    # ハイパラ
    safe_save_json(best_params, os.path.join(out_dir, f"best_params_{task}.json"))
    # 特徴量情報（予測時に使用）
    feature_info = {
        "feature_names": feature_cols,
        "categorical_features": categorical_cols,
    }
    safe_save_json(feature_info, os.path.join(out_dir, f"feature_info_{task}.json"))
    # 重要度
    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": gbm.feature_importance(importance_type="gain"),
            "split": gbm.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    safe_save_df(fi, os.path.join(out_dir, f"feature_importance_{task}.csv"), fmt="csv")
    # メトリクス
    safe_save_json(metrics, os.path.join(out_dir, f"metrics_{task}.json"))
    # 予測
    key_cols_present = [k for k in RECORD_KEY_COLS if k in te.columns]
    preds_df = te[key_cols_present].copy()
    if task in ("win", "top3"):
        preds_df["proba"] = pred_te
        preds_df["pred_label"] = (pred_te >= 0.5).astype(int)
    else:
        # 各クラス確率（1..num_class を列名に）
        for i in range(pred_te.shape[1]):
            preds_df[f"p{i + 1}"] = pred_te[:, i]
        preds_df["pred_label"] = pred_te.argmax(axis=1) + 1
    safe_save_df(
        preds_df, os.path.join(out_dir, f"preds_{task}.parquet"), fmt="parquet"
    )

    print(f"[Done] {task}: model={model_path}, metrics={metrics}")

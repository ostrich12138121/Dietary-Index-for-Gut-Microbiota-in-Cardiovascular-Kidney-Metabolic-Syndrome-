import os
import re
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    brier_score,
    integrated_brier_score,
)

# =========================
# 参数区
# =========================
FILE_PATH = r"D:\python\food\c1xin.csv"
ENDPOINT = "cv_death"          # "all_cause" 或 "cv_death"
CV_DEATH_LABEL = "heart"

ID_COL = "seqn"
TIME_COL = "time"
STATUS_COL = "status"
CAUSE_COL = "death-cause"

DIGM_GROUP_COL = "DIGMQ group"
DIGM_TREND_COL = "DIGMQ.median"

DESIGN_COLS = ["sdmvpsu", "sdmvstra", "wtsaf2yr", "nhs_wt"]

N_SPLITS = 5
RANDOM_STATE = 2026

# 评价时间点：请按你的 time 单位调整
EVAL_TIMES = np.array([36, 60, 96], dtype=float)

# 与当前 AFT 设定一致
AFT_DIST = "normal"
AFT_SCALE = 1.0

# =========================
# 工具函数
# =========================
def read_csv_auto(path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"文件编码无法识别，最后错误：{last_err}")

def make_event(df, endpoint="all_cause", status_col="status", cause_col="death-cause", cv_label="heart"):
    if endpoint == "all_cause":
        event = (df[status_col] == 1).astype(int)
    elif endpoint == "cv_death":
        event = (
            (df[status_col] == 1) &
            (df[cause_col].astype(str).str.lower() == cv_label.lower())
        ).astype(int)
    else:
        raise ValueError("ENDPOINT 只能是 'all_cause' 或 'cv_death'")
    return event

def safe_colname(x):
    x = str(x)
    x = re.sub(r"[^0-9a-zA-Z_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    if x == "":
        x = "col"
    return x

def get_output_dir():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, "xgb_no_year_metrics_outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def preprocess_fit_transform(X_train_raw, X_valid_raw):
    X_train = X_train_raw.copy()
    X_valid = X_valid_raw.copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    for c in num_cols:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_valid[c] = X_valid[c].fillna(med)

    for c in cat_cols:
        X_train[c] = X_train[c].fillna("Missing").astype(str)
        X_valid[c] = X_valid[c].fillna("Missing").astype(str)

    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
    X_valid = pd.get_dummies(X_valid, columns=cat_cols, drop_first=False)

    X_train.columns = [safe_colname(c) for c in X_train.columns]
    X_valid.columns = [safe_colname(c) for c in X_valid.columns]

    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

    X_train = X_train.astype(float)
    X_valid = X_valid.astype(float)

    return X_train, X_valid

def summarize_metric(df, cols):
    out = {}
    for c in cols:
        out[f"{c}_mean"] = round(float(df[c].mean()), 6)
        out[f"{c}_sd"] = round(float(df[c].std(ddof=1)), 6)
        out[f"{c}_median"] = round(float(df[c].median()), 6)
        out[f"{c}_min"] = round(float(df[c].min()), 6)
        out[f"{c}_max"] = round(float(df[c].max()), 6)
    return out

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    out_dir = get_output_dir()
    print("输出目录：", out_dir)

    df = read_csv_auto(FILE_PATH)
    df["event_ml"] = make_event(df, ENDPOINT, STATUS_COL, CAUSE_COL, CV_DEATH_LABEL)

    exclude_cols = [
        ID_COL, TIME_COL, STATUS_COL, CAUSE_COL,
        DIGM_GROUP_COL, DIGM_TREND_COL
    ] + DESIGN_COLS

    feature_cols = [c for c in df.columns if c not in exclude_cols + ["event_ml"]]
    X_raw = df[feature_cols].copy()

    year_cols = [c for c in X_raw.columns if str(c).startswith("Year")]
    X_raw = X_raw.drop(columns=year_cols, errors="ignore")

    time_all = df[TIME_COL].astype(float).values
    event_all = df["event_ml"].astype(int).values

    print("数据维度：", df.shape)
    print("终点：", ENDPOINT)
    print("事件数：", int(event_all.sum()))
    print("排除 Year 后原始特征数：", X_raw.shape[1])

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_raw, event_all), start=1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

        X_train_raw = X_raw.iloc[train_idx].copy()
        X_valid_raw = X_raw.iloc[valid_idx].copy()

        time_train = time_all[train_idx]
        time_valid = time_all[valid_idx]

        event_train = event_all[train_idx].astype(bool)
        event_valid = event_all[valid_idx].astype(bool)

        X_train, X_valid = preprocess_fit_transform(X_train_raw, X_valid_raw)

        y_train = Surv.from_arrays(event=event_train, time=time_train)
        y_valid = Surv.from_arrays(event=event_valid, time=time_valid)

        y_lower_train = time_train.copy()
        y_upper_train = np.where(event_train, time_train, np.inf)

        y_lower_valid = time_valid.copy()
        y_upper_valid = np.where(event_valid, time_valid, np.inf)

        dtrain = xgb.DMatrix(X_train, feature_names=X_train.columns.tolist())
        dtrain.set_float_info("label_lower_bound", y_lower_train)
        dtrain.set_float_info("label_upper_bound", y_upper_train)

        dvalid = xgb.DMatrix(X_valid, feature_names=X_valid.columns.tolist())
        dvalid.set_float_info("label_lower_bound", y_lower_valid)
        dvalid.set_float_info("label_upper_bound", y_upper_valid)

        xgb_params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 30,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "lambda": 1.0,
            "alpha": 0.0,
            "seed": RANDOM_STATE,
        }

        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        # AFT location parameter（更适合后续转 survival probability）
        mu_valid = model.predict(dvalid, output_margin=True)

        # 1) C-index
        risk_valid = -mu_valid
        cindex = concordance_index_censored(event_valid, time_valid, risk_valid)[0]

        # 2) time-dependent AUC
        auc_vals, mean_auc = cumulative_dynamic_auc(
            y_train, y_valid, risk_valid, EVAL_TIMES
        )

        # 3) Brier score / IBS
        # AFT-normal 下：S(t|x) = 1 - Phi((log t - mu)/sigma)
        log_t = np.log(EVAL_TIMES)[None, :]
        z = (log_t - mu_valid[:, None]) / AFT_SCALE
        surv_mat = 1.0 - norm.cdf(z)
        surv_mat = np.clip(surv_mat, 1e-6, 1.0)

        bs_times, bs_vals = brier_score(
            y_train, y_valid, surv_mat, EVAL_TIMES
        )
        ibs = integrated_brier_score(
            y_train, y_valid, surv_mat, EVAL_TIMES
        )

        row = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "n_train_event": int(event_train.sum()),
            "n_valid_event": int(event_valid.sum()),
            "n_features": X_train.shape[1],
            "best_iteration": int(model.best_iteration),
            "cindex": float(cindex),
            "mean_auc": float(mean_auc),
            "ibs": float(ibs),
        }

        for t, auc in zip(EVAL_TIMES, auc_vals):
            row[f"auc_{int(t)}"] = float(auc)

        for t, bs in zip(bs_times, bs_vals):
            row[f"brier_{int(t)}"] = float(bs)

        fold_results.append(row)

        print("C-index:", round(cindex, 4))
        print("Mean AUC:", round(mean_auc, 4))
        print("IBS:", round(ibs, 4))
        print("best_iteration:", model.best_iteration)

    fold_results_df = pd.DataFrame(fold_results)

    metric_cols = ["cindex", "mean_auc", "ibs"] + \
                  [f"auc_{int(t)}" for t in EVAL_TIMES] + \
                  [f"brier_{int(t)}" for t in EVAL_TIMES]

    summary = {
        "file_path": FILE_PATH,
        "endpoint": ENDPOINT,
        "cv_death_label": CV_DEATH_LABEL,
        "n_total": int(df.shape[0]),
        "n_event": int(event_all.sum()),
        "event_rate_percent": round(float(event_all.mean() * 100), 4),
        "n_features_raw_after_no_year": int(X_raw.shape[1]),
        "year_cols_removed": year_cols,
        "n_splits": N_SPLITS,
        "eval_times": EVAL_TIMES.tolist(),
        "aft_loss_distribution": AFT_DIST,
        "aft_loss_distribution_scale": AFT_SCALE,
    }
    summary.update(summarize_metric(fold_results_df, metric_cols))
    summary["best_iteration_mean"] = round(float(fold_results_df["best_iteration"].mean()), 6)
    summary["best_iteration_sd"] = round(float(fold_results_df["best_iteration"].std(ddof=1)), 6)

    fold_path = os.path.join(out_dir, f"XGBoost_{ENDPOINT}_metrics_5fold.csv")
    summary_path = os.path.join(out_dir, f"XGBoost_{ENDPOINT}_metrics_summary.txt")

    fold_results_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n已保存：")
    print(fold_path)
    print(summary_path)
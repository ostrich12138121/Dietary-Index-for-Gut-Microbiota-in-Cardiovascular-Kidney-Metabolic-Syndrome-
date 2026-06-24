import os
import re
import json
import numpy as np
import pandas as pd
import xgboost as xgb

# ====== Windows 临时目录，避免中文路径报错 ======
os.makedirs(r"D:\temp_joblib", exist_ok=True)
os.environ["TMP"] = r"D:\temp_joblib"
os.environ["TEMP"] = r"D:\temp_joblib"
os.environ["JOBLIB_TEMP_FOLDER"] = r"D:\temp_joblib"

from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored


# =========================================================
# 1. 参数区
# =========================================================
FILE_PATH = r"D:\python\food\c1xin.csv"   # 改成你的实际路径
ENDPOINT = "cv_death"                    # "all_cause" 或 "cv_death"
CV_DEATH_LABEL = "heart"                  # death-cause 中心血管死亡对应值

ID_COL = "seqn"
TIME_COL = "time"
STATUS_COL = "status"
CAUSE_COL = "death-cause"

DIGM_GROUP_COL = "DIGMQ group"
DIGM_TREND_COL = "DIGMQ.median"

DESIGN_COLS = ["sdmvpsu", "sdmvstra", "wtsaf2yr", "nhs_wt"]

N_SPLITS = 5
RANDOM_STATE = 2026


# =========================================================
# 2. 工具函数
# =========================================================
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
    out_dir = os.path.join(base_dir, "xgb_no_year_5fold_cv_outputs")
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


# =========================================================
# 3. 主程序
# =========================================================
if __name__ == "__main__":
    out_dir = get_output_dir()
    print("输出目录：", out_dir)

    df = read_csv_auto(FILE_PATH)
    df["event_ml"] = make_event(
        df,
        endpoint=ENDPOINT,
        status_col=STATUS_COL,
        cause_col=CAUSE_COL,
        cv_label=CV_DEATH_LABEL
    )

    exclude_cols = [
        ID_COL,
        TIME_COL,
        STATUS_COL,
        CAUSE_COL,
        DIGM_GROUP_COL,
        DIGM_TREND_COL,
    ] + DESIGN_COLS

    feature_cols = [c for c in df.columns if c not in exclude_cols + ["event_ml"]]
    X_raw = df[feature_cols].copy()

    # 排除 Year
    year_cols = [c for c in X_raw.columns if str(c).startswith("Year")]
    X_raw = X_raw.drop(columns=year_cols, errors="ignore")

    time_all = df[TIME_COL].astype(float).values
    event_all = df["event_ml"].astype(int).values

    print("数据维度：", df.shape)
    print("终点类型：", ENDPOINT)
    print("事件数：", int(event_all.sum()))
    print("事件率：", round(event_all.mean() * 100, 2), "%")
    print("排除 Year 后原始特征数：", X_raw.shape[1])

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []
    oof_rows = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_raw, event_all), start=1):
        print(f"\n========== Fold {fold} / {N_SPLITS} ==========")

        X_train_raw = X_raw.iloc[train_idx].copy()
        X_valid_raw = X_raw.iloc[valid_idx].copy()

        time_train = time_all[train_idx]
        time_valid = time_all[valid_idx]

        event_train = event_all[train_idx].astype(bool)
        event_valid = event_all[valid_idx].astype(bool)

        X_train, X_valid = preprocess_fit_transform(X_train_raw, X_valid_raw)

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
            # 如果你要试显卡可打开下一行
            # "device": "cuda",
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

        pred_time_valid = model.predict(dvalid)
        risk_valid = -pred_time_valid

        cindex = concordance_index_censored(event_valid, time_valid, risk_valid)[0]
        print("Fold C-index:", round(cindex, 4), "| best_iteration:", model.best_iteration)

        fold_results.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
            "n_train_event": int(event_train.sum()),
            "n_valid_event": int(event_valid.sum()),
            "n_features": X_train.shape[1],
            "best_iteration": int(model.best_iteration),
            "cindex": float(cindex)
        })

        fold_df = pd.DataFrame({
            "fold": fold,
            "row_index": valid_idx,
            "time": time_valid,
            "event": event_valid.astype(int),
            "xgb_predicted_time": pred_time_valid,
            "xgb_risk_score": risk_valid
        })
        oof_rows.append(fold_df)

    fold_results_df = pd.DataFrame(fold_results)
    oof_df = pd.concat(oof_rows, axis=0).sort_values("row_index").reset_index(drop=True)

    mean_cindex = fold_results_df["cindex"].mean()
    std_cindex = fold_results_df["cindex"].std(ddof=1)
    median_cindex = fold_results_df["cindex"].median()
    min_cindex = fold_results_df["cindex"].min()
    max_cindex = fold_results_df["cindex"].max()
    mean_best_iter = fold_results_df["best_iteration"].mean()

    print("\n================ 5-fold CV 结果 ================\n")
    print(f"Mean C-index   : {mean_cindex:.4f}")
    print(f"SD             : {std_cindex:.4f}")
    print(f"Median C-index : {median_cindex:.4f}")
    print(f"Min ~ Max      : {min_cindex:.4f} ~ {max_cindex:.4f}")
    print(f"Mean best iter : {mean_best_iter:.1f}")

    fold_results_path = os.path.join(out_dir, "XGBoost_no_year_5fold_fold_results.csv")
    oof_path = os.path.join(out_dir, "XGBoost_no_year_5fold_oof_predictions.csv")
    feature_path = os.path.join(out_dir, "XGBoost_no_year_5fold_features.txt")
    summary_path = os.path.join(out_dir, "XGBoost_no_year_5fold_summary.txt")

    fold_results_df.to_csv(fold_results_path, index=False, encoding="utf-8-sig")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    with open(feature_path, "w", encoding="utf-8") as f:
        for col in X_raw.columns.tolist():
            f.write(str(col) + "\n")

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
        "mean_cindex": round(float(mean_cindex), 6),
        "sd_cindex": round(float(std_cindex), 6),
        "median_cindex": round(float(median_cindex), 6),
        "min_cindex": round(float(min_cindex), 6),
        "max_cindex": round(float(max_cindex), 6),
        "mean_best_iteration": round(float(mean_best_iter), 2),
        "output_files": {
            "fold_results": fold_results_path,
            "oof_predictions": oof_path,
            "features": feature_path
        }
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n结果文件保存在：", out_dir)
    print("1) XGBoost_no_year_5fold_summary.txt")
    print("2) XGBoost_no_year_5fold_fold_results.csv")
    print("3) XGBoost_no_year_5fold_oof_predictions.csv")
    print("4) XGBoost_no_year_5fold_features.txt")
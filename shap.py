import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# =========================
# 1. 参数区
# =========================
FILE_PATH = r"D:\python\food\c1xin.csv"   # 改成你的实际路径
ENDPOINT = "cv_death"                    # "all_cause" 或 "cv_death"
CV_DEATH_LABEL = "heart"

# 你前面 5-fold 得到的较稳定迭代轮数
BEST_ITER_MAP = {
    "all_cause": 308,
    "cv_death": 290
}

ID_COL = "seqn"
TIME_COL = "time"
STATUS_COL = "status"
CAUSE_COL = "death-cause"

DIGM_GROUP_COL = "DIGMQ group"
DIGM_TREND_COL = "DIGMQ.median"

DESIGN_COLS = ["sdmvpsu", "sdmvstra", "wtsaf2yr", "nhs_wt"]

OUT_DIR = r"D:\python\food\shap_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 为了出图更清楚，可对 SHAP 可视化采样
MAX_PLOT_SAMPLES = 3000
RANDOM_STATE = 2026


# =========================
# 2. 工具函数
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


def preprocess_full_data(df):
    exclude_cols = [
        ID_COL, TIME_COL, STATUS_COL, CAUSE_COL,
        DIGM_GROUP_COL, DIGM_TREND_COL
    ] + DESIGN_COLS

    feature_cols = [c for c in df.columns if c not in exclude_cols + ["event_ml"]]
    X = df[feature_cols].copy()

    # 去掉 Year
    year_cols = [c for c in X.columns if str(c).startswith("Year")]
    X = X.drop(columns=year_cols, errors="ignore")

    # 缺失值处理
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].fillna("Missing").astype(str)

    # one-hot
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    X.columns = [safe_colname(c) for c in X.columns]
    X = X.astype(float)

    return X


# =========================
# 3. 读数据并构造结局
# =========================
df = read_csv_auto(FILE_PATH)
df["event_ml"] = make_event(df, ENDPOINT, STATUS_COL, CAUSE_COL, CV_DEATH_LABEL)

X = preprocess_full_data(df)
time_all = df[TIME_COL].astype(float).values
event_all = df["event_ml"].astype(int).values

print("数据维度：", df.shape)
print("终点：", ENDPOINT)
print("事件数：", int(event_all.sum()))
print("进入模型特征数：", X.shape[1])

# 检查 DIGM 是否存在
if "DIGM" not in X.columns:
    raise ValueError("当前特征矩阵中未找到 DIGM，请检查列名。")

# =========================
# 4. 拟合最终 XGBoost AFT 模型
#    这里用全数据拟合，轮数用你前面 CV 的 best iteration
# =========================
best_iter = BEST_ITER_MAP[ENDPOINT]

y_lower = time_all.copy()
y_upper = np.where(event_all.astype(bool), time_all, np.inf)

dall = xgb.DMatrix(X, feature_names=X.columns.tolist())
dall.set_float_info("label_lower_bound", y_lower)
dall.set_float_info("label_upper_bound", y_upper)

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
    dtrain=dall,
    num_boost_round=best_iter,
    verbose_eval=False
)

print(f"模型已训练完成，num_boost_round = {best_iter}")

# =========================
# 5. 计算 SHAP
# =========================
# 为了画图更快，可以采样
if X.shape[0] > MAX_PLOT_SAMPLES:
    X_plot = X.sample(MAX_PLOT_SAMPLES, random_state=RANDOM_STATE)
else:
    X_plot = X.copy()

explainer = shap.TreeExplainer(model)

# 注意：
# 对于 survival:AFT，TreeExplainer 解释的是模型原始输出。
# 原始输出越大，通常表示预测生存时间越长（风险越低）。
# 为了让“正的 SHAP 值 = 更高风险”更直观，
# 这里将 SHAP 值乘以 -1，构造 risk-oriented SHAP。
shap_values_raw = explainer.shap_values(X_plot)
risk_shap_values = -1 * shap_values_raw

# =========================
# 6. 全局特征重要性
# =========================
mean_abs_shap = np.abs(risk_shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": X_plot.columns,
    "mean_abs_risk_shap": mean_abs_shap
}).sort_values("mean_abs_risk_shap", ascending=False).reset_index(drop=True)

importance_df["rank"] = np.arange(1, len(importance_df) + 1)

importance_path = os.path.join(OUT_DIR, f"SHAP_importance_{ENDPOINT}.csv")
importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

print("已保存特征重要性：", importance_path)

# 打印 DIGM 的排名
digm_row = importance_df[importance_df["feature"] == "DIGM"]
if len(digm_row) > 0:
    digm_rank = int(digm_row["rank"].iloc[0])
    digm_value = float(digm_row["mean_abs_risk_shap"].iloc[0])
    print(f"DIGM 的 SHAP 排名：{digm_rank}")
    print(f"DIGM 的 mean_abs_risk_shap：{digm_value:.6f}")
else:
    print("未在重要性表中找到 DIGM。")

# =========================
# 7. SHAP bar plot
# =========================
plt.figure()
shap.summary_plot(
    risk_shap_values,
    X_plot,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title(f"SHAP feature importance ({ENDPOINT})")
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, f"SHAP_bar_{ENDPOINT}.png")
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()
print("已保存：", bar_path)

# =========================
# 8. SHAP beeswarm plot
# =========================
plt.figure()
shap.summary_plot(
    risk_shap_values,
    X_plot,
    max_display=20,
    show=False
)
plt.title(f"SHAP summary plot ({ENDPOINT})")
plt.tight_layout()
beeswarm_path = os.path.join(OUT_DIR, f"SHAP_beeswarm_{ENDPOINT}.png")
plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
plt.close()
print("已保存：", beeswarm_path)

# =========================
# 9. DIGM dependence plot
# =========================
plt.figure()
shap.dependence_plot(
    "DIGM",
    risk_shap_values,
    X_plot,
    interaction_index=None,
    show=False
)
plt.title(f"DIGM SHAP dependence plot ({ENDPOINT})")
plt.tight_layout()
digm_dep_path = os.path.join(OUT_DIR, f"SHAP_dependence_DIGM_{ENDPOINT}.png")
plt.savefig(digm_dep_path, dpi=300, bbox_inches="tight")
plt.close()
print("已保存：", digm_dep_path)

# =========================
# 10. 导出 DIGM 的 SHAP 值与原值
# =========================
digm_shap_df = pd.DataFrame({
    "DIGM": X_plot["DIGM"].values,
    "DIGM_risk_shap": risk_shap_values[:, list(X_plot.columns).index("DIGM")]
}).sort_values("DIGM")

digm_shap_path = os.path.join(OUT_DIR, f"DIGM_SHAP_values_{ENDPOINT}.csv")
digm_shap_df.to_csv(digm_shap_path, index=False, encoding="utf-8-sig")
print("已保存：", digm_shap_path)

# =========================
# 11. DIGM SHAP 简要统计
# =========================
digm_summary = digm_shap_df["DIGM_risk_shap"].describe(percentiles=[0.25, 0.5, 0.75])
summary_path = os.path.join(OUT_DIR, f"DIGM_SHAP_summary_{ENDPOINT}.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(str(digm_summary))
print("已保存：", summary_path)

print("\n全部完成。")
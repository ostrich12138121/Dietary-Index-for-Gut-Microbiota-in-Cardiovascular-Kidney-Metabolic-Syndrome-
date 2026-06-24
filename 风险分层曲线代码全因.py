import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# =========================
# 路径设置
# =========================
OUT_DIR = r"D:\python\food\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 全因死亡 OOF =====
RSF_ALL_OOF = r"D:\python\food\allcause\RSF_no_year_5fold_oof_predictions.csv"
XGB_ALL_OOF = r"D:\python\food\allcause\XGBoost_no_year_5fold_oof_predictions.csv"


def add_risk_group(df, score_col):
    df = df.copy()
    df["risk_group"] = pd.qcut(df[score_col], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    return df


def plot_km_by_risk(oof_file, time_col, event_col, score_col, title, save_name):
    df = pd.read_csv(oof_file)
    df = add_risk_group(df, score_col)

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(7, 5))
    for g in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        sub = df[df["risk_group"] == g]
        kmf.fit(sub[time_col], sub[event_col], label=f"{g} (n={len(sub)})")
        kmf.plot_survival_function(ci_show=False)

    plt.title(title)
    plt.xlabel("Follow-up time")
    plt.ylabel("Survival probability")
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("已保存：", save_path)


# RSF 全因死亡
plot_km_by_risk(
    RSF_ALL_OOF,
    time_col="time",
    event_col="event",
    score_col="rsf_risk_score",
    title="Risk-stratified KM curves by RSF OOF prediction (All-cause mortality)",
    save_name="Figure_KM_RSF_AllCause.png"
)

# XGBoost 全因死亡
plot_km_by_risk(
    XGB_ALL_OOF,
    time_col="time",
    event_col="event",
    score_col="xgb_risk_score",
    title="Risk-stratified KM curves by XGBoost OOF prediction (All-cause mortality)",
    save_name="Figure_KM_XGB_AllCause.png"
)
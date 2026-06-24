import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# =========================
# 路径设置
# =========================
OUT_DIR = r"D:\python\food\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# 全因死亡 XGBoost OOF
XGB_ALL_OOF = r"D:\python\food\allcause\XGBoost_no_year_5fold_oof_predictions.csv"

# 心血管死亡 XGBoost OOF
XGB_CV_OOF = r"D:\python\food\cvdeath\XGBoost_no_year_5fold_oof_predictions.csv"


# =========================
# 工具函数
# =========================
def add_risk_group(df, score_col):
    df = df.copy()
    # 按预测风险四分位分组：Q1最低风险，Q4最高风险
    df["risk_group"] = pd.qcut(
        df[score_col],
        4,
        labels=[
            "Q1: Lowest predicted risk",
            "Q2",
            "Q3",
            "Q4: Highest predicted risk"
        ]
    )
    return df


def format_p_value(p):
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}"


def plot_km_panel(
    ax,
    df,
    time_col,
    event_col,
    score_col,
    panel_title,
    y_label,
    ylim=None
):
    df = add_risk_group(df, score_col)

    # 计算多组 log-rank 检验
    lr = multivariate_logrank_test(
        event_durations=df[time_col],
        groups=df["risk_group"],
        event_observed=df[event_col]
    )
    p_text = format_p_value(lr.p_value)

    kmf = KaplanMeierFitter()

    order = [
        "Q1: Lowest predicted risk",
        "Q2",
        "Q3",
        "Q4: Highest predicted risk"
    ]

    for g in order:
        sub = df[df["risk_group"] == g]
        label = f"{g} (n={len(sub)})"
        kmf.fit(sub[time_col], sub[event_col], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2.0)

    ax.set_title(panel_title, fontsize=14)
    ax.set_xlabel("Follow-up time", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis="both", labelsize=11)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # 去掉上右边框，更像期刊图
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例
    leg = ax.legend(
        loc="lower left",
        fontsize=10,
        frameon=True,
        title=None
    )
    leg.get_frame().set_alpha(0.9)



# =========================
# 读取数据
# =========================
df_all = pd.read_csv(XGB_ALL_OOF)
df_cv = pd.read_csv(XGB_CV_OOF)

# =========================
# 绘图
# =========================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

# A. 全因死亡
plot_km_panel(
    ax=axes[0],
    df=df_all,
    time_col="time",
    event_col="event",
    score_col="xgb_risk_score",
    panel_title="A. All-cause mortality",
    y_label="Survival probability",
    ylim=(0.40, 1.01)
)

# B. 心血管死亡
# 你当前机器学习定义更接近 cause-specific cardiovascular death
plot_km_panel(
    ax=axes[1],
    df=df_cv,
    time_col="time",
    event_col="event",
    score_col="xgb_risk_score",
    panel_title="B. Cardiovascular mortality",
    y_label="Event-free probability",
    ylim=(0.75, 1.01)
)

plt.tight_layout()

save_path = os.path.join(OUT_DIR, "Figure_XGBoost_OOF_RiskStratified_Combined.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("已保存：", save_path)
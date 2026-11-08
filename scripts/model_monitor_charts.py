"""Draw monitoring charts from every monitoring partition up to the snapshot date.

    python scripts/model_monitor_charts.py --snapshotdate 2025-07-01

Writes PNGs to datamart/gold/model_monitoring/charts/ for the deck.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from utils import model_utils as mu


# reference palette (light chart surface)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
CHART_DIR = mu.MONITORING + "charts/"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def load_monitoring(snapshot_date_str):
    paths = sorted(glob.glob(mu.MONITORING + "monitoring_*.parquet"))
    frames = [pd.read_parquet(p) for p in paths if mu.partition_date(p) <= snapshot_date_str]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["cohort_date"] = pd.to_datetime(df["cohort_date"])
    return df


def short_label(version):
    # credit_model_2024_02_01_xgboost -> "XGB Feb 24"
    date = pd.to_datetime(mu.version_date(version))
    family = "LR" if "_logreg" in version else "XGB"
    return f"{family} {date:%b %y}"


def mark_promotions(ax, snapshot_date_str):
    # a dotted line where each new champion goes live; labels alternate between two heights
    registry = mu.read_registry()
    registry = registry[(registry["promoted"].astype(str) == "True") &
                        (registry["decision_date"] <= snapshot_date_str)].sort_values("decision_date")
    for i, (_, row) in enumerate(registry.iterrows()):
        date = pd.to_datetime(row["decision_date"])
        ax.axvline(date, color=MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(short_label(row["model_version"]), (date, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -10 if i % 2 == 0 else -21), textcoords="offset points",
                    color=TEXT_SECONDARY, fontsize=7)


def legend_below(ax, ncol=2):
    # legend under the plot so it never covers data or threshold labels
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=ncol, fontsize=8)


def format_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))


# Each chart follows the model in production at each date, so it stays readable however often
# the champion changes. Dotted vertical lines mark each promotion.

def plot_gini(ax, perf, snapshot_date_str, min_ratio):
    rows = perf.sort_values("cohort_date")
    ax.plot(rows["cohort_date"], rows["gini"], color=SERIES[0], linewidth=2, marker="o", markersize=5,
            label="Live Gini of the model in production (one cohort)")
    ax.plot(rows["cohort_date"], rows["gini_pooled"], color=SERIES[0], linewidth=1.2, linestyle="-.",
            label="Pooled Gini, latest 3 cohorts of that model (retrain test)")
    ax.step(rows["cohort_date"], rows["baseline_gini"], where="post", color=SERIES[1], linewidth=1.2,
            linestyle="--", label="Its OOT Gini at training (baseline)")
    ax.step(rows["cohort_date"], rows["baseline_gini"] * min_ratio, where="post", color=SERIES[1],
            linewidth=1, linestyle=":", label=f"Retrain line ({min_ratio:.0%} of baseline)")
    mark_promotions(ax, snapshot_date_str)
    ax.set_title("Performance: Gini by application month")
    ax.set_ylabel("Gini (2·AUC − 1)")
    ax.set_xlabel("Application month (labels observed 6 months later)")
    top = max(rows["gini"].max(), rows["baseline_gini"].max())
    bottom = min(rows["gini"].min(), (rows["baseline_gini"] * min_ratio).min())
    ax.set_ylim(bottom - 0.05, top + 0.1)
    legend_below(ax)
    format_dates(ax)


def plot_psi(ax, stab, config, snapshot_date_str):
    rows = stab.sort_values("cohort_date")
    ax.plot(rows["cohort_date"], rows["psi"], color=SERIES[0], linewidth=2, marker="o", markersize=5,
            label="Score PSI of the model in production")
    top = max(rows["psi"].max(), config["monitoring"]["psi_act"])
    if "max_csi" in rows and rows["max_csi"].notna().any():
        ax.plot(rows["cohort_date"], rows["max_csi"], color=SERIES[1], linewidth=1.2, linestyle="--",
                marker="s", markersize=4, label="Largest feature CSI that month")
        top = max(top, rows["max_csi"].max())
    for value, name in [(config["monitoring"]["psi_watch"], "watch 0.10"),
                        (config["monitoring"]["psi_act"], "act 0.25")]:
        ax.axhline(value, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(name, (1, value), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", color=TEXT_SECONDARY, fontsize=8)
    mark_promotions(ax, snapshot_date_str)
    ax.set_title("Stability: score PSI and largest feature CSI vs training OOT")
    ax.set_ylabel("Stability index")
    ax.set_xlabel("Scoring month")
    ax.set_ylim(bottom=0, top=top * 1.35)
    legend_below(ax)
    format_dates(ax)


def plot_calibration(ax, perf, snapshot_date_str):
    rows = perf.sort_values("cohort_date")
    ax.plot(rows["cohort_date"], rows["bad_rate"], color=TEXT_SECONDARY, linewidth=2, marker="o",
            markersize=5, label="Actual default rate")
    ax.plot(rows["cohort_date"], rows["mean_prediction"], color=SERIES[0], linewidth=2, marker="o",
            markersize=5, label="Mean predicted probability (model in production)")
    mark_promotions(ax, snapshot_date_str)
    ax.set_title("Calibration: actual vs predicted default rate")
    ax.set_ylabel("Default rate")
    ax.set_xlabel("Application month")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(bottom=0, top=max(rows["bad_rate"].max(), rows["mean_prediction"].max()) * 1.3)
    legend_below(ax)
    format_dates(ax)


def plot_shadow(ax, perf, shadow, snapshot_date_str):
    # same applicants, pooled live Gini: the champion vs its runner-up vs the latest rejected challenger
    champion = perf.sort_values("cohort_date")
    ax.plot(champion["cohort_date"], champion["gini_pooled"], color=SERIES[0], linewidth=2, marker="o",
            markersize=5, label="Champion (model in production)")
    is_runner_up = shadow["shadow_model"].str.endswith("_runner_up")
    for rows, label, color, style in [
            (shadow[is_runner_up], "Shadow: champion's runner-up (other model family)", SERIES[1], "--"),
            # a different model each time, so markers only: a line would join unrelated models
            (shadow[~is_runner_up], "Shadow: latest rejected challenger", SERIES[2], "none")]:
        if rows.empty:
            continue
        rows = rows.sort_values("cohort_date")
        ax.plot(rows["cohort_date"], rows["gini_pooled"], color=color, linewidth=1.5, linestyle=style,
                marker="s", markersize=4, label=label)
        flagged = rows[rows["shadow_flag"].astype(bool)]
        ax.scatter(flagged["cohort_date"], flagged["gini_pooled"], s=90, facecolors="none",
                   edgecolors=TEXT, zorder=5)
    mark_promotions(ax, snapshot_date_str)
    ax.set_title("Shadow models: pooled live Gini vs the champion")
    ax.set_ylabel("Gini, latest cohorts pooled")
    ax.set_xlabel("Application month (circled = shadow ahead, promotion review)")
    values = pd.concat([perf["gini_pooled"], shadow["gini_pooled"]])
    ax.set_ylim(values.min() - 0.05, values.max() + 0.1)
    legend_below(ax, ncol=1)
    format_dates(ax)


def plot_model_lines(ax, perf, shadow, snapshot_date_str):
    """One line per model: solid while it is the champion, dashed while it scores in shadow.

    Every point is that model's Gini on one month's applicants, so where two lines meet in the same
    month they were scored on the same people and can be compared directly.
    """
    lines = [(v, rows, "champion") for v, rows in perf.groupby("model_version")]
    for name, rows in shadow.groupby("shadow_model"):
        lines.append((name, rows, "runner-up" if name.endswith("_runner_up") else "rejected challenger"))
    lines.sort(key=lambda line: (mu.version_date(line[0]), line[2] != "champion"))
    colours = matplotlib.colormaps["viridis"](np.linspace(0, 0.9, len(lines)))

    # the model in production, month by month: a wide grey band under the champion segments
    production = perf.sort_values("cohort_date")
    ax.plot(production["cohort_date"], production["gini"], color=TEXT, linewidth=9, alpha=0.12,
            solid_capstyle="round", zorder=1, label="Model in production (champion each month)")

    for (name, rows, role), colour in zip(lines, colours):
        rows = rows.sort_values("cohort_date")
        champion = role == "champion"
        ax.plot(rows["cohort_date"], rows["gini"], color=colour, linewidth=3 if champion else 1,
                alpha=1 if champion else 0.6, linestyle="-" if champion else "--",
                marker="s" if "_logreg" in name else "o", markersize=7 if champion else 4,
                markerfacecolor=colour if champion else SURFACE, zorder=3 if champion else 2,
                label=f"{short_label(name)} ({role})")

    # winner each month: the highest Gini among every model scored on that month's applicants
    scored = pd.concat([perf[["cohort_date", "gini"]], shadow[["cohort_date", "gini"]]], ignore_index=True)
    best = scored.groupby("cohort_date", as_index=False)["gini"].max()
    ax.scatter(best["cohort_date"], best["gini"] + 0.025, marker="*", s=110, color="#f2b705",
               edgecolors=TEXT, linewidths=0.5, zorder=5, label="Best Gini that month")

    mark_promotions(ax, snapshot_date_str)
    ax.set_title("Live Gini by model: bold = champion, dashed = shadow, star = best")
    ax.set_ylabel("Gini on that month's applicants")
    ax.set_xlabel("Application month (circle = XGBoost, square = logistic regression)")
    values = pd.concat([perf["gini"], shadow["gini"]])
    ax.set_ylim(values.min() - 0.05, values.max() + 0.13)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, fontsize=7)
    format_dates(ax)


def load_feature_drift(snapshot_date_str):
    paths = sorted(glob.glob(mu.FEATURE_DRIFT + "feature_drift_*.parquet"))
    frames = [pd.read_parquet(p) for p in paths if mu.partition_date(p) <= snapshot_date_str]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_feature_drift(ax, drift, config, top_n=12):
    # rows: the features that drifted most at any point; columns: scoring months
    table = drift.pivot_table(index="feature", columns="monitor_date", values="csi")
    table = table.loc[table.max(axis=1).sort_values(ascending=False).index[:top_n]]
    act = config["monitoring"]["psi_act"]
    image = ax.imshow(table.values, aspect="auto", cmap="Oranges", vmin=0, vmax=max(act * 1.5, 0.05))
    ax.set_yticks(range(len(table.index)), table.index, fontsize=8)
    months = pd.to_datetime(table.columns)
    step = max(1, len(months) // 10)
    ax.set_xticks(range(0, len(months), step), [m.strftime("%b %y") for m in months[::step]])
    for (i, j), value in np.ndenumerate(table.values):
        if value >= config["monitoring"]["psi_watch"]:
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if value >= act else TEXT)
    ax.grid(False)
    ax.set_title(f"Feature drift: CSI of the {len(table.index)} most-drifted inputs")
    ax.set_xlabel("Scoring month (values shown at CSI ≥ 0.10)")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("CSI vs training OOT", color=TEXT_SECONDARY)


def main(snapshot_date_str):
    config = mu.load_config()
    df = load_monitoring(snapshot_date_str)
    if df.empty:
        print("no monitoring results yet; skipping charts")
        return

    os.makedirs(CHART_DIR, exist_ok=True)
    perf = df[df["metric_type"] == "performance"]
    stab = df[df["metric_type"] == "stability"]
    shadow = df[df["metric_type"] == "shadow"]

    panels = []
    if not perf.empty:
        panels += [("gini", lambda ax: plot_gini(ax, perf, snapshot_date_str,
                                                    config["monitoring"]["min_gini_ratio"])),
                   ("calibration", lambda ax: plot_calibration(ax, perf, snapshot_date_str))]
    if not stab.empty:
        panels.insert(1 if panels else 0, ("psi", lambda ax: plot_psi(ax, stab, config, snapshot_date_str)))
    if not shadow.empty and not perf.empty:
        panels.append(("shadow", lambda ax: plot_shadow(ax, perf, shadow, snapshot_date_str)))
        panels.append(("model_lines", lambda ax: plot_model_lines(ax, perf, shadow, snapshot_date_str)))
    drift = load_feature_drift(snapshot_date_str)
    if not drift.empty:
        panels.append(("feature_drift", lambda ax: plot_feature_drift(ax, drift, config)))

    # one PNG per chart for slides, plus a combined dashboard
    for name, draw in panels:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        draw(ax)
        fig.tight_layout()
        fig.savefig(CHART_DIR + f"{name}.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 4.4 * len(panels)), squeeze=False)
    for (name, draw), ax in zip(panels, axes[:, 0]):
        draw(ax)
    fig.suptitle(f"Loan default model monitoring (as of {snapshot_date_str})", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(CHART_DIR + "monitoring_dashboard.png", dpi=200)
    plt.close(fig)

    # monitoring table as CSV so readers can check the numbers behind the charts
    df.sort_values(["monitor_date", "metric_type"]).to_csv(CHART_DIR + "monitoring_table.csv", index=False)
    print("saved charts to:", CHART_DIR, [name for name, _ in panels])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="draw monitoring charts")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)

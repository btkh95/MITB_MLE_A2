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


def version_colors(df):
    # colour follows the model version (in training order), never its rank
    versions = sorted(df["model_version"].unique())
    return {v: SERIES[i % len(SERIES)] for i, v in enumerate(versions)}


def label_of(version):
    return version.replace("credit_model_", "model ").replace("_", "-")


def mark_promotions(ax, snapshot_date_str):
    registry = mu.read_registry()
    registry = registry[(registry["promoted"].astype(str) == "True") &
                        (registry["decision_date"] <= snapshot_date_str)]
    for _, row in registry.iterrows():
        date = pd.to_datetime(row["decision_date"])
        ax.axvline(date, color=MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(f"{label_of(row['model_version'])} live", (date, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -10), textcoords="offset points", color=TEXT_SECONDARY, fontsize=8)


def legend_below(ax):
    # legend under the plot so it never covers data or threshold labels
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=8)


def headroom(ax, top):
    # space above the data for the "model live" labels
    ax.set_ylim(top=top)


def format_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))


def plot_gini(ax, perf, colors, snapshot_date_str):
    for version, rows in perf.groupby("model_version"):
        rows = rows.sort_values("cohort_date")
        ax.plot(rows["cohort_date"], rows["gini"], color=colors[version], linewidth=2,
                marker="o", markersize=5, label=f"{label_of(version)} actual Gini")
        ax.axhline(rows["baseline_gini"].iloc[0], color=colors[version], linewidth=1, linestyle="--",
                   alpha=0.6, label=f"{label_of(version)} OOT Gini at training")
    mark_promotions(ax, snapshot_date_str)
    ax.set_title("Performance: Gini by application month")
    ax.set_ylabel("Gini (2·AUC − 1)")
    ax.set_xlabel("Application month (labels observed 6 months later)")
    headroom(ax, max(perf["gini"].max(), perf["baseline_gini"].max()) + 0.05)
    legend_below(ax)
    format_dates(ax)


def plot_psi(ax, stab, colors, config, snapshot_date_str):
    for version, rows in stab.groupby("model_version"):
        rows = rows.sort_values("cohort_date")
        ax.plot(rows["cohort_date"], rows["psi"], color=colors[version], linewidth=2,
                marker="o", markersize=5, label=label_of(version))
    for value, name in [(config["monitoring"]["psi_watch"], "watch 0.10"),
                        (config["monitoring"]["psi_act"], "act 0.25")]:
        ax.axhline(value, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(name, (1, value), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", color=TEXT_SECONDARY, fontsize=8)
    mark_promotions(ax, snapshot_date_str)
    ax.set_title("Stability: PSI of scores vs training OOT")
    ax.set_ylabel("Population stability index")
    ax.set_xlabel("Scoring month")
    ax.set_ylim(bottom=0, top=max(stab["psi"].max(), config["monitoring"]["psi_act"]) * 1.25)
    legend_below(ax)
    format_dates(ax)


def plot_calibration(ax, perf, colors, snapshot_date_str):
    # actual rate is the neutral reference; predictions keep their model-version colour
    rows = perf.sort_values("cohort_date")
    ax.plot(rows["cohort_date"], rows["bad_rate"], color=TEXT_SECONDARY, linewidth=2, marker="o",
            markersize=5, label="Actual default rate")
    for version, group in rows.groupby("model_version"):
        ax.plot(group["cohort_date"], group["mean_prediction"], color=colors[version], linewidth=2,
                marker="o", markersize=5, label=f"{label_of(version)} mean predicted probability")
    mark_promotions(ax, snapshot_date_str)
    ax.set_title("Calibration: actual vs predicted default rate")
    ax.set_ylabel("Default rate")
    ax.set_xlabel("Application month")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(bottom=0, top=max(rows["bad_rate"].max(), rows["mean_prediction"].max()) * 1.2)
    legend_below(ax)
    format_dates(ax)


def main(snapshot_date_str):
    config = mu.load_config()
    df = load_monitoring(snapshot_date_str)
    if df.empty:
        print("no monitoring results yet; skipping charts")
        return

    os.makedirs(CHART_DIR, exist_ok=True)
    colors = version_colors(df)
    perf = df[df["metric_type"] == "performance"]
    stab = df[df["metric_type"] == "stability"]

    panels = []
    if not perf.empty:
        panels += [("gini", lambda ax: plot_gini(ax, perf, colors, snapshot_date_str)),
                   ("calibration", lambda ax: plot_calibration(ax, perf, colors, snapshot_date_str))]
    if not stab.empty:
        panels.insert(1 if panels else 0, ("psi", lambda ax: plot_psi(ax, stab, colors, config, snapshot_date_str)))

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

"""
Reusable Visualization Library.

Provides clean, modern chart functions that the AI agent can call.
All charts use a dark premium theme and save to outputs/temp_chart.png.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# --- Premium Dark Theme Setup ---
COLORS = {
    "bg": "#0E1117",
    "surface": "#1A1D23",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "accent": "#58A6FF",
    "gradient": ["#58A6FF", "#BC8CFF", "#F778BA", "#FF7B72", "#FFA657", "#56D364"],
}

CHART_DIR = "outputs"


def _setup_style():
    """Apply the premium dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["muted"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "grid.color": "#21262D",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })


def _save_chart(fig, filename="temp_chart.png"):
    """Save chart to outputs directory."""
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_distribution(df: pd.DataFrame, column: str, title: str = None) -> str:
    """Generate a histogram with KDE for a numeric column."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    data = df[column].dropna()
    ax.hist(data, bins=30, color=COLORS["accent"], alpha=0.7, edgecolor=COLORS["surface"], linewidth=0.5)

    # KDE overlay
    try:
        from scipy import stats
        kde_x = np.linspace(data.min(), data.max(), 200)
        kde_y = stats.gaussian_kde(data)(kde_x)
        kde_y = kde_y * len(data) * (data.max() - data.min()) / 30  # Scale to histogram
        ax2 = ax.twinx()
        ax2.plot(kde_x, kde_y, color=COLORS["gradient"][1], linewidth=2, alpha=0.8)
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)
    except ImportError:
        pass

    ax.set_title(title or f"Distribution of {column}", pad=15)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)

    return _save_chart(fig)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str = None, top_n: int = None) -> str:
    """Generate a vertical bar chart."""
    _setup_style()

    plot_df = df.copy()
    if top_n:
        plot_df = plot_df.nlargest(top_n, y)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette(COLORS["gradient"][:3], n_colors=len(plot_df))
    bars = ax.bar(range(len(plot_df)), plot_df[y].values, color=colors, edgecolor=COLORS["surface"], linewidth=0.5)

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df[x].values, rotation=45, ha="right")
    ax.set_title(title or f"{y} by {x}", pad=15)
    ax.set_ylabel(y)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:,.0f}', ha='center', va='bottom',
                fontsize=9, color=COLORS["muted"])

    for spine in ax.spines.values():
        spine.set_alpha(0.3)

    fig.tight_layout()
    return _save_chart(fig)


def plot_trend(df: pd.DataFrame, date_col: str, metric_col: str, title: str = None) -> str:
    """Generate a time-series line chart with gradient fill."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_df = df.sort_values(date_col)

    ax.plot(plot_df[date_col], plot_df[metric_col],
            color=COLORS["accent"], linewidth=2.5, zorder=5)
    ax.fill_between(plot_df[date_col], plot_df[metric_col],
                    alpha=0.15, color=COLORS["accent"])

    # Highlight max point
    max_idx = plot_df[metric_col].idxmax()
    max_row = plot_df.loc[max_idx]
    ax.scatter(max_row[date_col], max_row[metric_col],
               color=COLORS["gradient"][5], s=100, zorder=10, edgecolors="white", linewidth=2)
    ax.annotate(f'Peak: {max_row[metric_col]:,.0f}',
                xy=(max_row[date_col], max_row[metric_col]),
                xytext=(10, 15), textcoords="offset points",
                fontsize=10, color=COLORS["gradient"][5],
                arrowprops=dict(arrowstyle="->", color=COLORS["gradient"][5], lw=1.5))

    ax.set_title(title or f"{metric_col} Over Time", pad=15)
    ax.set_xlabel(date_col)
    ax.set_ylabel(metric_col)
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    return _save_chart(fig)


def plot_top_n(df: pd.DataFrame, category_col: str, value_col: str,
               n: int = 10, title: str = None) -> str:
    """Generate a horizontal bar chart showing top N items."""
    _setup_style()

    plot_df = df.nlargest(n, value_col).sort_values(value_col)

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))

    gradient_colors = sns.color_palette("Blues_d", n_colors=len(plot_df))
    bars = ax.barh(range(len(plot_df)), plot_df[value_col].values,
                   color=gradient_colors, edgecolor=COLORS["surface"], linewidth=0.5)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df[category_col].values)
    ax.set_title(title or f"Top {n} {category_col} by {value_col}", pad=15)
    ax.set_xlabel(value_col)
    ax.grid(True, axis="x", alpha=0.3)

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2.,
                f' {width:,.0f}', ha='left', va='center',
                fontsize=9, color=COLORS["muted"])

    for spine in ax.spines.values():
        spine.set_alpha(0.3)

    fig.tight_layout()
    return _save_chart(fig)


def plot_correlation_heatmap(df: pd.DataFrame, title: str = None) -> str:
    """Generate a correlation heatmap for numeric columns."""
    _setup_style()

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        # Fallback: create a simple info chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation",
                ha="center", va="center", fontsize=14, color=COLORS["muted"])
        ax.set_axis_off()
        return _save_chart(fig)

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(corr.columns)), max(6, len(corr.columns) * 0.8)))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                square=True, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                annot_kws={"size": 10, "color": COLORS["text"]})

    ax.set_title(title or "Correlation Heatmap", pad=15)

    fig.tight_layout()
    return _save_chart(fig)


def plot_pie(df: pd.DataFrame, category_col: str, value_col: str,
             title: str = None, top_n: int = 8) -> str:
    """Generate a pie/donut chart for categorical proportions."""
    _setup_style()

    plot_df = df.nlargest(top_n, value_col)
    remaining = df[~df.index.isin(plot_df.index)][value_col].sum()
    if remaining > 0:
        other_row = pd.DataFrame({category_col: ["Other"], value_col: [remaining]})
        plot_df = pd.concat([plot_df, other_row], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette(COLORS["gradient"], n_colors=len(plot_df))
    wedges, texts, autotexts = ax.pie(
        plot_df[value_col], labels=plot_df[category_col],
        colors=colors, autopct="%1.1f%%", startangle=90,
        pctdistance=0.85, wedgeprops=dict(width=0.4, edgecolor=COLORS["bg"], linewidth=2)
    )

    for t in texts:
        t.set_color(COLORS["text"])
        t.set_fontsize(10)
    for t in autotexts:
        t.set_color(COLORS["text"])
        t.set_fontsize(9)

    ax.set_title(title or f"{value_col} by {category_col}", pad=20)

    fig.tight_layout()
    return _save_chart(fig)

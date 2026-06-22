# %% [markdown]
# # Cambridge, UK – Daily Temperature vs Climatology
#
# This script reproduces the style of ERA5 temperature charts showing:
# - Grey shading: 5th–95th percentile range across a 30-year climatology
# - Black line: climatological daily mean (smoothed)
# - Red fill: days warmer than average
# - Blue fill: days cooler than average
# - Dashed vertical line: today
# - Translucent red/blue fill: 16-day forecast
#
# Data source: Open-Meteo API (uses ERA5 reanalysis for historical, ECMWF for forecast)
# No API key required.
#
# To install dependencies, run in your terminal:
#   pip install requests pandas numpy matplotlib scipy
#
# To run in VS Code: open this file, use "Run Cell" (▶) on each # %% block
# To run in Jupyter:  copy each cell block into a notebook cell


# %% [markdown]
# ## 1. Imports & Configuration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import requests
from datetime import date
from scipy.ndimage import uniform_filter1d
import warnings
import os

warnings.filterwarnings("ignore")

# ── Location ──────────────────────────────────────────────────────────────────
LAT          = 52.2053   # Cambridge, UK
LON          = 0.1218
LOCATION     = "Cambridge, UK"

# ── Climatology period ────────────────────────────────────────────────────────
CLIM_START   = 1991      # WMO standard 1991–2020 period
CLIM_END     = 2020

# ── Current year to plot ──────────────────────────────────────────────────────
CURRENT_YEAR = 2026
TODAY        = date.today()

# ── Smoothing window for climatology lines (days) ─────────────────────────────
SMOOTH_WIN   = 21        # Larger = smoother curve; try 15–31

# ── Cache directory: avoid re-downloading 30 years of data every run ──────────
CACHE_DIR    = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Location  : {LOCATION} ({LAT}°N, {LON}°E)")
print(f"Clim. period: {CLIM_START}–{CLIM_END}")
print(f"Plotting  : {CURRENT_YEAR}")
print(f"Today     : {TODAY}")


# %% [markdown]
# ## 2. Data Download Functions
#
# We use two Open-Meteo endpoints:
# - **archive-api** → historical ERA5 data (past dates)
# - **api.open-meteo.com** → 16-day ECMWF forecast (future dates)

# %%
def fetch_historical(lat, lon, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily maximum 2 m temperature from the Open-Meteo archive API.

    Parameters
    ----------
    lat, lon       : float  – coordinates
    start_date     : str    – "YYYY-MM-DD"
    end_date       : str    – "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame with columns ['date', 'tmax']
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"     : lat,
        "longitude"    : lon,
        "start_date"   : start_date,
        "end_date"     : end_date,
        "daily"        : "temperature_2m_max",
        "timezone"     : "Europe/London",
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]),
                         "tmax": d["temperature_2m_max"]})


def fetch_forecast(lat, lon, forecast_days: int = 16) -> pd.DataFrame:
    """
    Download ECMWF daily maximum temperature forecast from Open-Meteo.

    Returns
    -------
    pd.DataFrame with columns ['date', 'tmax', 'is_forecast']
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"     : lat,
        "longitude"    : lon,
        "daily"        : "temperature_2m_max",
        "timezone"     : "Europe/London",
        "forecast_days": forecast_days,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"]),
                       "tmax": d["temperature_2m_max"],
                       "is_forecast": True})
    return df


# %% [markdown]
# ## 3. Download Data
#
# The 30-year download is large (~11 000 daily rows).  We cache it to a CSV
# so subsequent runs are instant.

# %%
# ── Climatology period ────────────────────────────────────────────────────────
clim_cache = os.path.join(CACHE_DIR, f"clim_{CLIM_START}_{CLIM_END}.csv")

if os.path.exists(clim_cache):
    print(f"Loading climatology from cache: {clim_cache}")
    clim_raw = pd.read_csv(clim_cache, parse_dates=["date"])
else:
    print(f"Downloading {CLIM_START}–{CLIM_END} climatology data (may take ~30 s)…")
    clim_raw = fetch_historical(LAT, LON,
                                f"{CLIM_START}-01-01",
                                f"{CLIM_END}-12-31")
    clim_raw.to_csv(clim_cache, index=False)
    print(f"  Saved to {clim_cache}")

print(f"  Climatology rows: {len(clim_raw)}")

# ── Current-year observations ─────────────────────────────────────────────────
# ERA5 has a ~5-day latency, so we stop 5 days before today to be safe
obs_end   = (pd.Timestamp(TODAY) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
obs_start = f"{CURRENT_YEAR}-01-01"

print(f"\nDownloading {CURRENT_YEAR} observations ({obs_start} → {obs_end})…")
curr_obs = fetch_historical(LAT, LON, obs_start, obs_end)
curr_obs["is_forecast"] = False
print(f"  Observation rows: {len(curr_obs)}")

# ── 16-day forecast ───────────────────────────────────────────────────────────
print("\nDownloading 16-day forecast…")
curr_fcast = fetch_forecast(LAT, LON, forecast_days=16)
# Keep only future dates (no overlap with observations)
curr_fcast = curr_fcast[curr_fcast["date"] > pd.to_datetime(obs_end)]
print(f"  Forecast rows: {len(curr_fcast)}")

# ── Combine into one DataFrame ────────────────────────────────────────────────
curr_all = (
    pd.concat([curr_obs, curr_fcast], ignore_index=True)
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)
print(f"\nTotal current-year rows: {len(curr_all)}")


# %% [markdown]
# ## 4. Compute 30-Year Climatology
#
# Key steps:
# 1. Group by calendar day (MM-DD) to get the same day across all 30 years
# 2. Remove Feb 29 – it would have only ~8 values and break leap-year indexing
# 3. Compute mean, 5th percentile, 95th percentile at each calendar day
# 4. Apply **circular** (wrap-around) smoothing so the curve doesn't have
#    hard edges at Jan 1 / Dec 31

# %%
def compute_climatology(df: pd.DataFrame, smooth_window: int = 21) -> pd.DataFrame:
    """
    Compute smoothed daily climatology statistics.

    Parameters
    ----------
    df            : DataFrame with 'date' and 'tmax' columns
    smooth_window : number of days for circular smoothing

    Returns
    -------
    DataFrame indexed by 'mmdd' (e.g. '01-01' … '12-31') with columns:
        clim_mean, p05, p95
    """
    df = df.copy()
    df["mmdd"] = df["date"].dt.strftime("%m-%d")
    df = df[df["mmdd"] != "02-29"]          # drop leap-day rows

    # Aggregate: one row per calendar day
    stats = df.groupby("mmdd")["tmax"].agg(
        clim_mean = lambda x: np.nanmean(x),
        p05       = lambda x: np.nanpercentile(x, 5),
        p95       = lambda x: np.nanpercentile(x, 95),
    ).reset_index()

    # Sort rows by calendar date using a fixed non-leap reference year
    ref = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    ref = ref[ref.strftime("%m-%d") != "02-29"]
    stats = stats.set_index("mmdd").reindex(ref.strftime("%m-%d")).reset_index()

    # ── Circular smoothing ────────────────────────────────────────────────────
    # Pad the array with values from the other end to avoid Jan/Dec edge artefacts
    pad = smooth_window
    for col in ["clim_mean", "p05", "p95"]:
        vals   = stats[col].values.astype(float)
        padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
        stats[col] = uniform_filter1d(padded, size=smooth_window)[pad:-pad]

    return stats


clim_stats = compute_climatology(clim_raw, smooth_window=SMOOTH_WIN)
print(f"Climatology computed: {len(clim_stats)} calendar days")
print(clim_stats.head())


# %% [markdown]
# ## 5. Merge & Compute Anomaly

# %%
# Add MM-DD key to current year, then merge climatology onto it
curr_all["mmdd"] = curr_all["date"].dt.strftime("%m-%d")
curr_all = curr_all[curr_all["mmdd"] != "02-29"]

curr = curr_all.merge(
    clim_stats[["mmdd", "clim_mean", "p05", "p95"]],
    on="mmdd", how="left"
).sort_values("date").reset_index(drop=True)

# The daily anomaly determines whether a day is red or blue
curr["anomaly"]      = curr["tmax"] - curr["clim_mean"]
curr["is_forecast"]  = curr["is_forecast"].fillna(False)

print(curr[["date", "tmax", "clim_mean", "anomaly", "is_forecast"]].tail(10))


# %% [markdown]
# ## 6. Plot
#
# The plot has five layers (bottom → top):
# 1. **Grey band** – 5th–95th percentile range
# 2. **Red fill** – temperature above climatological mean
# 3. **Blue fill** – temperature below climatological mean
# 4. **Black line** – climatological mean
# 5. **TODAY dashed line** + annotations

# %%
def plot_temperature_climatology(
    curr: pd.DataFrame,
    clim_start: int,
    clim_end:   int,
    location:   str,
    today:      date,
    year:       int,
    date_range: tuple = None,   # e.g. ("2026-03-01", "2026-07-01") or None for full year
    figsize:    tuple = (14, 7),
) -> tuple:
    """
    Build the full temperature-vs-climatology figure.

    Parameters
    ----------
    curr       : merged DataFrame (date, tmax, clim_mean, p05, p95, is_forecast)
    clim_start, clim_end : int – climatology period labels
    location   : str – city name for title
    today      : date object – used for TODAY line
    year       : int – year being plotted
    date_range : (str, str) or None – restrict x-axis to a window
    figsize    : tuple

    Returns
    -------
    fig, ax
    """
    # ── Optionally restrict date window ──────────────────────────────────────
    df = curr.copy()
    if date_range:
        mask = (df["date"] >= pd.to_datetime(date_range[0])) & \
               (df["date"] <= pd.to_datetime(date_range[1]))
        df = df[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 1: Percentile shading (5th–95th)
    # ─────────────────────────────────────────────────────────────────────────
    ax.fill_between(df["date"], df["p05"], df["p95"],
                    color="#D0D0D0", alpha=0.95, zorder=1,
                    label=f"5–95th percentile ({clim_start}–{clim_end})")

    # ─────────────────────────────────────────────────────────────────────────
    # Layers 2 & 3: Above / below average fill
    #
    # fill_between with where= fills the region between two y-arrays.
    # interpolate=True ensures the fill crosses the climatology line cleanly
    # (without it you sometimes get awkward gaps at the crossing point).
    #
    # We do this separately for observed data (solid colour) and
    # forecast data (lighter/translucent) so you can visually distinguish them.
    # ─────────────────────────────────────────────────────────────────────────
    observed = df[~df["is_forecast"]]
    forecast = df[df["is_forecast"]]

    for subset, alpha_val in [(observed, 0.85), (forecast, 0.45)]:
        if subset.empty:
            continue

        # Above-average: fill from clim_mean up to tmax (red)
        ax.fill_between(subset["date"],
                        subset["clim_mean"],
                        subset["tmax"],
                        where=subset["tmax"] >= subset["clim_mean"],
                        color="#C0392B",
                        alpha=alpha_val,
                        interpolate=True,
                        zorder=2)

        # Below-average: fill from tmax up to clim_mean (blue)
        ax.fill_between(subset["date"],
                        subset["clim_mean"],
                        subset["tmax"],
                        where=subset["tmax"] < subset["clim_mean"],
                        color="#2980B9",
                        alpha=alpha_val,
                        interpolate=True,
                        zorder=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 4: Climatological mean line
    # ─────────────────────────────────────────────────────────────────────────
    ax.plot(df["date"], df["clim_mean"],
            color="black", linewidth=1.5, zorder=3,
            label=f"Climatology mean ({clim_start}–{clim_end})")

    # ─────────────────────────────────────────────────────────────────────────
    # Layer 5: TODAY vertical dashed line
    # ─────────────────────────────────────────────────────────────────────────
    today_ts = pd.Timestamp(today)
    if df["date"].min() <= today_ts <= df["date"].max():
        ax.axvline(today_ts, color="#555555", linestyle="--",
                   linewidth=1.0, zorder=4, alpha=0.8)
        ylims = ax.get_ylim()
        ax.text(today_ts + pd.Timedelta(hours=10),
                ylims[0] + 0.5,
                "TODAY",
                rotation=90, va="bottom", ha="left",
                fontsize=8, color="#555555", zorder=5)

    # ─────────────────────────────────────────────────────────────────────────
    # X-axis: month labels + weekly minor ticks
    # ─────────────────────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))   # e.g. "Mar 8"
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0)) # weekly minor ticks
    plt.setp(ax.xaxis.get_majorticklabels(), ha="center", fontsize=10)

    # ─────────────────────────────────────────────────────────────────────────
    # Y-axis
    # ─────────────────────────────────────────────────────────────────────────
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax.set_ylabel("Maximum Temperature [°C]", fontsize=11)

    # ─────────────────────────────────────────────────────────────────────────
    # Grid & spines
    # ─────────────────────────────────────────────────────────────────────────
    ax.grid(axis="y", which="major", alpha=0.35, linewidth=0.7, color="#AAAAAA")
    ax.grid(axis="x", which="minor", alpha=0.20, linewidth=0.5, color="#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ─────────────────────────────────────────────────────────────────────────
    # Axis limits
    # ─────────────────────────────────────────────────────────────────────────
    ax.set_xlim(df["date"].min() - pd.Timedelta(days=1),
                df["date"].max() + pd.Timedelta(days=1))

    # ─────────────────────────────────────────────────────────────────────────
    # Legend (manual, to avoid duplicate entries from the two fill_between loops)
    # ─────────────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#D0D0D0", label=f"5–95th percentile ({clim_start}–{clim_end})"),
        plt.Line2D([0], [0], color="black", linewidth=1.5,
                   label=f"Climatology ({clim_start}–{clim_end})"),
        mpatches.Patch(color="#C0392B", label="Above average"),
        mpatches.Patch(color="#2980B9", label="Below average"),
        mpatches.Patch(color="#C0392B", alpha=0.45, label="Above average (forecast)"),
        mpatches.Patch(color="#2980B9", alpha=0.45, label="Below average (forecast)"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper left", fontsize=8.5,
              framealpha=0.92, edgecolor="#BBBBBB",
              ncol=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Title & metadata text
    # ─────────────────────────────────────────────────────────────────────────
    ax.set_title(
        f"{location}  |  Daily Maximum Temperature {year} vs Climatology",
        fontsize=13, fontweight="bold", pad=12, loc="left"
    )
    ax.text(0.0, 1.01,
            f"Model: ERA5  |  Climatology: {clim_start}–{clim_end}  |  Source: Open-Meteo",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="#666666")

    plt.tight_layout()
    return fig, ax


# ── Full-year plot ────────────────────────────────────────────────────────────
fig_full, ax_full = plot_temperature_climatology(
    curr, CLIM_START, CLIM_END, LOCATION, TODAY, CURRENT_YEAR,
    date_range=None,
    figsize=(15, 6),
)
plt.savefig("cambridge_temp_full_year.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: cambridge_temp_full_year.png")


# %% [markdown]
# ## 7. (Optional) Plot a Specific Date Window
#
# To zoom in on a season (like the reference image Mar–Jun), just pass
# a `date_range` tuple.

# %%
# ── Spring window: same style as the reference figure ────────────────────────
fig_spring, ax_spring = plot_temperature_climatology(
    curr, CLIM_START, CLIM_END, LOCATION, TODAY, CURRENT_YEAR,
    date_range=(f"{CURRENT_YEAR}-03-01", f"{CURRENT_YEAR}-07-15"),
    figsize=(13, 6),
)
plt.savefig("cambridge_temp_spring.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: cambridge_temp_spring.png")


# %% [markdown]
# ## 8. Bonus – Inspect the Climatology Table
#
# Useful for debugging or understanding the underlying numbers.

# %%
# Show the climatology for the days in the current dataset
check = curr[["date", "mmdd", "tmax", "clim_mean", "p05", "p95", "anomaly"]].copy()
check["anomaly"] = check["anomaly"].round(1)
print(check.to_string(index=False))


# %% [markdown]
# ## Extending This Script
#
# | What you might want                      | How to do it                                         |
# |------------------------------------------|------------------------------------------------------|
# | Different location                       | Change `LAT`, `LON`, `LOCATION` at the top           |
# | Different base period (e.g. 1981–2010)   | Change `CLIM_START` / `CLIM_END`                     |
# | Plot minimum or mean temperature         | Change `"temperature_2m_max"` → `_min` or `_mean`    |
# | Smoother / rougher climatology           | Increase / decrease `SMOOTH_WIN`                     |
# | Add heatwave labels like the reference   | Use `ax.annotate(…)` on peak anomaly dates            |
# | ERA5 directly (instead of Open-Meteo)    | Use the `cdsapi` library + Copernicus CDS account    |
# | Save data to NetCDF/xarray               | `pip install xarray netcdf4` and wrap output          |

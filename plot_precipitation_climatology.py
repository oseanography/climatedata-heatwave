# %% [markdown]
# # Daily Precipitation vs Climatology
#
# Mirrors the temperature climatology plot but for precipitation:
# - Grey band   : 5th–95th percentile range of daily precipitation
# - Black line  : smoothed climatological daily mean
# - Red fill    : above-average precipitation  (flood / wet signal)
# - Blue fill   : below-average precipitation  (drought / dry signal)
# - Translucent : forecast period
#
# A second optional cell plots a 30-day rolling total — this smooths out
# day-to-day noise and makes sustained flood or drought signals much clearer.
#
# Install dependencies (run once in your terminal):
#   pip install requests pandas numpy matplotlib scipy


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

# ── Locations ─────────────────────────────────────────────────────────────────
LOCATIONS = {
    "cambridge" : {"name": "Cambridge, UK",    "lat": 52.2053, "lon":   0.1218},
    "mainz"     : {"name": "Mainz, Germany",   "lat": 49.9929, "lon":   8.2473},
    "madrid"    : {"name": "Madrid, Spain",    "lat": 40.4168, "lon":  -3.7038},
    "hong_kong" : {"name": "Hong Kong",        "lat": 22.3193, "lon": 114.1694},
    "xiamen"    : {"name": "Xiamen, China",    "lat": 24.4798, "lon": 118.0894},
    "tokyo"     : {"name": "Tokyo, Japan",     "lat": 35.6762, "lon": 139.6503},
    "boston"    : {"name": "Boston, USA",      "lat": 42.3601, "lon": -71.0589},
    "london"    : {"name": "London, UK",       "lat": 51.5074, "lon":  -0.1278},
    "paris"     : {"name": "Paris, France",    "lat": 48.8566, "lon":   2.3522},
}

# ── Active location ───────────────────────────────────────────────────────────
ACTIVE = "madrid"   # ← change to any key above

LAT      = LOCATIONS[ACTIVE]["lat"]
LON      = LOCATIONS[ACTIVE]["lon"]
LOCATION = LOCATIONS[ACTIVE]["name"]

# ── Climatology period ────────────────────────────────────────────────────────
CLIM_START = 1991
CLIM_END   = 2020

# ── Current year ──────────────────────────────────────────────────────────────
CURRENT_YEAR = 2026
TODAY        = date.today()

# ── Smoothing window for climatology lines (days) ─────────────────────────────
# Precipitation is much noisier than temperature so a wider window helps
SMOOTH_WIN = 29

# ── Rolling window for optional second plot (days) ────────────────────────────
ROLL_WIN = 30

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Location     : {LOCATION} ({LAT}°N, {LON}°E)")
print(f"Clim. period : {CLIM_START}–{CLIM_END}")
print(f"Plotting     : {CURRENT_YEAR}")
print(f"Today        : {TODAY}")


# %% [markdown]
# ## 2. Download Functions

# %%
def fetch_historical_precip(lat, lon, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily precipitation sum from the Open-Meteo archive API (ERA5).

    Returns
    -------
    pd.DataFrame with columns ['date', 'precip']
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"   : lat,
        "longitude"  : lon,
        "start_date" : start_date,
        "end_date"   : end_date,
        "daily"      : "precipitation_sum",
        "timezone"   : "auto",
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date"  : pd.to_datetime(d["time"]),
        "precip": d["precipitation_sum"],
    })
    return df


def fetch_forecast_precip(lat, lon, forecast_days: int = 16,
                          past_days: int = 7) -> pd.DataFrame:
    """
    Download forecast + recent-past precipitation from Open-Meteo (ECMWF IFS).
    past_days fills the ~5-day ERA5 latency gap so there is no hole in the plot.

    Returns
    -------
    pd.DataFrame with columns ['date', 'precip', 'is_forecast']
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"     : lat,
        "longitude"    : lon,
        "daily"        : "precipitation_sum",
        "timezone"     : "auto",
        "forecast_days": forecast_days,
        "past_days"    : past_days,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date"       : pd.to_datetime(d["time"]),
        "precip"     : d["precipitation_sum"],
        "is_forecast": True,
    })
    return df


# %% [markdown]
# ## 3. Download Data

# %%
# ── Climatology period ────────────────────────────────────────────────────────
loc_slug   = LOCATION.replace(", ", "_").replace(" ", "_")
clim_cache = os.path.join(CACHE_DIR, f"precip_clim_{loc_slug}_{CLIM_START}_{CLIM_END}.csv")

if os.path.exists(clim_cache):
    print(f"Loading precipitation climatology from cache: {clim_cache}")
    clim_raw = pd.read_csv(clim_cache, parse_dates=["date"])
else:
    print(f"Downloading {CLIM_START}–{CLIM_END} precipitation data (may take ~30 s)…")
    clim_raw = fetch_historical_precip(LAT, LON,
                                       f"{CLIM_START}-01-01",
                                       f"{CLIM_END}-12-31")
    clim_raw.to_csv(clim_cache, index=False)
    print(f"  Saved to {clim_cache}")

print(f"  Climatology rows: {len(clim_raw)}")

# ── Current-year observations ─────────────────────────────────────────────────
obs_end   = (pd.Timestamp(TODAY) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
obs_start = f"{CURRENT_YEAR}-01-01"

print(f"\nDownloading {CURRENT_YEAR} observations ({obs_start} → {obs_end})…")
curr_obs = fetch_historical_precip(LAT, LON, obs_start, obs_end)
curr_obs["is_forecast"] = False
print(f"  Observation rows: {len(curr_obs)}")

# ── 16-day forecast ───────────────────────────────────────────────────────────
print("\nDownloading 16-day forecast…")
curr_fcast = fetch_forecast_precip(LAT, LON, forecast_days=16, past_days=7)

# Days between ERA5 cutoff and today = NWP analysis, not true forecast
today_ts = pd.Timestamp(TODAY)
curr_fcast["is_forecast"] = curr_fcast["date"] >= today_ts

# Only keep dates after ERA5 observations end
curr_fcast = curr_fcast[curr_fcast["date"] > pd.to_datetime(obs_end)]
print(f"  Forecast rows: {len(curr_fcast)}")

# ── Combine ───────────────────────────────────────────────────────────────────
curr_all = (
    pd.concat([curr_obs, curr_fcast], ignore_index=True)
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)
print(f"\nTotal current-year rows: {len(curr_all)}")


# %% [markdown]
# ## 4. Compute Precipitation Climatology
#
# Same approach as temperature: group by MM-DD, compute mean and percentiles,
# apply circular smoothing.
#
# Note: precipitation distributions are right-skewed (many near-zero days,
# rare very high values), so the 95th percentile will be much higher relative
# to the mean than it is for temperature — this is expected and meaningful.

# %%
def compute_precip_climatology(df: pd.DataFrame,
                                smooth_window: int = 29) -> pd.DataFrame:
    """
    Compute smoothed daily climatology of precipitation.

    Parameters
    ----------
    df            : DataFrame with 'date' and 'precip' columns
    smooth_window : days for circular smoothing

    Returns
    -------
    DataFrame with columns ['mmdd', 'clim_mean', 'p05', 'p95']
    """
    df = df.copy()
    df["mmdd"] = df["date"].dt.strftime("%m-%d")
    df = df[df["mmdd"] != "02-29"]          # remove leap days

    # Fill NaN precipitation as 0 (no rain = 0 mm, not missing)
    df["precip"] = df["precip"].fillna(0.0)

    stats = df.groupby("mmdd")["precip"].agg(
        clim_mean = lambda x: np.nanmean(x),
        p05       = lambda x: np.nanpercentile(x, 5),
        p95       = lambda x: np.nanpercentile(x, 95),
    ).reset_index()

    # Sort chronologically
    stats = stats.sort_values("mmdd").reset_index(drop=True)

    # Circular smoothing (wrap-around to avoid Jan/Dec edge effects)
    pad = smooth_window
    for col in ["clim_mean", "p05", "p95"]:
        vals   = stats[col].values.astype(float)
        padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
        stats[col] = uniform_filter1d(padded, size=smooth_window)[pad:-pad]

    # Precipitation cannot be negative — clamp after smoothing
    stats["clim_mean"] = stats["clim_mean"].clip(lower=0)
    stats["p05"]       = stats["p05"].clip(lower=0)
    stats["p95"]       = stats["p95"].clip(lower=0)

    return stats


clim_stats = compute_precip_climatology(clim_raw, smooth_window=SMOOTH_WIN)
print(f"Climatology computed: {len(clim_stats)} calendar days")


# %% [markdown]
# ## 5. Merge & Compute Anomaly

# %%
curr_all["mmdd"]   = curr_all["date"].dt.strftime("%m-%d")
curr_all           = curr_all[curr_all["mmdd"] != "02-29"]
curr_all["precip"] = curr_all["precip"].fillna(0.0)

curr = curr_all.merge(
    clim_stats[["mmdd", "clim_mean", "p05", "p95"]],
    on="mmdd", how="left"
).sort_values("date").reset_index(drop=True)

curr["anomaly"]     = curr["precip"] - curr["clim_mean"]
curr["is_forecast"] = curr["is_forecast"].fillna(False)

print(curr[["date", "precip", "clim_mean", "anomaly", "is_forecast"]].tail(12).to_string(index=False))


# %% [markdown]
# ## 6. Daily Precipitation Plot
#
# Direct equivalent of the temperature plot:
# - Red fill = wetter than average  (flood / wet signal)
# - Blue fill = drier than average  (drought / dry signal)

# %%
def plot_precip_climatology(
    curr        : pd.DataFrame,
    clim_start  : int,
    clim_end    : int,
    location    : str,
    today       : date,
    year        : int,
    date_range  : tuple = None,
    figsize     : tuple = (15, 6),
) -> tuple:
    """
    Plot daily precipitation against climatological mean and percentile range.

    Parameters
    ----------
    curr       : merged DataFrame (date, precip, clim_mean, p05, p95, is_forecast)
    date_range : (str, str) or None – restrict x-axis, e.g. ("2026-01-01","2026-07-01")
    """
    df = curr.copy()
    if date_range:
        mask = ((df["date"] >= pd.to_datetime(date_range[0])) &
                (df["date"] <= pd.to_datetime(date_range[1])))
        df = df[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # ── Layer 1: Percentile shading ───────────────────────────────────────────
    ax.fill_between(df["date"], df["p05"], df["p95"],
                    color="#D0D0D0", alpha=0.95, zorder=1,
                    label=f"5–95th percentile ({clim_start}–{clim_end})")

    # ── Layers 2 & 3: Above / below average fill ──────────────────────────────
    observed = df[~df["is_forecast"]]
    forecast = df[df["is_forecast"]]

    # Bridge the shading gap at the observed / forecast boundary
    if not observed.empty and not forecast.empty:
        forecast = pd.concat(
            [observed.iloc[[-1]], forecast]
        ).reset_index(drop=True)

    for subset, alpha_val in [(observed, 0.85), (forecast, 0.45)]:
        if subset.empty:
            continue

        # Above average: wetter than normal → red (flood/wet signal)
        ax.fill_between(subset["date"],
                        subset["clim_mean"],
                        subset["precip"],
                        where=subset["precip"] >= subset["clim_mean"],
                        color="#C0392B",
                        alpha=alpha_val,
                        interpolate=True,
                        zorder=2)

        # Below average: drier than normal → blue (drought/dry signal)
        ax.fill_between(subset["date"],
                        subset["clim_mean"],
                        subset["precip"],
                        where=subset["precip"] < subset["clim_mean"],
                        color="#2980B9",
                        alpha=alpha_val,
                        interpolate=True,
                        zorder=2)

    # ── Layer 4: Climatological mean line ─────────────────────────────────────
    ax.plot(df["date"], df["clim_mean"],
            color="black", linewidth=1.5, zorder=3,
            label=f"Climatology mean ({clim_start}–{clim_end})")

    # ── Layer 5: TODAY line ───────────────────────────────────────────────────
    today_ts = pd.Timestamp(today)
    if df["date"].min() <= today_ts <= df["date"].max():
        ax.axvline(today_ts, color="#555555", linestyle="--",
                   linewidth=1.0, zorder=4, alpha=0.8)
        ylims = ax.get_ylim()
        ax.text(today_ts + pd.Timedelta(hours=10),
                ylims[0] + 0.3, "TODAY",
                rotation=90, va="bottom", ha="left",
                fontsize=8, color="#555555", zorder=5)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_ylim(bottom=0)       # precipitation cannot be negative
    ax.set_ylabel("Daily Precipitation [mm]", fontsize=11)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), ha="center", fontsize=10)

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.35, linewidth=0.7, color="#AAAAAA")
    ax.grid(axis="x", which="minor", alpha=0.20, linewidth=0.5, color="#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(df["date"].min() - pd.Timedelta(days=1),
                df["date"].max() + pd.Timedelta(days=1))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#D0D0D0",
                       label=f"5–95th percentile ({clim_start}–{clim_end})"),
        plt.Line2D([0], [0], color="black", linewidth=1.5,
                   label=f"Climatology ({clim_start}–{clim_end})"),
        mpatches.Patch(color="#C0392B", label="Above average (wet / flood risk)"),
        mpatches.Patch(color="#2980B9", label="Below average (dry / drought risk)"),
        mpatches.Patch(color="#C0392B", alpha=0.45,
                       label="Above average – forecast"),
        mpatches.Patch(color="#2980B9", alpha=0.45,
                       label="Below average – forecast"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper left", fontsize=8.5,
              framealpha=0.92, edgecolor="#BBBBBB",
              ncol=2)

    # ── Title & metadata ──────────────────────────────────────────────────────
    ax.set_title(
        f"{location}  |  Daily Precipitation {year} vs Climatology",
        fontsize=13, fontweight="bold", pad=12, loc="left"
    )
    ax.text(0.0, 1.01,
            f"Model: ERA5  |  Climatology: {clim_start}–{clim_end}  |  "
            f"Source: Open-Meteo",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="#666666")

    plt.tight_layout()
    return fig, ax


# ── Full-year plot ────────────────────────────────────────────────────────────
fig_full, ax_full = plot_precip_climatology(
    curr, CLIM_START, CLIM_END, LOCATION, TODAY, CURRENT_YEAR,
    date_range=None, figsize=(15, 6),
)
plt.savefig(f"{ACTIVE}_precip_full_year.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {ACTIVE}_precip_full_year.png")


# %% [markdown]
# ## 7. (Optional) Date-Window Plot
#
# Zoom into a season, e.g. the monsoon season for Asian cities.

# %%
fig_window, ax_window = plot_precip_climatology(
    curr, CLIM_START, CLIM_END, LOCATION, TODAY, CURRENT_YEAR,
    date_range=(f"{CURRENT_YEAR}-01-01", f"{CURRENT_YEAR}-07-15"),
    figsize=(13, 6),
)
plt.savefig(f"{ACTIVE}_precip_window.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {ACTIVE}_precip_window.png")


# %% [markdown]
# ## 8. (Optional) 30-Day Rolling Total Plot
#
# Daily precipitation is inherently noisy — a single heavy shower can dominate
# the plot.  A rolling N-day total smooths this out and reveals whether a
# location is in a sustained wet (flood) or dry (drought) period.
#
# The rolling climatology is computed by summing the daily climatological mean
# over the same window, giving a meaningful baseline to compare against.

# %%
def plot_rolling_precip(
    curr        : pd.DataFrame,
    clim_start  : int,
    clim_end    : int,
    location    : str,
    today       : date,
    year        : int,
    roll_win    : int   = 30,
    date_range  : tuple = None,
    figsize     : tuple = (15, 6),
) -> tuple:
    """
    Plot rolling N-day precipitation total vs rolling climatological total.

    Red fill = sustained wet period above rolling climatological mean
    Blue fill = sustained dry period below rolling climatological mean
    """
    df = curr.copy().sort_values("date").reset_index(drop=True)

    # ── Compute rolling totals ────────────────────────────────────────────────
    df["roll_precip"] = df["precip"].rolling(roll_win, min_periods=1).sum()
    df["roll_clim"]   = df["clim_mean"].rolling(roll_win, min_periods=1).sum()
    df["roll_p05"]    = df["p05"].rolling(roll_win, min_periods=1).sum()
    df["roll_p95"]    = df["p95"].rolling(roll_win, min_periods=1).sum()
    df["roll_anom"]   = df["roll_precip"] - df["roll_clim"]

    if date_range:
        mask = ((df["date"] >= pd.to_datetime(date_range[0])) &
                (df["date"] <= pd.to_datetime(date_range[1])))
        df = df[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # ── Percentile shading ────────────────────────────────────────────────────
    ax.fill_between(df["date"], df["roll_p05"], df["roll_p95"],
                    color="#D0D0D0", alpha=0.95, zorder=1,
                    label=f"5–95th percentile ({clim_start}–{clim_end})")

    # ── Above / below fill ────────────────────────────────────────────────────
    observed = df[~df["is_forecast"]]
    forecast = df[df["is_forecast"]]

    if not observed.empty and not forecast.empty:
        forecast = pd.concat(
            [observed.iloc[[-1]], forecast]
        ).reset_index(drop=True)

    for subset, alpha_val in [(observed, 0.85), (forecast, 0.45)]:
        if subset.empty:
            continue

        ax.fill_between(subset["date"],
                        subset["roll_clim"],
                        subset["roll_precip"],
                        where=subset["roll_precip"] >= subset["roll_clim"],
                        color="#C0392B", alpha=alpha_val,
                        interpolate=True, zorder=2)

        ax.fill_between(subset["date"],
                        subset["roll_clim"],
                        subset["roll_precip"],
                        where=subset["roll_precip"] < subset["roll_clim"],
                        color="#2980B9", alpha=alpha_val,
                        interpolate=True, zorder=2)

    # ── Climatological mean ───────────────────────────────────────────────────
    ax.plot(df["date"], df["roll_clim"],
            color="black", linewidth=1.5, zorder=3,
            label=f"{roll_win}-day climatology ({clim_start}–{clim_end})")

    # ── TODAY line ────────────────────────────────────────────────────────────
    today_ts = pd.Timestamp(today)
    if df["date"].min() <= today_ts <= df["date"].max():
        ax.axvline(today_ts, color="#555555", linestyle="--",
                   linewidth=1.0, zorder=4, alpha=0.8)
        ylims = ax.get_ylim()
        ax.text(today_ts + pd.Timedelta(hours=10),
                ylims[0] + 0.5, "TODAY",
                rotation=90, va="bottom", ha="left",
                fontsize=8, color="#555555", zorder=5)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f"{roll_win}-Day Rolling Precipitation Total [mm]", fontsize=11)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), ha="center", fontsize=10)

    ax.grid(axis="y", which="major", alpha=0.35, linewidth=0.7, color="#AAAAAA")
    ax.grid(axis="x", which="minor", alpha=0.20, linewidth=0.5, color="#BBBBBB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(df["date"].min() - pd.Timedelta(days=1),
                df["date"].max() + pd.Timedelta(days=1))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#D0D0D0",
                       label=f"5–95th percentile ({clim_start}–{clim_end})"),
        plt.Line2D([0], [0], color="black", linewidth=1.5,
                   label=f"{roll_win}-day climatology"),
        mpatches.Patch(color="#C0392B",
                       label=f"Above average – sustained wet / flood risk"),
        mpatches.Patch(color="#2980B9",
                       label=f"Below average – sustained dry / drought risk"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper left", fontsize=8.5,
              framealpha=0.92, edgecolor="#BBBBBB")

    ax.set_title(
        f"{location}  |  {roll_win}-Day Rolling Precipitation {year} vs Climatology",
        fontsize=13, fontweight="bold", pad=12, loc="left"
    )
    ax.text(0.0, 1.01,
            f"Model: ERA5  |  Climatology: {clim_start}–{clim_end}  |  "
            f"Source: Open-Meteo",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="#666666")

    plt.tight_layout()
    return fig, ax


# ── Rolling plot – full year ──────────────────────────────────────────────────
fig_roll, ax_roll = plot_rolling_precip(
    curr, CLIM_START, CLIM_END, LOCATION, TODAY, CURRENT_YEAR,
    roll_win=ROLL_WIN, date_range=None, figsize=(15, 6),
)
plt.savefig(f"{ACTIVE}_precip_{ROLL_WIN}day_rolling.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {ACTIVE}_precip_{ROLL_WIN}day_rolling.png")


# %% [markdown]
# ## Extending This Script
#
# | What you might want                       | How to do it                                    |
# |-------------------------------------------|-------------------------------------------------|
# | Different location                        | Change `ACTIVE` at the top                      |
# | Longer/shorter rolling window             | Change `ROLL_WIN` (e.g. 7 for flash floods,     |
# |                                           | 90 for seasonal drought)                        |
# | Show only monsoon season                  | Pass date_range to plot functions               |
# | Combine temp + precip on one figure       | Use plt.subplots(2,1) and call both plot funcs  |
# | Compute SPI (Standardised Precip Index)   | Fit gamma distribution per DOY to clim_raw      |
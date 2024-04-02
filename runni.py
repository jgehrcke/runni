import argparse
import io
import logging
import re
import os
import tempfile
import time
from datetime import datetime

import requests

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%y%m%d-%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--window-width-days",
        type=int,
        default=14,
        help="Window width for rolling window analysis (days). Default: 14",
    )

    opts = parser.parse_args()
    plot(
        **process_data(
            pd.read_csv(get_csv(), usecols=["date", "km", "min", "sec"]), opts
        )
    )


def process_data(df, opts):
    # Keep only those rows that have a value set in the `km` column. That is
    # the criterion for having made a run on the corresponding day.
    df = df[df.km.notnull()]

    # Parse text in `date` column into a pd.Series of `datetime64` type values
    # and make this series be the new index of the data frame.
    df.index = pd.to_datetime(df["date"])

    # Sort data frame by index (sort from past to future).
    df = df.sort_index()

    # Turn the data frame into a `pd.Series` object, representing the distance
    # ran over time. Every row / data point in this series represents a run:
    # the index field is the date (day) of the run and the value is the
    # distance of the run.
    km_per_run = df["km"]

    hours_per_run = (60 * df["min"] + df["sec"]) / 3600

    # minutes per km
    avgspeed_per_run = hours_per_run * 60 / km_per_run

    # There may have been more than one run per day. In these cases, sum up the
    # distances and have a single row represent all runs of the day.

    # Example: two runs on 07-11:
    # 2019-07-10    3.2
    # 2019-07-11    4.5
    # 2019-07-11    5.4
    # 2019-07-17    4.5

    # Group events per day and sum up the run distance:
    km_per_run = km_per_run.groupby(km_per_run.index).sum()
    hours_per_run = hours_per_run.groupby(hours_per_run.index).sum()
    avgspeed_per_run = avgspeed_per_run.groupby(avgspeed_per_run.index).mean()

    # Outcome for above's example (for `km_per_run`):
    # 2019-07-10    3.2
    # 2019-07-11    9.9
    # 2019-07-17    4.5

    # The time series index is expected to have gaps: days on which no run was
    # recorded. Up-sample the time index to fill these gaps, with 1 day
    # resolution. Fill the missing values with zeros. This is not strictly
    # necessary for the subsequent analysis but makes the series easier to
    # reason about, and makes the rolling window analysis a little simpler: it
    # will contain one data point per day, precisely, within the represented
    # time interval.
    #
    # Before:
    #   In [28]: len(km_per_run)
    #   Out[28]: 75
    #
    #   In[27]: km_per_run.head()
    #   Out[27]:
    #   2019-05-27    2.7
    #   2019-06-06    2.9
    #   2019-06-11    4.6
    #   ...
    #
    # After:
    #   In [30]: len(km_per_run)
    #   Out[30]: 229
    #
    #   In [31]: km_per_run.head()
    #   Out[31]:
    #   2019-05-27    2.7
    #   2019-05-28    0.0
    #   2019-05-29    0.0
    #   2019-05-30    0.0
    #   ...
    #
    km_per_run = km_per_run.asfreq("1D", fill_value=0)
    hours_per_run = hours_per_run.asfreq("1D", fill_value=0)
    avgspeed_per_run = avgspeed_per_run.asfreq("1D", fill_value=None)

    # Should be >= 7 to be meaningful.
    window_width_days = opts.window_width_days
    window = km_per_run.rolling(window="%sD" % window_width_days)

    # For each window position get the sum of distances. For normalization,
    # divide this by the window width (in days) to get values of the unit
    # km/day -- and then convert to the new desired unit of km/week with an
    # additional factor of 7.
    km_per_week = window.sum() / (window_width_days / 7.0)

    # Do the same for run duration and speed, in less code
    hours_per_week = hours_per_run.rolling(window="%sD" % window_width_days).sum() / (
        window_width_days / 7.0
    )
    avgspeed_per_week = avgspeed_per_run.rolling(
        window="%sD" % window_width_days
    ).mean()

    # During the rolling window analysis the value derived from the current
    # window position is assigned to the right window boundary (i.e. to the
    # newest timestamp in the window). For presentation it is more convenient
    # and intuitive to have it assigned to the temporal center of the time
    # window. Invoking `rolling(..., center=True)` however yields
    # `NotImplementedError: center is not implemented for datetimelike and
    # offset based windows`. As a workaround, shift the data by half the window
    # size to 'the left': shift the timestamp index by a constant / offset.
    offset = pd.DateOffset(days=window_width_days / 2.0)
    km_per_week.index = km_per_week.index - offset
    hours_per_week.index = hours_per_week.index - offset
    avgspeed_per_week.index = avgspeed_per_week.index - offset

    returndict = {}
    for k in (
        "km_per_run",
        "km_per_week",
        "hours_per_run",
        "hours_per_week",
        "avgspeed_per_run",
        "avgspeed_per_week",
        "window_width_days",
    ):
        returndict[k] = locals()[k]

    return returndict


def plot(
    km_per_week,
    km_per_run,
    hours_per_run,
    hours_per_week,
    avgspeed_per_run,
    avgspeed_per_week,
    window_width_days,
):

    plt.style.use("ggplot")
    matplotlib_config()

    # First, distance over time.
    plt.figure()
    ax = km_per_week.plot(linestyle="solid", color="black")
    ax2 = km_per_run.plot(
        linestyle="None", marker="x", color="gray", markersize=3, ax=ax
    )

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Distance [km]")

    ax2.legend(
        [
            "distance per week, rolling window mean (%s days)" % window_width_days,
            "distance per day (raw data)",
        ],
        numpoints=4,
    )

    title = "Running distance per week, over time"
    plt.title(title)
    plt.tight_layout()
    savefig(title)

    # Now, run duration over time
    plt.figure()
    ax = hours_per_week.plot(linestyle="solid", color="black")
    ax2 = hours_per_run.plot(
        linestyle="None", marker="x", color="gray", markersize=3, ax=ax
    )

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Duration [hours]")

    ax2.legend(
        [
            "duration per week, rolling window mean (%s days)" % window_width_days,
            "duration per day (raw data)",
        ],
        numpoints=4,
    )

    title = "Running duration per week, over time"
    plt.title(title)
    plt.tight_layout()
    savefig(title)

    # Now, run velocity over time
    plt.figure()
    ax = avgspeed_per_week.plot(linestyle="solid", color="black")
    ax2 = avgspeed_per_run.plot(
        linestyle="None", marker="x", color="gray", markersize=3, ax=ax
    )

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Avg speed [min/km]")

    ax2.legend(
        [
            "avg speed per week, rolling window mean (%s days)" % window_width_days,
            "avg speed per day (raw data)",
        ],
        numpoints=4,
    )

    title = "Running velocity per week, over time"
    plt.title(title)
    plt.tight_layout()
    savefig(title)

    plt.show()


def get_csv():
    dockey = os.environ["RUNNI_GSHEET_KEY"]
    tmpdirpath = tempfile.gettempdir()
    cachepath = os.path.join(tmpdirpath, f"runni-{dockey[:5]}.csv.cache")

    def _get_csv_text_from_file_cache_or_web():
        # Read from cache if it exists and is not too old.
        maxage_minutes = 10
        if os.path.exists(cachepath):
            if time.time() - os.stat(cachepath).st_mtime < 60 * maxage_minutes:
                with open(cachepath, "rb") as f:
                    log.info("read data from file cache")
                    return f.read().decode("utf-8")

        # Cache miss. Read from web, store in cache.
        url = f"https://docs.google.com/spreadsheet/ccc?key={dockey}&output=csv"
        log.info("read data from web")
        resp = requests.get(url)
        resp.raise_for_status()

        with open(cachepath, "wb") as f:
            log.info("write data to file cache")
            f.write(resp.text.encode("utf-8"))

        return resp.text

    return io.StringIO(_get_csv_text_from_file_cache_or_web())


def savefig(title):
    now = datetime.utcnow()
    today = now.strftime("%Y-%m-%d")
    # Lowercase, replace special chars with whitespace, join on whitespace.
    cleantitle = "-".join(re.sub("[^a-z0-9]+", " ", title.lower()).split())
    fname = today + "_" + cleantitle
    fpath_figure = fname + ".png"
    log.info("Writing PNG figure to %s", fpath_figure)
    plt.savefig(fpath_figure, dpi=150)


def matplotlib_config():
    matplotlib.rcParams["figure.figsize"] = [10.5, 7.0]
    matplotlib.rcParams["figure.dpi"] = 100
    matplotlib.rcParams["savefig.dpi"] = 150


if __name__ == "__main__":
    main()

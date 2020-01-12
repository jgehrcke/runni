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
    plot(**process_data(pd.read_csv(get_csv(), usecols=["date", "km"]), opts))


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

    # There may have been more than one run per day. In these cases, sum up the
    # distances and have a single row represent all runs of the day.

    # Example: two runs on 07-11:
    # 2019-07-10    3.2
    # 2019-07-11    4.5
    # 2019-07-11    5.4
    # 2019-07-17    4.5

    # Group events per day and sum up the run distance:
    km_per_run = km_per_run.groupby(km_per_run.index).sum()

    # Outcome for above's example:
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
    # Before: In [28]: len(km_per_run) Out[28]: 75
    #
    # In[27]: km_per_run.head() Out[27]: date 2019-05-27    2.7 2019-06-06 2.9
    # 2019-06-11    4.6
    # ...
    #
    # After: In [30]: len(km_per_run)
    # # Out[30]: 229 Out[30]: 229
    #
    # In [31]: km_per_run.head() Out[31]: date 2019-05-27    2.7 2019-05-28 0.0
    # 2019-05-29    0.0 2019-05-30    0.0
    # ...
    #
    km_per_run = km_per_run.asfreq("1D", fill_value=0)

    # Should be >= 7 to be meaningful.
    window_width_days = opts.window_width_days
    # Count the number of events (requests) within the rolling window.
    window_width_days = 12  ## must at least 7 to be meaningful
    window = km_per_run.rolling(window="%sD" % window_width_days)
    s = window.sum()
    km_per_week = s / (window_width_days / 7.0)

    new_column_name = "km_per_week_%sd_window" % window_width_days
    km_per_week.rename(new_column_name, inplace=True)

    offset = pd.DateOffset(days=window_width_days / 2.0)
    km_per_week.index = km_per_week.index - offset

    returndict = {}
    for k in ("km_per_run", "km_per_week", "window_width_days"):
        returndict[k] = locals()[k]

    return returndict


def plot(km_per_week, km_per_run, window_width_days):

    plt.style.use("ggplot")
    matplotlib_config()
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

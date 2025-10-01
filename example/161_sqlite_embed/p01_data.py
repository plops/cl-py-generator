import matplotlib

matplotlib.use("qtagg")
import matplotlib.pyplot as plt

plt.ion()
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sqlite_minutils import *
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger.info("Logger configured")
db = Database("/home/kiel/summaries_20250929.db")
tab = db.table("items")
logger.info("Sqlite file opened")
cols = [
    "identifier",
    "model",
    "summary",
    "summary_timestamp_start",
    "summary_timestamp_end",
    "summary_done",
    "summary_input_tokens",
    "summary_output_tokens",
    "host",
    "original_source_link",
    "embedding",
    "full_embedding",
]
sql = (
    ("SELECT ")
    + (", ".join(cols))
    + (" FROM items WHERE summary is NOT NULL AND summary !=''")
)
df = pd.read_sql_query(sql, db.conn)
# Convert Timestamps from string into datetime type
df["ts_start"] = pd.to_datetime(
    df["summary_timestamp_start"], errors="coerce", utc=True
)
df["ts_end"] = pd.to_datetime(df["summary_timestamp_end"], errors="coerce", utc=True)
df["duration_s"] = ((df["ts_end"]) - (df["ts_start"])).dt.total_seconds()
# Find which inferences were performed during the work week and
logger.info("Read columns from sqlite into pandas dataframe")
# Add continental US workhour filter
# define continental US timezones
tzs = dict(
    eastern="US/Eastern",
    central="US/Central",
    mountain="US/Mountain",
    pacific="US/Pacific",
)
# create localized columns and boolean workhour masks per timezone
work_masks = []
for name, tz in tzs.items():
    col = f"ts_{name}"
    df[col] = df["ts_start"].dt.tz_convert(tz)
    # workday Mon-Fri -> dayofweek 0..4
    is_weekday = (df[col].dt.dayofweek) < (5)
    # workhours 09:00 <= local_time < 17:00 (hours 9..16
    is_workhour = df[col].dt.hour.between(9, 16)
    work_masks.append(((is_weekday) & (is_workhour)))
df["is_workhours_continental"] = np.logical_or(reduce(work_masks))
# filter invalid durations and tokens and keep only workhours rows
df_valid = df[
    (
        (df["is_workhours_continental"])
        & (df["duration_s"].notna())
        & ((df["duration_s"]) > (0))
        & (df["summary_input_tokens"].notna())
    )
]
for s in ["-flash", "-pro"]:
    mask = df.model.str.contains(s, case=False, na=False)
    dfm = df.loc[mask]
    dat = (dfm.summary_input_tokens) / (dfm.duration_s)
    bins = np.linspace(0, np.percentile(dat.dropna(), 99), 300)
    plt.hist(dat, log=True, bins=bins, label=s)
plt.xlabel("tokens/s")
plt.legend()
plt.show()
#
# >>> df
#       identifier                                              model  ... embedding full_embedding
# 0              1                            gemini-1.5-pro-exp-0827  ...      None           None
# 1              2                            gemini-1.5-pro-exp-0827  ...      None           None
# 2              3                            gemini-1.5-pro-exp-0827  ...      None           None
# 3              4                            gemini-1.5-pro-exp-0827  ...      None           None
# 4              5                            gemini-1.5-pro-exp-0827  ...      None           None
# ...          ...                                                ...  ...       ...            ...
# 7972        8155  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
# 7973        8156  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
# 7974        8157  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
# 7975        8158  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
# 7976        8159  gemini-2.5-pro| input-price: 1.25 output-price...  ...      None           None
#
# [7977 rows x 12 columns]
# >>> df.iloc[-1]
# identifier                                                              8159
# model                      gemini-2.5-pro| input-price: 1.25 output-price...
# summary                    **Abstract:**nnThis personal essay by Georgi...
# summary_timestamp_start                           2025-09-29T21:32:46.405084
# summary_timestamp_end                             2025-09-29T21:33:08.668613
# summary_done                                                             1.0
# summary_input_tokens                                                 14343.0
# summary_output_tokens                                                  742.0
# host                                                          194.230.161.72
# original_source_link       https://www.huffpost.com/entry/weight-loss-sur...
# embedding                                                               None
# full_embedding                                                          None
# Name: 7976, dtype: object
#
#
#
#  convert a subset of the columns to pandas (identifier model summary summary_timestamp_start summary_timestamp_end summary_done summary_input_tokens summary_output_tokens host original_source_link embedding full_embedding)
#  filter out bad summaries
#  compute 4d and 2d umap
#  compute hdbscan clusters
#  find names for the clusters
#
#  store the umap so that new emedding entries can be assigned a 2d position and a cluster name
#
#  visualize the full map
#
#  display a segment of the full map in the neighborhood of a new embedding entry
#
#
#

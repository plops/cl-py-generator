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
for name, df in [["work", df_valid], ["off", df_valid_off]]:
    for s in ["-flash"]:
        mask = df.model.str.contains(s, case=False, na=False)
        dfm = df.loc[mask]
        dat = ((dfm.summary_input_tokens) + (dfm.summary_output_tokens)) / (
            dfm.duration_s
        )
        bins = np.linspace(0, np.percentile(dat.dropna(), 99), 300)
        plt.hist(dat, log=True, bins=bins, label=((name) + (s)), alpha=(0.60))
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
# identifier                                                               8123
# model                       gemini-2.5-pro| input-price: 1.25 output-price...
# summary                     **Abstract:**nnThis presentation by Thomas B...
# summary_timestamp_start                            2025-09-29T11:11:19.613310
# summary_timestamp_end                              2025-09-29T11:12:00.599444
# summary_done                                                              1.0
# summary_input_tokens                                                  22888.0
# summary_output_tokens                                                  1162.0
# host                                                             193.8.40.126
# original_source_link              https://www.youtube.com/watch?v=0CepUaVqSeQ
# embedding                                                                None
# full_embedding                                                           None
# ts_start                                     2025-09-29 11:11:19.613310+00:00
# ts_end                                       2025-09-29 11:12:00.599444+00:00
# duration_s                                                          40.986134
# ts_eastern                                   2025-09-29 07:11:19.613310-04:00
# ts_central                                   2025-09-29 06:11:19.613310-05:00
# ts_mountain                                  2025-09-29 05:11:19.613310-06:00
# ts_pacific                                   2025-09-29 04:11:19.613310-07:00
# is_workhours_continental                                                False
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

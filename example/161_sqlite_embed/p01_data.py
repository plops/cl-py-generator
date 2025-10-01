import matplotlib

matplotlib.use("webagg")
import matplotlib.pyplot as plt
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
df["ts_start"] = pd.to_datetime(df["summary_timestamp_start"], errors="coerce")
df["ts_end"] = pd.to_datetime(df["summary_timestamp_end"], errors="coerce")
df["duration_s"] = ((df["ts_end"]) - (df["ts_start"])).dt.total_seconds()
logger.info("Read columns from sqlite into pandas dataframe")
plt.hist(((df.summary_input_tokens) / (df.duration_s)), log=True, bins=300)
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

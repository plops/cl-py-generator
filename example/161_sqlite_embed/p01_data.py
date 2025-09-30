import pandas as pd
import sys
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
logger.info("Read columns from sqlite into pandas dataframe")
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

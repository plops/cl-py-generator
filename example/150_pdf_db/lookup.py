import argparse
import sqlite_utils
#import subprocess
import time
import json
#from pathlib import Path
#from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default="pdfs.db", help="Path to the SQLite database.")
args = parser.parse_args()


db = sqlite_utils.Database(args.db_path)

try:
    fts_columns = ['text','path']
    # ['text','path','pdfinfo','pdfinfo_url']
    db['pdfs'].enable_fts(fts_columns)
    db['pdfs'].populate_fts(fts_columns)
except Exception as e:
    print(e)
    pass

# after update of the table you need to run .populate_fts to update search index
# or use create_trigger=True. However, this code doesn't change the database entries

rows = list(db['pdfs'].search(db.quote_fts('H.264')))
print(rows)

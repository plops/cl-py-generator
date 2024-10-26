import argparse
import sqlite_utils

parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default="pdfs.db", help="Path to the SQLite database.")
parser.add_argument("--search", default="OpenGL", help="Search term to search in the SQLite database.")
args = parser.parse_args()

db = sqlite_utils.Database(args.db_path)

try:
    if db['pdfs'].detect_fts() is None:
        fts_columns = ['text','path','pdfinfo','pdfinfo_url']
        db['pdfs'].enable_fts(fts_columns)
        db['pdfs'].populate_fts(fts_columns)
except Exception as e:
    print(e)
    pass

# after update of the table you need to run .populate_fts to update search index
# or use create_trigger=True. However, this code doesn't change the database entries

rows = list(db['pdfs'].search(db.quote_fts(args.search)))
for row in rows:
    print(row['path'])

import sqlite_minutils

db = sqlite_minutils.Database("/home/kiel/summaries_20251105.db")

tab = db['items']
# <Table items (identifier, model, transcript, host, summary, summary_done, summary_input_tokens, summary_output_tokens, summary_timestamp_start, summary_timestamp_end, timestamps, timestamps_done, timestamps_input_tokens, timestamps_output_tokens, timestamps_timestamp_start, timestamps_timestamp_end, timestamped_summary_in_youtube_format, cost, original_source_link, include_comments, include_timestamps, include_glossary, output_language, embedding, full_embedding)>

fts_columns = ['transcript','summary','original_source_link']

try:
    if tab.detect_fts() is None:
        tab.enable_fts(fts_columns)
        tab.populate_fts(fts_columns)
        db.conn.commit()
except Exception as e:
    print(e)
    pass
#
# # after update of the table you need to run .populate_fts to update search index
# # or use create_trigger=True. However, this code doesn't change the database entries
#
# rows = list(db['documents'].search(db.quote_fts(args.search)))
# for row in rows:
#     print(row['path'])

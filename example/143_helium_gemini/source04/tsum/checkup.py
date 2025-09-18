import sqlite3
import os

db_file = os.path.join("data", "summaries.db")

print(f"Opening and closing the database to trigger a checkpoint: {db_file}")

try:
    # Connect to the database
    con = sqlite3.connect(db_file)

    # You can optionally run a checkpoint command explicitly
    print("Running WAL checkpoint...")
    con.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    # Closing the connection will also trigger a checkpoint
    con.close()
    print("Connection closed. The WAL file should now be merged.")

except sqlite3.Error as e:
    print(f"An error occurred: {e}")

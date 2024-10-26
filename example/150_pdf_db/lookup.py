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

#!/usr/bin/env python3
# given a text file containing parts of filenames and a directory create a list of files of the files that match the parts
import time
import tqdm
import pathlib
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("input_paths", nargs="+", help="Path(s) to search for matching files.")
parser.add_argument("--file-parts-from", help="A text file with parts that shall occur in the filename.")
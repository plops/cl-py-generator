#!/usr/bin/env python3
# given a text file containing parts of filenames and a directory create a list of files of the files that match the parts
import time
import sys
import tqdm
import pathlib
import pandas as pd
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("input_paths", nargs="+", help="Path(s) to search for matching files.")
parser.add_argument("--file-parts-from", help="A text file with parts that shall occur in the filename.")
parser.add_argument("--min-size", type=int, default=0, help="Minimum size in bytes for a file to be selected.")
parser.add_argument("--suffix", type=str, default="*", help="File suffix pattern that must match the filename (e.g. *.mp4). The default pattern accepts all.")
args=parser.parse_args()
# stop if input_paths is empty
if ( ((len(args.input_paths))==(0)) ):
    sys.exit(0)
files=[]
for input_path in args.input_paths:
    path=pathlib.Path(input_path)
    if ( path.is_dir() ):
        files.extend(path.rglob(args.suffix))
    elif ( ((path.is_file()) and (((path.suffix)==(args.suffix)))) ):
        files.append(path)
df=pd.DataFrame(dict(file=files))
# load parts
with open(args.file_parts_from) as f:
    parts0=f.readlines()
# remove newlines
parts=[]
for part in parts0:
    parts.append(part.strip("\n"))
print("collect file sizes ".format())
res=[]
for file in tqdm.tqdm(files):
    # check if file has a match with any of the entries of parts
    isMatch=False
    for p in parts:
        if ( (p in str(file)) ):
            isMatch=True
    if ( isMatch ):
        st_size=0
        try:
            # this throws for dangling symlinks
            st_size=file.stat().st_size
        except Exception as e:
            pass
        # keep only rows that have st_size>=args.min_size
        if ( ((args.min_size)<=(st_size)) ):
            res.append(dict(file=str(file), st_size=st_size))
df=pd.DataFrame(res)
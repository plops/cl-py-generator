#!/usr/bin/env python3
import os
import time
import pathlib
import re
import pandas as pd
directory=pathlib.Path("/home/martin/stage/cl-py-generator")
training_data=[]
gen_files0=list(((directory)/("example")).rglob("gen*.lisp"))
gen_files1=[]
for f in gen_files0:
    # exclude C++ generating files
    content=f.read_text()
    if ( re.search(r"""\(ql:quickload "cl-cpp-generator2"\)""", content) ):
        print(f"Info 0: Skip C++ generator {f}.")
        continue
    folder=f.parent
    # count the number of python files
    py_files=list(folder.rglob("*.py"))
    n_py=len(py_files)
    gen_files1.append(dict(file=f, folder=folder, n_py=n_py, py_files=py_files))
g1=pd.DataFrame(gen_files1)
# count number of python-generating lisp files in this directory
folder_counts=g1.groupby("folder").size()
g1=g1.merge(folder_counts.rename("n_lisp"), left_on="folder", right_index=True)
#!/usr/bin/env python3
import os
import time
import pathlib
import re
import pandas as pd
start_time=time.time()
debug=True
_code_git_version="5f299f3712fc6eaef6abbf65bd0807547c04998a"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/train_llm/source/"
_code_generation_time="23:40:54 of Thursday, 2024-05-09 (GMT+1)"
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
    ipynb_files=list(folder.rglob("*.ipynb"))
    n_py=len(py_files)
    n_ipynb=len(ipynb_files)
    gen_files1.append(dict(file=f, folder=folder, n_py=n_py, n_ipynb=n_ipynb, py_files=py_files, ipynb_files=ipynb_files, short=f"{folder.stem}/{f.stem}"))
g1=pd.DataFrame(gen_files1)
# count number of python-generating lisp files in this directory
folder_counts=g1.groupby("folder").size()
g1=g1.merge(folder_counts.rename("n_lisp"), left_on="folder", right_index=True)
# find folder with one python-generating .lisp input and no .py file. that should be generated, then
g20=g1[((((g1.n_lisp)==(1))) & (((g1.n_py)==(0))) & (((g1.n_ipynb)!=(1))))].sort_values(by="short")
print("{} the following folders need python file g20.short={}".format(((time.time())-(start_time)), g20.short))
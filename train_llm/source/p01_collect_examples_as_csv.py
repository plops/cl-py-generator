#!/usr/bin/env python3
import os
import time
import pathlib
import re
import pandas as pd
start_time=time.time()
debug=True
_code_git_version="cc706f2fa373353427aa34c29b8b6018efe1996d"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/train_llm/source/"
_code_generation_time="23:47:47 of Thursday, 2024-05-09 (GMT+1)"
directory=pathlib.Path("/home/martin/stage/cl-py-generator")
training_data=[]
gen_files0=list(((directory)/("example")).rglob("gen*.lisp"))
gen_files1=[]
for f in gen_files0:
    # exclude C++ generating files
    content=f.read_text()
    if ( re.search(r"""\(ql:quickload "cl-cpp-generator2"\)""", content) ):
        print(f"Info 0: Skip C++ generator {f.parent.stem}/{f.stem}.")
        continue
    folder=f.parent
    # count the number of python files. and also the characters (in column len_py)
    py_files=list(folder.rglob("*.py"))
    len_py=0
    for p in py_files:
        len_py += len(p.read_text())
    n_py=len(py_files)
    # same stats for notebooks
    ipynb_files=list(folder.rglob("*.ipynb"))
    len_ipynb=0
    for p in ipynb_files:
        len_ipynb += len(p.read_text())
    n_ipynb=len(ipynb_files)
    # count characters in lisp file
    len_lisp=len(f.read_text())
    gen_files1.append(dict(file=f, len_lisp=len_lisp, folder=folder, n_py=n_py, len_py=len_py, n_ipynb=n_ipynb, len_ipynb=len_ipynb, py_files=py_files, ipynb_files=ipynb_files, short=f"{folder.stem}/{f.stem}"))
g1=pd.DataFrame(gen_files1)
# count number of python-generating lisp files in this directory
folder_counts=g1.groupby("folder").size()
g1=g1.merge(folder_counts.rename("n_lisp"), left_on="folder", right_index=True)
# find folder with one python-generating .lisp input and no .py file. that should be generated, then
g20=g1[((((g1.n_lisp)==(1))) & (((g1.n_py)==(0))) & (((g1.n_ipynb)!=(1))))].sort_values(by="short")
print("{} the following folders need python file g20.short={}".format(((time.time())-(start_time)), g20.short))
# find folder with one python-generating .lisp input and one .py file
g2=g1[((((g1.n_lisp)==(1))) & (((g1.n_py)==(1))) & (((g1.len_py)<(40000))) & (((g1.len_lisp)<(5000))))]
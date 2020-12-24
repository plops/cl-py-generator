# %% imports
import pathlib
import numpy as np
import pandas as pd
from tkinter import *
_code_git_version="6e66a6c4fc711e22adc8b5e7d6baf7538ab96291"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="22:34:04 of Thursday, 2020-12-24 (GMT+1)"
root=Tk()
def myclick():
    mylab=Label(root, text="look!")
    mylab.pack()
but=Button(root, text="click", padx=23, command=myclick)
but.pack()
lab0=Label(root, text="hello").grid(row=0, column=0)
lab1=Label(root, text="my name is").grid(row=1, column=1)
root.mainloop()
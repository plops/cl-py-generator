# %% imports
import pathlib
import numpy as np
import pandas as pd
from tkinter import *
_code_git_version="c5f8cd917b43fd230a48ab7ae3ba9dc5408836eb"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="04:35:44 of Friday, 2020-12-25 (GMT+1)"
root=Tk()
root.title("simple calculator")
e=Entry(root, width=35, borderwidth=5).grid(row=0, column=0, columnspan=3, padx=10, pady=10)
def button_add():
    return
button_0=Button(root, text="0", padx=40, pady=20, command=button_add).grid(row=4, column=0, columnspan=1)
button_1=Button(root, text="1", padx=40, pady=20, command=button_add).grid(row=3, column=0, columnspan=1)
button_2=Button(root, text="2", padx=40, pady=20, command=button_add).grid(row=3, column=1, columnspan=1)
button_3=Button(root, text="3", padx=40, pady=20, command=button_add).grid(row=3, column=2, columnspan=1)
button_4=Button(root, text="4", padx=40, pady=20, command=button_add).grid(row=2, column=0, columnspan=1)
button_5=Button(root, text="5", padx=40, pady=20, command=button_add).grid(row=2, column=1, columnspan=1)
button_6=Button(root, text="6", padx=40, pady=20, command=button_add).grid(row=2, column=2, columnspan=1)
button_7=Button(root, text="7", padx=40, pady=20, command=button_add).grid(row=1, column=0, columnspan=1)
button_8=Button(root, text="8", padx=40, pady=20, command=button_add).grid(row=1, column=1, columnspan=1)
button_9=Button(root, text="9", padx=40, pady=20, command=button_add).grid(row=1, column=2, columnspan=1)
root.mainloop()
# %% imports
import pathlib
import numpy as np
import pandas as pd
from tkinter import *
_code_git_version="9e5b04c3fddad20454ebc9d679dea20097f87752"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="05:01:05 of Friday, 2020-12-25 (GMT+1)"
root=Tk()
root.title("simple calculator")
entry=Entry(root, width=35, borderwidth=5)
entry.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
def button_click(n):
    cur=entry.get()
    entry.delete(0, END)
    entry.insert(0, ((str(cur))+(str(n))))
def button_clear():
    entry.delete(0, END)
def button_add():
    first_number=entry.get()
    global f_num
    f_num=int(first_number)
    entry.delete(0, END)
def button_equal():
    second_number=entry.get()
    entry.delete(0, END)
    entry.insert(0, ((f_num)+(int(second_number))))
button_0=Button(root, text="0", padx=40, pady=20, command=lambda : button_click(0))
button_0.grid(row=4, column=0, columnspan=1)
button_1=Button(root, text="1", padx=40, pady=20, command=lambda : button_click(1))
button_1.grid(row=3, column=0, columnspan=1)
button_2=Button(root, text="2", padx=40, pady=20, command=lambda : button_click(2))
button_2.grid(row=3, column=1, columnspan=1)
button_3=Button(root, text="3", padx=40, pady=20, command=lambda : button_click(3))
button_3.grid(row=3, column=2, columnspan=1)
button_4=Button(root, text="4", padx=40, pady=20, command=lambda : button_click(4))
button_4.grid(row=2, column=0, columnspan=1)
button_5=Button(root, text="5", padx=40, pady=20, command=lambda : button_click(5))
button_5.grid(row=2, column=1, columnspan=1)
button_6=Button(root, text="6", padx=40, pady=20, command=lambda : button_click(6))
button_6.grid(row=2, column=2, columnspan=1)
button_7=Button(root, text="7", padx=40, pady=20, command=lambda : button_click(7))
button_7.grid(row=1, column=0, columnspan=1)
button_8=Button(root, text="8", padx=40, pady=20, command=lambda : button_click(8))
button_8.grid(row=1, column=1, columnspan=1)
button_9=Button(root, text="9", padx=40, pady=20, command=lambda : button_click(9))
button_9.grid(row=1, column=2, columnspan=1)
button_add=Button(root, text="add", padx=39, pady=20, command=button_add)
button_add.grid(row=5, column=0, columnspan=1)
button_eq=Button(root, text="eq", padx=91, pady=20, command=button_equal)
button_eq.grid(row=5, column=1, columnspan=2)
button_clear=Button(root, text="clear", padx=79, pady=20, command=button_clear)
button_clear.grid(row=4, column=1, columnspan=2)
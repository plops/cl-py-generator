#!/usr/bin/env python3
# Visualize sqlite database with youtube video summaries
import pandas as pd
from tkinter import *
from sqlite_minutils import *
from tkinter import ttk
db=Database("/home/martin/summaries.db")
items=Table(db, "items")
res=[]
for row in items.rows:
    d={}
    d["summary"]=row["summary"]
    d["summary_done"]=row["summary_done"]
    d["model"]=row["model"]
    d["cost"]=row["cost"]
    d["summary_input_tokens"]=row["summary_input_tokens"]
    d["summary_output_tokens"]=row["summary_output_tokens"]
    d["summary_timestamp_start"]=row["summary_timestamp_start"]
    d["summary_timestamp_end"]=row["summary_timestamp_end"]
    d["original_source_link"]=row["original_source_link"]
    d["host"]=row["host"]
    res.append(d)
df=pd.DataFrame(res)
root=Tk()
frm=ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Hello World").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
root.mainloop()
print("finished ".format())
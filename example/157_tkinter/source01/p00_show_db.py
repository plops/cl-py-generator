#!/usr/bin/env python3
# Visualize sqlite database with youtube video summaries
import pandas as pd
from tkinter import *
from sqlite_minutils import *
from tkinter.ttk import *
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
    title=row["summary"].split("\n")[0]
    max_len=100
    if ( ((max_len)<(len(title))) ):
        title=title[:max_len]
    d["title"]=title
    res.append(d)
df=pd.DataFrame(res)
root=Tk()
frm=Frame(root, padding=10)
frm.grid()
Label(frm, text="summary_timestamp_start").grid(column=0, row=0)
Label(frm, text="cost").grid(column=1, row=0)
Label(frm, text="summary_input_tokens").grid(column=2, row=0)
Label(frm, text="summary_output_tokens").grid(column=3, row=0)
Label(frm, text="model").grid(column=4, row=0)
Label(frm, text="title").grid(column=5, row=0)
count=1
df=df[::-1]
for idx, row in df.iterrows():
    Label(frm, justify="right", text=row["summary_timestamp_start"]).grid(column=0, row=count)
    Label(frm, justify="right", text=row["cost"]).grid(column=1, row=count)
    Label(frm, justify="right", text=row["summary_input_tokens"]).grid(column=2, row=count)
    Label(frm, justify="right", text=row["summary_output_tokens"]).grid(column=3, row=count)
    Label(frm, justify="right", text=row["model"]).grid(column=4, row=count)
    Button(frm, command=lambda : print(df.iloc[idx].summary), text=row["title"]).grid(column=5, row=count)
    count += 1
root.mainloop()
print("finished ".format())
#!/usr/bin/env python3
# Visualize sqlite database with youtube video summaries
import pandas as pd
import tkinter as tk
import tkinter.ttk as ttk
from sqlite_minutils import *
from datetime import datetime

db = Database("/home/martin/summaries.db")  # Replace with your database path
items = Table(db, "items")
res = []
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
    try:
        start_time = datetime.fromisoformat(row["summary_timestamp_start"])
        end_time = datetime.fromisoformat(row["summary_timestamp_end"])
        time_delta = end_time - start_time
        # convert time_delta to seconds (float)
        time_delta_f = time_delta.total_seconds()
        d["time_diff"] = time_delta_f
        d["time_per_1ktoken"] = time_delta_f*1000/float(row["summary_input_tokens"]) # typically 0.2 to 2 seconds / 1k token
    except Exception as e:
        d["time_diff"] = 0
    title=row["summary"].split("\n")[0]
    max_len=100
    if ( ((max_len)<(len(title))) ):
        title=title[:max_len]
    d["title"]=title
    res.append(d)

df = pd.DataFrame(res)

#df = df.sort_values(by=["time_per_1ktoken"], ascending=False)
#df = df.reset_index(drop=True)

root = tk.Tk()
root.title("DataFrame Viewer")

#frame = ttk.Frame(root)
#frame.pack(fill=tk.BOTH, expand=True)

tree = ttk.Treeview(columns=list(df.columns) + ["Time Difference"], show="headings")
tree.pack(fill=tk.BOTH, expand=True)

vsb = ttk.Scrollbar(tree, orient="vertical", command=tree.yview)
hsb = ttk.Scrollbar(tree, orient="horizontal", command=tree.xview)
tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
vsb.pack(side="right", fill="y")
hsb.pack(side="bottom", fill="x")

for col in df.columns:
    tree.heading(col, text=col)
    tree.column(col, width=80, anchor='w')  # Adjust as needed
#
# tree.heading("Time Difference", text="Time Difference")
# tree.column("Time Difference", width=100, anchor='center')

# for index, row in df.iterrows():
#     try: # Handle potential errors during time conversion
#         start_time = datetime.fromisoformat(row["summary_timestamp_start"])
#         end_time = datetime.fromisoformat(row["summary_timestamp_end"])
#         time_diff = end_time - start_time
#         values = list(row) + [str(time_diff)]
#
#
#     except (ValueError, TypeError):
#         print(f"Error converting times for row {index}. Skipping time difference.") # More informative error message
#         values = list(row) + ["N/A"] # Or choose a different placeholder
#
#
#     tree.insert("", tk.END, values=values, iid=index) # Use index as iid
#


for index, row in df.iterrows():
    values = list(row)
    tree.insert("", tk.END, values=values, iid=index) # Use index as iid


def on_double_click(event):
    item = tree.selection()[0]
    idx = int(item)
    print(df.iloc[idx].summary)

tree.bind("<Double-1>", on_double_click)






root.mainloop()
# print("finished")
#
#

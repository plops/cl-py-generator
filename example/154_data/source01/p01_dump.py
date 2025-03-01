#!/usr/bin/env python3
# micromamba install requests sqlite_minutils
import time
import sys
import tqdm
import pathlib
import pandas as pd
import json
import numpy as np
import argparse
from sqlite_minutils import *
db_fns=list(pathlib.Path("data/").rglob("*.db"))
print(f"{len(db_fns)} files: {db_fns}")
res=[]
parser=argparse.ArgumentParser()
parser.add_argument("--personality", type=str, default="INFJ", help="Select personality type.")
parser.add_argument("--smoking", type=str, default="Non-smoker")
parser.add_argument("--debug", type=bool, default=False, help="Enable debug output")
args=parser.parse_args()
for db_fn in tqdm.tqdm(db_fns):
    db=Database(db_fn)
    users=Table(db, "Users")
    for row in users.rows:
        data=json.loads(row["data"])
        d=dict(id=row["id"], data=data)
        try:
            d["name"]=data.get("name", "Unknown")
        except Exception as e:
            pass
        try:
            d["bio"]=data.get("bio", "")
        except Exception as e:
            pass
        try:
            d["distance"]=((data.get("distance_mi", 0))/((1.609340    )))
        except Exception as e:
            pass
        try:
            d["birth_date"]=(int(data["birth_date"][0:4])) if (data.get("birth_date", False)) else (0)
        except Exception as e:
            pass
        try:
            d["images"]=list(map(lambda p: p["url"], data.get("photos", [])))
        except Exception as e:
            pass
        try:
            for s in data["selected_descriptors"]:
                try:
                    d[s["name"]]=s["choice_selections"][0]["name"]
                except Exception as e:
                    pass
        except Exception as e:
            pass
        res.append(d)
df2=pd.DataFrame(res)
df1=df2.drop_duplicates(subset="id")
df0=df1.set_index("id", drop=True, append=False, inplace=False)
print(f"number of entries: {len(df1)}")
df0["age"]=df0.birth_date.apply(lambda x: ((2025)-(x)), 1)
df0["Smoking"]=df0["Smoking"].astype(str)
df0["Family Plans"]=df0["Family Plans"].astype(str)
df=df0[((((df0["Personality Type"])==(args.personality))) & (((((df0["Smoking"])==(args.smoking))) | ((("nan")==(df0["Smoking"]))))) & (((df0.age)<(39))) & ((("I don't want children")!=(df0["Family Plans"]))) & ((("I have children and don't want more")!=(df0["Family Plans"]))) & ((("I have children and want more")!=(df0["Family Plans"]))))].sort_values(by="age")
print(f"number of entries of type {args.personality}: {len(df)} ({len(df)/len(df1)*100}% of {len(df1)})")
count=0
q=df.drop(columns=["bio", "data", "images", "distance", "Blood Type", "Personality Type"])
for idx, row in df.iterrows():
    count += 1
    print(f"#### {count} # {row['name']} ({row.age}): \n{row.bio}\n{q.loc[idx]}\n")
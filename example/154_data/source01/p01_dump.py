#!/usr/bin/env python3
# micromamba install requests sqlite_minutils
import time
import sys
import tqdm
import pathlib
import pandas as pd
import json
from sqlite_minutils import *
db_fns=list(pathlib.Path("data/").rglob("*.db"))
print(f"{len(db_fns)} files: {db_fns}")
res=[]
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
            d["gender"]=["Male", "Female", "Unknown"][data.get("gender", 2)]
        except Exception as e:
            pass
        try:
            d["images"]=list(map(lambda p: p["url"], data.get("photos", [])))
        except Exception as e:
            pass
        try:
            d["family_plans"]=next((s.get("choice_selections")[0].get("name")) if (((s.get("name"))==("Family Plans"))) else ("") for s in data["selected_descriptors"])
        except Exception as e:
            pass
        try:
            d["smoking"]=next((s.get("choice_selections")[0].get("name")) if (((s.get("name"))==("Smoking"))) else ("") for s in data["selected_descriptors"])
        except Exception as e:
            pass
        try:
            d["drinking"]=next((s.get("choice_selections")[0].get("name")) if (((s.get("name"))==("Drinking"))) else ("") for s in data["selected_descriptors"])
        except Exception as e:
            pass
        try:
            d["workout"]=next((s.get("choice_selections")[0].get("name")) if (((s.get("name"))==("Workout"))) else ("") for s in data["selected_descriptors"])
        except Exception as e:
            pass
        try:
            d["education"]=next((s.get("choice_selections")[0].get("name")) if (((s.get("name"))==("Education"))) else ("") for s in data["selected_descriptors"])
        except Exception as e:
            pass
        try:
            d["personality_type"]=next((s.get("choice_selections")[0].get("name")) if (((s.get("name"))==("Personality Type"))) else ("") for s in data["selected_descriptors"])
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
df0["age"]=df0.birth_date.apply(lambda x: ((2025)-(x)), 1)
df=df0[((df0["Personality Type"])==("INFJ"))].sort_values(by="age")
#!/usr/bin/env python3
import json
import tqdm
import pandas as pd
import numpy as np
from sqlite_minutils import *
db=Database("tide.db")
users=Table(db, "Users")
res=[]
for row in tqdm.tqdm(users.rows):
    q=json.loads(row["data"])
    d=dict()
    try:
        d["name"]=q["name"]
    except Exception as e:
        pass
    try:
        d["id"]=q["id"]
    except Exception as e:
        pass
    try:
        d["birth_date"]=q["birth_date"]
    except Exception as e:
        pass
    try:
        d["bio"]=q["bio"]
    except Exception as e:
        pass
    try:
        d["schools"]=q["schools"]
    except Exception as e:
        pass
    try:
        d["jobs"]=q["jobs"]
    except Exception as e:
        pass
    try:
        d["locations"]=q["locations"]
    except Exception as e:
        pass
    try:
        d["distance"]=q["distance"]
    except Exception as e:
        pass
    try:
        for s in q["selected_descriptors"]:
            try:
                d[s["name"]]=s["choice_selections"][0]["name"]
            except Exception as e:
                pass
    except Exception as e:
        pass
    res.append(d)
df0=pd.DataFrame(res)
df=df0[((((df0.Smoking)==("Non-smoker"))) | (df0.Smoking.isna()))]
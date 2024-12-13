#!/usr/bin/env python3
import time
import json
import tqdm
import pandas as pd
import numpy as np
import requests
import random
from sqlite_minutils import *
db=Database("tide.db")
users=Table(db, "Users")
res=[]
for row in tqdm.tqdm(users.rows):
    q=json.loads(row["data"])
    d=dict(id=row["id"])
    try:
        d["name"]=q["name"]
    except Exception as e:
        print("no name")
        pass
    try:
        d["birth_date"]=q["birth_date"]
    except Exception as e:
        print("no birth_date")
        pass
    try:
        d["bio"]=q["bio"]
    except Exception as e:
        print("no bio")
        pass
    try:
        d["schools"]=q["schools"]
    except Exception as e:
        print("no schools")
        pass
    try:
        d["jobs"]=q["jobs"]
    except Exception as e:
        print("no jobs")
        pass
    try:
        d["_id"]=q["_id"]
    except Exception as e:
        print("no _id")
        pass
    try:
        d["locations"]=q["locations"]
    except Exception as e:
        print("no locations")
        pass
    try:
        d["distance"]=q["distance"]
    except Exception as e:
        print("no distance")
        pass
    images=[]
    photos=q["photos"]
    num_photos=len(photos)
    for image_data in photos:
        try:
            url=image_data["processedFiles"][0]["url"]
            images.append(url)
        except Exception as e:
            pass
    try:
        d["num_photos"]=num_photos
        d["images"]=images
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
df2=pd.DataFrame(res)
df1=df2.drop_duplicates(subset="id")
df0=df1.set_index("id", drop=True, append=False, inplace=False)
df0["age"]=df0.birth_date.apply(lambda x: (((2024)-(int(x[0:4])))) if (((type("bla"))==(type(x)))) else (0), 1)
df=df0[((((((df0["Smoking"])==("Non-smoker"))) | (df0["Smoking"].isna()))) & (((((df0["Family Plans"])==("Not sure yet"))) | (((df0["Family Plans"])==("I want children"))) | (df0["Family Plans"].isna()))) & (((((df0["Drinking"])==("Not for me"))) | (((df0["Drinking"])==("Sober"))) | (df0["Drinking"].isna()))) & (((((df0["Workout"])==("Often"))) | (((df0["Workout"])==("Everyday"))) | (((df0["Workout"])==("Gym rat"))) | (df0["Workout"].isna()))) & (((((df0["Education"])==("Bachelors"))) | (((df0["Education"])==("Masters"))) | (((df0["Education"])==("PhD"))) | (df0["Education"].isna()))))]
def computeWeight(row):
    sum=0
    try:
        if ( ((row["Smoking"])==("Non-smoker")) ):
            sum += 10
    except Exception as e:
        pass
    try:
        if ( ((row["Family Plans"])==("Not sure yet")) ):
            sum += 3
    except Exception as e:
        pass
    try:
        if ( ((row["Family Plans"])==("I want children")) ):
            sum += 10
    except Exception as e:
        pass
    try:
        if ( ((row["Drinking"])==("Not for me")) ):
            sum += 7
    except Exception as e:
        pass
    try:
        if ( ((row["Drinking"])==("Sober")) ):
            sum += 5
    except Exception as e:
        pass
    try:
        if ( ((row["Workout"])==("Often")) ):
            sum += 4
    except Exception as e:
        pass
    try:
        if ( ((row["Workout"])==("Everyday")) ):
            sum += 3
    except Exception as e:
        pass
    try:
        if ( ((row["Workout"])==("Gym rat")) ):
            sum += 3
    except Exception as e:
        pass
    try:
        if ( ((row["Education"])==("Bachelors")) ):
            sum += 5
    except Exception as e:
        pass
    try:
        if ( ((row["Education"])==("Masters")) ):
            sum += 6
    except Exception as e:
        pass
    try:
        if ( ((row["Education"])==("PhD")) ):
            sum += 7
    except Exception as e:
        pass
    return sum
df["weight"]=df.apply(computeWeight, axis=1)
df=df.sort_values(by="weight", ascending=False)
print(df[["name", "weight", "age", "bio", "num_photos"]])
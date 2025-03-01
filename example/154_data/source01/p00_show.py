#!/usr/bin/env python3
import time
import pathlib
import json
import tqdm
import pandas as pd
import numpy as np
import requests
import random
from sqlite_minutils import *
db_fns=list(pathlib.Path("data/").rglob("*.db"))
res=[]
for fn in tqdm.tqdm(db_fns):
    db=Database(fn)
    users=Table(db, "Users")
    for row in users.rows:
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
df0["age"]=df0.birth_date.apply(lambda x: (((2025)-(int(x[0:4])))) if (((type("bla"))==(type(x)))) else (0), 1)
df=df0[((((((df0["Smoking"])==("Non-smoker"))) | (((df0["Smoking"])==("Trying to quit"))) | (((df0["Smoking"])==("Smoker when drinking"))) | (df0["Smoking"].isna()))) & (((((df0["Personality Type"])==("INFJ"))) | (False))) & (((((df0["Family Plans"])==("Not sure yet"))) | (((df0["Family Plans"])==("I want children"))) | (((df0["Family Plans"])==(""))) | (((df0["Family Plans"])==(""))) | (((df0["Family Plans"])==("I don't want children"))) | (((df0["Family Plans"])==("I have children and want more"))) | (((df0["Family Plans"])==("I have children and don't want more"))) | (df0["Family Plans"].isna()))) & (((((df0["Drinking"])==("Not for me"))) | (((df0["Drinking"])==("Sober"))) | (((df0["Drinking"])==("On special occasions"))) | (((df0["Drinking"])==("Sober curious"))) | (((df0["Drinking"])==("Socially on weekends"))) | (((df0["Drinking"])==("Most Nights"))) | (((df0["Drinking"])==("I don't drink"))) | (df0["Drinking"].isna()))) & (((((df0["Workout"])==("Ocasionally"))) | (((df0["Workout"])==("Sometimes"))) | (((df0["Workout"])==("Never"))) | (((df0["Workout"])==("Often"))) | (((df0["Workout"])==("Everyday"))) | (((df0["Workout"])==("Gym rat"))) | (df0["Workout"].isna()))) & (((((df0["Education"])==("High School"))) | (((df0["Education"])==("Trade School"))) | (((df0["Education"])==("Bachelors"))) | (((df0["Education"])==("In College"))) | (((df0["Education"])==("Masters"))) | (((df0["Education"])==("In Grad School"))) | (((df0["Education"])==("PhD"))) | (df0["Education"].isna()))))]
def computeWeight(row):
    sum=0
    try:
        if ( ((row["Smoking"])==("Non-smoker")) ):
            sum += 10
    except Exception as e:
        pass
    try:
        if ( ((row["Smoking"])==("Trying to quit")) ):
            sum += 1
    except Exception as e:
        pass
    try:
        if ( ((row["Smoking"])==("Smoker when drinking")) ):
            sum += 0
    except Exception as e:
        pass
    try:
        if ( ((row["Personality Type"])==("INFJ")) ):
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
        if ( ((row["Family Plans"])==("")) ):
            sum += 3
    except Exception as e:
        pass
    try:
        if ( ((row["Family Plans"])==("")) ):
            sum += 3
    except Exception as e:
        pass
    try:
        if ( ((row["Family Plans"])==("I don't want children")) ):
            sum += 0
    except Exception as e:
        pass
    try:
        if ( ((row["Family Plans"])==("I have children and want more")) ):
            sum += 1
    except Exception as e:
        pass
    try:
        if ( ((row["Family Plans"])==("I have children and don't want more")) ):
            sum += 0
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
        if ( ((row["Drinking"])==("On special occasions")) ):
            sum += 4
    except Exception as e:
        pass
    try:
        if ( ((row["Drinking"])==("Sober curious")) ):
            sum += 4
    except Exception as e:
        pass
    try:
        if ( ((row["Drinking"])==("Socially on weekends")) ):
            sum += 3
    except Exception as e:
        pass
    try:
        if ( ((row["Drinking"])==("Most Nights")) ):
            sum += 0
    except Exception as e:
        pass
    try:
        if ( ((row["Drinking"])==("I don't drink")) ):
            sum += 7
    except Exception as e:
        pass
    try:
        if ( ((row["Workout"])==("Ocasionally")) ):
            sum += 2
    except Exception as e:
        pass
    try:
        if ( ((row["Workout"])==("Sometimes")) ):
            sum += 1
    except Exception as e:
        pass
    try:
        if ( ((row["Workout"])==("Never")) ):
            sum += 0
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
        if ( ((row["Education"])==("High School")) ):
            sum += 1
    except Exception as e:
        pass
    try:
        if ( ((row["Education"])==("Trade School")) ):
            sum += 2
    except Exception as e:
        pass
    try:
        if ( ((row["Education"])==("Bachelors")) ):
            sum += 5
    except Exception as e:
        pass
    try:
        if ( ((row["Education"])==("In College")) ):
            sum += 5
    except Exception as e:
        pass
    try:
        if ( ((row["Education"])==("Masters")) ):
            sum += 6
    except Exception as e:
        pass
    try:
        if ( ((row["Education"])==("In Grad School")) ):
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
sleep_max_for=(1.30    )
for idx, (row_idx,row,) in tqdm.tqdm(enumerate(df.iterrows())):
    for i, url in enumerate(row.images):
        req=requests.get(url, stream=True)
        if ( ((req.status_code)==(200)) ):
            with open(f"img/{idx:04}_{row._id}_{row['name']}_{i}.jpg", "wb") as f:
                f.write(req.content)
        time.sleep(((random.random())*(sleep_max_for)))
#!/usr/bin/env python3
# micromamba install requests sqlite_minutils
import time
import sys
import requests
import random
import datetime
from sqlite_minutils import *
URL="https://api.gotinder.com"
with open("token") as f:
    token=f.read().strip()
class Person(object):
    def __init__(self, data, api):
        self._api=api
        self.data=data
        self.id=data["_id"]
    def parse(self, add_data = True):
        data=self.data
        d=dict(id=self.id)
        if ( add_data ):
            d["data"]=data
        try:
            for s in data["selected_descriptors"]:
                try:
                    d[s["name"]]=s["choice_selections"][0]["name"]
                except Exception as e:
                    pass
        except Exception as e:
            pass
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
        return d
class API():
    def __init__(self, token):
        self._token=token
    def get(self, link):
        headers={}
        headers["X-Auth-Token"]=self._token
        combined_url=((URL)+(link))
        print(combined_url)
        data=requests.get(combined_url, headers=headers)
        return data.json()
    def like(self, id):
        return self.get(f"/like/{id}")
    def dislike(self, id):
        return self.get(f"/pass/{id}")
    def nearby_persons(self):
        data=self.get("/v2/recs/core")
        global q
        q=data
        return list(map(lambda user: Person(user["user"], self), data.get("data").get("results")))
datetime_str=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
db_fn=f"tide_{datetime_str}.db"
db=Database(db_fn)
users=Table(db, "Users")
schema=dict(id=str, data=str, name=str, bio=str, distance=str, birth_date=int, gender=str, images=str, family_plans=str, smoking=str, drinking=str, workout=str, education=str, personality_type=str)
users.create(columns=schema, pk="id")
api=API(token)
while (True):
    time.sleep(((random.random())*((3.20    ))))
    try:
        persons=api.nearby_persons()
        l=len(persons)
        print(f"len(persons)={l}")
        for person in persons:
            try:
                # don't update existing entry
                p=person.parse()
                users.insert(p, ignore=True)
                name=p["name"]
                smoking=p["smoking"]
                family=p["family_plans"]
                print("nil person.id={} name={} smoking={} family={}".format(person.id, name, smoking, family))
                if ( ((((((p.get("smoking"))==("Non-smoker"))) or (((p.get("smoking"))==(""))))) and (((((p.get("family_plans"))==("Not sure yet"))) or (((p.get("family_plans"))==("I want children"))) or (((p.get("family_plans"))==(""))))) and (((p.get("personality_type"))==("INFJ"))) and (((((p.get("drinking"))==("Not for me"))) or (((p.get("drinking"))==("Sober"))) or (((p.get("drinking"))==(""))))) and (((((p.get("workout"))==("Often"))) or (((p.get("workout"))==("Everyday"))) or (((p.get("workout"))==("Gym rat"))) or (((p.get("workout"))==("")))))) ):
                    print(f"liking {p['name']}")
                    like_result=api.like(person.id)
                    print(like_result)
                    if ( like_result.get("rate_limited_until", False) ):
                        sys.exit(1)
            except Exception as e:
                print(f"169: {e}")
                pass
    except Exception as e:
        print(f"162: {e}")
        pass
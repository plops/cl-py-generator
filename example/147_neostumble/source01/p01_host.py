#!/usr/bin/env python3
import pandas as pd
import datetime
import time
b=pd.read_csv("../data/beacons.csv")
c=pd.read_csv("../data/cells.csv")
w=pd.read_csv("../data/wifis.csv")
u=w.ssid.unique()
w1=w.set_index("ssid")
w2=w1.sort_values(by="signalStrength", ascending=False)
w3=w2.reset_index().set_index(["ssid", "signalStrength"])
print(w.sort_values(by="signalStrength", ascending=False).iloc[0:300])
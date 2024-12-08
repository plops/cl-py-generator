#!/usr/bin/env python3
from sqlite_minutils import *
db=Database("tide.db")
users=Table(db, "Users")
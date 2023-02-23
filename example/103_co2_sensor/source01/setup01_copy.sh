#!/bin/bash

scp -i ~/.ssh/id_slim \
    main.cpp \
    192.168.202.37:/home/martin/src/my_fancy_app_name/main

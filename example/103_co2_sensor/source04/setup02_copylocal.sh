#!/bin/bash

#cp -r \
#   /home/martin/stage/cl-py-generator/example/103_co2_sensor/source04/proto \
#   /home/martin/src/my_fancy_app_name/main
./setup03_proto.sh

cp \
    /home/martin/stage/cl-py-generator/example/103_co2_sensor/source04/gen/*.{c,h} \
    /home/martin/src/my_fancy_app_name/main
cp \
 /home/martin/stage/cl-py-generator/example/103_co2_sensor/source04/{CMake*,*.{c,cpp,h,hpp}} \
 /home/martin/src/my_fancy_app_name/main

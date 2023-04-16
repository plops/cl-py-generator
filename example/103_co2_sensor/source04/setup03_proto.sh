/usr/bin/protoc \
    -I/home/martin/src/nanopb/generator/proto/ \
    --python_out=/home/martin/stage/cl-py-generator/example/103_co2_sensor/source04/gen \
    /home/martin/src/nanopb/generator/proto/nanopb.proto



/usr/bin/protoc \
    -I/home/martin/stage/cl-py-generator/example/103_co2_sensor/source04/proto \
    -I/home/martin/src/nanopb/generator \
    -I/home/martin/src/nanopb/generator/proto \
    --plugin=protoc-gen-nanopb=/home/martin/src/nanopb/generator/protoc-gen-nanopb \
    --nanopb_opt= \
    --nanopb_out=/home/martin/stage/cl-py-generator/example/103_co2_sensor/source04/gen \
    /home/martin/stage/cl-py-generator/example/103_co2_sensor/source04/proto/data.proto

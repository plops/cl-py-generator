#!/bin/bash

./setup01_build_image.sh \
    && ./setup03_copy_from_container.sh \
    && sudo ./setup04_create_qemu.sh \
    && sudo ./setup05_run_qemu.sh 

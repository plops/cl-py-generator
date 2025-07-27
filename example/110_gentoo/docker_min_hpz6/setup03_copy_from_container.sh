# Run the docker image and copy the results (squashfs image and kernel) to the host
TODAY=$(date +%Y%m%d)
TARGET=gentoo-z6-min
mkdir -p /dev/shm/${TARGET}
cp copy_files.sh /dev/shm/${TARGET}
docker run \
--rm \
--privileged \
-v /dev/shm:/tmp/outside \
gentoo-z6-min \
/bin/bash /tmp/outside/${TARGET}/copy_files.sh

    
# Run the docker image and copy the results (squashfs image and kernel) to the host
TODAY=$(date +%Y%m%d)
TARGET=gentoo-ideapad-min_${TODAY}
mkdir -p /dev/shm/${TARGET}
cp copy_files.sh /dev/shm/${TARGET}
cp config/40_custom /dev/shm/${TARGET}
docker run \
--rm \
--privileged \
-v /dev/shm:/tmp/outside \
gentoo-ideapad-min_20250305 \
/bin/bash /tmp/outside/${TARGET}/copy_files.sh

    
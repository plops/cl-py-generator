TODAY=$(date +%Y%m%d)
TARGET=gentoo-z6-min-openrc_${TODAY}
mkdir -p /dev/shm/${TARGET}
cp copy_files.sh /dev/shm/${TARGET}
docker run \
--rm \
--privileged \
-v /dev/shm:/tmp/outside \
gentoo-z6-min-openrc \
/bin/bash /tmp/outside/${TARGET}/copy_files.sh

    
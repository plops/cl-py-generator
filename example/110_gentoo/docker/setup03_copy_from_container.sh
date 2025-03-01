# Run the docker image and copy the results (squashfs image and kernel) to the host
cp copy_files.sh /dev/shm/
docker run --rm --privileged -v /dev/shm:/tmp/outside gentoo-ideapad /bin/bash /tmp/outside/copy_files.sh

    
# Run the docker image. If you want to copy files from inside of the container
# to the host, copy them to /tmp/outside.

docker run -it --privileged \
    --tmpfs /var/tmp/portage:size=32G,mode=1777 \
    -v /dev/shm:/tmp/outside gentoo-z6-min-openrc         

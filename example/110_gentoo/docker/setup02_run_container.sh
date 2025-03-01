# Run the docker image. If you want to copy files from inside of the container
# to the host, copy them to /tmp/outside.

docker run -it --privileged -v /dev/shm:/tmp/outside gentoo-ideapad

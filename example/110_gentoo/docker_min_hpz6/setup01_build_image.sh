# Build the docker image based on a gentoo stage3 image

# use plain to show container output (e.g. user password)
#sudo cp /etc/{shadow,passwd} config
DOCKER_BUILDKIT=1 docker build -t gentoo-z6-min --progress=plain .

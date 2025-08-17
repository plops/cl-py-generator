# Build the docker image based on a gentoo stage3 image

# use plain to show container output (e.g. user password)
DOCKER_BUILDKIT=1 docker build \
    --build-arg http_proxy="http://10.60.120.81:3142" \
    -t gentoo-z6-min --progress=plain .

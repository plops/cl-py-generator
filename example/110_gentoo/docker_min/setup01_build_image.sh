# Build the docker image based on a gentoo stage3 image

# use plain to show container output (e.g. user password)
DOCKER_BUILDKIT=1 docker build \
-t gentoo-ideapad-min_20250305 \
--progress=plain \
.

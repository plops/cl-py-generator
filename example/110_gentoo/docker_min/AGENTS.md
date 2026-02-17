This is docker configuration to build a minimal Gentoo image. It is based on the official Gentoo squashfs image that can
run from ram. Execute `cd /home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min && ./setup01_build_image.sh` to build the image. The resulting image will be in the current directory with the name "gentoo_minimal.img".

# Goals
- debloat (don't use systemd)
- when iterating don't modify early stages in the docker file (or files that are loaded there), this helps reduce reruns
- occasionally consolidate docker file to reduce the overhead (every week when the stage3 version is updated)
# Introduction

- this project creates a squashfs image with gentoo that i use on a lenova ideapad
- the main component is the docker file, that performs the following tasks:

- start a stage3 (no multilib, systemd) from https://github.com/gentoo/gentoo-docker-images

- specific world set of packages that i found useful:
    - wifi configuration using iwgtk
    - support for my bluetooth headphones
    - kernel with support for the hardware of the laptop
    - firefox 
    - common lisp and emacs and a c++ compiler
    - minimal xorg with dwm
    - turn screen dark and red in the evening

- the dockerfile creates a squashfs image (~2GB), kernel and initramfs
- files are placed on an encrypted hard drive
- when booting the initramfs copies the squashfs into ram, this allows very fast startup of the programs
- any changes are made persistent by storing to another encrypted harddriver that is combined with the root filesystem of the squashfs image using overlayfs.

    



## Appendix: Useful Docker commands

```
FROM <base-image>
WORKDIR <directory-in-image>
COPY <srcs> <dst>
RUN <cmd>
CMD <cmd>

# .dockerignore to exclude directories like .git
# docker build -t <image-tag> --file Dockerfile .

# docker run -d -p 3000:3000 -e PORT=3000 <image-tag>
# -d .. detached mode in background
# -p .. bind container port to host port
# -e .. set environment variable


# docker container list
# show the running containers

# docker container list --all
# also show stopped containers

# docker logs <container-id>


# connect to running container:
# docker exec -it <container-id> bash


# docker stop <container-id>
# after stopping you can still shell into it

# docker rm <container-id>


# create a registry on azure:
# create a resource
# search; container registry

# <name>.azurecr.io

# authenticate with registry

# docker login <name>.azurecr.io --username <username> --password <password>
# docker tag <image-tag> <name>.azurecr.io/<image-tag>:latest

# check if tag was applied with: docker image list

# docker push <name>.azurecr.io/<image-tag>:latest

# instead of latest you would use the actual version number

# once the image is in the registry you can rm the local version an will later be able to get it from the registry


# docker run -d -p 3000:3000 -e PORT=3000 <name>.azurecr.io/<image-tag>:<version>


```
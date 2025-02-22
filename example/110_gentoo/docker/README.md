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

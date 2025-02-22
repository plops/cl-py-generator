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

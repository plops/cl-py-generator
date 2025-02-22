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

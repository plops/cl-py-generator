# Gentoo SquashFS Live Environment for Lenovo Ideapad

This project describes the creation and deployment of a customized Gentoo Linux environment, packaged as a SquashFS image, optimized for fast boot and persistent storage on a Lenovo Ideapad.

## Overview

The core of this project is a Dockerfile that automates the building of a minimal, yet functional Gentoo system. This system is then compressed into a SquashFS image for rapid deployment and execution.  The image is designed to be loaded into RAM during boot, resulting in significantly faster application startup times.

## System Architecture

The system utilizes the following key components:

1.  **Gentoo Stage3 Base:**  The foundation is a Gentoo Stage3 tarball (non-multilib, systemd-based) sourced from [https://github.com/gentoo/gentoo-docker-images](https://github.com/gentoo/gentoo-docker-images). This provides a minimal Gentoo system within a Docker container.

2.  **Customized World Set:** Within the Docker environment, a specific set of packages is compiled and installed, tailored for the target hardware and user preferences. This includes:
    *   **Wireless Networking:** `iwgtk` for Wi-Fi configuration.
    *   **Bluetooth Audio:** Support for Bluetooth headphones.
    *   **Hardware-Specific Kernel:** A kernel configured for the Lenovo Ideapad's hardware.
    *   **Web Browsing:** Firefox.
    *   **Development Tools:** Common Lisp, Emacs, and a C++ compiler.
    *   **Window Management:** A minimal Xorg setup with the `dwm` window manager.
    *   **Night Mode:** Functionality to adjust the screen to a dark red hue in the evening.

3.  **SquashFS Image Creation:**  The Dockerfile culminates in the creation of a SquashFS image (approximately 2GB), along with a corresponding kernel and initramfs.

4.  **Encrypted Storage:**  The generated kernel, initramfs, and SquashFS image are stored on an encrypted partition of the target laptop's hard drive.

5.  **Boot Process & OverlayFS:**
    *   During boot, the initramfs copies the SquashFS image into RAM.
    *   An OverlayFS is used to combine the read-only SquashFS root filesystem with a separate, encrypted persistent storage partition.  This allows modifications to the system to be saved across reboots.

## Building and Interacting with the Docker Image

This section details the process of building the Docker image, running a container for testing, and accessing the build artifacts.

**Prerequisites:**

*   **Docker Engine and Buildx:** Docker Engine must be installed and running.  The `docker-buildx` plugin is also required.  On a Gentoo system, install `docker-buildx` with:

    ```bash
    sudo emerge -av app-containers/docker-buildx
    ```

**Build Process:**

1.  **Navigate to the Dockerfile Directory:** Open a terminal and navigate to the directory containing the `Dockerfile` and associated files.

2.  **Execute the Build Command:** Build the image using the following command:

    ```bash
    DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 -t gentoo-ideapad .
    ```

    *   `DOCKER_BUILDKIT=1`: Enables the BuildKit builder (for performance and features).
    *   `docker buildx build`: Uses the Buildx plugin.
    *   `--platform linux/amd64`: **Crucially Important:** Specifies the target architecture (amd64) to ensure compatibility with the Lenovo Ideapad, regardless of the build machine's architecture.
    *   `-t gentoo-ideapad`: Tags the image as `gentoo-ideapad`.
    *   `.`: Specifies the current directory as the build context.

**Running a Development Container:**

After building the image, you can launch a container for testing and development.  This provides an interactive environment to inspect the built system.

```bash
docker run -it --privileged -v /dev/shm:/tmp/outside gentoo-ideapad
```

*   `-it`: Runs the container interactively with a pseudo-TTY (for shell access).
*   `--privileged`:  Grants elevated privileges to the container. **Caution:**  `--privileged` should *not* be used in production due to significant security risks. It's used here for debugging and build-process tasks that require extensive system access.
*   `-v /dev/shm:/tmp/outside`: Mounts the host's shared memory (`/dev/shm`) into the container at `/tmp/outside`.  This facilitates efficient file transfer between the host and container, as explained below. 

**Accessing Build Artifacts (Kernel, Initramfs, SquashFS):**

The build process generates the kernel, initramfs, and SquashFS image *within* the container.  The volume mount (`-v /dev/shm:/tmp/outside`) provides a convenient way to copy these artifacts to the host system:

1.  **Inside the Container:**  After building, and while running the container interactively, copy the generated files (kernel, initramfs, and the SquashFS image) to the `/tmp/outside` directory *within the container*.  Since this directory is a shared volume, the files will also appear in `/dev/shm` on your host machine.

2.  **On the Host:**  Access the files directly from your host's `/dev/shm` directory.  You can then move them to their final destination on the encrypted hard drive.

**Alternative Build and Execution Scripts:**

Helper scripts (`setup01.sh`, `setup02.sh`, `setup03.sh`, etc.) are provided to potentially automate parts of the build and execution.  Examine these scripts for their specific functionality. 






## Dockerfile Details (Key Commands)

The Dockerfile uses standard Docker commands to build the environment.  Here's a summary of common directives:

| Command        | Description                                                                       |
|----------------|-----------------------------------------------------------------------------------|
| `FROM`         | Specifies the base image (in this case, a Gentoo Stage3 image).                 |
| `WORKDIR`      | Sets the working directory within the container.                                 |
| `COPY`         | Copies files and directories from the build context into the container.         |
| `RUN`          | Executes a command within the container (e.g., for installing packages).        |
| `CMD`          | Specifies the default command to run when the container starts (often not used in build images). |

A `.dockerignore` file should be used to exclude unnecessary directories (e.g., `.git`) from the build context, improving build speed and reducing image size.

## Appendix: Docker Commands & Azure Container Registry (ACR)

This section provides a quick reference for common Docker commands and instructions for using Azure Container Registry.

### Basic Docker Commands

| Command                                  | Description                                                                                       |
|------------------------------------------|---------------------------------------------------------------------------------------------------|
| `docker build -t <image-tag> -f Dockerfile .` | Builds a Docker image from a Dockerfile.  `-t` specifies a tag, `-f` specifies the Dockerfile (defaults to `Dockerfile` in the current directory). |
| `docker run -d -p <host-port>:<container-port> -e <VAR>=<value> <image-tag>` | Runs a container. `-d` (detached) runs in the background. `-p` binds ports. `-e` sets environment variables. |
| `docker container list`                   | Lists running containers.                                                                     |
| `docker container list --all`            | Lists all containers (including stopped ones).                                                 |
| `docker logs <container-id>`             | Displays the logs of a container.                                                                 |
| `docker exec -it <container-id> bash`    | Opens an interactive shell (bash) inside a running container.                                   |
| `docker stop <container-id>`             | Stops a running container.                                                                      |
| `docker rm <container-id>`               | Removes a stopped container.                                                                    |
| `docker image list`                       | list local images |

### Azure Container Registry (ACR)

1.  **Create a Registry:** In the Azure portal, create a new "Container Registry" resource.  This will provide a private registry at `<name>.azurecr.io`.

2.  **Authentication:** Authenticate with the registry using:
    ```bash
    docker login <name>.azurecr.io --username <username> --password <password>
    ```
    (Replace `<name>`, `<username>`, and `<password>` with your ACR credentials.)

3.  **Tagging:** Tag your local image for the registry:
    ```bash
    docker tag <image-tag> <name>.azurecr.io/<image-tag>:<version>
    ```
    Use a specific version number instead of `latest` for proper version control.

4.  **Pushing:** Push the tagged image to your ACR:
    ```bash
    docker push <name>.azurecr.io/<image-tag>:<version>
    ```

5.  **Pulling (Running from ACR):**  Run a container from your ACR image:
    ```bash
    docker run -d -p <host-port>:<container-port> -e <VAR>=<value> <name>.azurecr.io/<image-tag>:<version>
    ```

6. **Removing Local Image (Optional):**
    Once pushed to ACR, you may remove the local version using: `docker rmi <name>.azurecr.io/<image-tag>:<version>`


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
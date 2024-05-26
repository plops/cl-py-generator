# Installation
1. Download the Kubler master archive from [GitHub](https://github.com/edannenberg/kubler?tab=readme-ov-file) and extract it to a directory in your home folder:

```bash
mkdir ~/tools
cd ~/tools
curl -L https://github.com/edannenberg/kubler/archive/master.tar.gz | tar xz
```

Add the Kubler binary to your PATH. You can do this for the current session with the following command, or add it to your ~/.bashrc file to make it permanent:

```bash
export PATH=$PATH:~/tools/kubler-master/bin
source ~/tools/kubler-master/lib/kubler-completion.bash 
```

Kubler will read its configuration from /etc/kubler.conf or kubler-master/kubler.conf.

By default, Kubler uses ~/.kubler as its data directory. This is where it will store downloads and custom scripts.

# Uninstall Kubler

To uninstall Kubler, you need to remove the Kubler directory and clean up the environment variables. Follow these steps:

```bash
# Clean up Kubler's data
kubler clean -N

# Remove the Kubler binary from your PATH
export PATH=$(echo $PATH | sed -e 's,:?~/tools/kubler-master/bin,,g')

# Unload Kubler bash completion
unalias kubler 2>/dev/null

# Remove the Kubler directory
rm -rf ~/tools/kubler-master

# Remove Kubler's data directory
rm -rf ~/.kubler
```

# Kubler Usage Guide

## Building an Image containing the Figlet Tool

To build an image containing the Figlet tool, follow these steps:

```bash
cd ~/projects
kubler new namespace mytest
# Select `multi` when prompted

cd kubler-images
kubler update
kubler new image mytest/figlet
# Select `kubler-bash` for the parent image and `bt` for testing when prompted

kubler build mytest/figlet -i
```

## Troubleshooting Signature Check Issues
If you encounter a signature check issue while using Kubler on Ubuntu 24.04 (amd64), you can try importing the keys manually:

```
gpg --keyserver keys.gentoo.org --recv-keys E1D6ABB63BFCFB4BA02FDF1CEC590EEAC9189250
```
￼
If the import doesn't work, you can skip the signature check with the -s flag, although this is not recommended:

```
kubler build mytest/figlet -i -v -s
```

## Optimizing Emerge Performance

If you notice that the `emerge` process is slow, consider increasing the number of jobs to utilize more cores for the compilation of individual packages. This can be done by adding `BOB_MAKEOPTS='-j32'` to your `kubler.conf` file. 

Please note, however, that not all packages have enough files to fully utilize 32 cores. 

For CPUs with a high number of cores, an alternative and potentially faster approach would be to allow `emerge` to compile multiple packages simultaneously. This can be achieved by setting `S='--jobs 24 --load-average 32'MAKEOPT` in your `kubler.conf` file.

This setting is particularly effective for compiling a large number of small packages, such as Python or Perl libraries. However, for larger packages like Rust, LLVM, or GCC, compiling with 24 jobs in parallel may consume all your memory. Unfortunately, there isn't a one-size-fits-all `MAKEOPTS` setting that is optimal for all scenarios.

## Building and Testing the Figlet Tool

- After the image has been built, you will be inside the build container for figlet (kubler/bob-bash). You can verify this by running the `eix figlet` command:

```
kubler-bob-bash / # eix figlet
* app-misc/figlet
     Available versions:  2.2.5-r1 **9999*l
     Homepage:            http://www.figlet.org/
     Description:         program for making large letters out of ordinary text
```

Inside this build container, you can use typical Gentoo tools like eix and emerge -av to investigate dependencies and use flags of the packages you want to install.

Kubler itself suggests to use `history | cut -c 8-` to collect the commands you performed here.

Once you have determined what you want to install, you can modify the build.sh file inside the container and start the build. Alternatively, you can edit the build.sh file outside of the container.


```
nano /config/build.sh

_packages="app-misc/figlet"

nano /etc/portage/make.conf

MAKEOPTS="-j32"

kubler-build-root # 6.5sec

exit
```

To test the build, edit the build-test.sh file and add the following content:

```
emacs  mytest/images/figlet/build-test.sh

#!/usr/bin/env sh
set -eo pipefail

# check figlet version string
figlet -v | grep -A 2 'FIGlet Copyright' || exit 1
	

```
Finally, build the image and time the process:
```
time kubler build mytest/figlet -nF
```
Here is the output:

```
agum:~/projects/kubler-images$ time kubler build mytest/figlet --no-deps --force-full-image-build
»[✔]»[mytest/figlet]» done.

real    0m4.021s
```

You can then run the new container in docker:
```
docker run -it --rm mytest/figlet figlet foo

@agum:~/projects/kubler-images$ docker run -it --rm mytest/figlet figlet foo
  __             
 / _| ___   ___  
| |_ / _ \ / _ \ 
|  _| (_) | (_) |
|_|  \___/ \___/ 
```
Lets have a look at some of the containers that this process produced:
```
kiel@agum:~/projects/kubler-images$ docker images 
REPOSITORY                                              TAG                IMAGE ID       CREATED             SIZE
mytest/bob-figlet                                       20240525           76fc81f071a2   2 minutes ago       1.97GB
mytest/bob-figlet                                       latest             76fc81f071a2   2 minutes ago       1.97GB
mytest/figlet                                           20240525           ed13cf1111df   2 minutes ago       50.6MB
mytest/figlet                                           latest             ed13cf1111df   2 minutes ago       50.6MB
```
The image starting with `bob-` is the build container

## dependency graph

```
 time kubler build  kubler/graph-easy 
2m57s

fails with
!!! All ebuilds that could satisfy ">=x11-libs/pango-1.12" have been masked.
!!! One of the following masked packages is required to complete your request:
- x11-libs/pango-1.52.2::gentoo (masked by: package.mask, ~amd64 keyword)

```

- if i could get graph-easy to build. the following should work

```
kubler dep-graph -b kubler/nginx mytest

```

# Create an image with some Python libraries

Here we create a Docker image with several Python libraries using the Kubler tool.

The first step is to create a new image named "mytest/lmfit" using the command `kubler new image mytest/lmfit`. When asked enter "kubler/python3" as base image and select "bt" for testing.

Next, start the "kubler/bob-python3" builder image using the command `kubler build mytest/lmfit -i`.

Specify the Python libraries that should be included in the image: lmfit, pandas, tqdm, numpy, scikit-learn, xarray, and matplotlib. Obtain the full package names using the command `emerge --search scikit`.

Note that the "zarr" library is not yet available in Gentoo's portage, so it won't be included in the image for now.

To allow the inclusion of some packages that are not fully stable yet, modify the "build.sh" file in the "mytest/images/lmfit" directory. Use the "update_keywords" function to add the required keywords for the "lmfit", "asteval", and "uncertainties" packages.

Here are my modifications of `mytest/images/lmfit/build.sh`
```
_packages="dev-python/scipy dev-python/lmfit dev-python/pandas dev-python/tqdm dev-python/numpy dev-python/scikit-learn dev-python/xarray dev-python/matplotlib"

# in configure_rootfs_build
# dependencies of matplotlib:
    update_use   'dev-python/pillow' '+webp'
    update_use   'media-gfx/imagemagick' '+jpeg' '+tiff'
    update_use   'virtual/imagemagick-tools' '+jpeg' '+tiff'
# this is needed for lmfit:
    update_keywords 'dev-python/lmfit' '+~amd64'
    update_keywords 'dev-python/asteval' '+~amd64'
    update_keywords 'dev-python/uncertainties' '+~amd64'
# inf finish_rootfs_build
# numpy needs fortran library
    cp -rp "${DISTRIBUTION_DIR}/usr/lib/gcc/x86_64-pc-linux-gnu/13/libgfortran"*".so"* "${_EMERGE_ROOT}/usr/lib64"

```

Build the image using the command
```
cd ~/projects/kubler-images
kubler build mytest/lmfit -v
```

Finally, run a new container using the "mytest/lmfit" image, and execute a Python script inside the container. However, an error occurs indicating that the "libgfortran.so.5" shared object file is missing.

```
docker run -it --rm mytest/lmfit python
import numpy
```

```
ImportError: scipy is not installed.
scikit-learn requires scipy >= 1.6.0.

```

Attempts to run `kubler build mytest/lmfit -v` have consistently failed after building 100 packages. In an attempt to find a workaround I entered the builder image using `kubler build mytest/lmfit -i` and then execute kubler-build-root within it.

Artifacts from previous build attempts can be found in the /backup-rootfs directory of the builder image.

Compiled binary packages are stored in the /packages directory. This allows for repeated calls to `kubler build mytest/lmfit` without the need to rebuild packages that have already been compiled.

Source code tar files are located in the /distfiles directory.

Successful build of scikit-learn was achieved by explicitly building it with `emerge -av scipy` followed by `emerge -av scikit-learn` in the build image. Following this, the kubler-build-root script executed successfully. This might suggest that scipy and scikit-learn should be added to the `configure_builder` function in the `build.sh` script, but further investigation is needed.

# Explanation of kubler-build-root

[Note: I consulted with Gemini 1.5 Pro for an explanation of how kubler-build-root operates, after providing the entire kubler repository for analysis. The explanation provided seemed logical, but there's a possibility that it might contain inaccuracies.]

`kubler-build-root` is the heart of Kubler's Docker build engine and the script responsible for the "heavy lifting" during the first phase of image creation. Let's break down its actions step-by-step:

**Inside the Build Container:**

1. **Environment Setup:**
   - Sources `/etc/profile` to load system-wide environment variables and settings.
   - Determines the correct `emerge` binary to use based on the target architecture (`CHOST`).
   - Sets up important paths like:
      - `_EMERGE_ROOT`: The custom root directory where packages will be installed.
      - `_CONFIG`: The mounted image directory from the host.
      - `_ROOTFS_BACKUP`: A directory to store backup files and build artifacts.

2. **Configuration from `build.sh`:**
   - Sources the `build.sh` file from the mounted image directory (`/config/build.sh`). This is where you define the packages to install (`_packages`) and customize the build process using hooks.
   - Executes the `configure_builder()` hook, if defined. This allows you to install packages specific to the build container itself (e.g., development tools or additional libraries).

3. **Preparing for Portage:**
   - Sets the `ROOT` environment variable to point to `_EMERGE_ROOT`, telling Portage to install packages there.
   - Executes the `configure_rootfs_build()` hook, if defined. This hook is for configuring Portage before installing the main packages (e.g., updating USE flags, adding patches).

4. **Package Installation:**
   - Generates a `package.installed` file, listing the packages about to be installed. This helps with dependency management for child images.
   - Runs `emerge` to install the packages specified in `_packages` into the `_EMERGE_ROOT` directory.

5. **Post-Installation:**
   - Executes the `finish_rootfs_build()` hook, if defined. Use this hook for tasks like cleanup, configuration tweaks inside the `_EMERGE_ROOT`, or manual software installation.

6. **Artifact Creation:**
   - Creates a `rootfs.tar` archive of the `_EMERGE_ROOT` directory.
   - Generates `PACKAGES.md` documentation, listing all installed packages and their versions.

7. **Cleanup:**
   - Removes temporary files and directories, leaving behind a clean image directory on the host with the `rootfs.tar` and `PACKAGES.md`.

**The Big Picture:**

`kubler-build-root` leverages the flexibility of Gentoo's Portage to install a precise set of packages and their dependencies in a controlled environment.  By setting the `ROOT` environment variable, it avoids interfering with the build container's base system while creating a custom root filesystem tailored to your image's needs. The various hooks in `build.sh` provide points for customization, giving you fine-grained control over the build process.

In essence, `kubler-build-root` does the hard work of preparing the image's filesystem, leaving Docker's `build` command to simply layer it into a container image in the second phase. This approach combines the power of Gentoo's package management with the convenience of Docker, allowing you to build small, efficient, and highly customized images. 

## How does kubler handle MAKEOPTS?

[Please note, this response is generated by Gemini 1.5 Pro. While it aims to be helpful, it may not be entirely accurate and I haven't manually verified it, yet.]

Kubler handles `MAKEOPTS` for `emerge` in a slightly nuanced way to provide both flexibility and consistency across builds. Here's the breakdown:

1. **Default `MAKEOPTS`:**
   - The default `MAKEOPTS` for the build container is defined in the `build-root.sh` script itself: `export MAKEOPTS="${BOB_MAKEOPTS:--j9}"`.
   - This sets `MAKEOPTS` to the value of the environment variable `BOB_MAKEOPTS`. If `BOB_MAKEOPTS` is not set, it defaults to `-j9`.

2. **`BOB_MAKEOPTS` from `kubler.conf`:**
   - The `BOB_MAKEOPTS` variable is typically defined in your global `kubler.conf` file or a namespace-specific `kubler.conf`. This allows you to set a default value for all your builds within that namespace or globally.

3. **Overriding with `USE_BUILDER_FLAGS`:**
   - Inside `build-root.sh`, if the `USE_BUILDER_FLAGS` environment variable is set, it sources `/etc/profile` again. This will load the `portage-defaults.sh` script located in the `engine/docker/bob-core/etc/profile.d/` directory, which is part of the Kubler core image.

4. **`portage-defaults.sh` and `DEF_BUILDER_MAKEOPTS`:**
   - `portage-defaults.sh` checks if `USE_BUILDER_FLAGS` is set. If it is, it exports `MAKEOPTS` as: `export MAKEOPTS="${DEF_BUILDER_MAKEOPTS:--j9}"`.
   - This essentially means that if `USE_BUILDER_FLAGS` is active (which is the case when using `crossdev`), the `MAKEOPTS` will be taken from the `DEF_BUILDER_MAKEOPTS` environment variable instead of `BOB_MAKEOPTS`.
   - `DEF_BUILDER_MAKEOPTS` is, like `BOB_MAKEOPTS`,  a variable defined in your `kubler.conf` files, but it's specifically intended for configuring the "host" compiler when using `crossdev`.

**In summary:**

- For normal builds, Kubler uses the `BOB_MAKEOPTS` value from `kubler.conf` (or defaults to `-j9`).
- When `crossdev` is involved, Kubler switches to using `DEF_BUILDER_MAKEOPTS` from `kubler.conf` to configure the "host" compiler.
- The actual setting of `MAKEOPTS` happens through sourcing `/etc/profile` within the build container, which loads the necessary scripts to export the correct value.

**Important Note:**  Kubler does not directly read `/etc/portage/make.conf` for `MAKEOPTS`. The values in `kubler.conf` are used to dynamically set the environment variable within the build container.


# create an image with x11

```
cd ~/project_ram
kubler new namespace gentooram
# multi

git init /home/kiel/project_ram/kubler-images/gentooram

kiel@agum:~/project_ram/kubler-images/gentooram$ git add .gitignore README.md kubler.conf 
kiel@agum:~/project_ram/kubler-images/gentooram$ git commit -m "initial commit"

»[!]» To create images in the new namespace run:
»»»
»»» $ cd /home/kiel/project_ram/kubler-images
    $ kubler new image gentooram/<image_name>

kiel@agum:~/project_ram/kubler-images$ kubler new image gentooram/x11
»»»
»»» <enter> to accept default value
»»»
»»» Extend an existing Kubler managed image? Fully qualified image id (i.e. kubler/busybox) or scratch
»[?]» Parent Image (scratch): kubler/bash
»»»
»»» Add test template(s)? Possible choices:
»»»   hc  - Add a stub for Docker's HEALTH-CHECK, recommended for images that run daemons
»»»   bt  - Add a stub for a custom build-test.sh script, a good choice if HEALTH-CHECK is not suitable
»»»   yes - Add stubs for both test types
»»»   no  - Fck it, we'll do it live!
»[?]» Tests (hc): bt
»»»
»[✔]» Successfully created new image at /home/kiel/project_ram/kubler-images/gentooram/images/x11
»»»

emacs gentooram/kubler.conf
BOB_MAKEOPTS='-j32'


time kubler build gentooram/x11 -s
```

- i want to have systemd instead of openrc. don't know yet how to configure this

# References

1. [Kubler on GitHub](https://github.com/edannenberg/kubler?tab=readme-ov-file)
2. [Elttam Blog on Kubler](https://www.elttam.com/blog/kubler/#content)
3. [Kubler YouTube Video](https://youtu.be/bbC6HXUUjjg)
4. [Kubler on Gentoo Wiki](https://wiki.gentoo.org/wiki/Kubler)
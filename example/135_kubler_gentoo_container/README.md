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

For CPUs with a high number of cores, an alternative and potentially faster approach would be to allow `emerge` to compile multiple packages in parallel. This can be achieved by setting `MAKEOPTS='--jobs 8 --load-average 32'` in your `kubler.conf` file.

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
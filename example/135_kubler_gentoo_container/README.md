# installation
- from https://github.com/edannenberg/kubler?tab=readme-ov-file
```
mkdir ~/tools
cd ~/tools
curl -L https://github.com/edannenberg/kubler/archive/master.tar.gz | tar xz

# run the following or add to ~/.bashrc
export PATH=$PATH:~/tools/kubler-master/bin
source ~/tools/kubler-master/lib/kubler-completion.bash 

# configuration will be read  from /etc/kubler.conf
# or kubler-master/kubler.conf 

# data dir is ~/.kubler by default. this will hold downloads or custom scripts


sudo usermod -aG docker $USER
# log $USER out and log back in 
```


# remove kubler

```
export PATH=$PATH:~/tools/kubler-master/bin
source ~/tools/kubler-master/lib/kubler-completion.bash 

kubler clean -N
rm -rf ~/tools/kubler-master
rm -rf ~/.kubler
```


# try kubler

```
cd ~/projects
kubler new namespace mytest

# multi
cd kubler-images
kubler update
kubler new image mytest/figlet
# kubler-bash
# bt

# 

kubler build mytest/figlet -i

```


```
agum:~/projects/kubler-images$ kubler build mytest/figlet -i
»»»»»[init]» generate build graph for interactive build of mytest/figlet
»»» required engines:    docker
»»» required stage3:     stage3-amd64-musl-hardened stage3-amd64-hardened-nomultilib-openrc
»»» required builders:   kubler/bob-musl kubler/bob
»»» build sequence:      kubler/busybox kubler/glibc kubler/s6 kubler/openssl kubler/bash
»[✔]»[init]» done.
»[⠼]»[portage]» download portage snapshot [ 45M ]

# why am i not in the interactive build container?
```

- the flag `-v` says `»[✘]»[portage]» Signature check failed`

```
gpg --keyserver keys.gentoo.org --recv-keys E1D6ABB63BFCFB4BA02FDF1CEC590EEAC9189250

```

- this import doesn't work
- apparently i can skip the key check with flag `-s`, i don't like it
  but this is what i'll do

```
kubler build mytest/figlet -i -v -s

```

- emerge is quite slow. looks like it is not taking advantage of all
  the cores

```
kubler-bob-bash / # eix figlet
* app-misc/figlet
     Available versions:  2.2.5-r1 **9999*l
     Homepage:            http://www.figlet.org/
     Description:         program for making large letters out of ordinary text

```

```
nano /config/build.sh

_packages="app-misc/figlet"

nano /etc/portage/make.conf

MAKEOPTS="-j32"

kubler-build-root # 6.5sec

exit

emacs  mytest/images/figlet/build-test.sh

#!/usr/bin/env sh
set -eo pipefail

# check figlet version string
figlet -v | grep -A 2 'FIGlet Copyright' || exit 1
	
time kubler build mytest/figlet -nF

```

```
agum:~/projects/kubler-images$ time kubler build mytest/figlet --no-deps --force-full-image-build
»[✔]»[mytest/figlet]» done.

real    0m4.021s

```

```
docker run -it --rm mytest/figlet figlet foo

@agum:~/projects/kubler-images$ docker run -it --rm mytest/figlet figlet foo
  __             
 / _| ___   ___  
| |_ / _ \ / _ \ 
|  _| (_) | (_) |
|_|  \___/ \___/ 
                 
kiel@agum:~/projects/kubler-images$ docker images 
REPOSITORY                                              TAG                IMAGE ID       CREATED             SIZE
mytest/bob-figlet                                       20240525           76fc81f071a2   2 minutes ago       1.97GB
mytest/bob-figlet                                       latest             76fc81f071a2   2 minutes ago       1.97GB
mytest/figlet                                           20240525           ed13cf1111df   2 minutes ago       50.6MB
mytest/figlet                                           latest             ed13cf1111df   2 minutes ago       50.6MB
```
- the image starting with bob- is the build container

## dependency graph

```
 time kubler build  kubler/graph-easy 

```

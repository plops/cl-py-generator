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

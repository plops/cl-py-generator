
# Current Build information

#64 195.2 Filesystem size 7664398.69 Kbytes (7484.76 Mbytes)
#64 195.2       36.94% of uncompressed filesystem size (20746320.61 Kbytes)

real    142m11.172s


# How to configure a kernel given the output of lsmod of another kernel

make LSMOD=~/lsmod.txt localmodconfig


# How to configure colemak keyboard layout in the linux command line

Open your terminal and type the following command:
```bash
loadkeys /usr/share/keymaps/i386/colemak/en-latin9.map.gz 
```

Your keyboard layout should now be set to Colemak for the current TTY
session. You can test this by typing in the console. To switch back to
the standard US QWERTY layout temporarily, you can use `loadkeys us`.

# GCC 16 errors

I switched to the experimental GCC 16 and had an error compiling jemalloc.
Annoyingly, such an error aborts `emerge -e world` and I have to start over again after fixing the error. 

The following command creates a list of packages that will be compiled by `emerge -e world`, perhaps I could split
the build into multiple docker RUN stages so that I can have it pick up from an intermediate stage instead of starting over again after fixing the error.
```
emerge -pqe @world
```

Unreleased GCCs default to extra runtime checks even with USE=-debug. 
The checks (sometimes substantially) increase build time but provide important protection.
This behaviour can be disabled with `GCC_CHECKS_LIST="release"`


Messages for package dev-util/nvidia-cuda-toolkit-12.9.1-r1:                                                                                       
gcc > 14 will not work with CUDA


# Misc

```
#56 1645.7  * Messages for package net-wireless/bluez-5.85:                     
#56 1645.7                                                                                                             
#56 1645.7  *   CONFIG_BT_RFCOMM_TTY:    is not set when it should be.

#56 1645.7  * Messages for package media-sound/pulseaudio-daemon-17.0-r1:
#56 1645.7                                                                                                             
#56 1645.7  * You have enabled bluetooth USE flag for pulseaudio. Daemon will now handle                     
#56 1645.7  * bluetooth Headset (HSP HS and HSP AG) and Handsfree (HFP HF) profiles using                    
#56 1645.7  * native headset backend by default. This can be selectively disabled
#56 1645.7  * via runtime configuration arguments to module-bluetooth-discover                               
#56 1645.7  * in /etc/pulse/default.pa                                                                                 
#56 1645.7  * To disable HFP HF append enable_native_hfp_hf=false
#56 1645.7  * To disable HSP HS append enable_native_hsp_hs=false                                            
VHOST_NET#56 1645.7  * To disable HSP AG append headset=auto or headset=ofono         
```

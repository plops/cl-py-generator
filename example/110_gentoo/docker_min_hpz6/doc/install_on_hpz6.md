in this file i will describe how to install the image on a HP Z6 workstation. it is not an installation from scratch,
but i will use a different persistent partition than the one used in the existing installation. this changes is so
fundamental that it is basically like a new installation, but it is not a clean-slate installation because the existing GRUB
will be used (which i typically install with an ubuntu or fedora anyway).

we booted into an existing linux and look at the disks first:

```
$ lsblk
NAME                        MAJ:MIN RM   SIZE RO TYPE  MOUNTPOINTS
loop0                         7:0    0   7.6G  0 loop  
nvme1n1                     259:0    0 953.9G  0 disk  
├─nvme1n1p1                 259:2    0     1G  0 part  
├─nvme1n1p2                 259:3    0     2G  0 part  
└─nvme1n1p3                 259:4    0 950.8G  0 part  
  └─vg                      253:0    0 950.8G  0 crypt 
    └─ubuntu--vg-ubuntu--lv 253:1    0 950.8G  0 lvm   
nvme0n1                     259:1    0 953.9G  0 disk  
├─nvme0n1p1                 259:5    0     1G  0 part  
├─nvme0n1p2                 259:6    0     8G  0 part  
├─nvme0n1p3                 259:7    0 195.3G  0 part  
├─nvme0n1p4                 259:8    0  22.5G  0 part  
└─nvme0n1p5                 259:9    0 687.1G  0 part  /mnt5
```

i deceided that i will clear nvme0n1p5 and replace it with a luks partition containing an ext4 filesystem.
this will become the persistent root partition

we still need a place to store the kernel, initramfs, and squashfs image. and we will have to configure the pre-existing GRUB.

ideally i want kernel, initramfs and squashfs on the same unencrypted disk. i'm not worried about being attacked, i'm
worried about the drive being stolen. the only data that needs protection is on the persistent layer, and that will be encrypted.


i have already a pre-existing installation that is currently running.
Schauen wir mal von wo der aktuelle Kernel gebootet wurde. Das kann ich mir nie merken
kiel@localhost ~/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6/gentoo-z6-min_20260225 $ cat /proc/cmdline 
BOOT_IMAGE=/boot/vmlinuz root=UUID=4f708c84-185d-437b-a03a-7a565f598a23 ro squashfs.part=/dev/disk/by-label/gentoo squashfs.file=gentoo.squashfs persist.part=/dev/disk/by-uuid/42bbab57-7cb0-465f-8ef9-6b34379da7d3 persist.lv=/dev/mapper/ubuntu--vg-ubuntu--lv persist.mapname=vg

Aha, die Position hat ein Label gent. Wo zeigt das genau hin?

kiel@localhost ~/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6/gentoo-z6-min_20260225 $ ls -ltr /dev/disk/by-label/gentoo
lrwxrwxrwx 1 root root 15 Feb 27 13:18 /dev/disk/by-label/gentoo -> ../../nvme0n1p3

Gut, das ist Position 3 der ersten NVME Festplatte. Die monten wir jetzt mal.


localhost /home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6/gentoo-z6-min_20260225 # mount /dev/nvme0n1p3 /0p3/
localhost /home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6/gentoo-z6-min_20260225 # ls /0p3
bin   dev  etc              gentoo.squashfs_0206  home  lib64  mnt   opt   root  sbin  sys  tools  var
boot  efi  gentoo.squashfs  gentoo.squashfs_1124  lib   media  mnt1  proc  run   srv   tmp  usr

Im Root verzeichnet befinden sich die SquashfS-Dateien. Ich werde neuen Squash FS ist immer das aktuelle Datum im Format MMDD geben. So dass ich zur Not wieder zurückgehen kann. Nach einem Update.

Als nächstes schauen wir uns das Bootverzeichnis dieser Position an.

localhost /home/kiel/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6/gentoo-z6-min_20260225 # ls /0p3/boot/
0806                                        initramfs-6.6.67-gentoo-gentoo-dist.img  initramfs_squash_sda1-x86_64.img_1023  kernel-6.6.67-gentoo-gentoo-dist
System.map                                  initramfs-6.6.74-gentoo-gentoo-dist.img  initramfs_squash_sda1-x86_64.img_1026  kernel-6.6.74-gentoo-gentoo-dist
amd-uc.img                                  initramfs-hpz6-x86_64.img_0206           initramfs_squash_sda1-x86_64.img_1104  vmlinuz
grub                                        initramfs-with-ssh.img                   initramfs_squash_sda1-x86_64.img_1124  vmlinuz_0121
initramfs-6.12.16-gentoo-gentoo-dist.img    initramfs_squash_from_disk.img           kernel-6.12.16-gentoo                  vmlinuz_0128
initramfs-6.12.16-gentoo.img                initramfs_squash_from_disk.img_1026      kernel-6.12.16-gentoo-gentoo-dist      vmlinuz_0206
initramfs-6.12.21-gentoo-gentoo-dist.img    initramfs_squash_from_disk.img_1104      kernel-6.12.21-gentoo-gentoo-dist      vmlinuz_0926
initramfs-6.12.31-gentoo-gentoo-dist.img    initramfs_squash_sda1-x86_64.img         kernel-6.12.31-gentoo-gentoo-dist      vmlinuz_1023
initramfs-6.6.52-gentoo-gentoo-dist.img     initramfs_squash_sda1-x86_64.img_0121    kernel-6.6.52-gentoo-gentoo-dist       vmlinuz_1026
initramfs-6.6.58-gentoo-r1-gentoo-dist.img  initramfs_squash_sda1-x86_64.img_0128    kernel-6.6.58-gentoo-r1-gentoo-dist    vmlinuz_1104
initramfs-6.6.62-gentoo-gentoo-dist.img     initramfs_squash_sda1-x86_64.img_0926    kernel-6.6.62-gentoo-gentoo-dist       vmlinuz_1124


Gut, hier befinden sich die VM Linus Dateien mit dem Unterstrich und die Innendramefest Dateien, die zu unserem Gento-System gehören. Da werden wir auch die neuen Dateien hinlegen.
Als nächstes schauen wir ob die Gruppkonfiguration auf dieser Position auch die richtige ist.

localhost /0p3/boot/grub # cat grub.cfg |grep -i gentoo -A 10|head -n 12
menuentry 'Gentoo from ram (Configurable) nvme0n1p3' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-4f708c84-185d-437b-a03a-7a565f598a23' {
        load_video
        insmod gzio
        insmod part_gpt
        insmod btrfs
        search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23
        echo    'Loading Linux 6.12.31-gentoo-gentoo-dist ...'
        linux   /boot/vmlinuz root=UUID=4f708c84-185d-437b-a03a-7a565f598a23 ro squashfs.part=/dev/disk/by-label/gentoo squashfs.file=gentoo.squashfs persist.part=/dev/disk/by-uuid/42bbab57-7cb0-465f-8ef9-6b34379da7d3 persist.lv=/dev/mapper/ubuntu--vg-ubuntu--lv persist.mapname=vg 
        echo    'Loading initial ramdisk ...'
        initrd  /boot/initramfs_squash_sda1-x86_64.img
}

Ja, das sieht gut aus. Das ist meine ursprüngliche Konfiguration. Wir sehen, dass die Variable squashfs.part auf die Disk mit Labelgänge zeigt.


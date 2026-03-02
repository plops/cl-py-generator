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

Gut, das ist Position 3 der ersten NVME Festplatte. Die mounten wir jetzt mal.


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

Ich habe die aktuellen Bildresultate bereits mit dem Skript setup03_copy_from_container.sh aus dem Docker Container herauskopiert.
Sie befinden sich in folgendem verzeichnis.

kiel@localhost ~/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6/gentoo-z6-min_20260225 $ ls -lh
total 1.4G
-rwxrwxrwx 1 kiel kiel 1.4G Mar  2 09:38 gentoo.squashfs
-rwxrwxrwx 1 kiel kiel  13M Mar  2 09:38 initramfs-hpz6-x86_64.img
-rwxrwxrwx 1 kiel kiel  13M Mar  2 09:38 initramfs_squash_sda1-x86_64.img
-rwxrwxrwx 1 kiel kiel  20K Mar  2 09:38 packages.txt
-rwxrwxrwx 1 kiel kiel  15M Mar  2 09:38 vmlinuz

Dieser Build wurde mit Portage von 0225 erzeugt. Daher wird dieses Suffix für die Dateien verwendet werden.
Wir brauchen nur die initramfs mit squash im namen. Von den initramFS-Dateien. Die mit HPZ6 im namen ist nur ein Experiment, was nicht funktioniert hat.


# Neue Dateien mit Datums-Suffix ablegen (Rollback-freundlich)
sudo cp -av gentoo.squashfs /0p3/gentoo.squashfs_0225
sudo cp -av vmlinuz /0p3/boot/vmlinuz_0225
sudo cp -av initramfs_squash_sda1-x86_64.img /0p3/boot/initramfs_squash_sda1-x86_64.img_0225


Der schwierigste Schritt wird gleich die GRUB-Konfiguration anzugehen. Dabei ist der wesentliche Punkt, die UUIDs für die einzelnen Festplattenpartitionen herauszufinden. Aber vorher müssen wir die persistente Partition vorbereiten.

export PERSIST_PART=/dev/nvme0n1p5
export CRYPT_NAME=persist

# 1) LUKS-Container anlegen
sudo cryptsetup luksFormat --type luks2 "$PERSIST_PART"

# 2) LUKS-Container öffnen -> /dev/mapper/persist
sudo cryptsetup open "$PERSIST_PART" "$CRYPT_NAME"

# 3) ext4 im entschlüsselten Device erstellen
sudo mkfs.ext4 -L persist "/dev/mapper/$CRYPT_NAME"

# 4) Mountpoint anlegen und testen
sudo mkdir -p /mnt/persist
sudo mount "/dev/mapper/$CRYPT_NAME" /mnt/persist

# 5) UUIDs notieren (für Kernel-Parameter/Config)
sudo blkid "$PERSIST_PART" "/dev/mapper/$CRYPT_NAME"

# 6) Optional: sauber aushängen und schließen
sudo cryptsetup close "$CRYPT_NAME"

Hier sind die UUIDs für die neuen Partitionen.

kiel@localhost ~/stage/cl-py-generator/example/110_gentoo/docker_min_hpz6/gentoo-z6-min_20260225 $ sudo blkid "$PERSIST_PART" "/dev/mapper/$CRYPT_NAME"
/dev/nvme0n1p5: UUID="0d7c5e23-6bab-4dce-b744-a5d61d497aca" TYPE="crypto_LUKS" PARTLABEL="docker" PARTUUID="6ea05cc3-7d72-45fb-8305-7fbddd0781e2"
/dev/mapper/persist: LABEL="persist" UUID="3ca5dfb2-35c9-4ed5-906a-f965dbcd1c7b" BLOCK_SIZE="4096" TYPE="ext4"

Die UUID für die Position von wo das squash FS geladen wird, bleibt dieselbe wie vorher.

$ sudo blkid /dev/disk/by-label/gentoo
/dev/disk/by-label/gentoo: LABEL="gentoo" UUID="4f708c84-185d-437b-a03a-7a565f598a23" UUID_SUB="02afc0f2-fe58-4d88-8cbd-0bb98ad50d74" BLOCK_SIZE="4096" TYPE="btrfs" PARTLABEL="/" PARTUUID="c3240922-6cf5-431f-8238-a7cae7e72746"

Im Grub müssen wir die uuid eintragen, die mit 4f708... beginnt


Jetzt müssen wir den neuen Eintrag für die Grubkonfiguration schreiben.
Im Gegensatz zu der alten Installation, wo ich das DRACuT-Skript manuel neu geschrieben hatte um persistente partition auf LVM->LUKS->Ext4 zu erlauben, benutze ich jetzt das Original Dracut. Daher müssen sich die Parameter etwas ändern.

Ich habe es beim besten Billen nicht hinbekommen, mit dem Original Draw Card die persistente Position in einem LVM-Container zu haben. Aber das ist eigentlich nicht so wichtig. Wichtiger ist, dass die Konfiguration robust ist und sich nicht bei jedem Update ändern kann. Daher der Umbau jetzt.

In ../grub.txt habe ich beispiele fuer die neun Grubeinträge. Diese sind aber für einen anderen Computer . wir nehmen
beispiel `Gentoo Dracut (Fixed)`



menuentry 'Gentoo Dracut (persist on nvme0n1p5)' {
    insmod part_gpt
    insmod fat
    insmod btrfs
    search --no-floppy --fs-uuid --set=root 4f708c84-185d-437b-a03a-7a565f598a23
    
    linux /boot/vmlinuz \
    root=live:UUID=4f708c84-185d-437b-a03a-7a565f598a23 \
    rd.live.dir=/ \
    rd.live.squashimg=gentoo.squashfs_0225 \
    rd.live.ram=1 \
    rd.luks.uuid=0d7c5e23-6bab-4dce-b744-a5d61d497aca \
    rd.luks.name=0d7c5e23-6bab-4dce-b744-a5d61d497aca=enc \
        rd.overlay=/dev/mapper/enc:persistent \
    rd.live.overlay.overlayfs=1
    initrd /boot/initramfs_squash_sda1-x86_64.img_0225
}


Ich bin mir nicht sicher, was für ein Passwort im Squash festgesetzt wird. Das ist auch nicht wichtig. Am besten ist, wir überschreiben einfach das Passwort in der persistenten Partition mit den existierenden.

sudo mount "/dev/mapper/$CRYPT_NAME" /mnt
sudo mkdir -p /mnt/etc
sudo cp -av /etc/shadow /mnt/etc/shadow
sudo cp -av /etc/passwd /mnt/etc/passwd
sudo cp -av /etc/group /mnt/etc/group
sudo umount /mnt


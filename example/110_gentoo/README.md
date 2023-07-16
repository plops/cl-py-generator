# script to install gentoo into a chroot

- this file documents the process of creating a rather minimal gentoo
  on a laptop that has been booted with another linux
  distribution. the initial linux distribution is helpful because we
  can use its general kernel to identify which kernel modules are used
  in this particular system (and which ones aren't)

- i am running this on an arch linux 

## requirements

### must support
- ideapad 3 15ABA7
- usb tether with android device
- bluetooth headphones
- sbcl
- external usb harddisk
- python: pandas, numpy, opencv, lmfit, tqdm
- charge battery at most to 80% (requires tlp)

- rust is slowing the whole compilation down. so perhaps install firefox-bin and rust-bin

### can support (nice to have)
- nvme harddiskø
- wifi rtw_8852be. at time of writing 6.1 is gentoo stable and has no support. i fix kernel to 6.3.12, because i don't want to recompile kernel every week. 
- webcam
- bluetooth network tether with android
- rootfs on squashfs

## install instructions

```
wget https://bouncer.gentoo.org/fetch/root/all/releases/amd64/autobuilds/20230611T170207Z/stage3-amd64-nomultilib-systemd-mergedusr-20230611T170207Z.tar.xz
mkdir gentoo
cd gentoo
sudo tar xpvf ../stage3-*.tar.xz --xattrs-include='*.*' --numeric-owner
# 1.2GB extracted



cd /mnt/gentoo

# create /etc/portage/env directory if it doesn't exist
mkdir -p etc/portage/env

# create /etc/portage/env/rustenv and add MAKEOPTS
# rust needs 4GB per process
echo "MAKEOPTS=\"-j3\"" > etc/portage/env/rustenv

# add dev-lang/rust to /etc/portage/package.env
echo "dev-lang/rust rustenv" >> etc/portage/package.env


#EMERGE_DEFAULT_OPTS="--jobs 4 --load-average 8"
#MAKEOPTS="-j2"

cat << EOF > etc/portage/make.conf
COMMON_FLAGS="-march=native -fomit-frame-pointer -O2 -pipe"
CFLAGS="${COMMON_FLAGS}"
CXXFLAGS="${COMMON_FLAGS}"
FCFLAGS="${COMMON_FLAGS}"
FFLAGS="${COMMON_FLAGS}"
LC_MESSAGES=C.utf8
MAKEOPTS="-j12"
USE="X"
VIDEO_CARDS="radeon"
FEATURES="buildpkg"
PKGDIR="/var/cache/binpkgs"
BINPKG_FORMAT="gpkg"
BINPKG_COMPRESS="zstd"
BINPKG_COMPRESS_FLAG_ZSTD="-T0"
L10N="en-GB"
LLVM_TARGETS="X86 AMDGPU"
EOF

cp --dereference /etc/resolv.conf /mnt/gentoo/etc/
mount --types proc /proc /mnt/gentoo/proc
mount --rbind /sys /mnt/gentoo/sys
mount --make-rslave /mnt/gentoo/sys
mount --rbind /dev /mnt/gentoo/dev
mount --make-rslave /mnt/gentoo/dev
mount --bind /run /mnt/gentoo/run
mount --make-slave /mnt/gentoo/run 

chroot /mnt/gentoo /bin/bash 
source /etc/profile 
export PS1="(chroot) ${PS1}"

# /boot partition should be mounted here (but i don't do that)

emerge-webrsync
# eselect profile list
emerge --ask=n --verbose --update --deep --newuse @world
emerge --depclean

# less /var/db/repos/gentoo/profiles/use.desc

emerge --ask=n app-portage/cpuid2cpuflags
echo "*/* $(cpuid2cpuflags)" > /etc/portage/package.use/00cpu-flags

ln -sf ../usr/share/zoneinfo/Europe/Zurich /etc/localtime

cat <<EOF > /etc/locale.gen
en_US ISO-8859-1
en_US.UTF-8 UTF-8
EOF
locale-gen
eselect locale set C.UTF8

env-update && source /etc/profile && export PS1="(chroot) ${PS1}"

cat <<EOF > /etc/portage/package.license
sys-kernel/linux-firmware @BINARY-REDISTRIBUTABLE
EOF

emerge --ask=n sys-kernel/linux-firmware
# rm /bin/cpio # what is up with that?
emerge sys-kernel/genkernel

emerge sys-kernel/gentoo-sources
eselect kernel set 1
cd /usr/src/linux
# make sure all the devices (wifi, bluetooth, usb...) are connected and
# running in the host system. i used arch. initially i forgot to attach
# my external ssd to the usb port and then the resulting kernel couldn't
# boot from it
make localmodconfig
cp .config /usr/src/linux-config
genkernel --kernel-config=/usr/src/linux-config --microcode=amd --lvm --luks all

cat << EOF > /etc/portage/package.accept_keywords/package.accept_keywords
virtual/dotnet-sdk ~amd64
net-wireless/iwgtk  ~amd64
sys-kernel/gentoo-sources ~amd64
sys-kernel/linux-headers ~amd64
sys-power/tlp ~amd64
dev-python/lmfit ~amd64
dev-python/asteval ~amd64
dev-python/uncertainties ~amd64
EOF

cat << EOF > /etc/portage/package.mask/package.mask
>=sys-kernel/gentoo-sources-6.3.13
>=sys-kernel/linux-headers-6.3.13
EOF

cat << EOF > /etc/portage/package.use/package.use
www-client/firefox -clang -gmp-autoupdate -openh264 system-av1 system-harfbuzz system-icu system-jpeg system-libevent -system-libvpx -system-webp -dbus -debug -eme-free -geckodriver -hardened -hwaccel -jack -libproxy -lto -pgo pulseaudio -screencast -selinux -sndio -system-png -system-python-libs -wayland -wifi
www-client/chromium X -hangouts -official -pic -proprietary-codecs suid system-harfbuzz system-icu system-png -component-build -cups -custom-cflags -debug -gtk4 -headless -kerberos -libcxx -lto -pax-kernel -pgo -pulseaudio -qt5 -screencast -selinux -system-av1 -system-ffmpeg -vaapi -wayland -widevine
x11-base/xorg-server systemd udev xorg -debug -elogind -minimal -selinux -suid -test -unwind -xcsecurity -xephyr -xnest -xvfb
app-editors/emacs -acl gmp inotify ssl systemd threads xpm zlib Xaw3d -alsa -aqua athena -cairo -dbus dynamic-loading -games -gfile -gif -gpm -gsettings -gtk -gui -gzip-el -harfbuzz -imagemagick -jit -jpeg -json -kerberos -lcms -libxml2 -livecd -m17n-lib -mailutils -motif -png -selinux -sound -source -svg -tiff -toolkit-scroll-bars -valgrind -wide-int -xft -xwidgets
x11-terms/xterm openpty unicode -Xaw3d -sixel -toolbar -truetype -verify-sig -xinerama
net-wireless/bluez -mesh -obex readline systemd udev -btpclient -cups -debug -deprecated -doc -experimental -extra-tools -midi -selinux -test -test-programs 
net-wireless/iwd client -crda -monitor systemd -ofono -standalone -wired
net-misc/dhcp client ipv6 -server ssl -ldap -selinux -vim-syntax
dev-vcs/git blksha1 curl gpg iconv nls pcre -perl safe-directory -webdav -cgi -cvs -doc -highlight -keyring -mediawiki -perforce -selinux -subversion -test tk -xinet
sci-libs/nlopt -cxx -guile -octave python -test
dev-python/numpy lapack -test
sci-libs/openblas openmp -dynamic -eselect-ldso -index-64bit pthread -relapack -test
media-libs/opencv eigen features2d openmp python -contrib -contribcvv -contribdnn -contribfreetype -contribhdf -contribovis -contribsfm -contribxfeatures2d -cuda -debug -dnnsamples -download -examples ffmpeg -gdal -gflags -glog -gphoto2 gstreamer -gtk3 -ieee1394 -java -jpeg -jpeg2k -lapack -lto opencl -opencvapps -openexr -opengl -png -qt5 -tesseract -testprograms -threads -tiff v4l -vaapi -vtk -webp -xine
dev-python/matplotlib -cairo -debug -doc -examples -excel -gtk3 -latex -qt5 -test -tk -webagg -wxwidgets
dev-python/pandas X -doc -full-support -minimal -test
dev-lang/python ensurepip gdbm ncurses readline sqlite ssl -bluetooth -build -debug -examples -hardened -libedit -lto -pgo -test tk -valgrind -verify-sig
dev-python/pillow jpeg zlib -debug -examples -imagequant -jpeg2k -lcms -test -tiff tk -truetype webp -xcb
media-gfx/imagemagick X bzip2 cxx openmp png zlib -corefonts -djvu -fftw -fontconfig -fpx -graphviz -hdri -heif -jbig jpeg -jpeg2k jpegxl -lcms -lqr -lzma -opencl -openexr -pango -perl -postscript -q8 -q32 -raw -static-libs -svg -test tiff -truetype webp -wmf -xml -zip
virtual/imagemagick-tools jpeg -perl -png -svg tiff
dev-lang/rust clippy -debug -dist -doc -llvm-libunwind -miri -nightly parallel-compiler -profiler rust-analyzer rust-src rustfmt -system-bootstrap system-llvm -test -verify-sig -wasm
media-plugins/alsa-plugins mix usb_stream -arcam_av -debug -ffmpeg -jack -libsamplerate -oss pulseaudio -speex
media-libs/libaom -examples -doc -test
sys-kernel/dracut -selinux -test
media-sound/pulseaudio glib bluetooth -daemon -jack ofono-headset
media-libs/libcanberra gtk3 sound udev alsa pulseaudio
net-wireless/blueman nls network -policykit pulseaudio
media-libs/libpulse X asyncns glib systemd dbus -doc -gtk -selinux -test -valgrind
media-sound/pulseaudio-daemon X alsa alsa-plugin asyncns gdbm glib orc ssl systemd udev webrtc-aec -aptx bluetooth dbus -elogind -equalizer -fftw -gstreamer -jack -ldac -lirc ofono-headset -oss -selinux -sox -system-wide -tcpd -test -valgrind -zeroconf
net-misc/ofono atmodem cdmamodem datafiles isimodem phonesim provision qmimodem udev bluetooth -doc -dundee -examples -tools -upower
dev-python/lmfit -test
dev-python/tqdm -examples -test
x11-wm/dwm savedconfig -xinerama

EOF

# charge battery at most to 80%
cat /etc/tlp.conf|grep CHARGE_TH
START_CHARGE_THRESH_BAT0=75
STOP_CHARGE_THRESH_BAT0=80

# eselect blas set openblas
# eselect lapack set openblas

# emerge -av --fetchonly # downloads all source packages without compilation

emerge -av xorg-server firefox \
gentoolkit eix \
dwm xterm \
emacs sbcl slime \
magit paredit \
bluez iwd dhcp \
dev-vcs/git \
dev-python/pip \
numpy scipy scikit-learn nlopt matplotlib opencv python 

eix-update
# only install what isn't already there
emerge -av $(for pkg in xorg-server firefox gentoolkit dwm xterm emacs sbcl slime magit paredit bluez iwd dhcp dev-vcs/git dev-python/pip numpy scipy scikit-learn nlopt matplotlib opencv python lmfit tqdm ofono pulseaudio-daemon pulseaudio blueman dracut iwgtk; do eix -I "$pkg" >/dev/null || echo "$pkg"; done)


emacs /etc/portage/savedconfig/x11-wm/dwm-6.4
#define MODKEY Mod4Mask
static const char *termcmd[]  = { "xterm", NULL };


emerge --ask --verbose --update --newuse --deep --with-bdeps=y @world 

#12GB for rust, 6.6GB for firefox
emerge -uDN @world --buildpkg --buildpkg-exclude "virtual/* sys-kernel/*-sources"
emerge @world --buildpkg --buildpkg-exclude "virtual/* sys-kernel/*-sources"

```
# boot from squashfs

https://unix.stackexchange.com/questions/235145/how-to-boot-using-a-squashfs-image-as-rootfs

```
{   cd /tmp; cat >fstab
    mkdir -p sfs/sfs sfs/usb
    dracut  -i fstab /etc/fstab     \
            -i sfs sfs              \ # i think only one include is allowed
            --add-drivers overlay   \
            --add-drivers squashfs  \
            initramfs.img 
}   <<"" #FSTAB
    UUID={USB-UUID}     /sfs/usb    $usbfs      defaults    0 0
    /sfs/usb/img.sfs    /sfs/sfs    squashfs    defaults    0 0

root=overlay \
rootfstype=overlay \
rootflags=\
lowerdir=/sfs/sfs,\
upperdir=/sfs/usb/persist,\
workdir=/sfs/usb/tmp


lsinitrd

dracut --print-cmdline

--include -i (can only be given once)
--install -I

rd.luks=0 rd.lvm=0 rd.md=0 rd.dm=0

           This turns off every automatic assembly of LVM, MD raids, DM
           raids and crypto LUKS.

```


# what does march native mean on amd laptop?

```
[martin@archlinux 110_gentoo]$ gcc -v -E -x c /dev/null -o /dev/null -march=native 2>&1 | grep /cc1 | grep mtune


 /usr/lib/gcc/x86_64-pc-linux-gnu/13.1.1/cc1 -E -quiet -v /dev/null -o /dev/null -march=znver3 -mmmx -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -msse4a -mno-fma4 -mno-xop -mfma -mno-avx512f -mbmi -mbmi2 -maes -mpclmul -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd -mno-avx512er -mno-avx512pf -mno-avx512vbmi -mno-avx512ifma -mno-avx5124vnniw -mno-avx5124fmaps -mno-avx512vpopcntdq -mno-avx512vbmi2 -mno-gfni -mvpclmulqdq -mno-avx512vnni -mno-avx512bitalg -mno-avx512bf16 -mno-avx512vp2intersect -mno-3dnow -madx -mabm -mno-cldemote -mclflushopt -mclwb -mclzero -mcx16 -mno-enqcmd -mf16c -mfsgsbase -mfxsr -mno-hle -msahf -mno-lwp -mlzcnt -mmovbe -mno-movdir64b -mno-movdiri -mmwaitx -mno-pconfig -mpku -mno-prefetchwt1 -mprfchw -mno-ptwrite -mrdpid -mrdrnd -mrdseed -mno-rtm -mno-serialize -mno-sgx -msha -mshstk -mno-tbm -mno-tsxldtrk -mvaes -mno-waitpkg -mwbnoinvd -mxsave -mxsavec -mxsaveopt -mxsaves -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-uintr -mno-hreset -mno-kl -mno-widekl -mno-avxvnni -mno-avx512fp16 -mno-avxifma -mno-avxvnniint8 -mno-avxneconvert -mno-cmpccxadd -mno-amx-fp16 -mno-prefetchi -mno-raoint -mno-amx-complex --param l1-cache-size=32 --param l1-cache-line-size=64 --param l2-cache-size=512 -mtune=znver3 -dumpbase null


```

- on same system inside gentoo chroot:

```
(chroot) archlinux / # cpuid2cpuflags
CPU_FLAGS_X86: aes avx avx2 f16c fma3 mmx mmxext pclmul popcnt rdrand sha sse sse2 sse3 sse4_1 sse4_2 sse4a ssse3

```

# dwm


 * Your configuration for x11-wm/dwm-6.3 has been saved in 
 * "/etc/portage/savedconfig/x11-wm/dwm-6.3" for your editing pleasure.
 * You can edit these files by hand and remerge this package with
 * USE=savedconfig to customise the configuration.
 * You can rename this file/directory to one of the following for

# install to usb disk

```
fdisk /dev/sda
d
n 1

+256M
n


w
mkfs.vfat -F 32 /dev/sda1
mkfs.ext4 /dev/sda3

mkdir /media/boot
mkdir /media/root
mount /dev/sda1 /media/boot
mount /dev/sda2 /media/root

cp -ar /mnt/gentoo/{bin,dev,etc,home,lib,lib64,media,mnt,opt,proc,root,run,sbin,sys,tmp} /media/root
mkdir -p /media/root/var/cache
cp -ar /mnt/gentoo/var/{db,empty,lib,lock,log,run,spool,tmp} /media/root/var
cp -ar /mnt/gentoo/var/cache/{edb,eix,fontconfig,genkernel,ldconfig,man,revdep-rebuild} /media/root/var/cache
mkdir -p /media/root/usr/src/
cp -ar /mnt/gentoo/usr/{bin,include,lib,lib64,libexec,local,sbin,share,x86_64-pc-linux-gnu} /media/root/usr/
cp -ar /mnt/gentoo/usr/src/linux-config /media/root/usr/src/
cp -ar /mnt/genoot/boot/* /media/boot/
```

- create a squashfs
https://github.com/plougher/squashfs-tools/blob/master/USAGE-4.6

```
pacman -S squashfs-tools
time \
mksquashfs \
/mnt/gentoo \
/home/martin/gentoo_20230716.squashfs \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 1 \
-progress \
-mem 7G \
-wildcards \
-e \
usr/src/linux* \
var/cache/binpkgs \
var/cache/distfiles \
gentoo*squashfs \
usr/share/genkernel/distfiles \
tmp \
proc \
sys \
run \
dev/pts \
dev/shm \
dev/hugepages \
dev/mqueue 

```
- runtime of squashfs 56sec

```
Exportable Squashfs 4.0 filesystem, zstd compressed, data block size 131072
        compressed data, compressed metadata, compressed fragments,
        compressed xattrs, compressed ids
        duplicates are removed
Filesystem size 2036756.07 Kbytes (1989.02 Mbytes)
        36.86% of uncompressed filesystem size (5525563.52 Kbytes)
		
[root@archlinux 110_gentoo]# ls -trhl /home/martin/gentoo.squashfs 
-rw-r--r-- 1 root root 2.0G Jun 19 20:54 /home/martin/gentoo.squashfs

```


# install grub

- make sure the following kernel modules exist: squashfs, overlay,
  loop, vfat. you may also compile those into the kernel, then you
  might be able to boot without initrd
- also we need to have support for /dev/sda3 and /dev/nvme...,
  dependening on what disk we use

- check output of blkid
- modify /etc/fstab
```
cat << EOF > etc/fstab
UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 / ext4 noatime 0 1
UUID=F63D-5318 /boot vfat noauto,noatime 1 2
EOF

/dev/sda1               /boot           ext4            noauto,noatime  1 2
/dev/sda3               /               ext4            noatime         0 1
```


```
mount /dev/sda3 /mnt/gentoo

cp --dereference /etc/resolv.conf /mnt/gentoo/etc/
mount --types proc /proc /mnt/gentoo/proc
mount --rbind /sys /mnt/gentoo/sys
mount --make-rslave /mnt/gentoo/sys
mount --rbind /dev /mnt/gentoo/dev
mount --make-rslave /mnt/gentoo/dev
mount --bind /run /mnt/gentoo/run
mount --make-slave /mnt/gentoo/run 
mount /dev/sda4 /mnt/gentoo/boot/efi

chroot /mnt/gentoo /bin/bash 
source /etc/profile 
export PS1="(chroot) ${PS1}"


emerge --ask --verbose sys-boot/grub

#grub-install /dev/sda
grub-install --target=x86_64-efi --efi-directory=/boot/efi

grub-mkconfig -o /boot/grub/grub.cfg
```

- the kernel boot line should look like this:
```
linux   /vmlinuz-6.1.31-gentoo-x86_64 root=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 ro rd.shell

```
- the flag rd.shell tells initrd to go into shell if a root disk is not found
- alternatively write
```
linux   /vmlinuz-6.1.31-gentoo-x86_64 root=/dev/sda3 ro rd.shell

```

# configure initramfs with dracut
```
dracut -m "kernel-modules base rootfs-block " --kver=6.1.38-gentoo-x86_64 --filesystems "squashfs vfat overlay" --force
```

# umount gentoo system

```
mount|grep /mnt/gentoo|cut -d  " " -f3|tac|xargs -n1 umount
```

# update existing boot partition 

i have created a gentoo system in /mnt/gentoo. i also have an external
harddisk that already contains an older version. it is mounted as
/media. show how to use rsync to copy /mnt/gentoo to /media. files
which have not changed shall not be overwritten. make sure that xattrs
and userid's stay are not different between the two disks

- make sure you are out of all chroots and no extra filesystems are
  mounted
- use rsync to only copy changes

```
mount /dev/sda3 /media
rsync -avhHAX --progress --numeric-ids /mnt/gentoo/ /media/

```

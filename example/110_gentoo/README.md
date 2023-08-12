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

- compile vfat, ext4, squashfs, loop, BLK_DEV_NVME, scsi into the
  kernel (not as module). this makes the system more resilent because
  i can boot without initramfs

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
USE="X vaapi"
VIDEO_CARDS="radeon radeonsi amdgpu"
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
net-wireless/sdrplay *
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
app-misc/radeontop ~amd64

dev-dotnet/dotnet-sdk-bin ~amd64
net-wireless/sdrplay ~amd64 
net-wireless/soapysdr ~amd64
net-wireless/soapysdrplay ~amd64
net-wireless/soapyplutosdr ~amd64
net-libs/libad9361-iio ~amd64
net-libs/libiio ~amd64

EOF

cat << EOF > /etc/portage/package.mask/package.mask
>=sys-kernel/gentoo-sources-6.3.13
>=sys-kernel/linux-headers-6.3.13
<sys-kernel/gentoo-sources-6.3.12
<=sys-kernel/linux-headers-6.2
EOF

cat << EOF > /etc/portage/package.use/package.use
www-client/firefox -clang -gmp-autoupdate -openh264 system-av1 system-harfbuzz system-icu system-jpeg system-libevent -system-libvpx -system-webp -dbus -debug -eme-free -geckodriver -hardened -hwaccel -jack -libproxy -lto -pgo pulseaudio -screencast -selinux -sndio -system-png -system-python-libs -wayland -wifi
www-client/chromium X -hangouts -official -pic -proprietary-codecs suid system-harfbuzz system-icu system-png -component-build -cups -custom-cflags -debug -gtk4 -headless -kerberos -libcxx -lto -pax-kernel -pgo -pulseaudio -qt5 -screencast -selinux -system-av1 -system-ffmpeg -vaapi -wayland -widevine
x11-base/xorg-server systemd udev xorg -debug -elogind -minimal -selinux -suid -test -unwind -xcsecurity -xephyr -xnest -xvfb
app-emacs/emacs-common -games gui
app-editors/emacs -acl gmp inotify ssl systemd threads xpm zlib Xaw3d -alsa -aqua athena -cairo dbus dynamic-loading -games -gfile -gif -gpm -gsettings -gtk gui -gzip-el -harfbuzz -imagemagick -jit -jpeg -json -kerberos -lcms -libxml2 -livecd -m17n-lib -mailutils -motif -png -selinux -sound -source -svg -tiff -toolkit-scroll-bars -valgrind -wide-int -xft -xwidgets
x11-terms/xterm openpty unicode -Xaw3d -sixel -toolbar -truetype -verify-sig -xinerama
net-wireless/bluez -mesh -obex readline systemd udev -btpclient -cups -debug -deprecated -doc -experimental -extra-tools -midi -selinux -test -test-programs 
net-wireless/iwd client -crda -monitor systemd -ofono -standalone -wired
net-misc/dhcp client ipv6 -server ssl -ldap -selinux -vim-syntax
dev-vcs/git blksha1 curl gpg iconv nls pcre -perl safe-directory -webdav -cgi -cvs -doc -highlight -keyring -mediawiki -perforce -selinux -subversion -test tk -xinet
sci-libs/nlopt -cxx -guile -octave python -test
dev-python/numpy lapack -test
sci-libs/openblas openmp -dynamic -eselect-ldso -index-64bit pthread -relapack -test
 media-video/ffmpeg X bzip2 -dav1d encode gnutls gpl iconv network postproc threads vaapi zlib alsa -amf -amr -amrenc -appkit -bluray -bs2b -cdio -chromaprint -chromium -codec2 -cpudetection -cuda -debug -doc -fdk -flite -fontconfig -frei0r -fribidi -gcrypt -gme -gmp -gsm -hardcoded-tables -iec61883 -ieee1394 -jack -jpeg2k -kvazaar -ladspa -libaom -libaribb24 -libass -libcaca -libdrm -libilbc -librtmp -libsoxr -libtesseract -libv4l -libxml2 -lv2 -lzma -mipsdspr1 -mipsdspr2 -mipsfpu -mmal -modplug -mp3 -nvenc -openal -opencl -opengl -openh264 -openssl -opus -oss -pic pulseaudio -qsv -rav1e -rubberband -samba -sdl -snappy -sndio -speex -srt -ssh -static-libs -svg -svt-av1 -test -theora -truetype -twolame -v4l -vdpau -verify-sig -vidstab -vmaf -vorbis -vpx -vulkan -webp -x264 -x265 -xvid -zeromq -zimg -zvbi
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
media-video/mpv X alsa cli -drm -egl -iconv libmpv -libplacebo -lua -uchardet -xv zlib -aqua -archive -bluray -cdda -coreaudio -debug -dvb -dvd -gamepad -jack -javascript -jpeg -lcms -libcaca -mmal -nvenc openal opengl -pipewire pulseaudio -raspberry-pi -rubberband -sdl -selinux -sixel -sndio -test -tools vaapi -vdpau -vulkan -wayland -zimg

sys-fs/squashfs-tools xattr -debug -lz4 -lzma -lzo zstd 

# tor firefox binary requires libdbus-glib 
dev-libs/glib elf mime xattr dbus -debug -gtk-doc -selinux -static-libs -sysprof -systemtap -test -utils
dev-libs/dbus-glib -debug -static-libs -test


# google chrome binary needs libcups, rpm2targz can be used to extract the rpm with the binary
# watching video with google chrome uses 4 or 5W, while firefox consumes 12W
net-print/cups-filters -foomatic -postscript -dbus -exif -jpeg -ldap -pclm -pdf -perl -png -test -tiff -zeroconf
net-print/cups -X -acl -pam -ssl -systemd -dbus -debug -kerberos -openssl -selinux -static-libs -test -usb -xinetd -zeroconf
app-text/poppler cxx -introspection jpeg -jpeg2k lcms utils -boost -cairo -cjk -curl -debug -doc -nss -png -qt5 -test -tiff -verify-sig
sys-fs/lvm2 readline systemd udev lvm -sanlock -selinux -static -static-libs -thin -valgrind

# qdirstat
dev-qt/qtcore systemd -debug -icu -old-kernel -test
dev-qt/qtgui X libinput png udev -accessibility dbus -debug -egl -eglfs -evdev -gles2-only -ibus jpeg -linuxfb -test -tslib -tuio -vnc -vulkan -wayland
dev-qt/qtwidgets X png dbus -debug -gles2-only -gtk -test
sys-apps/qdirstat

net-wireless/soapysdr -bladerf -hackrf plutosdr python -rtlsdr -uhd

x11-libs/wxGTK X lzma spell -curl -debug -doc -gstreamer -keyring libnotify opengl -pch -sdl -test -tiff -wayland -webkit
dev-libs/libpcre2 bzip2 jit pcre16 pcre32 readline unicode zlib -libedit -split-usr -static-libs

sci-libs/fftw -fortran openmp -doc -mpi -test threads -zbus

EOF

# charge battery at most to 80%
cat /etc/tlp.conf|grep CHARGE_TH
START_CHARGE_THRESH_BAT0=75
STOP_CHARGE_THRESH_BAT0=1

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

cp config-6.3.12-gentoo-x86_64 /usr/src/linux/.config
cp dwm-6.4 /etc/portage/savedconfig/x11-wm/dwm-6.4


useradd martin
cd /mnt/gentoo/home/martin
mkdir -p src
cd src
git clone https://git.suckless.org/slstatus

cp slstatus_config.h /mnt/gentoo/home/martin/src/slstatus

cd /mnt/gentoo/home/martin
curl -O https://beta.quicklisp.org/quicklisp.lisp
sbcl --load quicklisp.lisp
(quicklisp-quickstart:install)
(ql:add-to-init-file)
(ql:quickload "quicklisp-slime-helper")

cat << EOF > /home/martin/.emacs
(require 'package)
(setq package-enable-at-startup nil)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/"))
(package-initialize)
(unless (package-installed-p 'ido-hacks)
  (package-refresh-contents)
  (package-install 'ido-hacks))
(unless (package-installed-p 'eval-in-repl)
  (package-refresh-contents)
  (package-install 'eval-in-repl))
(require 'ido-hacks)
(ido-hacks-mode)
(eval-after-load "vc" '(remove-hook 'find-file-hooks 'vc-find-file-hook))
(setq vc-handled-backends nil)
(custom-set-variables
 '(inhibit-startup-screen t)
 '(package-selected-packages
   (quote
    (ido-hacks eval-in-repl)))
 '(show-paren-mode t)
 '(tool-bar-mode nil)
 '(ido-mode t)
 ;'(transient-mark-mode nil)
 )
(setq ido-enable-flex-matching t)
(setq ido-everywhere t)
(ido-mode +1)
;(show-paren-mode 1)
(global-set-key (kbd "<f4>") 'magit-status)
(load (expand-file-name "~/quicklisp/slime-helper.el"))
(require 'slime-autoloads)
(slime-setup '(slime-fancy))
(defun cliki:start-slime ()
  (unless (slime-connected-p)
    (save-excursion (slime))))
(add-hook 'slime-mode-hook 'cliki:start-slime)
(setq inferior-lisp-program "sbcl")
(require 'eval-in-repl-slime)
(add-hook 'lisp-mode-hook
		  '(lambda ()
		     (local-set-key (kbd "<C-return>") 'eir-eval-in-slime)))

(custom-set-faces
 '(default ((t (:family "fixed" :foundry "misc" :slant normal :weight normal :height 98 :width semi-condensed)))))


EOF

ln -s /home/martin/stage/cl-py-generator/ /home/martin/quicklisp/local-projects/



# only install what isn't already there
emerge -av $(for pkg in xorg-server firefox gentoolkit dwm xterm emacs sbcl slime magit paredit bluez iwd dhcp dev-vcs/git dev-python/pip numpy scipy scikit-learn nlopt matplotlib redshift opencv python lmfit tqdm ofono pulseaudio-daemon pulseaudio blueman dracut iwgtk glib dbus-glib mpv mksquashfs-tools radeontop sys-fs/lvm2 nvme-cli hdparm cryptsetup dev-python/mss soapysdr wxGTK; do eix -I "$pkg" >/dev/null || echo "$pkg"; done)


emacs /etc/portage/savedconfig/x11-wm/dwm-6.4
#define MODKEY Mod4Mask
static const char *termcmd[]  = { "xterm", NULL };


emerge --ask --verbose --update --newuse --deep --with-bdeps=y @world 

#12GB for rust, 6.6GB for firefox
emerge -uDN @world --buildpkg --buildpkg-exclude "virtual/* sys-kernel/*-sources"
emerge @world --buildpkg --buildpkg-exclude "virtual/* sys-kernel/*-sources"

```

# gpu power levels

- https://wiki.gentoo.org/wiki/AMDGPU

```
archlinux / # cat /sys/class/drm/card0/device/pp_dpm_sclk
0: 200Mhz 
1: 400Mhz *
2: 1800Mhz 

archlinux / # cat /sys/class/drm/card0/device/pp_dpm_mclk
0: 400Mhz 
1: 800Mhz 
2: 1200Mhz *
3: 1333Mhz 

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

export INDIR=/mnt/gentoo
export OUTFILE=/home/martin/gentoo_20230716.squashfs



export INDIR=/
export OUTFILE=/gentoo_20230716b.squashfs
time \
mksquashfs \
$INDIR \
$OUTFILE \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 1 \
-progress \
-mem 10G \
-wildcards \
-e \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
proc/* \
sys/* \
run/* \
dev/pts/* \
dev/shm/* \
dev/hugepages/* \
dev/mqueue/* \
home/martin/.cache/mozilla \
home/martin/.cache/google-chrome \
home/martin/.b \
home/martin/Downloads/* \
home/martin/.config/* \
home/martin/.mozilla/* \
home/martin/src \
var/log/journal/* \
var/cache/genkernel/* \
tmp/* \
persistent




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
- with rd.break you can always go into the shell (even if root disk is found)
- alternatively write
```
linux   /vmlinuz-6.1.31-gentoo-x86_64 root=/dev/sda3 ro rd.shell

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


the objective is
           to locate your root volume and create a symlink /dev/root
           which points to the file system.

        5. Next, make a symbolic link to the unlocked root volume

                   # ln -s /dev/mapper/luks-$UUID /dev/root

            6. With the root volume available, you may continue booting
               the system by exiting the dracut shell

                   # exit

```

# mount squashfs from dracut initramfs 

```
umount /sysroot
mkdir /mnt
mkdir /squash
mount /dev/nvme0n1p3 /mnt
mount /mnt/gentoo_20230716b.squashfs /squash
mkdir -p /mnt/persistent/lower
mkdir -p /mnt/persistent/work
mount -t overlay overlay -o upperdir=/mnt/persistent/lower,lowerdir=/squash,workdir=/mnt/persistent/work /sysroot

```

# configure initramfs with dracut

https://man7.org/linux/man-pages/man7/dracut.cmdline.7.html
booting live images
```
# get old init
cd /dev/shm/
lsinitrd --unpack
cp /dev/shm/init init_dracut

# man dracut.modules
find /usr/lib/dracut/modules.d/
# maybe i can use hook pre-pivot to mount squashfs and set up overlay

 root=live:UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 rd.live.dir=/ rd.live.squashimg=gentoo_20230716b.squashfs rd.live.ram=1 rd.live.overlay=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 rd.live.overlay.overlayfs=1 rd.shell rd.debug

crypt



dracut \
  -m " kernel-modules base rootfs-block livenet " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs_live-6.3.12-gentoo-x86_64.img


# mv /usr/lib/dracut/modules.d/99base/init.sh /usr/lib/dracut/modules.d/99base/init_orig.sh
cp init_dracut.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs_squash-6.3.12-gentoo-x86_64.img


```
Here's an explanation of each of the options you listed:

- `root=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826`: This sets the root file system for the booting process. The value is specified as a UUID (Universally Unique Identifier), which is a way to uniquely identify a filesystem independent of its location.

- `rd.live.dir=/`: This is a dracut live option, specifying the base directory where live image content is located.

- `rd.live.squashimg=gentoo_20230716b.squashfs`: This tells dracut where the SquashFS image file (a compressed read-only file system for Linux) is located for live boot. In this case, the SquashFS image file is named `gentoo_20230716b.squashfs`.

- `rd.live.ram=1`: Another dracut live option, this forces the live image to be copied into RAM before being mounted. This is useful for freeing up the boot medium after the system has started.

- `rd.live.overlay=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826`: This specifies the UUID of the device to use as an overlay device for persistent data storage during a live session.

- `rd.live.overlay.overlayfs=1`: This specifies the use of overlayfs as the mechanism to provide a writable root file system on a read-only medium. Overlayfs allows the combination of two directories (an upper and a lower directory) and presents them as a single merged directory.

- `rd.shell`: This provides a shell in the initramfs environment in case dracut fails to locate or mount the root filesystem, or if the `rootfs` block device is not clean. It can also be helpful for debugging purposes.

These options are used to modify the way dracut and the Linux kernel handle the booting process, particularly for live systems. Please note that options starting with `rd.live` are specific to the dracut live module.


Add a custom entry to /etc/default/grub
```
grub-install --target=x86_64-efi --efi-directory=/boot/efi

grub-mkconfig -o /boot/grub/grub.cfg
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


# binary tor shim

- https://forums.gentoo.org/viewtopic-p-8683928.html?sid=01daca293e043d753608059b2cc00982

```

cat << EOF > /dev/shm/shim.c 
#include <stdlib.h>
void gdk_wayland_display_get_wl_compositor() { abort(); }
void gdk_wayland_device_get_wl_pointer() { abort(); }
void gdk_wayland_window_get_wl_surface() { abort(); }
void gdk_wayland_display_get_wl_display() { abort(); }
EOF

cc -shared -o /dev/shm/shim.so /dev/shm/shim.c

	
cd /dev/shm/tor* &&
sed '/LD_PRELOAD/!s,Exec[^=]*=,&env LD_PRELOAD=/dev/shm/shim.so ,' start-tor-browser.desktop >tmp &&
mv tmp start-tor-browser.desktop &&
./start-tor-browser.desktop 



```

# next

- install cryptsetup, dev-python/mss


# update 2023-07-29

## base system

- boot from ssd (not squashfs)
```
eix-sync

# config file '/etc/portage/savedconfig/sys-kernel/linux-firmware-20230515' needs updating.
# maybe i have to rebuild the kernel?
dispatch-conf 

# go through this file and update /etc/portage files

emerge --ask --verbose --update --newuse --deep --with-bdeps=y @world 

# this downloads and installs gentoo-sources-6.1.42
# but for wifi hardware i must use >= 6.3.12
# maybe i have to mask lower versions as well

# install a few things that I decided to add meanwhile. notably cryptsetup, mss, soapysdrplay (unfortunately not free software, binary only! note to self: never buy hardware before checking software first) and mc
# only packages that don't exist yet, will be added by the following command:

emerge -av $(for pkg in xorg-server firefox gentoolkit dwm xterm emacs sbcl slime magit paredit bluez iwd dhcp dev-vcs/git dev-python/pip numpy scipy scikit-learn nlopt matplotlib redshift opencv python lmfit tqdm ofono pulseaudio-daemon pulseaudio blueman dracut iwgtk glib dbus-glib mpv squashfs-tools radeontop sys-fs/lvm2 nvme-cli hdparm cryptsetup dev-python/mss soapysdr app-misc/mc soapysdrplay ; do eix -I "$pkg" >/dev/null || echo "$pkg"; done)

# I will not install wxGTK at this time

# try to get rid of the two slots of llvm

emerge -ac

# it removes llvm-15 and related clang, it also removes the 6.1.42 kernel sources that i masked

#
revdep-rebuild

# system is consistent

# delete old binary packages (35 files, 1.2GB)
eclean packages
# delete old sources (12 files, 404MB)
eclean distfiles 



```
### the new packages

```

sys-kernel/linux-firmware/linux-firmware-20230625_p20230724-1.gpkg.tar
app-crypt/libmd/libmd-1.1.0-1.gpkg.tar
media-libs/alsa-ucm-conf/alsa-ucm-conf-1.2.9-1.gpkg.tar
dev-python/ensurepip-setuptools/ensurepip-setuptools-68.0.0-1.gpkg.tar
media-libs/libpng/libpng-1.6.40-1.gpkg.tar
dev-libs/libassuan/libassuan-2.5.6-1.gpkg.tar
media-libs/tiff/tiff-4.5.1-1.gpkg.tar
sys-apps/sandbox/sandbox-2.37-1.gpkg.tar
dev-libs/libpcre2/libpcre2-10.42-r1-1.gpkg.tar
dev-libs/openssl/openssl-3.0.9-r2-1.gpkg.tar
dev-perl/Sub-Name/Sub-Name-0.270.0-1.gpkg.tar
dev-perl/Pod-Parser/Pod-Parser-1.660.0-1.gpkg.tar
dev-perl/YAML-Tiny/YAML-Tiny-1.740.0-1.gpkg.tar
dev-perl/Module-Build/Module-Build-0.423.400-1.gpkg.tar
dev-perl/IO-Socket-INET6/IO-Socket-INET6-2.730.0-1.gpkg.tar
dev-perl/Net-HTTP/Net-HTTP-6.230.0-1.gpkg.tar
dev-perl/HTML-Parser/HTML-Parser-3.810.0-1.gpkg.tar
dev-lisp/asdf/asdf-3.3.5-r1-1.gpkg.tar
sys-apps/acl/acl-2.3.1-r2-1.gpkg.tar
sys-apps/coreutils/coreutils-9.3-r3-1.gpkg.tar
sys-apps/systemd/systemd-253.6-1.gpkg.tar
net-libs/gnutls/gnutls-3.8.0-1.gpkg.tar
media-libs/flac/flac-1.4.3-1.gpkg.tar
sys-devel/automake/automake-1.16.5-r1-1.gpkg.tar
dev-python/pyparsing/pyparsing-3.1.0-1.gpkg.tar
dev-python/pathspec/pathspec-0.11.1-1.gpkg.tar
media-libs/alsa-lib/alsa-lib-1.2.9-1.gpkg.tar
dev-libs/libuv/libuv-1.46.0-1.gpkg.tar
dev-util/strace/strace-6.3-1.gpkg.tar
app-portage/eix/eix-0.36.7-1.gpkg.tar
dev-lisp/sbcl/sbcl-2.3.5-1.gpkg.tar
media-libs/openal/openal-1.23.1-r1-1.gpkg.tar
net-misc/openssh/openssh-9.3_p2-1.gpkg.tar
sys-apps/dbus/dbus-1.15.6-1.gpkg.tar
x11-drivers/xf86-video-amdgpu/xf86-video-amdgpu-23.0.0-1.gpkg.tar
dev-python/contourpy/contourpy-1.1.0-1.gpkg.tar
app-editors/emacs/emacs-28.2-r9-1.gpkg.tar
dev-python/editables/editables-0.3-1.gpkg.tar
dev-python/calver/calver-2022.06.26-1.gpkg.tar
dev-python/pluggy/pluggy-1.2.0-1.gpkg.tar
dev-python/trove-classifiers/trove-classifiers-2023.5.24-1.gpkg.tar
dev-python/fonttools/fonttools-4.40.0-1.gpkg.tar
dev-python/asteval/asteval-0.9.31-1.gpkg.tar
dev-python/hatchling/hatchling-1.18.0-1.gpkg.tar
dev-python/urllib3/urllib3-2.0.3-1.gpkg.tar
dev-python/pycairo/pycairo-1.24.0-1.gpkg.tar
dev-libs/libksba/libksba-1.6.4-1.gpkg.tar
media-libs/gstreamer/gstreamer-1.20.6-1.gpkg.tar
sys-kernel/gentoo-sources/gentoo-sources-6.1.42-1.gpkg.tar
dev-lang/vala/vala-0.56.8-1.gpkg.tar
media-libs/gst-plugins-base/gst-plugins-base-1.20.6-1.gpkg.tar
sys-kernel/genkernel/genkernel-4.3.5-1.gpkg.tar
media-video/ffmpeg/ffmpeg-4.4.4-r3-1.gpkg.tar
media-video/mpv/mpv-0.35.1-r2-1.gpkg.tar
acct-group/avahi/avahi-0-r2-1.gpkg.tar
app-crypt/argon2/argon2-20190702-r1-1.gpkg.tar
sys-apps/hdparm/hdparm-9.65-1.gpkg.tar
dev-libs/json-c/json-c-0.16-r1-1.gpkg.tar
acct-user/avahi/avahi-0-r2-1.gpkg.tar
dev-libs/libdaemon/libdaemon-0.14-r4-1.gpkg.tar
app-doc/xmltoman/xmltoman-0.6-1.gpkg.tar
dev-python/mss/mss-9.0.1-1.gpkg.tar
dev-libs/libpcre/libpcre-8.45-r1-1.gpkg.tar
net-dns/avahi/avahi-0.8-r7-1.gpkg.tar
sys-libs/libnvme/libnvme-1.4-1.gpkg.tar
sys-fs/cryptsetup/cryptsetup-2.6.1-1.gpkg.tar
net-libs/libiio/libiio-0.24-1.gpkg.tar
sys-libs/slang/slang-2.3.2-1.gpkg.tar
sys-apps/nvme-cli/nvme-cli-2.4-r2-1.gpkg.tar
net-libs/libad9361-iio/libad9361-iio-0.2-r1-1.gpkg.tar
app-misc/mc/mc-4.8.29-1.gpkg.tar
net-wireless/soapysdr/soapysdr-0.8.1-1.gpkg.tar
net-wireless/soapyplutosdr/soapyplutosdr-0.2.1-1.gpkg.tar
net-wireless/sdrplay/sdrplay-3.07.1-1.gpkg.tar
net-wireless/soapysdrplay/soapysdrplay-20220120-1.gpkg.tar
media-video/mpv/mpv-0.35.1-r2-2.gpkg.tar


```



## /home

- i did some changes to the home directory in the overlay directory
- these should be taken into the new squashfs
- some config files (.emacs, .xinitrc, some scripts)
- quicklisp/ installation
```
sbcl
(ql:update-client) # no change
(ql:update-dist "quicklisp") # no change

# build swank
emacs
M-x slime

# delete old swank
rm -rf ~/.cache/common-lisp/sbcl-2.2.9-linux-x64/

```
- stage/ directory with git repositories
- src/rysen-monitor (i think that did not change)
- src/slstatus	 (i think that did not change)
- indeed the two last ones didn't change

### not to be added into squashfs:
- clion installation, and sonarlint
- llama 2
- pythonai (a python environment i used to convert the llama 2 network to int4)

## update kernel

```
# i think this adds the new rtw89/rtw8852b_fw.bin file
# i'm not sure if it actually is different
cd /usr/src/linux
make -j12
make modules_install
make install
```

## create squashfs

```

export INDIR=/
export OUTFILE=/mnt/gentoo_20230729.squashfs
time \
mksquashfs \
$INDIR \
$OUTFILE \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 1 \
-progress \
-mem 10G \
-wildcards \
-e \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
boot/* \
proc \
sys/* \
run/* \
dev/pts/* \
dev/shm/* \
dev/hugepages/* \
dev/mqueue/* \
home/martin/.cache/mozilla \
home/martin/.cache/google-chrome \
home/martin/.cache/mesa_shader_cache \
home/martin/.cache/fontconfig \
home/martin/Downloads/* \
home/martin/.config/* \
home/martin/.mozilla/* \
home/martin/src \
var/log/journal/* \
var/cache/genkernel/* \
tmp/* \
mnt/ \
persistent

Filesystem size 2195231.74 Kbytes (2143.78 Mbytes)
        35.16% of uncompressed filesystem size (6242681.04 Kbytes)

20 sec

# it is not filtering out proc (one directory of a process that
# probably starts later than the wildcard expansion )
 ```

- use proc (not proc/*) and place the OUTFILE in an ignored directory
  so that the file isn't read after the wildcard expansion
- /usr/lib/firmware is 892MB, this surely can be reduced


## change overlay partition

- sharing a 50GB partition with the original system, my home directory
  and the persistent overlay turns out to be insufficient

- make a new partition to use as overlay
- make appropriate changes to initramfs


## rebuild initramfs

```

cp init_dracut.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20230729_squash-6.3.12-gentoo-x86_64.img

# new entry in 
# grub.cfg

menuentry 'Gentoo GNU/Linux 20230729 ram squash persist ssd' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.3.12-gentoo-x86_64 ...'
	linux	/vmlinuz-6.3.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init
	initrd	/initramfs20230729_squash-6.3.12-gentoo-x86_64.img
}

```

# Proposed changes

- add mold (81MB installed), ccache (7MB installed), include-what-you-use (18.5MB)
- add liquid-dsp (1.7MB)


# Encrypted Hard Drive Implementation Proposal

## Overview:
Storing website passwords and cookies on persistent storage enhances
convenience of the webbrowser and Linux system. However, in the event
that a laptop is misplaced or stolen, it is crucial to ensure
unauthorized individuals cannot easily access stored online
credentials. Disk encryption is a potential solution to this
challenge.

## Proposal Details:

1. **Reference**: Comprehensive documentation on Gentoo disk
   encryption can be found at the following link:
  [Gentoo Full Disk Encryption Guide](https://wiki.gentoo.org/wiki/Full_Disk_Encryption_From_Scratch_Simplified)

2. **Encryption Process**:
   - Set up an encrypted partition on the NVMe drive using LUKS (Linux
     Unified Key Setup).
   - Retain the grub partition, kernel, and initramfs in an
     unencrypted state. Although it's technically possible to decrypt
     a partition from grub, the process is notably intricate and can
     become problematic in case of errors.
   - The encrypted partition will house an ext4 filesystem. This
     filesystem will contain both the squashfs file and the upper
     directory for an overlayfs.

3. **Boot Process**:
   - Upon every boot, the initramfs will prompt the user to input the
     password.
   - Post-authentication, the initramfs will copy the squashfs file
     into RAM. Following this, the overlayfs is configured to:
     - Read from the squashfs (which is located in RAM).
     - Write any changes to the ext4 system within the encrypted LUKS
       partition.
   - Concluding the aforementioned processes, the initramfs will
     transition the rootfs over to the overlayfs.

## Limitations & Considerations:

It is essential to recognize potential vulnerabilities, even with
encryption:

- **Hardware Attacks**: An attacker with physical access to the
  hardware could potentially install a hardware key logger or alter
  the unencrypted kernel or initramfs to capture the user's password.

- **RAM Exploitation**: Technically advanced attackers might freeze
  and remove the RAM from a running system to access encrypted data.

However, it's worth noting that our security measures are not designed
to counter such sophisticated attacks. These specific threat models
lie outside the scope of the protection we intend to offer through
this proposal.

By implementing this encrypted hard drive proposal, we aim to bolster
security, ensuring data integrity and confidentiality against common
threat models.

## Implementation

### Preparing the Encrypted Partition

```
sudo su
fdisk /dev/nvme0n1

# Determine the available partitions.
# For this guide, we'll use partition p4.

cryptsetup luksFormat --key-size 512 /dev/nvme0n1p4

# Set up the encryption:

cryptsetup luksOpen /dev/nvme0n1p4 vg
mkfs.ext4 -L rootfs /dev/mapper/vg

# Now, copy the squashfs file to the encrypted disk:

mount /dev/nvme0n1p3 /mnt3/
mount /dev/mapper/vg /media/
cp -r /mnt3/gentoo_20230729.squashfs /media/

```

Important Note: If the disk ever gets corrupted, data recovery might
be possible using a backup of the LUKS header. However, before
creating a backup, ensure that the header will not be stored on an
unencrypted storage device, especially when there's a risk of losing
the laptop. As a precaution, turn off swap using the swapoff command
and save the backup in /dev/shm/. The resulting backup file
crypt_headers.img will have a size of 16MB.

```
 cryptsetup luksHeaderBackup /dev/nvme0n1p4 --header-backup-file /dev/shm/crypt_headers.img

```


### Configuring Initramfs 

```
# Replace the initramfs configuration script:

cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

# Generate the initramfs image. Note that crypt and dm modules are included:

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20230729_squash_crypt-6.3.12-gentoo-x86_64.img

# Please make sure that /boot is mounted /dev/nvme0n1p1 using `mount|grep boot`
# /dev/nvme0n1p1 on /boot type vfat

# Lastly, add a new entry in grub.cfg:
# sudo emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20230729 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.3.12-gentoo-x86_64 ...'
	linux	/vmlinuz-6.3.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init
	initrd	/initramfs20230729_squash_crypt-6.3.12-gentoo-x86_64.img
}


```

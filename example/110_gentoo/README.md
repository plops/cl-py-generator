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

dev-libs/rocm-opencl-runtime ~amd64
dev-libs/rocr-runtime ~amd64
dev-libs/rocm-comgr ~amd64
dev-libs/rocm-device-libs ~amd64
dev-util/rocm-cmake ~amd64
dev-libs/roct-thunk-interface ~amd64
sci-libs/clblast ~amd64
dev-util/rocminfo ~amd64
dev-util/cmake ~amd64
net-libs/liquid-dsp ~amd64
sci-libs/scikit-learn ~amd64
dev-python/pythran ~amd64
dev-cpp/xsimd ~amd64
EOF

cat << EOF > /etc/portage/package.mask/package.mask
>=sys-kernel/gentoo-sources-6.3.13
>=sys-kernel/linux-headers-6.3.13
<sys-kernel/gentoo-sources-6.3.12
<=sys-kernel/linux-headers-6.2
dev-lang/rust
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
numpy scipy scikit-learn nlopt matplotlib opencv python \
x11-misc/xclip \
nss nspr 

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
- add glfw (0.6MB)
- add fftw sci-libs/fftw -fortran openmp -doc -mpi -test threads -zbus
- add btrfs-progs and kernel driver

## Battery Low Warning

### Objective

Implement a warning system that alerts the user with a sound and
changes the desktop background when the battery charge drops below
20%.

### Dependencies

- *sox* - For generating and playing the warning beep.
- *xsetroot* - To change the desktop background color.

- add sox and xsetroot to warn when battery needs to be recharged 
```
media-sound/sox openmp -alsa -amr -ao -encode -flac -id3tag -ladspa -mad -magic -ogg -opus -oss -png pulseaudio -sndfile -sndio -static-libs -twolame -wavpack
```

- Install the script and configure systemd service with timer.

```
sudo cp battery_check.sh /usr/bin
sudo chmod +x /usr/bin/battery_check.sh


cat << EOF > /etc/systemd/system/battery_check.service
[Unit]
Description=Check Battery Status

[Service]
Type=oneshot
ExecStart=/usr/bin/battery_check.sh
EOF

cat << EOF > /etc/systemd/system/battery_check.timer
[Unit]
Description=Run battery_check.service every minute

[Timer]
OnBootSec=5min
OnUnitActiveSec=1min

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now battery_check.timer

```

# Collect System Statistics

The following procedure outlines the creation and management of a
systemd service and timer to periodically gather system statistics
using tlp-stat. The service will log these statistics every minute,
ensuring that the most recent system data is readily available for
diagnostics or analysis.

```
cat << EOF > /etc/systemd/system/tlp-stat-logger.service
[Unit]
Description=Log tlp-stat output

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo -n 'datetime: ' >> /var/log/tlp.log; date +%Y-%m-%d_%H:%M:%S >> /var/log/tlp.log && tlp-stat >> /var/log/tlp.log"
EOF


cat << EOF > /etc/systemd/system/tlp-stat-logger.timer
[Unit]
Description=Run tlp-stat-logger every minute

[Timer]
OnBootSec=1min
OnUnitActiveSec=1min

[Install]
WantedBy=timers.target
EOF


sudo systemctl daemon-reload
sudo systemctl enable tlp-stat-logger.timer
sudo systemctl start tlp-stat-logger.timer

# check active timers

sudo systemctl list-timers


```


The `tlp.log` file provides detailed insights into various hardware
and system performance metrics. An excerpt from such a log might look
like:


```
+++ AMD Radeon Graphics
/sys/class/drm/card0/device/power_dpm_force_performance_level     = auto

+++ Wireless
hci0(btusb)                   : bluetooth, connected
wlan0(rtw89_8852be)           : wifi, connected, power management = off


  SMART info:
    Critical Warning:                   0x00
    Temperature:                        47 Celsius
    Available Spare:                    100%
    Available Spare Threshold:          10%

/sys/class/power_supply/BAT0/energy_full                    =  36700 m
/sys/class/power_supply/BAT0/energy_now                     =  26780 m
/sys/class/power_supply/BAT0/power_now                      =  28148 m
/sys/class/power_supply/BAT0/status                         = Charging

Charge                                                      =   73.0 [%]

```

The `tlp.log` file captures a lot of different data, making it really
useful for checking system details or fixing issues. We record all
this information because even if we're not sure what we'll need,
having everything saved means we're prepared for any situation in the
future.

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

# Encrypted partition with btrfs

```
# build btrfs tools
emerge btrfs-progs
# compile btrfs kernel module
make modules_install
depmod -a
modprobe btrfs

```

- https://gist.github.com/MaxXor/ba1665f47d56c24018a943bb114640d7

```
cryptsetup luksFormat -v -c aes-xts-plain64 -h sha512 --key-size 512 /dev/nvme0n1p5

cryptsetup luksOpen /dev/nvme0n1p5 p5
mkfs.btrfs /dev/mapper/p5
mount -t btrfs -o defaults,noatime,compress=zstd /dev/mapper/p5 /mnt5

```

# next

- need support for sdcard (mmc_block and sdhci)


# update 2023-09-02

- start system from nvme0n1p3 (instead of squashfs)
- eix-sync

```
emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
```
- 163 packages need to be updated

- during the last package (firefox) the laptop became very sluggish
  and i had to reboot. i still want to compile firefox with -j12. so i
  turned on swap
```
archlinux /home/martin # mkswap /dev/nvme0n1p2
Setting up swapspace version 1, size = 18 GiB (19327348736 bytes)
no label, UUID=c8f97a2d-3dd9-4d87-bc6c-40cb7d75d34b
archlinux /home/martin # swapon /dev/nvme0n1p2

```

- compiling firefox takes too long (also its dependencies nodejs and
  librsvg). i think it would be an improvement to install the binary

- while compilations are running, we can check the kernel and modules

```
cd /usr/src/linux
sudo make menuconfig

# btrfs is already configured as a module
# just install the kernel and module
make modules_install
make install

# maybe i will have to update the initramfs. but i don't know how to
# do that anymore for the kernel that boots directly from the disk

```


- 4 config files in /etc need updating 
- /etc/sudoers has been misconfigured, would be unfortunate if this change goes into squashfs
- use the following to revert change
```
export EDITOR=emacs
visudo /etc/sudoers
```
- remove backup files that emacs creates. they seem to disturb portage (it tries to read them)
```
 rm /etc/portage/package.mask/#package.mask#
```

```
emerge --depclean
# libssh removed
eclean-dist
#  [  449.9 M ] Total space from 62 files were freed in the packages directory
eclean-pkg
# nothing
revdep-rebuild
# system is consistent
```

- list the new binpkgs, also show and sort by modification time. up to
  6 were compiled simultaneously, so from the modification times I
  can't derive the compile times.

```
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %p size=%s\n"|sort -n
# don't show the entire path, just the filename:
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 
```

```
2023-09-02 22:32:49.3177042130 +0200 baselayout-2.14-1.gpkg.tar size=61440
2023-09-02 22:33:01.0643064210 +0200 hwdata-0.372-1.gpkg.tar size=2109440
2023-09-02 22:34:46.3737285410 +0200 linux-firmware-20230804-1.gpkg.tar size=377477120
2023-09-02 22:36:10.5599332400 +0200 b2-4.10.1-1.gpkg.tar size=552960
2023-09-02 22:36:22.9665318270 +0200 ensurepip-pip-23.2.1-1.gpkg.tar size=2068480
2023-09-02 22:36:51.3530427240 +0200 ethtool-6.4-1.gpkg.tar size=266240
2023-09-02 22:38:39.1957842760 +0200 mpfr-4.2.0_p12-1.gpkg.tar size=491520
2023-09-02 22:38:56.8190209030 +0200 libpng-1.6.40-r1-1.gpkg.tar size=409600
2023-09-02 22:39:08.5122900700 +0200 iw-5.19-1.gpkg.tar size=133120
2023-09-02 22:41:38.4548006020 +0200 perl-5.38.0-r1-1.gpkg.tar size=15882240
2023-09-02 22:42:05.3046532640 +0200 libxcrypt-4.4.36-1.gpkg.tar size=174080
2023-09-02 22:42:16.6879241330 +0200 perl-ExtUtils-MakeMaker-7.700.0-1.gpkg.tar size=20480
2023-09-02 22:42:29.0678561980 +0200 perl-Exporter-5.770.0-r1-1.gpkg.tar size=20480
2023-09-02 22:42:40.0677958370 +0200 perl-Carp-1.540.0-1.gpkg.tar size=20480
2023-09-02 22:42:51.9310640710 +0200 perl-Encode-3.190.0-1.gpkg.tar size=20480
2023-09-02 22:43:02.8643374090 +0200 perl-File-Spec-3.880.0-1.gpkg.tar size=20480
2023-09-02 22:43:13.8742769920 +0200 perl-Scalar-List-Utils-1.630.0-1.gpkg.tar size=20480
2023-09-02 22:43:34.7208292650 +0200 perl-Data-Dumper-2.188.0-1.gpkg.tar size=20480
2023-09-02 22:43:45.5407698910 +0200 perl-MIME-Base64-3.160.100_rc-1.gpkg.tar size=20480
2023-09-02 22:43:57.2607055780 +0200 perl-Time-Local-1.300.0-r2-2.gpkg.tar size=20480
2023-09-02 22:44:54.0537272630 +0200 perl-IO-1.520.0-1.gpkg.tar size=20480
2023-09-02 22:45:04.7670018080 +0200 perl-Module-Metadata-1.0.37-r3-1.gpkg.tar size=20480
2023-09-02 22:45:18.6169258070 +0200 perl-Test-Harness-3.440.0-r1-1.gpkg.tar size=20480
2023-09-02 22:45:29.4968661040 +0200 perl-XSLoader-0.320.0-1.gpkg.tar size=20480
2023-09-02 22:45:41.2801347770 +0200 perl-Compress-Raw-Bzip2-2.204.1_rc-1.gpkg.tar size=20480
2023-09-02 22:45:52.1967415400 +0200 perl-Getopt-Long-2.540.0-1.gpkg.tar size=20480
2023-09-02 22:46:03.2133477530 +0200 perl-parent-0.241.0-1.gpkg.tar size=20480
2023-09-02 22:46:15.3632810810 +0200 HTML-Tagset-3.200.0-r2-2.gpkg.tar size=40960
2023-09-02 22:46:29.3965374080 +0200 Socket6-0.290.0-2.gpkg.tar size=51200
2023-09-02 22:46:42.6997977400 +0200 Text-CharWidth-0.40.0-r2-1.gpkg.tar size=40960
2023-09-02 22:46:55.6930597740 +0200 TimeDate-2.330.0-r1-2.gpkg.tar size=51200
2023-09-02 22:47:08.8229877240 +0200 File-Temp-0.231.100-1.gpkg.tar size=71680
2023-09-02 22:47:20.3362578790 +0200 perl-CPAN-Meta-Requirements-2.140.0-r9-2.gpkg.tar size=20480
2023-09-02 22:47:31.5961960900 +0200 perl-CPAN-Meta-YAML-0.18.0-r9-1.gpkg.tar size=20480
2023-09-02 22:47:43.4561310100 +0200 perl-ExtUtils-Manifest-1.730.0-r2-1.gpkg.tar size=20480
2023-09-02 22:47:54.0527395280 +0200 perl-Parse-CPAN-Meta-2.150.10-r7-1.gpkg.tar size=20480
2023-09-02 22:48:04.4860156090 +0200 perl-Perl-OSType-1.10.0-r7-1.gpkg.tar size=20480
2023-09-02 22:48:15.0359577170 +0200 perl-Text-ParseWords-3.310.0-r1-1.gpkg.tar size=20480
2023-09-02 22:48:25.5025669490 +0200 perl-version-0.992.900-r1-1.gpkg.tar size=20480
2023-09-02 22:48:38.1524975330 +0200 Regexp-IPv6-0.30.0-r2-1.gpkg.tar size=30720
2023-09-02 22:48:47.2457809670 +0200 perl-JSON-PP-4.160.0-r1-1.gpkg.tar size=20480
2023-09-02 22:48:58.0257218130 +0200 perl-CPAN-2.360.0-1.gpkg.tar size=20480
2023-09-02 22:49:08.5523307160 +0200 perl-ExtUtils-CBuilder-0.280.238-1.gpkg.tar size=20480
2023-09-02 22:49:19.5122705730 +0200 perl-ExtUtils-Install-2.220.0-1.gpkg.tar size=20480
2023-09-02 22:49:31.2388728910 +0200 perl-ExtUtils-ParseXS-3.510.0-1.gpkg.tar size=20480
2023-09-02 22:49:42.4821445280 +0200 perl-podlators-5.10.0-1.gpkg.tar size=20480
2023-09-02 22:49:53.8687487110 +0200 perl-IO-Socket-IP-0.410.100_rc-1.gpkg.tar size=20480
2023-09-02 22:50:06.6953449920 +0200 perl-Digest-MD5-2.580.100_rc-1.gpkg.tar size=20480
2023-09-02 22:50:19.3819420420 +0200 HTTP-Date-6.60.0-1.gpkg.tar size=40960
2023-09-02 22:50:32.9085344820 +0200 Encode-Locale-1.50.0-r1-2.gpkg.tar size=40960
2023-09-02 22:50:45.6551312020 +0200 LWP-MediaTypes-6.40.0-2.gpkg.tar size=51200
2023-09-02 22:50:57.1784013020 +0200 perl-File-Temp-0.231.100-1.gpkg.tar size=20480
2023-09-02 22:51:10.0649972550 +0200 Clone-0.460.0-2.gpkg.tar size=40960
2023-09-02 22:51:22.7882607700 +0200 IO-HTML-1.4.0-2.gpkg.tar size=40960
2023-09-02 22:53:22.7576024440 +0200 passwdqc-2.0.3-1.gpkg.tar size=112640
2023-09-02 22:53:37.9508524060 +0200 IO-HTML-1.4.0-3.gpkg.tar size=40960
2023-09-02 22:53:38.6975149750 +0200 MIME-Charset-1.13.1-2.gpkg.tar size=51200
2023-09-02 22:53:51.0974469310 +0200 Pod-Parser-1.660.0-2.gpkg.tar size=102400
2023-09-02 22:53:51.2741126280 +0200 Sub-Name-0.270.0-2.gpkg.tar size=40960
2023-09-02 22:53:51.5041113660 +0200 TermReadKey-2.380.0-r1-1.gpkg.tar size=51200
2023-09-02 22:53:54.4440952330 +0200 Net-SSLeay-1.920.0-r1-2.gpkg.tar size=286720
2023-09-02 22:54:01.3240574800 +0200 Text-WrapI18N-0.60.0-r2-1.gpkg.tar size=30720
2023-09-02 22:54:10.0106764790 +0200 YAML-Tiny-1.740.0-2.gpkg.tar size=51200
2023-09-02 22:54:31.0605609690 +0200 File-Listing-6.160.0-1.gpkg.tar size=40960
2023-09-02 22:54:38.5605198130 +0200 libudev-251-r1-1.gpkg.tar size=30720
2023-09-02 22:54:53.6937701040 +0200 perl-CPAN-Meta-2.150.10-r7-1.gpkg.tar size=20480
2023-09-02 22:55:02.7903868530 +0200 Devel-CheckLib-1.160.0-2.gpkg.tar size=40960
2023-09-02 22:55:26.1535919820 +0200 Module-Build-0.423.400-2.gpkg.tar size=204800
2023-09-02 22:55:26.6969223340 +0200 Try-Tiny-0.310.0-2.gpkg.tar size=40960
2023-09-02 22:55:29.8869048290 +0200 perl-Compress-Raw-Zlib-2.204.1_rc-1.gpkg.tar size=20480
2023-09-02 22:55:35.1302093900 +0200 perl-libnet-3.150.0-1.gpkg.tar size=20480
2023-09-02 22:55:35.7402060430 +0200 Unicode-LineBreak-2019.1.0-1.gpkg.tar size=122880
2023-09-02 22:55:56.1500940450 +0200 compat-29.1.4.2-1.gpkg.tar size=143360
2023-09-02 22:57:03.0197271010 +0200 python-3.11.5-1.gpkg.tar size=28221440
2023-09-02 22:57:31.4529044090 +0200 IO-Socket-INET6-2.730.0-2.gpkg.tar size=40960
2023-09-02 22:57:39.4028607840 +0200 SGMLSpm-1.1-r2-1.gpkg.tar size=51200
2023-09-02 22:57:40.3095224750 +0200 URI-5.190.0-r1-1.gpkg.tar size=81920
2023-09-02 22:57:53.4561170010 +0200 perl-IO-Compress-2.204.0-1.gpkg.tar size=20480
2023-09-02 22:57:55.8727704060 +0200 Locale-gettext-1.70.0-r1-1.gpkg.tar size=40960
2023-09-02 22:58:12.3260134530 +0200 WWW-RobotRules-6.20.0-r2-2.gpkg.tar size=40960
2023-09-02 22:58:21.3292973820 +0200 HTTP-Message-6.440.0-2.gpkg.tar size=81920
2023-09-02 22:58:34.8658897670 +0200 libarchive-3.7.1-1.gpkg.tar size=624640
2023-09-02 22:58:49.1024783110 +0200 HTML-Parser-3.810.0-2.gpkg.tar size=102400
2023-09-02 22:58:50.7458026270 +0200 HTTP-Cookies-6.100.0-2.gpkg.tar size=51200
2023-09-02 22:59:04.2057287660 +0200 HTTP-Negotiate-6.10.0-r2-2.gpkg.tar size=40960
2023-09-02 22:59:33.0222373040 +0200 curl-8.1.2-1.gpkg.tar size=1484800
2023-09-02 23:01:33.3415770580 +0200 arpack-3.8.0-r1-1.gpkg.tar size=153600
2023-09-02 23:01:50.5714825100 +0200 opensp-1.5.2-r10-1.gpkg.tar size=1218560
2023-09-02 23:02:21.2746473610 +0200 icu-73.2-1.gpkg.tar size=15165440
2023-09-02 23:02:36.7978955110 +0200 strace-6.3-2.gpkg.tar size=1249280
2023-09-02 23:02:46.8278404720 +0200 cmake-3.26.5-r2-1.gpkg.tar size=17326080
2023-09-02 23:03:06.9710632710 +0200 glib-utils-2.76.4-1.gpkg.tar size=61440
2023-09-02 23:03:41.8142054040 +0200 more-itertools-10.1.0-1.gpkg.tar size=174080
2023-09-02 23:28:59.4158776460 +0200 nodejs-20.5.1-1.gpkg.tar size=18421760
2023-09-02 23:29:25.4824012740 +0200 platformdirs-3.10.0-1.gpkg.tar size=81920
2023-09-02 23:29:44.0822992080 +0200 pyparsing-3.0.9-1.gpkg.tar size=358400
2023-09-02 23:29:58.9755508160 +0200 typing-extensions-4.7.1-1.gpkg.tar size=122880
2023-09-02 23:30:16.3221222940 +0200 editables-0.5-1.gpkg.tar size=51200
2023-09-02 23:30:32.3587009610 +0200 pathspec-0.11.2-1.gpkg.tar size=102400
2023-09-02 23:31:00.7618784330 +0200 gpgme-1.21.0-1.gpkg.tar size=870400
2023-09-02 23:32:27.7447344530 +0200 qpdf-11.5.0-1.gpkg.tar size=6184960
2023-09-02 23:32:28.3947308860 +0200 portage-utils-0.96.1-1.gpkg.tar size=215040
2023-09-02 23:32:59.2045618180 +0200 wheel-0.41.1-1.gpkg.tar size=112640
2023-09-02 23:32:59.9645576480 +0200 magit-3.3.0-r4-1.gpkg.tar size=788480
2023-09-02 23:33:14.1378132060 +0200 po4a-0.66-1.gpkg.tar size=942080
2023-09-02 23:33:36.6776895200 +0200 jaraco-text-3.11.1-r1-1.gpkg.tar size=61440
2023-09-02 23:33:56.5842469500 +0200 lcms-2.15-1.gpkg.tar size=286720
2023-09-02 23:34:07.3675211110 +0200 libnvme-1.5-1.gpkg.tar size=245760
2023-09-02 23:34:30.3373950650 +0200 tar-1.35-1.gpkg.tar size=1054720
2023-09-02 23:35:09.6505126700 +0200 nvme-cli-2.5-1.gpkg.tar size=798720
2023-09-02 23:35:20.2204546680 +0200 debianutils-5.8-1.gpkg.tar size=51200
2023-09-02 23:35:20.4704532960 +0200 poppler-23.08.0-1.gpkg.tar size=2078720
2023-09-02 23:35:40.8470081470 +0200 Mozilla-CA-20999999-r1-2.gpkg.tar size=30720
2023-09-02 23:35:51.8436144710 +0200 setuptools-68.0.0-r1-1.gpkg.tar size=1320960
2023-09-02 23:36:07.4735287020 +0200 IO-Socket-SSL-2.83.0-2.gpkg.tar size=215040
2023-09-02 23:36:49.4199651900 +0200 docutils-0.20.1-r1-1.gpkg.tar size=3031040
2023-09-02 23:37:06.8965359550 +0200 cython-0.29.36-1.gpkg.tar size=3225600
2023-09-02 23:37:25.5897667100 +0200 Net-HTTP-6.230.0-2.gpkg.tar size=51200
2023-09-02 23:37:36.5063734730 +0200 charset-normalizer-3.2.0-1.gpkg.tar size=153600
2023-09-02 23:37:38.4630294020 +0200 gdbus-codegen-2.76.4-1.gpkg.tar size=174080
2023-09-02 23:37:52.3096200870 +0200 pybind11-2.11.1-1.gpkg.tar size=266240
2023-09-02 23:37:53.0462827110 +0200 pillow-10.0.0-1.gpkg.tar size=1003520
2023-09-02 23:38:25.7794364230 +0200 trove-classifiers-2023.7.6-1.gpkg.tar size=61440
2023-09-02 23:38:48.1759801900 +0200 psutil-5.9.5-1.gpkg.tar size=737280
2023-09-02 23:38:58.4325905740 +0200 meson-python-0.13.2-r1-1.gpkg.tar size=122880
2023-09-02 23:39:11.5525185790 +0200 pip-23.2.1-1.gpkg.tar size=4474880
2023-09-02 23:39:24.4257812710 +0200 fonttools-4.42.0-1.gpkg.tar size=3799040
2023-09-02 23:39:38.1390393530 +0200 lxml-4.9.3-r1-1.gpkg.tar size=1638400
2023-09-02 23:39:57.1856015030 +0200 loky-3.4.1-1.gpkg.tar size=174080
2023-09-02 23:40:07.3855455310 +0200 urllib3-2.0.4-1.gpkg.tar size=337920
2023-09-02 23:41:25.1117856790 +0200 numpy-1.25.2-1.gpkg.tar size=10219520
2023-09-02 23:41:57.6649403790 +0200 joblib-1.3.1-1.gpkg.tar size=573440
2023-09-02 23:42:09.3415429710 +0200 portage-3.0.49-r2-1.gpkg.tar size=3645440
2023-09-02 23:42:30.3180945300 +0200 perl-cleaner-2.31-1.gpkg.tar size=30720
2023-09-02 23:43:18.2944979290 +0200 iproute2-6.4.0-1.gpkg.tar size=1300480
2023-09-02 23:43:30.5544306530 +0200 genkernel-4.3.6-1.gpkg.tar size=188364800
2023-09-02 23:44:51.4539867210 +0200 elfutils-0.189-r1-1.gpkg.tar size=1105920
2023-09-02 23:46:37.6134041770 +0200 sudo-1.9.14_p2-1.gpkg.tar size=2099200
2023-09-02 23:49:35.6657604570 +0200 scipy-1.11.1-1.gpkg.tar size=28999680
2023-09-02 23:50:00.9122885850 +0200 grub-2.06-r7-3.gpkg.tar size=17530880
2023-09-02 23:50:19.9521841050 +0200 libwww-perl-6.600.0-r1-2.gpkg.tar size=163840
2023-09-02 23:50:39.2420782520 +0200 XML-Parser-2.460.0-r2-2.gpkg.tar size=184320
2023-09-02 23:51:02.9719480360 +0200 LWP-Protocol-https-6.110.0-1.gpkg.tar size=30720
2023-09-02 23:52:39.2180865570 +0200 mesa-23.1.6-1.gpkg.tar size=9113600
2023-09-02 23:53:11.7879078320 +0200 libepoxy-1.5.10-r2-1.gpkg.tar size=522240
2023-09-02 23:53:37.2511014370 +0200 glib-2.76.4-1.gpkg.tar size=3901440
2023-09-02 23:54:12.0709103660 +0200 xorg-server-21.1.8-r2-1.gpkg.tar size=3532800
2023-09-02 23:54:56.6873322020 +0200 xterm-383-1.gpkg.tar size=675840
2023-09-02 23:55:15.5105622440 +0200 libiio-0.25-1.gpkg.tar size=194560
2023-09-02 23:55:26.4205023760 +0200 bluez-5.68-1.gpkg.tar size=1474560
2023-09-02 23:55:47.5603863720 +0200 harfbuzz-8.0.1-1.gpkg.tar size=3389440
2023-09-02 23:57:21.8898687440 +0200 imagemagick-7.1.1.6-r1-1.gpkg.tar size=9246720
2023-09-02 23:57:33.1064738600 +0200 qtcore-5.15.10-r1-1.gpkg.tar size=8069120
2023-09-02 23:57:55.4196847510 +0200 libad9361-iio-0.3-1.gpkg.tar size=112640
2023-09-02 23:58:36.1627945090 +0200 mpv-0.36.0-r1-1.gpkg.tar size=2140160
2023-09-02 23:59:25.6058565260 +0200 soapysdr-0.8.1-2.gpkg.tar size=952320
2023-09-03 00:00:44.1787586950 +0200 matplotlib-3.7.2-1.gpkg.tar size=33546240
2023-09-03 00:05:41.8171254200 +0200 qtgui-5.15.10-r1-1.gpkg.tar size=5212160
2023-09-03 00:11:57.6650629740 +0200 soapyplutosdr-0.2.1-2.gpkg.tar size=81920
2023-09-03 00:12:16.8982907660 +0200 librsvg-2.56.3-1.gpkg.tar size=4126720
2023-09-03 00:16:04.1337104910 +0200 qtwidgets-5.15.10-r2-1.gpkg.tar size=3471360
2023-09-03 00:16:05.5137029180 +0200 gtkmm-3.24.8-1.gpkg.tar size=2488320
2023-09-03 00:16:38.4735220530 +0200 gtk-4.10.5-1.gpkg.tar size=13465600
2023-09-03 02:21:29.2761406470 +0200 firefox-102.15.0-1.gpkg.tar size=74403840
```

- i have installed the following extra packages while running on
  overlayfs (on nvme0n1p4):

```
sci-libs/fftw -fortran openmp -doc -mpi -test threads -zbus

media-sound/sox openmp -alsa -amr -ao -encode -flac -id3tag -ladspa -mad -magic -ogg -opus -oss -png pulseaudio -sndfile -sndio -static-libs -twolame -wavpack# required by net-analyzer/wireshark-4.0.6::gentoo[minizip]
# required by wireshark (argument)
>=sys-libs/zlib-1.2.13-r1 minizip
# required by x11-misc/xdg-utils-1.1.3_p20210805-r1::gentoo
# required by net-analyzer/wireshark-4.0.6::gentoo[gui]
# required by wireshark (argument)
>=app-text/xmlto-0.0.28-r10 text
# required by app-text/evince-44.2::gentoo
# required by evince (argument)
>=app-text/poppler-23.05.0 cairo
# required by gnome-base/gnome-keyring-42.1-r2::gentoo
# required by virtual/secret-service-0::gentoo
# required by app-crypt/libsecret-0.20.5-r3::gentoo
>=app-crypt/gcr-3.41.1-r2:0 gtk
app-text/mupdf X drm javascript ssl opengl# required by media-gfx/gimp-2.10.34-r1::gentoo
# required by gimp (argument)
>=media-libs/gegl-0.4.44 cairo
net-misc/tigervnc drm nls -opengl -server viewer -dri3 -gnutls -java -xinerama

net-libs/liquid-dsp ~amd64
sys-fs/duf ~amd64

dev-python/grpcio ~amd64
dev-python/grpcio-tools ~amd64
dev-libs/protobuf ~amd64
net-libs/grpc ~amd64

```


- i also added some files and have to save the browser config. copy
  this data to encrypted partition p4 or p5 and copy into overlayfs on
  first boot from squashfs. don't forget network script in /

- in this update cycle i want to add:
  - fftw mupdf tigervnc (viewer only) duf liquid-dsp grpc grpcio grpcio-tools feh glfw fdupes
  - btrfs kernel module and btrfs-progs
  
- add mold (81MB installed), ccache (7MB installed), include-what-you-use (18.5MB)
- add liquid-dsp (1.7MB)
- add glfw (0.6MB)
- add fftw sci-libs/fftw -fortran openmp -doc -mpi -test threads -zbus (was already present)
- replace firefox with firefox-bin
- i don't think i need gimp wireshark evince openjdk bazel fltk fuse zeromq go svn bazel
- modify portage config (note: that I recreate the entire config plus updates here):

```
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

net-libs/liquid-dsp ~amd64
sys-fs/duf ~amd64

dev-python/grpcio ~amd64
dev-python/grpcio-tools ~amd64
dev-libs/protobuf ~amd64
net-libs/grpc ~amd64
EOF


cat << EOF > /etc/portage/package.use/package.use
#www-client/firefox -clang -gmp-autoupdate -openh264 system-av1 system-harfbuzz system-icu system-jpeg system-libevent -system-libvpx -system-webp -dbus -debug -eme-free -geckodriver -hardened -hwaccel -jack -libproxy -lto -pgo pulseaudio -screencast -selinux -sndio -system-png -system-python-libs -wayland -wifi
# gmp-autoupdate .. Allow Gecko Media Plugins (binary blobs) to be automatically downloaded and kept up-to-date in user profiles
# this affects gmpopenh264 and widewinecdm
# i don't think i need that
# dns-over-https has been disabled by default (avoid going through cloudflare, can be enabled in preferences)
# app.normandy.enabled = false by default (mozilla can push changes to settings or install add-ons remotely)
www-client/firefox-bin alsa ffmpeg -gmp-autoupdate pulseaudio -selinux -wayland
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
media-sound/sox openmp -alsa -amr -ao -encode -flac -id3tag -ladspa -mad -magic -ogg -opus -oss -png pulseaudio -sndfile -sndio -static-libs -twolame -wavpack
# opengl requires javascript:
app-text/mupdf X drm javascript ssl opengl
net-misc/tigervnc drm nls -opengl -server viewer -dri3 -gnutls -java -xinerama

EOF



```

- install these new packages:

```
emerge -av --jobs=6 --load-average=10  glfw mold ccache include-what-you-use liquid-dsp mupdf btrfs-progs firefox-bin sox tigervnc
```
- this pulls in 36 new packages
```
equery depends librsvg
# emacs gtk imagemagick freetype ffmpeg cairo gtk+
# looks like i can't get rid of rust easily
emerge --deselect www-client/firefox
emerge --ask --depclean
# gone without replacement: firefox nss zip cbindgen nodejs nspr icu
# updated: autoconf dav1d
eclean-dist
#  [  186.1 M ] Total space from 60 files were freed in the distfiles directory
eclean-pkg
# already clean
```

- check news
```
eselect news read
# 2022-11-19-lvm2-default-USE-flags
```

- this seems to explain why my old gentoo broke a long time ago. i
  will have to add +lvm to sys-fs/lvm2. or USE="lvm"

- by default they remvoed lvm2 components and only provide device-mapper functionality

- i tried 3 times and was never able to find a fix.

- something about pipewire vs pulseaudio
- xorg can be started as normal user but user has to logind provider
  like elogind or systemd
- something about genkernel filenames


```
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n

2023-09-03 09:37:04.0711178870 +0200 gsm-1.0.22_p1-1.gpkg.tar size=71680
2023-09-03 09:37:17.4577110960 +0200 alabaster-0.7.13-1.gpkg.tar size=61440
2023-09-03 09:38:56.1071697620 +0200 imagesize-1.4.1-1.gpkg.tar size=61440
2023-09-03 09:39:08.8171000170 +0200 jbig2dec-0.19-1.gpkg.tar size=102400
2023-09-03 09:39:09.0037656590 +0200 snowballstemmer-2.2.0-r1-1.gpkg.tar size=389120
2023-09-03 09:39:12.1804148940 +0200 gumbo-0.10.1-1.gpkg.tar size=194560
2023-09-03 09:39:28.4069925190 +0200 Babel-2.12.1-1.gpkg.tar size=9594880
2023-09-03 09:39:40.8035911600 +0200 fftw-3.3.10-1.gpkg.tar size=4167680
2023-09-03 09:39:55.7801756430 +0200 openjpeg-2.5.0-r5-1.gpkg.tar size=337920
2023-09-03 09:39:56.0968405720 +0200 shadowman-3-1.gpkg.tar size=20480
2023-09-03 09:40:02.2501401390 +0200 mujs-1.3.3-1.gpkg.tar size=256000
2023-09-03 09:40:14.8967374090 +0200 mimalloc-2.1.2-1.gpkg.tar size=122880
2023-09-03 09:40:21.2700357690 +0200 glu-9.0.3-1.gpkg.tar size=215040
2023-09-03 09:40:22.2333638160 +0200 jpeg-100-r1-1.gpkg.tar size=30720
2023-09-03 09:40:22.3766963630 +0200 glfw-3.3.8-1.gpkg.tar size=174080
2023-09-03 09:41:00.9031516180 +0200 glu-9.0-r2-1.gpkg.tar size=30720
2023-09-03 09:41:06.0264568370 +0200 liquid-dsp-1.6.0-1.gpkg.tar size=491520
2023-09-03 09:41:09.7231032190 +0200 ccache-4.8.2-1.gpkg.tar size=1320960
2023-09-03 09:41:25.6763490090 +0200 include-what-you-use-0.20-1.gpkg.tar size=1894400
2023-09-03 09:41:45.6795725760 +0200 firefox-bin-117.0-1.gpkg.tar size=85248000
2023-09-03 09:41:54.0628599060 +0200 freeglut-3.4.0-1.gpkg.tar size=215040
2023-09-03 09:41:54.3528583150 +0200 fltk-1.3.5-r4-1.gpkg.tar size=1208320
2023-09-03 09:42:00.7128234150 +0200 sox-14.4.2_p20210509-r2-1.gpkg.tar size=409600
2023-09-03 09:42:38.9192804260 +0200 hwloc-2.9.1-1.gpkg.tar size=2764800
2023-09-03 09:43:13.0890929210 +0200 tigervnc-1.13.1-r3-1.gpkg.tar size=460800
2023-09-03 09:43:15.0157490150 +0200 tbb-2021.9.0-1.gpkg.tar size=491520
2023-09-03 09:43:20.3823862330 +0200 sphinxcontrib-applehelp-1.0.4-1.gpkg.tar size=71680
2023-09-03 09:43:34.4023092990 +0200 mupdf-1.22.0-1.gpkg.tar size=34048000
2023-09-03 09:44:00.0221687110 +0200 sphinxcontrib-devhelp-1.0.2-r1-1.gpkg.tar size=61440
2023-09-03 09:44:56.1351941280 +0200 sphinxcontrib-jsmath-1.0.1-r3-1.gpkg.tar size=51200
2023-09-03 09:45:45.2549245860 +0200 mold-2.0.0-r1-1.gpkg.tar size=3522560
2023-09-03 09:46:00.6815066000 +0200 sphinxcontrib-htmlhelp-2.0.1-1.gpkg.tar size=81920
2023-09-03 09:46:14.6347633650 +0200 sphinxcontrib-serializinghtml-1.1.5-r1-1.gpkg.tar size=71680
2023-09-03 09:46:28.4080211190 +0200 sphinxcontrib-qthelp-1.0.3-r2-1.gpkg.tar size=71680
2023-09-03 09:46:45.7745924870 +0200 sphinx-7.0.1-1.gpkg.tar size=3112960
2023-09-03 09:47:20.1577371450 +0200 btrfs-progs-6.3.3-1.gpkg.tar size=1177600
```

- backup persistent storage
```
cryptsetup luksOpen /dev/nvme0n1p4 p4
cryptsetup luksOpen /dev/nvme0n1p5 p5
mount /dev/mapper/p4 /mnt4
mount /dev/mapper/p5 /mnt5

rsync -avhHAX --progress --numeric-ids /mnt4/persistent/lower/ /mnt5/persistent_backupon20230903

#    -a : Archive mode. This tells rsync to copy all files, including directories, links, special files, and ACLs.
#    v : Verbose. This tells rsync to print more information about the transfer.
#    h : Human-readable. This tells rsync to print sizes in human-readable format.
#    H : Hard links. This tells rsync to preserve hard links.
#    A : Acls. This tells rsync to preserve ACLs.
#    X : Exclude extended attributes. This tells rsync to exclude extended attributes.
#    --progress : Show progress during transfer.
#    --numeric-ids : Don't map uid/gid values by user/group name. This tells rsync to transfer the numeric IDs of the users and groups instead of their names.

# bit slow, try with less logging

rsync -ahHAX --numeric-ids /mnt4/persistent/lower/ /mnt5/persistent_backupon20230903
# this is 62GB and there isn't enough space on mnt5 (only 25GB)

cd /mnt4
mv persistent persistent_backup_on20230903

```
- clion and bazel stuff can probably go
- make sure the important directories are readable


- modify filename for squashfs in init_dracut_crypt.sh
```
# Mount the ext4 filesystems
mount -t ext4 /dev/mapper/vg /mnt

# Mount the squashfs
mount /mnt/gentoo_20230903.squashfs /squash
```


```
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20230903_squash_crypt-6.3.12-gentoo-x86_64.img


```

- we will have to place a new squashfs into /mnt4


```

export INDIR=/
export OUTFILE=/mnt4/gentoo_20230903.squashfs
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
mnt4/ \
mnt5/ \
persistent

# 36sec
Filesystem size 2259640.29 Kbytes (2206.68 Mbytes)
        35.33% of uncompressed filesystem size (6395377.19 Kbytes)
Unrecognised xattr prefix system.posix_acl_access

# the new squashfs is 100MB larger than the old one
ls -ltrh /mnt4/*.squashfs
-rw-r--r-- 1 root root 2.1G Aug 12 21:42 /mnt4/gentoo_20230729.squashfs
-rw-r--r-- 1 root root 2.2G Sep  3 10:25 /mnt4/gentoo_20230903.squashfs


```

- check grub config, add the new entry

```
 emacs /boot/grub/grub.cfg


menuentry 'Gentoo GNU/Linux 20230903 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.3.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.3.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20230903_squash_crypt-6.3.12-gentoo-x86_64.img
}

```

## After update

- download new: chrome clion sonarlint protocol_buffers markdown ideolog 
- move (required) files from /mnt4/backup/lower/home/ into /home

- some libraries that are needed for google-chrome were deleted
```
martin@archlinux ~/Downloads/chrome $ ldd chrome
        linux-vdso.so.1 (0x00007fff31dbb000)
        libdl.so.2 => /usr/lib64/libdl.so.2 (0x00007f89bb140000)
        libpthread.so.0 => /usr/lib64/libpthread.so.0 (0x00007f89bb13b000)
        libgobject-2.0.so.0 => /usr/lib64/libgobject-2.0.so.0 (0x00007f89bb0cb000)
        libglib-2.0.so.0 => /usr/lib64/libglib-2.0.so.0 (0x00007f89baf49000)
        libnss3.so => not found
        libnssutil3.so => not found
        libsmime3.so => not found
        libnspr4.so => not found
```
- maybe next time just install google chrome bin from gentoo (if available)?
- for now this works:
```
sudo emerge -av nss nspr
```
- ublock o, sponsor



- recompile ryzen kernel module
```
cd ~/src/ryzen_monitor/ryzen_smu
sudo mkdir /mnt3
sudo mount /dev/nvme0n1p3 /mnt3
sudo ln -s /mnt3/usr/src/linux-6.3.12-gentoo /usr/src/linux-6.3.12-gentoo
make
sudo cp ryzen_smu.ko /lib/modules/6.3.12-gentoo-x86_64/kernel/
sudo depmod -a
sudo modprobe ryzen_smu

```

- maybe next time don't try to move folder with mc
- takes quite long and it complained about two files
- also i'm not sure if all the attributes are maintained
- i think the dates are okay

- copy .ssh folder

## Forgotten
- i forgot to check if quicklisp has updates
- i forgot to install grpc (i think)
  - also duf liquid-dsp grpc grpcio grpcio-tools feh fdupes


## Build gRPC
- manually install grpc
  https://github.com/grpc/grpc/blob/v1.57.0/src/cpp/README.md
- they say one should not install grpc system wide

- https://github.com/grpc/grpc/blob/v1.57.0/BUILDING.md

- this file is good it also explains cross-compilation using
  `-DCMAKE_TOOLCHAIN_FILE=path/to/file`
  
```
git clone -b v1.57.0 https://github.com/grpc/grpc
cd grpc
git submodule update --init
# uses 1.6GB 
mkdir -p cmake/build
cd cmake/build
cmake ../.. -G Ninja \
      -DCMAKE_INSTALL_PREFIX=/home/martin/grpc \
      -DBUILD_SHARED_LIBS=ON           \
      -DgRPC_INSTALL=ON                \
      -DCMAKE_BUILD_TYPE=Release       \
      -DgRPC_ABSL_PROVIDER=module     \
      -DgRPC_CARES_PROVIDER=module    \
      -DgRPC_PROTOBUF_PROVIDER=module \
      -DgRPC_RE2_PROVIDER=module      \
      -DgRPC_SSL_PROVIDER=package      \
      -DgRPC_ZLIB_PROVIDER=package
time ninja
# 2353 steps
ninja install
# 51MB
```

- not installed: emerge abseil-cpp c-ares protobuf dev-libs/re2
- installed: openssl zlib

- here is an example of how to build the dependencies as modules:
  https://github.com/grpc/grpc/blob/v1.57.0/test/distrib/cpp/run_distrib_test_cmake_module_install.sh



## Proposed updates

- add grpc
- add tmux
- i installed texlive (but i don't think i'll need that always)
- add rocm

```
1694748431.3609125970 /var/cache/binpkgs/dev-libs/re2/re2-0.2022.12.01-1.gpkg.tar 317440
1694748441.1241923550 /var/cache/binpkgs/dev-libs/xxhash/xxhash-0.8.1-1.gpkg.tar 112640
1694748465.2307267380 /var/cache/binpkgs/dev-cpp/abseil-cpp/abseil-cpp-20220623.1-1.gpkg.tar 1679360
1694748743.0025358130 /var/cache/binpkgs/net-libs/grpc/grpc-1.52.1-1.gpkg.tar 12_892_160

1694812658.1767854940 /var/cache/binpkgs/app-misc/tmux/tmux-3.3a-r1-1.gpkg.tar 501_760

1694815270.8391153170 /var/cache/binpkgs/dev-util/rocm-smi/rocm-smi-5.4.2-1.gpkg.tar 1_361_920
1694815770.2497081610 /var/cache/binpkgs/app-editors/vim-core/vim-core-9.0.1503-1.gpkg.tar 9830400
1694815791.6362574700 /var/cache/binpkgs/sys-process/numactl/numactl-2.0.16-1.gpkg.tar 112640
1694815803.3795263630 /var/cache/binpkgs/dev-libs/roct-thunk-interface/roct-thunk-interface-5.4.3-1.gpkg.tar 133120
1694817697.1624676640 /var/cache/binpkgs/sys-devel/llvm/llvm-15.0.7-r3-1.gpkg.tar 84_551_680
1694817710.8890590060 /var/cache/binpkgs/sys-devel/llvm-toolchain-symlinks/llvm-toolchain-symlinks-15-r1-1.gpkg.tar 20480
1694817749.0821827570 /var/cache/binpkgs/sys-libs/compiler-rt/compiler-rt-15.0.7-1.gpkg.tar 92_160
1694818127.9901035200 /var/cache/binpkgs/sys-libs/compiler-rt-sanitizers/compiler-rt-sanitizers-15.0.7-1.gpkg.tar 4_229_120
1694818137.9467155500 /var/cache/binpkgs/sys-devel/clang-runtime/clang-runtime-15.0.7-1.gpkg.tar 30720
1694818234.7528509980 /var/cache/binpkgs/sys-devel/lld/lld-16.0.6-1.gpkg.tar 3_481_600
1694818244.6694632480 /var/cache/binpkgs/sys-devel/lld-toolchain-symlinks/lld-toolchain-symlinks-16-r2-1.gpkg.tar 20480
1694842429.6509723210 /var/cache/binpkgs/sys-devel/clang/clang-15.0.7-r3-1.gpkg.tar 91_996_160
1694842444.3875581220 /var/cache/binpkgs/sys-devel/clang-toolchain-symlinks/clang-toolchain-symlinks-15-r2-1.gpkg.tar 20480
1694842804.2889165160 /var/cache/binpkgs/dev-libs/rocm-device-libs/rocm-device-libs-5.4.3-1.gpkg.tar 655_360
1694842850.9053273770 /var/cache/binpkgs/dev-libs/rocr-runtime/rocr-runtime-5.4.3-r1-1.gpkg.tar 839_680
1694842863.0152609250 /var/cache/binpkgs/dev-util/rocminfo/rocminfo-5.4.3-1.gpkg.tar 61_440

```

### ROCM

```
dev-util/rocm-smi ~amd64
dev-libs/roct-thunk-interface ~amd64
dev-libs/rocm-device-libs ~amd64
```

```
sudo emerge -av rocminfo rocm-smi
```
- installs clang 15

- AMDGPU is present in LLVM_TARGET
- openmp USE is enabled for clang-runtime
- libomp requires offload
```
[ebuild   R    ] sys-devel/clang-runtime-16.0.6:16::gentoo  USE="compiler-rt openmp sanitize -libcxx" 0 KiB
[ebuild   R    ] sys-libs/libomp-16.0.6:0/16::gentoo  USE="-debug -gdb-plugin -hwloc -offload -ompt -test -verify-sig" LLVM_TARGETS="(-AMDGPU) -NVPTX" PYTHON_SINGLE_TARGET="python3_11 -python3_10 (-python3_12)" 0 KiB
```


``` 
# in package.use
*/* AMDGPU_TARGETS: gfx90c -gfx908 -gfx90a -gfx1030 -gfx1031
```

- test offloading
```
cat <<EOF > TestOffload.cpp
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    const int N = 1024;
    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    // Initialize
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Offloading to GPU
    #pragma omp target map(to: a[0:N], b[0:N]) map(from: c[0:N])
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }

    // Verification
    for (int i = 0; i < N; ++i) {
        if (std::fabs(c[i] - 3.0f) > 1e-6) {
            std::cout << "Verification failed at index " << i << std::endl;
            delete[] a;
            delete[] b;
            delete[] c;
            return 1;
        }
    }

    std::cout << "Verification passed" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}
EOF
clang -fno-stack-protector -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa \
  --rocm-path=/usr --rocm-device-lib-path=/usr/lib/amdgcn/bitcode TestOffload.cpp -o TestOffload -lstdc++
./TestOffload 
# Verification passed

```
- Intro to GPU Programming with the OpenMP API (OpenMP Webinar) https://www.youtube.com/watch?v=uVcvecgdW7g

- paralution looks interesting: https://www.paralution.com/downloads/paralution-um.pdf


# Update 2023-09-30


- try to replace rust with rust-bin
https://wiki.gentoo.org/wiki/User:Vazhnov/Knowledge_Base:replace_rust_with_rust-bin
```
emerge --ask --unmerge dev-lang/rust
emerge --ask --oneshot virtual/rust dev-lang/rust-bin
```

- forbid installing rust in the future

```
cat << EOF > /etc/portage/package.mask/package.mask
>=sys-kernel/gentoo-sources-6.3.13
<sys-kernel/gentoo-sources-6.3.12
>=sys-kernel/linux-headers-6.3.13
<=sys-kernel/linux-headers-6.2
dev-lang/rust
EOF

cat << EOF > /etc/portage/package.mask/package.mask
>=sys-kernel/gentoo-sources-6.6.13
<sys-kernel/gentoo-sources-6.6.11
>=sys-kernel/linux-headers-6.6.14
<=sys-kernel/linux-headers-6.2
dev-lang/rust
EOF
```

```
emerge --ask --verbose --update --newuse --deep --with-bdeps=y @world --fetchonly
CXXFLAGS="-std=c++14" emerge -av opencv
```

- i decided not to add rocm yet
- i add tmux, rust-bin, grpc, feh, fdupes, sdl2, tkdiff
- i can't get opencv compiled

- when gcc is installed abort emerge and run this:
```
emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
 
emerge --jobs=6 --load-average=10  --ask --verbose tmux  net-libs/grpc app-misc/fdupes media-gfx/feh media-libs/libsdl2 tkdiff
emerge --ask --verbose --depclean opencv
revdep-rebuild

emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
emerge --depclean
eclean-dist
eclean-pkg
```
 
```
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

net-libs/liquid-dsp ~amd64
sys-fs/duf ~amd64

dev-python/grpcio ~amd64
dev-python/grpcio-tools ~amd64
dev-libs/protobuf ~amd64
net-libs/grpc ~amd64

dev-cpp/abseil-cpp ~amd64
EOF


cat << EOF > /etc/portage/package.use/package.use
#www-client/firefox -clang -gmp-autoupdate -openh264 system-av1 system-harfbuzz system-icu system-jpeg system-libevent -system-libvpx -system-webp -dbus -debug -eme-free -geckodriver -hardened -hwaccel -jack -libproxy -lto -pgo pulseaudio -screencast -selinux -sndio -system-png -system-python-libs -wayland -wifi
# gmp-autoupdate .. Allow Gecko Media Plugins (binary blobs) to be automatically downloaded and kept up-to-date in user profiles
# this affects gmpopenh264 and widewinecdm
# i don't think i need that
# dns-over-https has been disabled by default (avoid going through cloudflare, can be enabled in preferences)
# app.normandy.enabled = false by default (mozilla can push changes to settings or install add-ons remotely)
www-client/firefox-bin alsa ffmpeg -gmp-autoupdate pulseaudio -selinux -wayland
dev-lang/rust-bin -big-endian -clippy -doc -prefix -rust-analyzer -rust-src -rustfmt -verify-sig
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
# media-libs/opencv eigen features2d openmp python -contrib -contribcvv -contribdnn -contribfreetype -contribhdf -contribovis -contribsfm -contribxfeatures2d -cuda -debug -dnnsamples -download -examples ffmpeg -gdal -gflags -glog -gphoto2 gstreamer -gtk3 -ieee1394 -java jpeg -jpeg2k lapack -lto -opencl -opencvapps -openexr opengl png qt5 -tesseract -testprograms threads -tiff v4l vaapi -vtk -webp -xine
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
media-sound/sox openmp -alsa -amr -ao -encode -flac -id3tag -ladspa -mad -magic -ogg -opus -oss -png pulseaudio -sndfile -sndio -static-libs -twolame -wavpack
# opengl requires javascript:
app-text/mupdf X drm javascript ssl opengl
net-misc/tigervnc drm nls -opengl -server viewer -dri3 -gnutls -java -xinerama

app-misc/tmux systemd -debug -selinux -utempter -vim-syntax
net-libs/grpc -doc -examples -test
app-misc/fdupes ncurses
media-gfx/feh -curl -debug -exif inotify -test -xinerama
media-libs/libsdl2 X -joystick sound threads udev video -alsa -aqua -custom-cflags -dbus -doc -fcitx4 -gles1 -gles2 -haptic -ibus -jack -kms -libsamplerate -nas opengl -oss -pipewire pulseaudio -sndio -static-libs -vulkan -wayland -xscreensaver
net-print/cups -X -acl -dbus -debug -kerberos -openssl -pam -selinux ssl -static-libs -systemd -test -usb -xinetd -zeroconf
EOF



```

- dispatch-conf want to mess up /etc/sudoers and /etc/tlp.conf


```
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 

2023-09-30 17:09:10.0848416250 +0200 commonlisp-0-r1-1.gpkg.tar size=20480
2023-09-30 17:57:38.7455471720 +0200 rust-bin-1.71.1-1.gpkg.tar size=167843840
2023-09-30 18:16:00.3128357130 +0200 rust-bin-1.71.1-2.gpkg.tar size=167843840
2023-09-30 18:16:13.7860951130 +0200 rust-1.71.1-1.gpkg.tar size=30720
2023-09-30 18:20:25.5280470260 +0200 gnuconfig-20230731-1.gpkg.tar size=51200
2023-09-30 18:20:38.4413094980 +0200 hwdata-0.373-1.gpkg.tar size=2119680
2023-09-30 18:21:00.1978567770 +0200 libpaper-2.1.0-1.gpkg.tar size=51200
2023-09-30 18:21:11.5611277550 +0200 ensurepip-setuptools-68.1.2-1.gpkg.tar size=737280
2023-09-30 18:22:03.6541752300 +0200 linux-firmware-20230919-1.gpkg.tar size=404080640
2023-09-30 18:22:31.2740236680 +0200 libpcre-8.45-r2-1.gpkg.tar size=901120
2023-09-30 18:23:21.4970814050 +0200 slang-2.3.3-1.gpkg.tar size=1024000
2023-09-30 18:23:47.4336057460 +0200 gzip-1.13-1.gpkg.tar size=184320
2023-09-30 18:24:11.6334729510 +0200 kbd-2.6.1-1.gpkg.tar size=1392640
2023-09-30 18:24:30.6800351000 +0200 emacs-common-1.9-1.gpkg.tar size=102400
2023-09-30 18:26:33.1726962620 +0200 openssl-3.0.10-1.gpkg.tar size=6727680
2023-09-30 18:26:52.7159223530 +0200 perl-Module-Load-0.360.0-r3-1.gpkg.tar size=20480
2023-09-30 18:27:04.2325258230 +0200 URI-5.210.0-1.gpkg.tar size=92160
2023-09-30 18:27:19.3491095380 +0200 Mozilla-PublicSuffix-1.0.6-1.gpkg.tar size=71680
2023-09-30 18:27:30.6957139410 +0200 HTTP-CookieJar-0.14.0-1.gpkg.tar size=40960
2023-09-30 18:27:42.6189818460 +0200 Compress-Raw-Zlib-2.206.0-1.gpkg.tar size=81920
2023-09-30 18:27:52.5655939310 +0200 perl-Compress-Raw-Zlib-2.206.0-1.gpkg.tar size=20480
2023-09-30 18:28:06.7555160640 +0200 libwww-perl-6.720.0-r1-1.gpkg.tar size=163840
2023-09-30 18:28:22.4020968710 +0200 eselect-1.4.26-1.gpkg.tar size=102400
2023-09-30 18:28:53.5352593630 +0200 libxml2-2.11.5-1.gpkg.tar size=1443840
2023-09-30 18:29:10.1718347370 +0200 gpep517-15-1.gpkg.tar size=61440
2023-09-30 18:29:51.4216083810 +0200 gmp-6.3.0-1.gpkg.tar size=1095680
2023-09-30 18:30:07.6948524160 +0200 pyparsing-3.1.1-1.gpkg.tar size=378880
2023-09-30 18:30:32.0113856470 +0200 mpfr-4.2.1-1.gpkg.tar size=491520
2023-09-30 18:30:47.6546331390 +0200 jaraco-functools-3.9.0-1.gpkg.tar size=61440
2023-09-30 18:31:05.0712042330 +0200 wheel-0.41.2-1.gpkg.tar size=112640
2023-09-30 18:31:20.4711197270 +0200 nspektr-0.5.0-1.gpkg.tar size=51200
2023-09-30 18:31:40.8443412630 +0200 libksba-1.6.4-r1-1.gpkg.tar size=194560
2023-09-30 19:03:31.1805250620 +0200 gcc-13.2.1_p20230826-1.gpkg.tar size=89047040
2023-09-30 19:04:14.1402893230 +0200 libgcrypt-1.10.2-1.gpkg.tar size=860160
2023-09-30 19:04:42.1634688800 +0200 file-5.45-r1-1.gpkg.tar size=1044480
2023-09-30 19:05:06.2333367980 +0200 freetype-2.13.2-1.gpkg.tar size=716800
2023-09-30 19:05:36.7565026370 +0200 tcl-8.6.13-r1-1.gpkg.tar size=3102720
2023-09-30 19:06:30.6495402350 +0200 curl-8.2.1-1.gpkg.tar size=1505280
2023-09-30 19:06:53.2527495350 +0200 mpg123-1.31.3-r1-1.gpkg.tar size=450560
2023-09-30 19:07:29.0425531400 +0200 elfutils-0.189-r4-1.gpkg.tar size=1105920
2023-09-30 19:07:56.4957358260 +0200 libsndfile-1.2.2-1.gpkg.tar size=389120
2023-09-30 19:08:16.2522940790 +0200 libjpeg-turbo-3.0.0-1.gpkg.tar size=696320
2023-09-30 19:11:06.3280274640 +0200 mold-2.1.0-1.gpkg.tar size=3911680
2023-09-30 19:11:44.0911535750 +0200 libwebp-1.3.1_p20230908-1.gpkg.tar size=542720
2023-09-30 19:12:02.1843876220 +0200 threadpoolctl-3.2.0-1.gpkg.tar size=81920
2023-09-30 19:12:21.7076138230 +0200 setuptools-68.1.2-1.gpkg.tar size=1269760
2023-09-30 19:12:47.9474698330 +0200 meson-1.2.1-r1-1.gpkg.tar size=2662400
2023-09-30 19:13:15.2473200270 +0200 pygments-2.16.1-1.gpkg.tar size=2795520
2023-09-30 19:13:34.3472152170 +0200 trove-classifiers-2023.8.7-1.gpkg.tar size=61440
2023-09-30 19:13:52.7671141390 +0200 joblib-1.3.2-1.gpkg.tar size=573440
2023-09-30 19:16:36.6362149160 +0200 mesa-23.1.8-1.gpkg.tar size=9113600
2023-09-30 19:16:54.4827836500 +0200 pluggy-1.3.0-1.gpkg.tar size=92160
2023-09-30 19:17:30.5825855550 +0200 fonttools-4.42.1-1.gpkg.tar size=3799040
2023-09-30 19:17:56.1257787210 +0200 kiwisolver-1.4.5-1.gpkg.tar size=204800
2023-09-30 19:18:13.4923500900 +0200 tqdm-4.66.1-1.gpkg.tar size=225280
2023-09-30 19:24:09.1003987090 +0200 scipy-1.11.2-1.gpkg.tar size=29009920
2023-09-30 19:24:30.5936141000 +0200 sphinxcontrib-applehelp-1.0.7-1.gpkg.tar size=71680
2023-09-30 19:24:47.2168562140 +0200 sphinxcontrib-devhelp-1.0.5-1.gpkg.tar size=61440
2023-09-30 19:25:02.8934368560 +0200 sphinxcontrib-htmlhelp-2.0.4-1.gpkg.tar size=71680
2023-09-30 19:25:18.6900168400 +0200 sphinxcontrib-serializinghtml-1.1.9-1.gpkg.tar size=71680
2023-09-30 19:32:20.0043715670 +0200 systemd-253.11-1.gpkg.tar size=9461760
2023-09-30 19:32:51.6108647950 +0200 fontconfig-2.14.2-r3-1.gpkg.tar size=819200
2023-09-30 19:33:16.9673923190 +0200 tk-8.6.13-1.gpkg.tar size=2447360
2023-09-30 19:33:37.6039457430 +0200 tlp-1.6.0-1.gpkg.tar size=122880
2023-09-30 19:33:38.1706093010 +0200 libnvme-1.5-r2-1.gpkg.tar size=245760
2023-09-30 19:34:26.1703459050 +0200 lvm2-2.03.21-r1-1.gpkg.tar size=2836480
2023-09-30 19:34:27.3170062790 +0200 abseil-cpp-20230802.0-1.gpkg.tar size=1986560
2023-09-30 19:34:27.8136702200 +0200 man-pages-6.05.01-1.gpkg.tar size=3184640
2023-09-30 19:34:43.6169168340 +0200 dracut-059-r3-1.gpkg.tar size=450560
2023-09-30 19:35:10.2167708690 +0200 portage-3.0.51-1.gpkg.tar size=3778560
2023-09-30 19:35:33.7633083250 +0200 emacs-29.1-r1-1.gpkg.tar size=48015360
2023-09-30 19:36:24.9530274240 +0200 editor-0-r6-1.gpkg.tar size=20480
2023-09-30 19:37:29.5560062520 +0200 xmlto-0.0.28-r11-1.gpkg.tar size=71680
2023-09-30 19:38:20.1390620140 +0200 protobuf-23.3-r2-1.gpkg.tar size=3860480
2023-09-30 19:40:24.3083806410 +0200 qtcore-5.15.10-r2-1.gpkg.tar size=8069120
2023-09-30 19:40:38.5083027200 +0200 imagemagick-7.1.1.11-1.gpkg.tar size=9000960
2023-09-30 19:41:19.2580791080 +0200 sphinxcontrib-serializinghtml-1.1.9-2.gpkg.tar size=71680
2023-09-30 19:43:29.7806962050 +0200 matplotlib-3.8.0-1.gpkg.tar size=33935360
2023-09-30 19:44:33.8803444620 +0200 ffmpeg-6.0-r6-1.gpkg.tar size=10096640
2023-09-30 19:45:32.9233538000 +0200 qtgui-5.15.10-r2-1.gpkg.tar size=5212160
2023-09-30 19:46:18.3397712460 +0200 grub-2.06-r8-1.gpkg.tar size=17530880
2023-09-30 19:52:47.4176362020 +0200 xxhash-0.8.1-1.gpkg.tar size=112640
2023-09-30 19:52:47.6809680900 +0200 re2-0.2022.12.01-1.gpkg.tar size=317440
2023-09-30 19:52:49.2409595300 +0200 fdupes-2.2.1-1.gpkg.tar size=71680
2023-09-30 19:53:04.3008768900 +0200 imlib2-1.9.1-r1-1.gpkg.tar size=614400
2023-09-30 19:53:29.2740731840 +0200 feh-3.10-1.gpkg.tar size=204800
2023-09-30 19:53:37.5340278580 +0200 tmux-3.3a-r1-1.gpkg.tar size=501760
2023-09-30 19:53:46.3606460890 +0200 abseil-cpp-20230125.3-r1-1.gpkg.tar size=1925120
2023-09-30 19:54:36.3670383480 +0200 libsdl2-2.28.3-1.gpkg.tar size=1157120
2023-09-30 19:55:41.1366829280 +0200 protobuf-23.3-r2-2.gpkg.tar size=3860480
2023-09-30 20:01:48.7279991240 +0200 grpc-1.57.0-r1-1.gpkg.tar size=14940160
2023-09-30 20:04:38.1937358560 +0200 tkdiff-5.5.2-1.gpkg.tar size=296960
2023-09-30 20:13:28.4441594690 +0200 tigervnc-1.13.1-r3-2.gpkg.tar size=460800
2023-09-30 20:13:54.0273524160 +0200 mpv-0.36.0-r1-2.gpkg.tar size=2140160
2023-09-30 20:13:57.9906640010 +0200 firefox-bin-118.0.1-1.gpkg.tar size=87572480
2023-09-30 20:15:14.1935791740 +0200 sphinxcontrib-qthelp-1.0.6-1.gpkg.tar size=71680
2023-09-30 20:16:10.6432694100 +0200 sudo-1.9.14_p3-1.gpkg.tar size=2099200
2023-09-30 20:17:07.1062929050 +0200 qtwidgets-5.15.10-r3-1.gpkg.tar size=3471360
2023-09-30 20:33:52.5707754730 +0200 qttest-5.15.10-1.gpkg.tar size=256000
2023-09-30 20:34:28.9639091010 +0200 qtconcurrent-5.15.10-1.gpkg.tar size=71680
2023-09-30 20:35:14.0703282480 +0200 qtopengl-5.15.10-1.gpkg.tar size=235520
2023-09-30 21:17:28.7364194040 +0200 cups-2.4.7-1.gpkg.tar size=5949440
2023-09-30 22:21:56.9951925410 +0200 sphinx-7.1.2-1.gpkg.tar size=3215360



```

- update lisp

```
sbcl
(ql:update-client) 
(ql:update-dist "quicklisp")
# no change
# client 2021-02-13
# quicklisp 2023-06-18

# build swank
emacs
M-x slime
(ql:quickload "cl-cpp-generator2")
(ql:quickload "cl-py-generator")
(ql:quickload "cl-unicode")
(ql:quickload "cl-change-case")

```



```
emerge --ask --unmerge virtual/rust dev-lang/rust-bin
eselect gcc set 2
source /etc/profile 
emerge --ask --oneshot --usepkg=n sys-devel/libtool

emerge --depclean
eclean-dist
eclean-pkg

export INDIR=/
export OUTFILE=/mnt4/gentoo_20230930.squashfs
rm $OUTFILE
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
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x} \
persistent

# 38sec
Filesystem size 2035980.92 Kbytes (1988.26 Mbytes)
        33.70% of uncompressed filesystem size (6040918.82 Kbytes)
# slightly bigger than previous build
Filesystem size 1866772.87 Kbytes (1823.02 Mbytes)
        33.83% of uncompressed filesystem size (5517829.14 Kbytes)


# the new squashfs is 200MB smaller than the older one
ls -ltrh /mnt4/*.squashfs
-rw-r--r-- 1 root root 2.2G Sep  3 10:25 /mnt4/gentoo_20230903.squashfs
-rw-r--r-- 1 root root 2.0G Oct  1 17:02 /mnt4/gentoo_20230930.squashfs

```
- large files are LLVM. in particular /opt/rust-bin comes with its own copy of llvm.
- i should probably remove rust-bin
- libmupdf is 47MB
- gcc 13 and 12 have both 140MB binary folder
  - maybe i should have called `eselect gcc set 2`
- /usr/lib/firmware contains lots of small files (that i probably never use)
- /var/tmp/portage/media-libs/opencv-4.7.0 is a temporary build
  directory and should most certainly not be present


```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20230930_squash_crypt-6.3.12-gentoo-x86_64.img

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20230930 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.3.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.3.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20230930_squash_crypt-6.3.12-gentoo-x86_64.img
}

```
## After update

- i deleted linux directories (usr, var, opt..) in /mnt4/persistent/lower
- i kept my user home directory on the persistent partition this time
- download new or update: chrome clion sonarlint protocol_buffers markdown ideolog 

- chrome doesn't start anymore. next time i have to make sure that nss
  is put back in (note: initially i didn't boot from new squashfs,
  ignore that)
```
sudo emerge -av nss nspr
```

- ERROR:root:code for hash blake2b was not found. Failed to create
  binpkg file
  
- lots of errors: /opt/firefox/firefox-bin: error while loading shared
  libraries: libstdc++.so.6: cannot open shared object file: No such
  file or directory


```
sudo ln -s /home/martin/Downloads/chrome/google-chrome /usr/bin/
ln -s /usr/lib/gcc/x86_64-pc-linux-gnu/13/libgcc_s.so.1 /usr/lib64/

```
- the ryzen monitor isn't working

## Rebuild world 2023-10-01

```
sudo emerge -av nss nspr
eix-sync
emerge --jobs=12 --load-average=13 -e @world
ldconfig
```

- it looks like ldconfig was the only thing that was needed to prevent
  these errors:

```
/opt/firefox/firefox-bin: error while loading shared libraries: libstdc++.so.6: cannot open shared object file: No such file or directory

```

- 750 packages
- compile kernel (with gcc 13)

```
cd /usr/src/linux
make -j12
make modules_install
make install
```

```
emerge --depclean
eclean-dist # 20MB
eclean-pkg  # clean already
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n  
2023-10-01 11:25:48.1309301130 +0200 autoconf-2.71-r6-2.gpkg.tar size=860160
..
2023-10-01 16:40:51.8705301840 +0200 clang-runtime-16.0.6-2.gpkg.tar size=30720
2023-10-01 16:41:28.2336639770 +0200 include-what-you-use-0.20-2.gpkg.tar size=1873920
```
- ryzen module

```

cd ~/src/ryzen_monitor/ryzen_smu
make
sudo cp ryzen_smu.ko /lib/modules/6.3.12-gentoo-x86_64/kernel/
sudo depmod -a
sudo modprobe ryzen_smu

```

- create new dracut
- create new squashfs

# Try to install opencl runtime

- https://wiki.gentoo.org/wiki/OpenCL

```
accept_keywords:

dev-libs/rocm-opencl-runtime ~amd64
dev-libs/rocr-runtime ~amd64
dev-libs/rocm-comgr ~amd64
dev-libs/rocm-device-libs ~amd64
dev-util/rocm-cmake ~amd64
dev-libs/roct-thunk-interface ~amd64
sci-libs/clblast ~amd64
dev-util/rocminfo ~amd64

sudo emerge -av rocm-opencl-runtime sci-libs/clblast rocminfo

```
# Lenovo  fan speed
- https://github.com/torvalds/linux/blob/0ec5a38bf8499f403f81cb81a0e3a60887d1993c/drivers/platform/x86/ideapad-laptop.c


- periodic full power
```
echo 1 > /sys/devices/pci0000:00/0000:00:14.3/PNP0C09:00/VPC2004:00/fan_mode
```

- apparently 0 sets it back to auto


# Get NVME temperature

```
sudo nvme smart-log /dev/nvme0n1 | grep temperature
temperature                             : 50 °C (323 K)
```

# Update 2023-11-01

```
eix-sync
dispatch-conf
# make sure locale.gen stays as it is
emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
emerge -av x11-misc/xclip nss nspr
```

- rocm pulls in clang 17, so i don't think i want to install this now.
- i also had bullet installed, but i don't think i need that anymore
```
# emerge -av rocm-opencl-runtime sci-libs/clblast rocminfo
# package.use: sci-physics/bullet openmp threads -doc double-precision examples extras tbb -test
```

```
emerge --depclean # nothing 
revdep-rebuild # nothing
eclean-dist # 182M
eclean-pkg # 306M
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 
```

```
2023-11-01 08:21:22.4335391860 +0100 glibc-2.37-r7-1.gpkg.tar size=15790080
2023-11-01 08:21:43.7100890990 +0100 hwdata-0.374-1.gpkg.tar size=2129920
2023-11-01 08:21:55.0833600220 +0100 ensurepip-setuptools-68.2.2-1.gpkg.tar size=737280
2023-11-01 08:22:01.7566567360 +0100 llvm-common-16.0.6-4.gpkg.tar size=30720
2023-11-01 08:22:23.0198733890 +0100 hdparm-9.65-r1-1.gpkg.tar size=133120
2023-11-01 08:22:30.0065017170 +0100 python-exec-2.4.10-3.gpkg.tar size=40960
2023-11-01 08:22:34.4864771330 +0100 xxhash-0.8.2-1.gpkg.tar size=133120
2023-11-01 08:22:58.4796788050 +0100 ethtool-6.5-1.gpkg.tar size=266240
2023-11-01 08:23:09.3429525260 +0100 xclip-0.13-1.gpkg.tar size=51200
2023-11-01 08:23:13.0895986340 +0100 sandbox-2.38-1.gpkg.tar size=184320
2023-11-01 08:25:27.6721934520 +0100 openssl-3.0.11-1.gpkg.tar size=6727680
2023-11-01 08:25:53.4387187260 +0100 eselect-1.4.27-1.gpkg.tar size=102400
2023-11-01 08:26:06.1853154470 +0100 clang-common-16.0.6-r2-3.gpkg.tar size=30720
2023-11-01 08:26:26.0318732070 +0100 rust-1.71.1-r1-1.gpkg.tar size=30720
2023-11-01 08:26:43.6851096690 +0100 gpgme-1.22.0-1.gpkg.tar size=716800
2023-11-01 08:27:01.4316789520 +0100 transient-0.4.3-1.gpkg.tar size=163840
2023-11-01 08:27:02.1483416860 +0100 with-editor-3.3.2-1.gpkg.tar size=51200
2023-11-01 08:27:22.5182299070 +0100 libidn2-2.3.4-r1-1.gpkg.tar size=245760
2023-11-01 08:27:38.2181437550 +0100 typing-extensions-4.8.0-1.gpkg.tar size=112640
2023-11-01 08:27:58.2780336770 +0100 iproute2-6.5.0-1.gpkg.tar size=1310720
2023-11-01 08:28:10.2413013630 +0100 file-5.45-r3-1.gpkg.tar size=1044480
2023-11-01 08:28:35.5011627510 +0100 font-util-1.4.1-1.gpkg.tar size=81920
2023-11-01 08:28:53.1210660620 +0100 xcb-proto-1.16.0-1.gpkg.tar size=204800
2023-11-01 08:29:08.0909839160 +0100 nasm-2.16.01-r1-1.gpkg.tar size=583680
2023-11-01 08:29:35.5474999160 +0100 sqlite-3.43.2-1.gpkg.tar size=1740800
2023-11-01 08:30:05.0440047220 +0100 libxcb-1.16-1.gpkg.tar size=604160
2023-11-01 08:30:25.2938936020 +0100 compose-tables-1.8.7-1.gpkg.tar size=163840
2023-11-01 08:30:57.4470504970 +0100 libX11-1.8.7-1.gpkg.tar size=972800
2023-11-01 08:31:23.3035752770 +0100 setuptools-68.2.2-1.gpkg.tar size=1269760
2023-11-01 08:31:23.4435745090 +0100 libglvnd-1.7.0-1.gpkg.tar size=542720
2023-11-01 08:31:23.8335723690 +0100 libdrm-2.4.116-1.gpkg.tar size=276480
2023-11-01 08:32:55.7930677460 +0100 cython-3.0.2-r1-1.gpkg.tar size=4608000
2023-11-01 08:33:13.3196382370 +0100 setuptools-scm-8.0.4-1.gpkg.tar size=143360
2023-11-01 08:33:30.0295465420 +0100 trove-classifiers-2023.9.19-1.gpkg.tar size=61440
2023-11-01 08:33:50.2861020520 +0100 pytz-2023.3_p1-1.gpkg.tar size=102400
2023-11-01 08:33:51.4760955220 +0100 meson-python-0.14.0-1.gpkg.tar size=122880
2023-11-01 08:33:52.9427541400 +0100 urllib3-2.0.6-1.gpkg.tar size=337920
2023-11-01 08:34:12.4193139300 +0100 pillow-10.0.1-1.gpkg.tar size=1003520
2023-11-01 08:34:36.7058473260 +0100 gemato-20.5-1.gpkg.tar size=174080
2023-11-01 08:36:36.1451919090 +0100 numpy-1.26.1-1.gpkg.tar size=10588160
2023-11-01 08:37:06.8116902950 +0100 libudev-251-r2-1.gpkg.tar size=30720
2023-11-01 08:37:15.9383068800 +0100 sphinx-7.2.6-1.gpkg.tar size=3246080
2023-11-01 08:37:29.4648993200 +0100 contourpy-1.1.1-1.gpkg.tar size=337920
2023-11-01 08:39:35.7708728890 +0100 systemd-254.5-1.gpkg.tar size=9922560
2023-11-01 08:39:51.6041193390 +0100 udev-217-r7-1.gpkg.tar size=20480
2023-11-01 08:40:20.5072940670 +0100 libinput-1.24.0-1.gpkg.tar size=378880
2023-11-01 08:40:37.7271995740 +0100 dracut-059-r4-1.gpkg.tar size=450560
2023-11-01 08:40:40.8038493580 +0100 libXpm-3.5.17-1.gpkg.tar size=112640
2023-11-01 08:41:08.3236983440 +0100 btrfs-progs-6.5.2-1.gpkg.tar size=1198080
2023-11-01 08:41:33.0435626950 +0100 openssh-9.4_p1-r1-1.gpkg.tar size=1607680
2023-11-01 08:42:14.4633354060 +0100 xorg-server-21.1.9-1.gpkg.tar size=3543040
2023-11-01 08:42:34.4065593020 +0100 harfbuzz-8.2.0-1.gpkg.tar size=3450880
2023-11-01 08:43:39.5128687010 +0100 cups-2.4.7-r1-1.gpkg.tar size=5939200
2023-11-01 08:44:16.1160011770 +0100 ffmpeg-6.0-r9-1.gpkg.tar size=10096640
2023-11-01 08:44:38.2158799050 +0100 grub-2.06-r9-1.gpkg.tar size=17530880
2023-11-01 08:45:02.0857489200 +0100 xf86-video-amdgpu-23.0.0-3.gpkg.tar size=174080
2023-11-01 08:45:17.5056643050 +0100 firefox-bin-119.0-r2-1.gpkg.tar size=88156160
2023-11-01 08:45:17.8256625490 +0100 nghttp2-1.57.0-1.gpkg.tar size=225280
2023-11-01 08:46:03.7354106210 +0100 xf86-video-ati-22.0.0-3.gpkg.tar size=471040
2023-11-01 08:46:17.4053356080 +0100 xterm-384-1.gpkg.tar size=675840
2023-11-01 08:46:51.7484804850 +0100 curl-8.4.0-1.gpkg.tar size=1556480
2023-11-01 08:47:29.3149410080 +0100 xf86-input-libinput-1.4.0-1.gpkg.tar size=81920
2023-11-01 08:47:46.6381792810 +0100 brotli-1.1.0-1.gpkg.tar size=501760
2023-11-01 08:47:46.9081778000 +0100 smartmontools-7.4-1.gpkg.tar size=778240
2023-11-01 08:47:53.1381436130 +0100 poppler-23.09.0-1.gpkg.tar size=2068480
2023-11-01 08:47:53.4548085420 +0100 magit-3.3.0.50_p20230912-1.gpkg.tar size=870400
2023-11-01 08:48:14.3813603750 +0100 json-c-0.17-1.gpkg.tar size=245760
2023-11-01 09:21:14.1938295940 +0100 nspr-4.35-r2-4.gpkg.tar size=266240
2023-11-01 09:21:40.5670182060 +0100 nss-3.91-3.gpkg.tar size=3307520


```

```
export INDIR=/
export OUTFILE=/mnt4/gentoo_20231101.squashfs
rm $OUTFILE
time \
mksquashfs \
$INDIR \
$OUTFILE \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 6 \
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
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x} \
persistent
# 34sec with compression level 1
#Filesystem size 2040829.95 Kbytes (1993.00 Mbytes)
#        33.66% of uncompressed filesystem size (6062643.23 Kbytes)
# 10MB bigger than last time

# with compressino level 6
```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20231101_squash_crypt-6.3.12-gentoo-x86_64.img

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20231101 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.3.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.3.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20231101_squash_crypt-6.3.12-gentoo-x86_64.img
}

```
## After update

- i deleted linux directories (usr, var, opt..) in /mnt4/persistent/lower
- delete:
```
cd /mnt4/persistent/lower
rm -rf usr var opt etc
cd home/martin
rm -rf .cache/torch Downloads/chrome Downloads/chrome_old .gradle .android .espressif .arduino15 grpc scraper_env
```
- i kept my the remainder of user home directory on the persistent partition this time
- download new or update: chrome clion sonarlint protocol_buffers markdown ideolog 

# Issues 2023-12-17

- i might want to include chromedriver in the next update
- i think i typically download chrome (and not build it with gentoo, so it might be best to just install the appropriate chromedriver for that download version of chrome)

# Update 2023-12-17

- i think the last update was 2023-11-01

```
eix-sync
emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
```

- i want cmake 3.28, by default it will install 3.27.7. modify
  package.accept_keywords accordingly (see up at cat << EOF >
  /etc/portage/package.accept...)

- i also installed unrar last time. but i don't think i will need that
  in the future

- 129 packages will be updated

- ninja didn't want to overwrite /usr/bin/ninja, i installed it again

```
dispatch-conf # wants to mess up sudoers
emerge --depclean
```

```
 dev-python/calver
    selected: 2022.06.26 
   protected: none 
     omitted: none 

 sys-devel/binutils
    selected: 2.40-r5 
   protected: none 
     omitted: 2.41-r2 

```

```
revdep-rebuild # nothing
emerge -av nss nspr # to be sure that chrome will work
emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world

eclean-dist # 1.5G 80 files
eclean-pkg # 2G 84 files
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 
```

```
2023-12-17 19:51:56.1847093780 +0100 debianutils-5.8-1.gpkg.tar size=51200
2023-12-17 19:51:57.2580368220 +0100 debianutils-5.8-2.gpkg.tar size=51200
2023-12-17 19:51:58.4746968120 +0100 installkernel-gentoo-7-1.gpkg.tar size=20480
2023-12-17 20:01:32.6115462670 +0100 baselayout-2.14-r1-1.gpkg.tar size=61440
2023-12-17 20:01:33.3382089460 +0100 ensurepip-pip-23.3.1-1.gpkg.tar size=2088960
2023-12-17 20:01:35.0515328770 +0100 hwdata-0.376-1.gpkg.tar size=2140160
2023-12-17 20:01:37.9248504440 +0100 install-xattr-0.8-r1-1.gpkg.tar size=30720
2023-12-17 20:01:43.0414890330 +0100 ensurepip-setuptools-69.0.2-1.gpkg.tar size=747520
2023-12-17 20:01:47.1381332190 +0100 libffi-3.4.4-r2-1.gpkg.tar size=194560
2023-12-17 20:01:53.7080971670 +0100 alsa-ucm-conf-1.2.10-r1-1.gpkg.tar size=81920
2023-12-17 20:02:05.6213651270 +0100 hdparm-9.65-r2-1.gpkg.tar size=133120
2023-12-17 20:02:18.7679596520 +0100 libpaper-2.1.2-1.gpkg.tar size=51200
2023-12-17 20:02:28.6379054920 +0100 c-ares-1.21.0-1.gpkg.tar size=245760
2023-12-17 20:06:49.3631414430 +0100 timezone-data-2023c-r1-1.gpkg.tar size=368640
2023-12-17 20:06:53.5664517100 +0100 gzip-1-1.gpkg.tar size=20480
2023-12-17 20:07:36.6695485180 +0100 linux-firmware-20231211-1.gpkg.tar size=455239680
2023-12-17 20:08:46.5624983170 +0100 zlib-1.3-r2-1.gpkg.tar size=194560
2023-12-17 20:09:22.1789695410 +0100 xz-utils-5.4.5-1.gpkg.tar size=542720
2023-12-17 20:09:51.4321423490 +0100 passwdqc-2.0.3-r1-1.gpkg.tar size=112640
2023-12-17 20:10:09.2053781520 +0100 libtirpc-1.3.4-1.gpkg.tar size=204800
2023-12-17 20:10:31.5852553440 +0100 perl-Unicode-Collate-1.310.0-r1-1.gpkg.tar size=20480
2023-12-17 20:10:43.2418580460 +0100 getuto-1.9.1-1.gpkg.tar size=20480
2023-12-17 20:10:48.8284940560 +0100 HTTP-Message-6.450.0-1.gpkg.tar size=81920
2023-12-17 20:10:53.6751341270 +0100 libnsl-2.0.1-1.gpkg.tar size=51200
2023-12-17 20:11:18.9783286100 +0100 kbd-2.6.4-1.gpkg.tar size=1413120
2023-12-17 20:14:09.5673925120 +0100 qtcore-5.15.11-r1-1.gpkg.tar size=8089600
2023-12-17 20:14:58.7337893810 +0100 gettext-0.22.4-1.gpkg.tar size=3276800
2023-12-17 20:15:49.7801759330 +0100 texinfo-7.1-r1-1.gpkg.tar size=2048000
2023-12-17 20:16:45.7365355420 +0100 binutils-libs-2.41-r2-1.gpkg.tar size=1843200
2023-12-17 20:16:56.6431423590 +0100 gmp-6.3.0-r1-1.gpkg.tar size=1095680
2023-12-17 20:18:09.8727405160 +0100 binutils-2.41-r2-1.gpkg.tar size=9287680
2023-12-17 20:18:34.8326035500 +0100 iproute2-6.6.0-1.gpkg.tar size=1300480
2023-12-17 20:18:37.4692557480 +0100 libXrandr-1.5.4-1.gpkg.tar size=71680
2023-12-17 20:18:42.0458973010 +0100 elfutils-0.190-1.gpkg.tar size=1116160
2023-12-17 20:19:07.4624244960 +0100 libgpg-error-1.47-r1-1.gpkg.tar size=389120
2023-12-17 20:19:40.1155786470 +0100 kmod-31-1.gpkg.tar size=174080
2023-12-17 20:20:10.9887425650 +0100 alsa-lib-1.2.10-r2-1.gpkg.tar size=614400
2023-12-17 20:20:21.7853499860 +0100 sqlite-3.44.2-r1-1.gpkg.tar size=1751040
2023-12-17 20:20:38.2252597730 +0100 fonttosfnt-1.2.3-1.gpkg.tar size=61440
2023-12-17 20:20:53.5851754860 +0100 mpg123-1.32.3-1.gpkg.tar size=460800
2023-12-17 20:20:53.7451746080 +0100 debianutils-5.14-1.gpkg.tar size=51200
2023-12-17 20:21:07.4650993210 +0100 arpack-3.9.1-1.gpkg.tar size=153600
2023-12-17 20:21:21.7216877550 +0100 strace-6.6-1.gpkg.tar size=1269760
2023-12-17 20:21:59.2348152370 +0100 libarchive-3.7.2-1.gpkg.tar size=624640
2023-12-17 20:22:22.4580211340 +0100 qtdbus-5.15.11-1.gpkg.tar size=450560
2023-12-17 20:24:19.0573813020 +0100 cmake-3.28.1-1.gpkg.tar size=18298880
2023-12-17 20:24:35.6506235800 +0100 editor-0-r7-1.gpkg.tar size=20480
2023-12-17 20:24:47.5438916500 +0100 usbutils-016-1.gpkg.tar size=133120
2023-12-17 20:24:59.5071593360 +0100 feh-3.10.1-1.gpkg.tar size=215040
2023-12-17 20:25:33.4036399970 +0100 hwloc-2.9.2-1.gpkg.tar size=2805760
2023-12-17 20:25:33.5036394480 +0100 cycler-0.12.1-1.gpkg.tar size=61440
2023-12-17 20:25:48.2602251390 +0100 groff-1.23.0-1.gpkg.tar size=4116480
2023-12-17 20:25:59.9001612660 +0100 contourpy-1.2.0-1.gpkg.tar size=368640
2023-12-17 20:26:17.5200645770 +0100 openssh-9.5_p1-r2-1.gpkg.tar size=1617920
2023-12-17 20:26:36.9266247520 +0100 emacs-29.1-r5-1.gpkg.tar size=48015360
2023-12-17 20:26:57.9231762010 +0100 cython-3.0.6-1.gpkg.tar size=4638720
2023-12-17 20:27:48.7195641250 +0100 compat-29.1.4.4-1.gpkg.tar size=122880
2023-12-17 20:28:01.7228261040 +0100 jaraco-functools-4.0.0-1.gpkg.tar size=61440
2023-12-17 20:28:17.5327393480 +0100 fonttools-4.46.0-1.gpkg.tar size=4065280
2023-12-17 20:28:32.6726562680 +0100 sudo-1.9.15_p2-1.gpkg.tar size=2181120
2023-12-17 20:28:33.1393203740 +0100 jaraco-text-3.12.0-1.gpkg.tar size=61440
2023-12-17 20:28:49.8225621590 +0100 man-db-2.12.0-1.gpkg.tar size=1269760
2023-12-17 20:29:31.1156688980 +0100 lxml-4.9.3-r2-1.gpkg.tar size=1751040
2023-12-17 20:29:55.9021995500 +0100 packaging-23.2-1.gpkg.tar size=163840
2023-12-17 20:29:56.1988645890 +0100 mako-1.3.0-1.gpkg.tar size=266240
2023-12-17 20:29:56.4021968070 +0100 meson-python-0.15.0-1.gpkg.tar size=122880
2023-12-17 20:30:27.5586925040 +0100 platformdirs-4.0.0-1.gpkg.tar size=81920
2023-12-17 20:30:39.4052941630 +0100 pillow-10.1.0-1.gpkg.tar size=1034240
2023-12-17 20:30:53.9318811160 +0100 pooch-1.8.0-1.gpkg.tar size=194560
2023-12-17 20:31:05.1284863420 +0100 setuptools-69.0.2-r1-1.gpkg.tar size=1269760
2023-12-17 20:32:03.1181681260 +0100 uncertainties-3.1.7-r1-3.gpkg.tar size=245760
2023-12-17 20:32:23.4213900470 +0100 meson-1.2.3-1.gpkg.tar size=2672640
2023-12-17 20:33:42.9542869480 +0100 numpy-1.26.2-1.gpkg.tar size=10588160
2023-12-17 20:34:26.4540482450 +0100 bluez-5.70-r1-1.gpkg.tar size=1546240
2023-12-17 20:35:07.4938230410 +0100 util-linux-2.38.1-r3-1.gpkg.tar size=4505600
2023-12-17 20:35:15.8237773310 +0100 qtgui-5.15.11-r2-1.gpkg.tar size=4853760
2023-12-17 20:41:45.6083050760 +0100 qtwidgets-5.15.11-r1-1.gpkg.tar size=3471360
2023-12-17 20:43:07.9778530780 +0100 python-3.11.6-1.gpkg.tar size=28139520
2023-12-17 20:43:26.0477539200 +0100 scipy-1.11.4-1.gpkg.tar size=30033920
2023-12-17 20:45:22.2104498170 +0100 python-3.12.1-1.gpkg.tar size=29081600
2023-12-17 20:45:46.1036520370 +0100 wheel-0.42.0-1.gpkg.tar size=112640
2023-12-17 20:46:06.4935401490 +0100 certifi-3021.3.16-r4-1.gpkg.tar size=61440
2023-12-17 20:46:07.1702031020 +0100 btrfs-progs-6.6.2-1.gpkg.tar size=1208320
2023-12-17 20:46:16.6934841770 +0100 pygments-2.17.2-1.gpkg.tar size=2816000
2023-12-17 20:46:41.3800153780 +0100 charset-normalizer-3.3.2-1.gpkg.tar size=163840
2023-12-17 20:46:56.1632675890 +0100 idna-3.6-1.gpkg.tar size=174080
2023-12-17 20:47:11.3565175500 +0100 urllib3-2.1.0-1.gpkg.tar size=286720
2023-12-17 20:47:25.6431058200 +0100 trove-classifiers-2023.11.29-1.gpkg.tar size=61440
2023-12-17 20:47:45.7729953580 +0100 lmfit-1.2.2-3.gpkg.tar size=307200
2023-12-17 20:47:48.6996459650 +0100 cloudpickle-3.0.0-1.gpkg.tar size=92160
2023-12-17 20:48:12.1328507100 +0100 ninja-1.11.1-r3-1.gpkg.tar size=225280
2023-12-17 20:48:13.3128442350 +0100 pip-23.3.1-1.gpkg.tar size=4536320
2023-12-17 20:48:37.4427118230 +0100 Babel-2.13.1-1.gpkg.tar size=9594880
2023-12-17 20:48:53.6026231470 +0100 matplotlib-3.8.2-r1-1.gpkg.tar size=33935360
2023-12-17 20:49:12.6758518170 +0100 ninja-1-1.gpkg.tar size=20480
2023-12-17 20:51:06.3185615420 +0100 systemd-254.5-r1-1.gpkg.tar size=9902080
2023-12-17 20:51:26.4084513000 +0100 desktop-file-utils-0.27-1.gpkg.tar size=122880
2023-12-17 20:51:38.1283869870 +0100 libdrm-2.4.117-1.gpkg.tar size=286720
2023-12-17 20:51:50.0383216320 +0100 portage-3.0.57-1.gpkg.tar size=3840000
2023-12-17 20:52:15.2815164450 +0100 vulkan-headers-1.3.268-1.gpkg.tar size=2017280
2023-12-17 20:52:23.1448066280 +0100 spirv-headers-1.3.268-1.gpkg.tar size=245760
2023-12-17 20:52:25.0647960930 +0100 xkeyboard-config-2.40-1.gpkg.tar size=1239040
2023-12-17 20:52:33.9847471450 +0100 blake3-1.5.0-1.gpkg.tar size=81920
2023-12-17 20:52:53.0913089650 +0100 libnvme-1.6-r1-1.gpkg.tar size=256000
2023-12-17 20:52:53.6979723030 +0100 cairo-1.18.0-1.gpkg.tar size=880640
2023-12-17 20:53:38.3110608240 +0100 soapyplutosdr-0.2.1_p20220710-1.gpkg.tar size=81920
2023-12-17 20:54:28.8907832710 +0100 abseil-cpp-20220623.1-1.gpkg.tar size=1648640
2023-12-17 20:54:40.6373854790 +0100 qpdf-11.6.3-r1-1.gpkg.tar size=5826560
2023-12-17 20:54:56.9372960340 +0100 protobuf-21.12-1.gpkg.tar size=3584000
2023-12-17 20:55:07.1239068020 +0100 ccache-4.8.3-1.gpkg.tar size=1320960
2023-12-17 20:55:44.6203677090 +0100 libva-2.20.0-1.gpkg.tar size=256000
2023-12-17 20:56:14.2735383220 +0100 libxkbcommon-1.6.0-1.gpkg.tar size=266240
2023-12-17 20:56:31.9801078250 +0100 vulkan-loader-1.3.268-1.gpkg.tar size=184320
2023-12-17 20:56:59.5699564270 +0100 gentoolkit-0.6.3-1.gpkg.tar size=3522560
2023-12-17 20:57:26.8031403200 +0100 spirv-tools-1.3.268-1.gpkg.tar size=5406720
2023-12-17 21:01:52.7750141460 +0100 grpc-1.52.1-1.gpkg.tar size=12523520
2023-12-17 21:02:39.6380903210 +0100 nvme-cli-2.6-1.gpkg.tar size=808960
2023-12-17 21:02:40.6114183130 +0100 tlp-1.6.1-1.gpkg.tar size=122880
2023-12-17 21:03:27.5611606790 +0100 dracut-059-r5-1.gpkg.tar size=450560
2023-12-17 21:05:55.4170159950 +0100 mold-2.4.0-1.gpkg.tar size=3461120
2023-12-17 21:06:20.3468791930 +0100 xterm-388-1.gpkg.tar size=686080
2023-12-17 21:06:37.0001211430 +0100 pycairo-1.25.1-1.gpkg.tar size=174080
2023-12-17 21:06:42.7900893710 +0100 firefox-bin-120.0.1-1.gpkg.tar size=89477120
2023-12-17 21:07:21.3998775020 +0100 glslang-1.3.268-r2-1.gpkg.tar size=6481920
2023-12-17 21:09:00.6626661360 +0100 mesa-23.1.9-1.gpkg.tar size=9093120
2023-12-17 21:09:20.9192216460 +0100 shaderc-2023.7-1.gpkg.tar size=409600
2023-12-17 21:09:47.9657398960 +0100 xorg-server-21.1.10-r1-1.gpkg.tar size=3543040
2023-12-17 21:10:00.4323381530 +0100 libplacebo-6.338.1-1.gpkg.tar size=624640
2023-12-17 21:10:10.6856152220 +0100 xf86-input-libinput-1.4.0-2.gpkg.tar size=81920
2023-12-17 21:10:32.5988283080 +0100 mpv-0.37.0-1.gpkg.tar size=2252800
2023-12-17 21:10:32.9954927980 +0100 xf86-video-amdgpu-23.0.0-4.gpkg.tar size=174080
2023-12-17 21:10:59.0153500150 +0100 xf86-video-ati-22.0.0-4.gpkg.tar size=471040
2023-12-17 21:18:38.6561610940 +0100 ninja-1.11.1-r3-2.gpkg.tar size=225280
2023-12-17 21:21:56.2550767810 +0100 nspr-4.35-r2-5.gpkg.tar size=266240
2023-12-17 21:22:22.1316014510 +0100 nss-3.91-4.gpkg.tar size=3307520
```

```
export INDIR=/
export OUTFILE=/mnt4/gentoo_20231217.squashfs
rm $OUTFILE
time \
mksquashfs \
$INDIR \
$OUTFILE \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 6 \
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
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x} \
persistent

# 1m21sec , nearly a minute slower than last time, i used compression level 6

Filesystem size 1974429.96 Kbytes (1928.15 Mbytes)
        30.67% of uncompressed filesystem size (6438024.75 Kbytes)
# compressed filesystem is 60MB smaller, uncompressid is 400MB larger
```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20231217_squash_crypt-6.3.12-gentoo-x86_64.img

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20231217 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.3.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.3.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20231217_squash_crypt-6.3.12-gentoo-x86_64.img
}

```

- I deleted etc usr var in /mnt4/persistent. the home directory i
  mostly kept as it is. i deleted an old version of clion but none of
  the dot-directories

## update chrome

- download the stable version
- the following chrome is only for testing (selenium)
```
wget -c https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/120.0.6099.71/linux64/chrome-linux64.zip
wget https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/120.0.6099.71/linux64/chromedriver-linux64.zip

```

- the normal version i download as rpm and convert to tar with rpm2tar ...


## Rebuild world 2023-12-18

```
eix-sync
emerge --jobs=12 --load-average=13 --fetchonly -e @world
emerge --jobs=12 --load-average=13 -e @world

```

- issue with scikit-learn and liquid-dsp
- 769 packages

```
emerge --depclean
revdep-rebuild
eclean-dist
eclean-pkg
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 

```

```
2023-12-17 21:22:22.1316014510 +0100 nss-3.91-4.gpkg.tar size=3307520
2023-12-18 08:34:32.8019104270 +0100 libintl-0-r2-3.gpkg.tar size=30720
2023-12-18 08:34:32.9085765090 +0100 gnuconfig-20230731-4.gpkg.tar size=51200
2023-12-18 08:34:36.7652220120 +0100 libiconv-0-r2-3.gpkg.tar size=30720
...
2023-12-18 13:12:34.4870372380 +0100 clang-runtime-16.0.6-3.gpkg.tar size=30720
2023-12-18 13:13:04.0000000000 +0100 Packages size=2483125
2023-12-18 13:13:04.0968747560 +0100 include-what-you-use-0.20-3.gpkg.tar size=1873920

```
```
sudo emerge -av xrandr
```

- complete build took 4h45min

```
export INDIR=/
export OUTFILE=/mnt4/gentoo_20231218.squashfs
rm $OUTFILE
time \
mksquashfs \
$INDIR \
$OUTFILE \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 6 \
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
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x} \
persistent

```

```
Filesystem size 1979134.07 Kbytes (1932.75 Mbytes)
        30.58% of uncompressed filesystem size (6472722.79 Kbytes)

```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20231218_squash_crypt-6.3.12-gentoo-x86_64.img

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20231218 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.3.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.3.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20231218_squash_crypt-6.3.12-gentoo-x86_64.img
}

```

# Update 2024-01-14

```
eix-sync
dispatch-conf
# make sure locale.gen stays as it is
emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
eclean-dist
eclean-pkg
revdep-rebuild # nothing
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 

```


```
  2140  2024-01-14 00:59:07.6754200530 +0100 xmltoman-0.6-1.gpkg.tar size=20480
  2141  2024-01-14 00:59:09.1787451370 +0100 xmltoman-0.6-2.gpkg.tar size=20480
  2142  2024-01-14 00:59:10.6820702210 +0100 xmltoman-0.6-3.gpkg.tar size=20480
  2143  2024-01-14 00:59:12.1953952500 +0100 ninja-1.11.1-r2-1.gpkg.tar size=225280
  2144  2024-01-14 00:59:13.7387201140 +0100 ninja-1.11.1-r3-1.gpkg.tar size=225280
  2145  2024-01-14 00:59:15.2520451430 +0100 ninja-1.11.1-r3-2.gpkg.tar size=225280
  2146  2024-01-14 00:59:16.7687034880 +0100 ninja-1.11.1-r3-3.gpkg.tar size=225280
,,,
  2512  2024-01-14 07:33:00.4824029520 +0100 gtk+-3.24.39-1.gpkg.tar size=11673600
  2513  2024-01-14 07:58:40.7472841640 +0100 adwaita-icon-theme-45.0-1.gpkg.tar size=2099200
  2514  2024-01-14 07:58:44.4739303800 +0100 clang-17.0.6-1.gpkg.tar size=101068800
  2515  2024-01-14 07:59:06.3071439050 +0100 clang-toolchain-symlinks-17-1.gpkg.tar size=20480
  2516  2024-01-14 07:59:15.8404249250 +0100 libnotify-0.8.3-1.gpkg.tar size=81920
  2517  2024-01-14 07:59:54.7202115740 +0100 firefox-bin-121.0.1-r1-1.gpkg.tar size=89671680
  2518  2024-01-14 08:00:25.4267097410 +0100 compiler-rt-17.0.6-1.gpkg.tar size=92160
  2519  2024-01-14 08:00:42.6766150830 +0100 gtk-4.12.4-1.gpkg.tar size=13731840
  2520  2024-01-14 08:07:22.8677523890 +0100 compiler-rt-sanitizers-17.0.6-1.gpkg.tar size=4618240
  2521  2024-01-14 08:07:33.1676958680 +0100 clang-runtime-17.0.6-1.gpkg.tar size=30720
```


```
cryptsetup luksOpen /dev/nvme0n1p4 p4
cryptsetup luksOpen /dev/nvme0n1p5 p5
mount /dev/mapper/p4 /mnt4
mount /dev/mapper/p5 /mnt5
```

```
export INDIR=/
export OUTFILE=/mnt4/gentoo_20240114.squashfs
rm $OUTFILE
time \
mksquashfs \
$INDIR \
$OUTFILE \
-comp zstd \
-xattrs \
-not-reproducible \
-Xcompression-level 6 \
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
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x} \
persistent

```

- quite a lot larger then before. is this because llvm was updated?
- i think next i should update kernel and then rebuild world
```
Filesystem size 2179491.07 Kbytes (2128.41 Mbytes)
        29.59% of uncompressed filesystem size (7365737.88 Kbytes)
```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.3.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20240114_squash_crypt-6.3.12-gentoo-x86_64.img

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240114 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.3.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.3.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240114_squash_crypt-6.3.12-gentoo-x86_64.img
}

```

* Update 2024-01-17

- update kernel from 6.3.12 to 6.6.12
- modify package mask

```
eix-sync
dispatch-conf
emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
emerge sys-kernel/gentoo-sources
eselect kernel set 1
cd /usr/src/linux
make oldconfig
make -j 12
make -j 12 modules
make modules_install install
emacs /boot/grub/grub.cfg
menuentry 'Gentoo GNU/Linux 6.6.12' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.12-gentoo-x86_64 ...'
	linux	/vmlinuz-6.6.12-gentoo-x86_64 root=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 ro  
	echo	'Loading initial ramdisk ...'
	initrd	/initramfs-6.6.12-gentoo-x86_64.img
}
```


```
dracut: *** Creating initramfs image file '/usr/src/linux-6.6.12-gentoo/arch/x86/boot/initrd' done ***
```
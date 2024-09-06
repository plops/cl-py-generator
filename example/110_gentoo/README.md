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
wget https://distfiles.gentoo.org/releases/amd64/autobuilds/20240505T170430Z/stage3-amd64-nomultilib-systemd-20240505T170430Z.tar.xz
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
CPU_FLAGS_X86="aes avx avx2 f16c fma3 mmx mmxext pclmul popcnt rdrand sha sse sse2 sse3 sse4_1 sse4_2 sse4a ssse3"
CFLAGS="${COMMON_FLAGS}"
CXXFLAGS="${COMMON_FLAGS}"
FCFLAGS="${COMMON_FLAGS}"
FFLAGS="${COMMON_FLAGS}"
LC_MESSAGES=C.utf8
MAKEOPTS="-j12"
EMERGE_DEFAULT_OPTS="--jobs 14 --load-average 32"
USE="X vaapi -doc -cups"
#VIDEO_CARDS="radeon radeonsi amdgpu"
VIDEO_CARDS="nvidia"
FEATURES="buildpkg"
PKGDIR="/var/cache/binpkgs"
BINPKG_FORMAT="gpkg"
BINPKG_COMPRESS="zstd"
BINPKG_COMPRESS_FLAG_ZSTD="-T0"
L10N="en-GB"
#LLVM_TARGETS="X86 AMDGPU"
LLVM_TARGETS="X86 NVPTR"
INPUT_DEVICES="libinput evdev synaptics"
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
dev-python/scikit-learn ~amd64
dev-python/pythran ~amd64
dev-cpp/xsimd ~amd64
media-video/obs-studio ~amd64
EOF

cat << EOF > /etc/portage/package.mask/package.mask
>=sys-kernel/gentoo-sources-6.6.18
<=sys-kernel/gentoo-sources-6.6.16
>=sys-kernel/linux-headers-6.6.19
<=sys-kernel/linux-headers-6.2
dev-lang/rust
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
 media-video/ffmpeg X bzip2 -dav1d encode gnutls gpl iconv network postproc threads vaapi zlib alsa -amf -amr -amrenc -appkit -bluray -bs2b -cdio -chromaprint -chromium -codec2 -cpudetection -cuda -debug -doc -fdk -flite -fontconfig -frei0r -fribidi -gcrypt -gme -gmp -gsm -hardcoded-tables -iec61883 -ieee1394 -jack -jpeg2k -kvazaar -ladspa -libaom -libaribb24 -libass -libcaca -libdrm -libilbc -librtmp -libsoxr -libtesseract -libv4l -libxml2 -lv2 -lzma -mipsdspr1 -mipsdspr2 -mipsfpu -mmal -modplug -mp3 -nvenc -openal -opencl -opengl -openh264 -openssl opus -oss -pic pulseaudio -qsv -rav1e -rubberband -samba -sdl -snappy -sndio -speex -srt -ssh -static-libs -svg -svt-av1 -test -theora -truetype -twolame -v4l -vdpau -verify-sig -vidstab -vmaf -vorbis -vpx -vulkan -webp x264 -x265 -xvid -zeromq -zimg -zvbi
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
dev-qt/qtgui X libinput png udev -accessibility dbus -debug -egl -eglfs -evdev -gles2-only -ibus jpeg -linuxfb -test -tslib -tuio -vnc vulkan -wayland
dev-qt/qtwidgets X png dbus -debug -gles2-only -gtk -test
sys-apps/qdirstat

net-wireless/soapysdr -bladerf -hackrf plutosdr python -rtlsdr -uhd

x11-libs/wxGTK X lzma spell -curl -debug -doc -gstreamer -keyring libnotify opengl -pch -sdl -test -tiff -wayland -webkit
dev-libs/libpcre2 bzip2 jit pcre16 pcre32 readline unicode zlib -libedit -split-usr -static-libs

sci-libs/fftw -fortran openmp -doc -mpi -test threads -zbus
media-sound/sox openmp -alsa -amr -ao -encode -flac -id3tag -ladspa -mad -magic -ogg -opus -oss -png pulseaudio -sndfile -sndio -static-libs -twolame -wavpack
# opengl requires javascript:
app-text/mupdf X drm -javascript ssl -opengl
net-misc/tigervnc drm nls -opengl -server viewer -dri3 -gnutls -java -xinerama

app-misc/tmux systemd -debug -selinux -utempter -vim-syntax
net-libs/grpc -doc -examples -test
app-misc/fdupes ncurses
media-gfx/feh -curl -debug -exif inotify -test -xinerama
media-libs/libsdl2 X -joystick sound threads udev video -alsa -aqua -custom-cflags -dbus -doc -fcitx4 -gles1 -gles2 -haptic -ibus -jack -kms -libsamplerate -nas opengl -oss -pipewire pulseaudio -sndio -static-libs -vulkan -wayland -xscreensaver
net-print/cups -X -acl -dbus -debug -kerberos -openssl -pam -selinux ssl -static-libs -systemd -test -usb -xinetd -zeroconf
media-libs/mesa X gles2 llvm proprietary-codecs vaapi zstd -d3d9 -debug -gles1 -lm-sensors -opencl -osmesa -selinux -test -unwind -valgrind -vdpau vulkan vulkan-overlay -wayland -xa -zink


media-video/mpv X alsa cli libmpv openal opengl pulseaudio vaapi zlib -aqua -archive -bluray -cdda -coreaudio -debug -drm -dvb -dvd -egl -gamepad -iconv -jack -javascript -jpeg -lcms -libcaca -lua -mmal -nvenc -pipewire -raspberry-pi -rubberband -sdl -selinux -sixel -sndio -test -tools -uchardet -vdpau vulkan -wayland -xv -zimg

# wireshark pulls in a lot of qt stuff
net-libs/libpcap -bluetooth -dbus -netlink -rdma -remote -static-libs -test usb -verify-sig -yydebug
net-analyzer/wireshark capinfos captype dftest dumpcap editcap filecaps gui mergecap minizip netlink pcap plugins randpkt randpktdump reordercap sharkd ssl text2pcap tshark udpdump zlib zstd -androiddump -bcg729 -brotli -ciscodump -doc -dpauxmon http2 -ilbc -kerberos -libxml2 -lua -lz4 -maxminddb -opus qt6 -sbc -sdjournal -selinux -smi -snappy -spandsp -sshdump -test -tfshark -verify-sig -wifi
dev-libs/boehm-gc large threads  cxx -static-libs
app-text/xmlto text -latex
dev-qt/qtmultimedia X ffmpeg -vaapi -alsa -eglfs -gstreamer -opengl -pulseaudio -qml -test -v4l -vulkan
sys-libs/zlib minizip -static-libs -verify-sig
dev-qt/qtbase X concurrent dbus gui libinput network nls -opengl sql sqlite ssl udev -vulkan widgets xml -accessibility -brotli -cups -eglfs -evdev -gles2-only -gssapi -gtk -icu -libproxy -mysql -oci8 -odbc -postgres -sctp -test -tslib -wayland -zstd
#dev-qt/qttools assistant linguist widgets -clang -designer -distancefieldgenerator -gles2-only -opengl -pixeltool -qdbus -qdoc -qml -qtattributionsscanner -qtdiag -qtplugininfo -test -vulkan -zstd
dev-qt/qtdeclarative jit widgets -debug -gles2-only -localstorage -test -vulkan

media-video/obs-studio alsa ssl -browser -decklink -fdk -jack -lua -mpegts -nvenc -pipewire pulseaudio -python -qsv -speex -test -truetype v4l -vlc -wayland -websocket
sci-libs/armadillo arpack blas -doc -examples lapack -mkl superlu -test
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


# dev-build/rocm-cmake ~amd64
# dev-libs/rccl ~amd64
# dev-libs/rocm-comgr ~amd64
# dev-libs/rocm-device-libs ~amd64
# dev-libs/rocm-opencl-runtime ~amd64
# dev-util/rocm-smi ~amd64
# dev-util/rocm_bandwidth_test ~amd64
# dev-util/rocminfo ~amd64
# dev-perl/URI-Encode ~amd64
# dev-util/hip ~amd64
# dev-util/hipcc ~amd64

# # emerge -avp sci-libs/pytorch --autounmask=y --autounmask-unrestricted-atoms=y --autounmask-keep-masks=y
# sci-libs/pytorch ~amd64
# sci-libs/caffe2 ~amd64
# dev-libs/cpuinfo ~amd64
# dev-libs/pthreadpool ~amd64
# sci-libs/onnx ~amd64
# sci-libs/foxi ~amd64
# dev-libs/psimd ~amd64
# dev-libs/FP16 ~amd64
# dev-libs/FXdiv ~amd64
# dev-libs/pocketfft ~amd64
# sci-libs/kineto ~amd64
# dev-python/Opcodes
# dev-libs/dynolog
# dev-python/PeachPy

# sci-libs/hipBLAS ~amd64
# sci-libs/hipFFT ~amd64
# sci-libs/hipRAND ~amd64
# sci-libs/hipSPARSE ~amd64
# sci-libs/rocBLAS ~amd64
# sci-libs/rocFFT ~amd64
# sci-libs/rocPRIM ~amd64
# sci-libs/rocSOLVER ~amd64

# sci-libs/rocSPARSE ~amd64
# dev-util/Tensile ~amd64
# sci-libs/rocRAND ~amd64

# dev-util/lcov ~amd64
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

```
emerge -e @world
```

- why llvm 16 and 17 are installed?

```
$ equery d llvm
 * These packages depend on llvm:
dev-util/include-what-you-use-0.20 (sys-devel/llvm:16)
...
```

- looks like iwyu has to go

```
emerge --deselect dev-util/include-what-you-use
emerge --deselect mold
emerge -a --depclean
```

```
 cryptsetup luksOpen /dev/nvme0n1p4 p4
 mount /dev/mapper/p4 /mnt4
```

- create new image

```
export INDIR=/
export OUTFILE=/mnt4/gentoo_20240118.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
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

- 200MB less compressed and 1GB less uncompressed

```
Filesystem size 2054328.17 Kbytes (2006.18 Mbytes)
        32.45% of uncompressed filesystem size (6331291.08 Kbytes)

```


```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.12-gentoo-x86_64 \
  --force \
  /boot/initramfs20240118_squash_crypt-6.6.12-gentoo-x86_64.img

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240118 6.6.12 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.6.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240118_squash_crypt-6.6.12-gentoo-x86_64.img
}

```

- need new build of ryzen module
```
mkdir /mnt3
mount /dev/nvme0n1p3 /mnt3
/home/martin/src/ryzen_monitor/ryzen_smu
ln -s /mnt3/usr/src/linux-6.6.12-gentoo  /usr/src/linux-6.6.12-gentoo
rm /usr/src/linux-6.6.12-gentoo
```
- bluetooth not always working for audio. i had success starting
  bluetoothctl, executing remove <mac> and then powercycling the
  headset. after 20s or so i see messages in bluetoothctl, the headset
  connects and plays audio (eventually)

- compile vulkan
```
mesa vulkan
mpv vaapi vulkan
vulkan-tools

```


# prepare next update

- install clang
```
LLVM_TARGETS="-AArch64 AMDGPU -ARM -AVR -BPF -Hexagon -Lanai -LoongArch -MSP430 -Mips -NVPTX -PowerPC -RISCV -Sparc -SystemZ -VE -WebAssembly X86 -XCore -ARC -CSKY -DirectX -M68k -SPIRV -Xtensa"
```

- the profile turns on all targets for llvm and clang. 
https://forums.gentoo.org/viewtopic-p-8772699.html?sid=d0ffa9c8b8c2041e1d23d4c0d50ed239

- they say it is because rust might need it.

- i can't figure out how to disable the other architectures and they seem quite adamant that this might break things (if you change it retroactively). so i guess i build all archs now




# Update 2024-01-20

- modify make.conf for -j4 (just in case, j12 doesn't seem to work)

```
 sudo emerge -av clang
```

```
media-libs/mesa X gles2 llvm proprietary-codecs vaapi zstd -d3d9 -debug -gles1 -lm-sensors -opencl -osmesa -selinux -test -unwind -valgrind -vdpau vulkan vulkan-overlay -wayland -xa -zink

media-video/mpv X alsa cli libmpv openal opengl pulseaudio vaapi zlib -aqua -archive -bluray -cdda -coreaudio -debug -drm -dvb -dvd -egl -gamepad -iconv -jack -javascript -jpeg -lcms -libcaca -lua -mmal -nvenc -pipewire -raspberry-pi -rubberband -sdl -selinux -sixel -sndio -test -tools -uchardet -vdpau vulkan -wayland -xv -zimg

# sudo emerge -av mesa mpv vulkan-tools
 emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
emerge --depclean
dispatch-conf # nothing

emerge -av vulkan-tools

```



```
 cryptsetup luksOpen /dev/nvme0n1p4 p4
 mount /dev/mapper/p4 /mnt4
```

- create new image

```
export TODAY=20240120
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
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

- 110MB more compressed and 500MB more uncompressed

```
Filesystem size 2166673.71 Kbytes (2115.89 Mbytes)
        31.80% of uncompressed filesystem size (6814451.47 Kbytes)
real    1m39.108s

```


```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.12-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.12-gentoo-x86_64.img"

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240120 6.6.12 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.6.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240120_squash_crypt-6.6.12-gentoo-x86_64.img
}

```

# Try to install halide

- this looks interesting https://github.com/KomputeProject/kompute

- install halide



```
git clone https://github.com/halide/Halide/
cmake -G Ninja -DTARGET_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=/usr/lib/llvm/17/lib64/cmake/llvm -DHalide_SHARED_LLVM=YES ..

cmake --build build --config Release
ctest -C Release
```

- needs lld (maybe only for webassembly backend?)


# Update 2024-02-18

- i don't think i want to have halide and rocm in the image
- download 100 packages
```
emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world --fetchonly
```

- quite a few packages complained about missing kernel source. next
  time i shall start 6.6.12
 

- update boot code
```
emerge --depclean
revdep-rebuild
eclean-dist
eclean-pkg
grub-install --target=x86_64-efi --efi-directory=/boot/efi
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n
```

- note: when trying to reboot grub failed with this error:
```
error: symbol `grub_is_shim_lock_enabled` not found.
```
- the cause for this is that the new grubx64.efi file was installed in /boot/EFI/EFI/gentoo and grub tried to load the file from last year. i just copied the new file into /boot/EFI/gentoo/grubx64.efi. i have not but i shall  delete /boot/EFI/EFI/gentoo/grubx64.efi 
```
$ find /boot/EFI/ -type f
/boot/EFI/gentoo/grubx64.efi
/boot/EFI/EFI/gentoo/grubx64.efi
```

- looks like the update took 2 hours
```
2024-02-18 16:23:01.6323503540 +0100 b2-4.10.1-1.gpkg.tar size=552960
2024-02-18 16:23:03.8490048560 +0100 b2-4.10.1-2.gpkg.tar size=552960
2024-02-18 16:23:06.1156590850 +0100 b2-4.10.1-3.gpkg.tar size=552960
2024-02-18 16:23:08.3289802730 +0100 b2-4.10.1-4.gpkg.tar size=552960
2024-02-18 16:23:10.5523014060 +0100 b2-4.10.1-5.gpkg.tar size=552960
2024-02-18 16:23:12.7722892240 +0100 b2-4.10.1-6.gpkg.tar size=552960
2024-02-18 16:23:15.0089436170 +0100 b2-4.10.1-7.gpkg.tar size=552960
2024-02-18 16:23:24.0455606960 +0100 strace-6.6-1.gpkg.tar size=1269760
2024-02-18 16:23:26.3288814990 +0100 strace-6.6-2.gpkg.tar size=1269760
2024-02-18 16:23:28.6155356180 +0100 strace-6.6-3.gpkg.tar size=1269760
2024-02-18 16:23:33.9421730550 +0100 autoconf-2.71-r6-5.gpkg.tar size=860160
2024-02-18 16:23:36.2321604890 +0100 autoconf-2.71-r6-6.gpkg.tar size=860160
2024-02-18 16:23:38.9954786590 +0100 boost-1.82.0-r1-1.gpkg.tar size=18657280
2024-02-18 16:23:41.5854644460 +0100 boost-1.82.0-r1-2.gpkg.tar size=18575360
2024-02-18 16:23:44.1621169740 +0100 boost-1.82.0-r1-3.gpkg.tar size=18585600
2024-02-18 16:23:46.7521027610 +0100 boost-1.82.0-r1-4.gpkg.tar size=18585600
2024-02-18 16:23:49.3487551790 +0100 boost-1.82.0-r1-5.gpkg.tar size=18575360
2024-02-18 16:23:51.9854073770 +0100 boost-1.82.0-r1-6.gpkg.tar size=18585600
2024-02-18 16:23:54.8453916830 +0100 sbcl-2.3.5-1.gpkg.tar size=11888640
2024-02-18 16:23:57.3720444850 +0100 sbcl-2.3.5-2.gpkg.tar size=11888640
2024-02-18 16:23:59.9086972320 +0100 sbcl-2.3.5-3.gpkg.tar size=11888640
2024-02-18 16:24:02.4620165540 +0100 sbcl-2.3.5-4.gpkg.tar size=11888640
2024-02-18 16:53:09.4257635180 +0100 glibc-2.38-r10-1.gpkg.tar size=15831040
2024-02-18 16:53:27.1589995410 +0100 baselayout-2.14-r2-1.gpkg.tar size=61440
2024-02-18 16:53:44.4455713480 +0100 libffi-3.4.4-r3-1.gpkg.tar size=194560
2024-02-18 16:54:01.0154804220 +0100 gperf-3.1-r2-1.gpkg.tar size=184320
2024-02-18 16:54:13.2287467360 +0100 autoconf-wrapper-20231224-1.gpkg.tar size=20480
2024-02-18 16:54:25.5786789660 +0100 mime-types-2.1.54-1.gpkg.tar size=40960
2024-02-18 16:54:55.7985131360 +0100 c-ares-1.25.0-r1-1.gpkg.tar size=286720
2024-02-18 16:55:27.7316712380 +0100 linux-firmware-20240115-1.gpkg.tar size=464373760
2024-02-18 16:55:58.0148383940 +0100 pkgconf-2.1.1-1.gpkg.tar size=112640
2024-02-18 16:56:12.9747563020 +0100 cpuid2cpuflags-14-1.gpkg.tar size=20480
2024-02-18 16:56:13.7280855020 +0100 timezone-data-2023d-1.gpkg.tar size=368640
2024-02-18 16:56:24.7780248660 +0100 b2-5.0.0-1.gpkg.tar size=727040
2024-02-18 16:56:44.7812484330 +0100 elt-patches-20240116-1.gpkg.tar size=71680
2024-02-18 16:56:58.6678388970 +0100 zlib-1.3-r4-1.gpkg.tar size=194560
2024-02-18 16:57:19.8577226190 +0100 libtirpc-1.3.4-r1-1.gpkg.tar size=204800
2024-02-18 16:57:35.4376371250 +0100 libxml2-2.12.5-1.gpkg.tar size=1443840
2024-02-18 16:58:10.4474450110 +0100 automake-1.16.5-r2-1.gpkg.tar size=870400
2024-02-18 16:58:10.5574444070 +0100 clang-common-17.0.6-r1-3.gpkg.tar size=40960
2024-02-18 16:59:57.0468600520 +0100 libgcrypt-1.10.3-r1-1.gpkg.tar size=860160
2024-02-18 17:01:45.7395969400 +0100 boost-1.84.0-r3-1.gpkg.tar size=18677760
2024-02-18 17:02:22.7627271110 +0100 more-itertools-10.2.0-1.gpkg.tar size=174080
2024-02-18 17:02:38.9593048990 +0100 platformdirs-4.2.0-1.gpkg.tar size=81920
2024-02-18 17:02:55.8425455870 +0100 markupsafe-2.1.5-1.gpkg.tar size=71680
2024-02-18 17:03:49.0389203410 +0100 binutils-2.41-r5-1.gpkg.tar size=9287680
2024-02-18 17:31:17.6832068270 +0100 gcc-13.2.1_p20240113-r1-1.gpkg.tar size=88739840
2024-02-18 17:31:57.7529869460 +0100 libxcb-1.16-r1-1.gpkg.tar size=604160
2024-02-18 17:32:30.3794745770 +0100 ninja-1.11.1-r5-1.gpkg.tar size=215040
2024-02-18 17:32:50.3226984730 +0100 shared-mime-info-2.4-r1-1.gpkg.tar size=942080
2024-02-18 17:34:47.3853894310 +0100 systemd-255.3-1.gpkg.tar size=10260480
2024-02-18 17:35:08.3552743600 +0100 ca-certificates-20230311.3.96.1-1.gpkg.tar size=184320
2024-02-18 17:35:26.1351767930 +0100 popt-1.19-r1-1.gpkg.tar size=92160
2024-02-18 17:35:51.0417067870 +0100 libidn2-2.3.7-1.gpkg.tar size=245760
2024-02-18 17:36:16.3282346950 +0100 gawk-5.3.0-r1-1.gpkg.tar size=1331200
2024-02-18 17:36:29.9348266960 +0100 installkernel-24-1.gpkg.tar size=20480
2024-02-18 17:36:49.2813872000 +0100 iproute2-6.6.0-r3-1.gpkg.tar size=1300480
2024-02-18 17:51:21.8299241090 +0100 cython-3.0.8-1.gpkg.tar size=4638720
2024-02-18 17:51:43.7298039350 +0100 pax-utils-1.3.7-1.gpkg.tar size=133120
2024-02-18 17:52:01.6563722300 +0100 gast-0.5.4-1.gpkg.tar size=112640
2024-02-18 17:52:02.1230363360 +0100 trove-classifiers-2024.1.31-1.gpkg.tar size=61440
2024-02-18 17:52:07.1096756390 +0100 pixman-0.43.2-1.gpkg.tar size=583680
2024-02-18 17:52:28.6528907550 +0100 alabaster-0.7.16-1.gpkg.tar size=61440
2024-02-18 17:52:34.3861926270 +0100 pytz-2024.1-1.gpkg.tar size=102400
2024-02-18 17:52:34.3928592570 +0100 libnvme-1.7.1-r1-1.gpkg.tar size=266240
2024-02-18 17:52:41.1828219970 +0100 mako-1.3.2-1.gpkg.tar size=266240
2024-02-18 17:52:51.6994309550 +0100 pip-23.3.2-r1-1.gpkg.tar size=4536320
2024-02-18 17:53:02.6293709770 +0100 binutils-libs-2.41-r5-1.gpkg.tar size=1853440
2024-02-18 17:53:06.7893481500 +0100 rsync-3.2.7-r3-1.gpkg.tar size=460800
2024-02-18 17:53:36.2491864900 +0100 libXaw3d-1.6.5-r1-1.gpkg.tar size=256000
2024-02-18 17:53:50.7291070330 +0100 libarchive-3.7.2-r1-1.gpkg.tar size=624640
2024-02-18 17:54:11.4623265940 +0100 feh-3.10.2-1.gpkg.tar size=215040
2024-02-18 17:54:22.8689306670 +0100 nvme-cli-2.7.1-1.gpkg.tar size=849920
2024-02-18 17:54:27.5055718900 +0100 pluggy-1.4.0-1.gpkg.tar size=102400
2024-02-18 17:54:27.5622382460 +0100 dracut-060_pre20240104-r2-1.gpkg.tar size=460800
2024-02-18 17:54:27.7089041080 +0100 pillow-10.2.0-1.gpkg.tar size=1044480
2024-02-18 17:54:27.8189035040 +0100 json-glib-1.8.0-1.gpkg.tar size=245760
2024-02-18 17:55:10.7186680940 +0100 hatchling-1.21.1-1.gpkg.tar size=245760
2024-02-18 17:55:31.9985513220 +0100 fonttools-4.47.2-1.gpkg.tar size=4065280
2024-02-18 17:55:37.2518558280 +0100 transient-0.5.3-1.gpkg.tar size=174080
2024-02-18 17:55:45.9084749920 +0100 libdrm-2.4.120-1.gpkg.tar size=286720
2024-02-18 17:55:50.9684472260 +0100 cups-2.4.7-r2-1.gpkg.tar size=5939200
2024-02-18 17:56:03.5117117280 +0100 sudo-1.9.15_p5-1.gpkg.tar size=2181120
2024-02-18 17:56:14.0249873710 +0100 ofono-2.1-1.gpkg.tar size=880640
2024-02-18 17:58:08.1443611470 +0100 curl-8.5.0-1.gpkg.tar size=1587200
2024-02-18 17:58:24.2409394840 +0100 grub-2.12-r1-1.gpkg.tar size=18012160
2024-02-18 17:58:34.6108825800 +0100 numpy-1.26.3-1.gpkg.tar size=10588160
2024-02-18 18:00:41.9635170720 +0100 cmake-3.28.3-1.gpkg.tar size=18227200
2024-02-18 18:01:15.4000002580 +0100 gnupg-2.2.42-r2-1.gpkg.tar size=3440640
2024-02-18 18:01:28.5399281530 +0100 vulkan-headers-1.3.275-1.gpkg.tar size=2078720
2024-02-18 18:01:33.9732316720 +0100 spirv-headers-1.3.275-1.gpkg.tar size=245760
2024-02-18 18:01:54.7264511230 +0100 gpgme-1.23.2-1.gpkg.tar size=849920
2024-02-18 18:03:08.2760475230 +0100 vulkan-loader-1.3.275-1.gpkg.tar size=194560
2024-02-18 18:03:43.1458561770 +0100 genkernel-4.3.10-1.gpkg.tar size=188375040
2024-02-18 18:04:11.5357003900 +0100 ccache-4.9.1-1.gpkg.tar size=1413120
2024-02-18 18:05:24.0119693470 +0100 libjxl-0.8.2-r1-1.gpkg.tar size=6133760
2024-02-18 18:06:58.8481156050 +0100 spirv-tools-1.3.275-1.gpkg.tar size=5376000
2024-02-18 18:10:47.4135280310 +0100 scipy-1.12.0-1.gpkg.tar size=31426560
2024-02-18 18:11:19.7900170330 +0100 volk-1.3.275-1.gpkg.tar size=102400
2024-02-18 18:11:25.9366499710 +0100 sphinxcontrib-applehelp-1.0.8-1.gpkg.tar size=71680
2024-02-18 18:11:37.0699222110 +0100 glslang-1.3.275-1.gpkg.tar size=3358720
2024-02-18 18:12:30.0896312680 +0100 vulkan-tools-1.3.275-1.gpkg.tar size=317440
2024-02-18 18:12:44.2062204710 +0100 sphinxcontrib-devhelp-1.0.6-1.gpkg.tar size=61440
2024-02-18 18:14:15.7057183720 +0100 ffmpeg-6.0.1-r2-1.gpkg.tar size=10096640
2024-02-18 18:14:39.3689218550 +0100 qtcore-5.15.12-r2-1.gpkg.tar size=8079360
2024-02-18 18:14:48.5622047410 +0100 imagemagick-7.1.1.25-1.gpkg.tar size=9379840
2024-02-18 18:16:03.4884602530 +0100 sphinxcontrib-htmlhelp-2.0.5-1.gpkg.tar size=81920
2024-02-18 18:16:10.3484226100 +0100 shaderc-2023.8-1.gpkg.tar size=409600
2024-02-18 18:17:20.0947065470 +0100 mesa-23.3.5-1.gpkg.tar size=13527040
2024-02-18 18:17:38.9312698490 +0100 sphinxcontrib-serializinghtml-1.1.10-1.gpkg.tar size=71680
2024-02-18 18:18:00.9044826060 +0100 libplacebo-6.338.2-1.gpkg.tar size=624640
2024-02-18 18:18:20.3477092460 +0100 xorg-server-21.1.11-1.gpkg.tar size=3543040
2024-02-18 18:18:33.5043037160 +0100 mupdf-1.23.3-r1-1.gpkg.tar size=34007040
2024-02-18 18:18:44.1275787550 +0100 firefox-bin-122.0.1-1.gpkg.tar size=91514880
2024-02-18 18:18:59.3408286060 +0100 libsdl2-2.28.5-1.gpkg.tar size=1157120
2024-02-18 18:19:07.7407825120 +0100 xf86-input-libinput-1.4.0-5.gpkg.tar size=81920
2024-02-18 18:19:11.3840958530 +0100 qtdbus-5.15.12-1.gpkg.tar size=450560
2024-02-18 18:19:29.1006653010 +0100 sphinxcontrib-qthelp-1.0.7-1.gpkg.tar size=71680
2024-02-18 18:19:34.4806357780 +0100 xf86-video-amdgpu-23.0.0-7.gpkg.tar size=174080
2024-02-18 18:19:55.3938543520 +0100 xf86-video-ati-22.0.0-7.gpkg.tar size=471040
2024-02-18 18:22:02.8998213360 +0100 qtgui-5.15.12-r2-1.gpkg.tar size=4853760
2024-02-18 18:24:03.9991568100 +0100 qtwidgets-5.15.12-r1-1.gpkg.tar size=3471360

```


- create new image

```
export TODAY=20240218
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
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

- 41MB more compressed

```
# old:
# Filesystem size 2166673.71 Kbytes (2115.89 Mbytes)

# new:
Filesystem size 2170782.41 Kbytes (2119.90 Mbytes)
        32.03% of uncompressed filesystem size (6777058.19 Kbytes)
real    1m14.266s
user    11m28.658s

```


```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.12-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.12-gentoo-x86_64.img"

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240218 6.6.12 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.12-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.6.12-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240218_squash_crypt-6.6.12-gentoo-x86_64.img
}

```

# Update 2024-02-19

- boot 6.6.12 with 6.3.12 initramfs (only way to get this kernel with unencrypted fs for now)

```
eix-sync
emerge -e @world
```

```
real    408m56.099s
user    1503m28.130s
sys     186m33.941s
```

- no make file in kernel directory. it is actually quite empty.
- perhaps i should switch to 6.6.17. it is listed as long-term on kernel.org

```
eselect kernel set 1
cd /usr/src/linux
make oldconfig

make menuconfig
 *   CONFIG_IP_NF_NAT:   is not set when it should be.
 *   CONFIG_IP_NF_TARGET_MASQUERADE:     is not set when it should be.

make -j12
make install modules_install

```

```
real    11m25.799s
user    89m32.173s
sys     8m39.992s

```

```
menuentry 'Gentoo GNU/Linux 6.6.17' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.17-gentoo-x86_64 ...'
	linux	/kernel-6.6.17-gentoo-x86_64 root=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 ro  
	echo	'Loading initial ramdisk (for 6.3.12) ...'
	initrd	/initramfs-6.3.12-gentoo-x86_64.img
}

```
- compile ryzen monitor modul


- build @world again with the new kernel and its sources:

```
real    292m10.822s
user    1552m58.187s
sys     197m19.246s
```

- update lisp stuff

```

sbcl
(ql:update-client) # no change
(ql:update-dist "quicklisp") # no change
(map nil 'ql-dist:clean (ql-dist:all-dists))
# delete old swank
rm -rf ~/.cache/common-lisp/

# build swank
emacs
M-x slime

# try to update packages, not needed


```


- create new image

```
export TODAY=20240219
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
lib/modules/6.6.12-gentoo-x86_64 \
usr/lib/modules/6.3.12-gentoo-x86_64 \
usr/lib/modules/6.6.12-gentoo-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
opt/rust-bin* \
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
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia} \
persistent \
initramfs-with-squashfs.img

```

- a lot bigger

```
# old:
# Filesystem size 2166673.71 Kbytes (2115.89 Mbytes)

# new:
Filesystem size 2363738.04 Kbytes (2308.34 Mbytes)
        33.90% of uncompressed filesystem size (6972349.94 Kbytes)

```

- i don't think i should put the stage repos in, in particular because there is a 100 MB gentoo build log in there
- exclude /opt/rust-bin*, /usr/lib/modules/..., nvidia firmware
- this really shaved off a lot from the image:

```
Filesystem size 1768582.55 Kbytes (1727.13 Mbytes)
        30.33% of uncompressed filesystem size (5831387.82 Kbytes)
real    0m59.446s
user    10m20.119s
sys     0m9.701s

```
```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.17-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.17-gentoo-x86_64.img"

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240219 6.6.17 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.17-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/kernel-6.6.17-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240219_squash_crypt-6.6.17-gentoo-x86_64.img
}

```



# Proposal

- i might want usbmon and whatever is needed to debug usb with wireshark

https://wiki.wireshark.org/CaptureSetup/USB

## Update 2024-02-24

- add usbmon to kernel
- add obs-studio (requires qt6), wireshark (use also qt6), mupdf without opengl and javascript
- update

```
* Messages for package media-sound/pulseaudio-daemon-16.99.1:

 * You have enabled bluetooth USE flag for pulseaudio. Daemon will now handle
 * bluetooth Headset (HSP HS and HSP AG) and Handsfree (HFP HF) profiles using
 * native headset backend by default. This can be selectively disabled
 * via runtime configuration arguments to module-bluetooth-discover
 * in /etc/pulse/default.pa
 * To disable HFP HF append enable_native_hfp_hf=false
 * To disable HSP HS append enable_native_hsp_hs=false
 * To disable HSP AG append headset=auto or headset=ofono
 * (note this does NOT require enabling USE ofono)
 * 
 * You have enabled both native and ofono headset profiles. The runtime decision
 * which to use is done via the 'headset' argument of module-bluetooth-discover.
 * 
 * Pulseaudio autospawn by client library is no longer enabled when systemd is available.
 * It's recommended to start pulseaudio via its systemd user units:
 * 
 *   systemctl --user enable pulseaudio.service pulseaudio.socket
 * 
 * Root user can change system default configuration for all users:
 * 
 *   systemctl --global enable pulseaudio.service pulseaudio.socket
 * 
 * If you would like to enable autospawn by client library, edit autospawn flag in /etc/pulse/client.conf like this:
 * 
 *   autospawn = yes
 * 
 * The change from autospawn to user units will take effect after restarting.
 * 
 * PulseAudio can be enhanced by installing the following:
 *   sys-auth/rtkit for restricted realtime capabilities via D-Bus


```

- qml (qtdeclarative) installs 400MB. perhaps i can get rid of it?
  yes, the depclean removed it again

```
emerge --depclean
eclean-dist
eclean-pkg
revdep-rebuild 
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 

```

```
2024-02-24 10:54:12.1546436930 +0100 ensurepip-pip-24.0-1.gpkg.tar size=2088960
2024-02-24 11:00:51.3824529520 +0100 python-3.11.8_p1-1.gpkg.tar size=28313600
2024-02-24 11:01:04.8223792010 +0100 linux-firmware-20240220-r1-1.gpkg.tar size=471306240
2024-02-24 11:01:45.4521562480 +0100 packaging-23.2-r1-1.gpkg.tar size=174080
2024-02-24 11:02:10.6320180740 +0100 meson-1.3.1-r1-1.gpkg.tar size=2723840
2024-02-24 11:04:20.3779727670 +0100 systemd-255.3-r1-1.gpkg.tar size=10260480
2024-02-24 11:06:07.3773856140 +0100 python-3.12.2_p1-1.gpkg.tar size=29163520
2024-02-24 11:06:59.7670981280 +0100 webrtc-audio-processing-1.3-r3-1.gpkg.tar size=1228800
2024-02-24 11:07:04.1004076820 +0100 gumbo-0.12.1-1.gpkg.tar size=194560
2024-02-24 11:07:04.1004076820 +0100 libpsl-0.21.5-1.gpkg.tar size=112640
2024-02-24 11:07:11.9736978110 +0100 pip-24.0-1.gpkg.tar size=4546560
2024-02-24 11:07:47.4401698580 +0100 libuv-1.48.0-1.gpkg.tar size=204800
2024-02-24 11:08:15.2566838830 +0100 soapyplutosdr-0.2.2-1.gpkg.tar size=81920
2024-02-24 11:08:27.3332842800 +0100 pythran-0.15.0-r1-1.gpkg.tar size=1218560
2024-02-24 11:08:32.2032575560 +0100 libpulse-16.99.1-1.gpkg.tar size=1024000
2024-02-24 11:08:41.2765411000 +0100 mupdf-1.23.3-r1-4.gpkg.tar size=34007040
2024-02-24 11:09:14.8130237370 +0100 pulseaudio-daemon-16.99.1-1.gpkg.tar size=1177600
2024-02-24 11:09:47.2461790950 +0100 firefox-bin-123.0-1.gpkg.tar size=92487680
2024-02-24 11:17:50.6868595740 +0100 pcap-0-r1-1.gpkg.tar size=20480
2024-02-24 11:18:10.3434183760 +0100 boehm-gc-8.2.4-1.gpkg.tar size=399360
2024-02-24 11:18:31.7633008350 +0100 zlib-1.3-r4-4.gpkg.tar size=225280
2024-02-24 11:18:49.2898713260 +0100 xprop-1.2.6-1.gpkg.tar size=71680
2024-02-24 11:19:00.5531428530 +0100 perl-File-Path-2.180.0-r2-1.gpkg.tar size=20480
2024-02-24 11:19:30.5396449700 +0100 w3m-0.5.3_p20230121-1.gpkg.tar size=1290240
2024-02-24 11:20:39.0959354380 +0100 qtnetwork-5.15.12-r1-1.gpkg.tar size=870400
2024-02-24 11:21:18.7857176420 +0100 qtconcurrent-5.15.12-1.gpkg.tar size=71680
2024-02-24 11:21:59.3154952370 +0100 qtxml-5.15.12-1.gpkg.tar size=174080
2024-02-24 11:22:43.2419208600 +0100 qttest-5.15.12-1.gpkg.tar size=256000
2024-02-24 11:22:56.5918476030 +0100 IPC-System-Simple-1.300.0-1.gpkg.tar size=51200
2024-02-24 11:23:08.1084510730 +0100 w3m-1-1.gpkg.tar size=20480
2024-02-24 11:23:20.5850492750 +0100 File-BaseDir-0.90.0-1.gpkg.tar size=40960
2024-02-24 11:23:43.3815908470 +0100 linguist-tools-5.15.12-1.gpkg.tar size=491520
2024-02-24 11:23:56.1448541430 +0100 File-DesktopEntry-0.220.0-r1-1.gpkg.tar size=40960
2024-02-24 11:24:09.0747831900 +0100 File-MimeInfo-0.330.0-1.gpkg.tar size=61440
2024-02-24 11:26:48.8272398900 +0100 qtgui-5.15.12-r2-4.gpkg.tar size=5068800
2024-02-24 11:27:35.0603195220 +0100 qtmultimedia-5.15.12-1.gpkg.tar size=450560
2024-02-24 11:28:20.6434027210 +0100 qtprintsupport-5.15.12-1.gpkg.tar size=225280
2024-02-24 11:33:16.9817765800 +0100 qtdeclarative-5.15.12-1.gpkg.tar size=8192000
2024-02-24 11:33:38.9616559660 +0100 libpcap-1.10.4-1.gpkg.tar size=348160
2024-02-24 11:33:57.8482189940 +0100 xmlto-0.0.28-r11-8.gpkg.tar size=71680
2024-02-24 11:34:16.0814522740 +0100 xdg-utils-1.1.3_p20210805-r1-1.gpkg.tar size=102400
2024-02-24 11:37:18.5504509850 +0100 wireshark-4.0.11-1.gpkg.tar size=33464320
2024-02-24 12:56:38.7809961400 +0100 jansson-2.14-r1-1.gpkg.tar size=71680
2024-02-24 12:57:19.8541040870 +0100 x264-0.0.20231114-r1-1.gpkg.tar size=860160
2024-02-24 12:57:43.6773066920 +0100 rnnoise-0.4.1_p20210122-r1-1.gpkg.tar size=133120
2024-02-24 12:58:19.1637786280 +0100 libv4l-1.22.1-2.gpkg.tar size=215040
2024-02-24 12:58:37.5236778790 +0100 xcb-util-cursor-0.1.5-1.gpkg.tar size=61440
2024-02-24 12:59:16.3734646930 +0100 mupdf-1.23.3-r1-5.gpkg.tar size=33853440
2024-02-24 12:59:34.2466999480 +0100 mbedtls-2.28.7-1.gpkg.tar size=839680
2024-02-24 13:07:30.9474174120 +0100 qtbase-6.6.2-1.gpkg.tar size=18298880
2024-02-24 13:08:09.6472050490 +0100 qtshadertools-6.6.2-1.gpkg.tar size=2140160
2024-02-24 13:08:30.5837568270 +0100 qt5compat-6.6.2-1.gpkg.tar size=655360
2024-02-24 13:22:26.9991670420 +0100 qtdeclarative-6.6.2-1.gpkg.tar size=33423360
2024-02-24 13:22:55.3690113640 +0100 qtsvg-6.6.2-1.gpkg.tar size=296960
2024-02-24 13:26:30.0611665860 +0100 mupdf-1.23.3-r1-6.gpkg.tar size=33853440
2024-02-24 13:27:56.0040283120 +0100 qttools-6.6.2-1.gpkg.tar size=2887680
2024-02-24 13:29:32.6534979530 +0100 ffmpeg-6.0.1-r2-4.gpkg.tar size=10127360
2024-02-24 13:30:46.2964271750 +0100 qtmultimedia-6.6.2-1.gpkg.tar size=3072000
2024-02-24 13:32:07.5559812680 +0100 obs-studio-30.0.2-1.gpkg.tar size=6543360
2024-02-24 13:36:11.3446434900 +0100 wireshark-4.0.11-2.gpkg.tar size=34027520
2024-02-24 13:40:21.8399355770 +0100 cmake-3.27.9-1.gpkg.tar size=17971200
2024-02-24 13:40:42.5064888380 +0100 qttranslations-6.6.2-1.gpkg.tar size=2529280

```


- create new image

```
export TODAY=20240224
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
lib/modules/6.6.12-gentoo-x86_64 \
usr/lib/modules/6.3.12-gentoo-x86_64 \
usr/lib/modules/6.6.12-gentoo-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
opt/rust-bin* \
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
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia} \
persistent \
initramfs-with-squashfs.img

```

- a bit bigger (80MB compressed):

```
# old:
Filesystem size 1768582.55 Kbytes (1727.13 Mbytes)
        30.33% of uncompressed filesystem size (5831387.82 Kbytes)
real    0m59.446s
# new:
Filesystem size 1849509.25 Kbytes (1806.16 Mbytes)
        30.03% of uncompressed filesystem size (6158277.84 Kbytes)
real    1m7.399s	
```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.17-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.17-gentoo-x86_64.img"

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240224 6.6.17 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.17-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/kernel-6.6.17-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240224_squash_crypt-6.6.17-gentoo-x86_64.img
}

```

## note:

- i forgot to enable usb in pcap
```
net-libs/libpcap-1.10.4::gentoo  USE="-bluetooth -dbus -netlink -rdma -remote -static-libs -test -usb -verify-sig -yydebug"
```


# proposal:


- add this boot parameter amd_pstate=active
  https://news.ycombinator.com/item?id=39582116

- https://www.reddit.com/r/linux_gaming/comments/13by61l/amd_pstateactive/


```
>> The amd_pstate_epp sets the governor to powersave by default (there is a performance one too but for whatever reason my kernel parameter to set the default governor to performance no longer works),
Best to leave it on powersave and then set the EPP profile (energy_performance_preference in sysfs) to performanc.
=======
- add this boot parameter amd_pstate=active https://news.ycombinator.com/item?id=39582116

# update 2024-03-26

- add usb use to pcap

```
eix-sync
dispatch-config

emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
emerge --depclean
revdep-rebuild
eclean-dist
eclean-pkg
```

```
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 

```

```
2024-03-26 08:09:20.8261874680 +0100 scikit-learn-1.3.2-r1-1.gpkg.tar size=13352960
2024-03-26 08:09:23.8195043760 +0100 scikit-learn-1.3.2-r1-2.gpkg.tar size=13352960
2024-03-26 08:09:26.8094879690 +0100 scikit-learn-1.3.2-r1-3.gpkg.tar size=13352960
2024-03-26 08:09:29.7994715610 +0100 scikit-learn-1.3.2-r1-4.gpkg.tar size=13352960
2024-03-26 08:12:08.5286005440 +0100 root-0-r2-1.gpkg.tar size=20480
2024-03-26 08:12:08.7919324320 +0100 man-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:12.5285785940 +0100 adm-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:18.5585455050 +0100 wheel-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:23.0251876610 +0100 kmem-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:23.6351843140 +0100 tty-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:30.7918117090 +0100 utmp-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:35.5584522190 +0100 audio-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:36.0817826800 +0100 dialout-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:36.1417823510 +0100 cdrom-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:45.9317286290 +0100 disk-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:45.9450618900 +0100 input-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:54.1383502630 +0100 kvm-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:59.3516549880 +0100 lp-0-r3-1.gpkg.tar size=20480
2024-03-26 08:12:59.8283190390 +0100 render-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:06.4182828770 +0100 sgx-0-r2-1.gpkg.tar size=20480
2024-03-26 08:13:07.0882792010 +0100 users-0-r2-1.gpkg.tar size=20480
2024-03-26 08:13:07.4382772800 +0100 tape-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:17.3915559950 +0100 video-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:22.8648592940 +0100 systemd-journal-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:22.9148590200 +0100 usb-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:29.8914874030 +0100 messagebus-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:29.9248205530 +0100 systemd-timesync-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:37.2547803300 +0100 systemd-resolve-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:38.1347755010 +0100 systemd-oom-0-r2-1.gpkg.tar size=20480
2024-03-26 08:13:46.2580642580 +0100 systemd-network-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:46.9147273220 +0100 systemd-coredump-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:53.4780246390 +0100 systemd-journal-remote-0-r3-1.gpkg.tar size=20480
2024-03-26 08:13:53.6446903910 +0100 nobody-0-r2-1.gpkg.tar size=20480
2024-03-26 08:14:00.6113188290 +0100 avahi-0-r3-1.gpkg.tar size=20480
2024-03-26 08:14:06.2479545650 +0100 libffi-3.4.4-r4-1.gpkg.tar size=194560
2024-03-26 08:14:06.9512840390 +0100 netdev-0-r3-1.gpkg.tar size=20480
2024-03-26 08:14:10.6445971050 +0100 dhcp-0-r3-1.gpkg.tar size=20480
2024-03-26 08:14:20.8312078740 +0100 sshd-0-r3-1.gpkg.tar size=20480
2024-03-26 08:14:20.8478744490 +0100 mail-0-r3-1.gpkg.tar size=20480
2024-03-26 08:14:32.7044760530 +0100 nullmail-0-r2-1.gpkg.tar size=20480
2024-03-26 08:14:37.8477811630 +0100 portage-0-r2-1.gpkg.tar size=20480
2024-03-26 08:14:45.2344072960 +0100 ensurepip-setuptools-69.1.1-1.gpkg.tar size=747520
2024-03-26 08:14:49.8310487380 +0100 libaio-0.3.113-r1-1.gpkg.tar size=51200
2024-03-26 08:15:01.4943180700 +0100 scdoc-1.11.3-1.gpkg.tar size=40960
2024-03-26 08:15:02.2776471050 +0100 pkgconf-2.1.1-4.gpkg.tar size=112640
2024-03-26 08:15:09.0709431610 +0100 c-ares-1.26.0-1.gpkg.tar size=286720
2024-03-26 08:15:09.8609388250 +0100 ell-0.62-1.gpkg.tar size=307200
2024-03-26 08:15:20.0608828540 +0100 lpadmin-0-r3-1.gpkg.tar size=20480
2024-03-26 08:15:29.5041643680 +0100 avahi-0-r3-1.gpkg.tar size=20480
2024-03-26 08:15:30.4274926340 +0100 timezone-data-2024a-r1-1.gpkg.tar size=368640
2024-03-26 08:15:41.9107629540 +0100 dhcp-0-r3-1.gpkg.tar size=20480
2024-03-26 08:15:42.5374261820 +0100 nullmail-0-r2-1.gpkg.tar size=20480
2024-03-26 08:15:48.1240621920 +0100 linux-firmware-20240312-1.gpkg.tar size=473415680
2024-03-26 08:15:59.9206641250 +0100 bzip2-1.0.8-r5-1.gpkg.tar size=317440
2024-03-26 08:16:07.3272901490 +0100 pkgconfig-3-1.gpkg.tar size=20480
2024-03-26 08:16:43.6304242700 +0100 readline-8.1_p2-r2-1.gpkg.tar size=1126400
2024-03-26 08:20:06.9759750890 +0100 root-0-r3-1.gpkg.tar size=20480
2024-03-26 08:20:07.9359698210 +0100 systemd-journal-remote-0-r3-1.gpkg.tar size=20480
2024-03-26 08:20:14.0426029780 +0100 systemd-coredump-0-r3-1.gpkg.tar size=20480
2024-03-26 08:20:14.6659328910 +0100 systemd-network-0-r3-1.gpkg.tar size=20480
2024-03-26 08:20:21.4458956860 +0100 systemd-oom-0-r2-1.gpkg.tar size=20480
2024-03-26 08:20:22.3625573230 +0100 systemd-resolve-0-r3-1.gpkg.tar size=20480
2024-03-26 08:20:30.4291797240 +0100 systemd-timesync-0-r3-1.gpkg.tar size=20480
2024-03-26 08:20:30.9025104600 +0100 messagebus-0-r3-1.gpkg.tar size=20480
2024-03-26 08:20:37.2624755600 +0100 man-1-r3-1.gpkg.tar size=20480
2024-03-26 08:20:37.2691421900 +0100 portage-0-r3-1.gpkg.tar size=20480
2024-03-26 08:20:43.7824397820 +0100 sshd-0-r3-1.gpkg.tar size=20480
2024-03-26 08:21:48.6087507180 +0100 nobody-0-r2-1.gpkg.tar size=20480
2024-03-26 08:22:01.4786800950 +0100 getuto-1.10.2-1.gpkg.tar size=20480
2024-03-26 08:22:46.0651020960 +0100 rust-bin-1.75.0-1.gpkg.tar size=159754240
2024-03-26 08:24:32.1711865110 +0100 perl-5.38.2-r2-1.gpkg.tar size=15861760
2024-03-26 08:24:50.8910837870 +0100 rust-1.75.0-1.gpkg.tar size=30720
2024-03-26 08:25:08.6443197000 +0100 ExtUtils-CChecker-0.110.0-1.gpkg.tar size=40960
2024-03-26 08:25:30.1908681310 +0100 qdirstat-1.9-1.gpkg.tar size=829440
2024-03-26 08:25:35.2475070500 +0100 XS-Parse-Keyword-0.380.0-1.gpkg.tar size=81920
2024-03-26 08:25:49.9640929600 +0100 Syntax-Keyword-Try-0.280.0-1.gpkg.tar size=61440
2024-03-26 08:28:01.8900356900 +0100 openssl-3.0.13-1.gpkg.tar size=6758400
2024-03-26 08:28:27.0865640920 +0100 luit-20240102-1.gpkg.tar size=61440
2024-03-26 08:28:38.0231707450 +0100 man-pages-6.06-1.gpkg.tar size=3256320
2024-03-26 08:28:42.9264771720 +0100 ca-certificates-20230311.3.97-1.gpkg.tar size=184320
2024-03-26 08:29:12.3296491570 +0100 dhcp-4.4.3_p1-r5-1.gpkg.tar size=1751040
2024-03-26 08:29:41.5994885400 +0100 typing-extensions-4.10.0-1.gpkg.tar size=122880
2024-03-26 08:29:58.7927275260 +0100 urllib3-2.2.1-1.gpkg.tar size=327680
2024-03-26 08:30:14.9526388500 +0100 trove-classifiers-2024.3.3-1.gpkg.tar size=61440
2024-03-26 08:30:31.8292129070 +0100 pyparsing-3.1.2-1.gpkg.tar size=378880
2024-03-26 08:30:39.3225051210 +0100 setuptools-69.1.1-1.gpkg.tar size=1269760
2024-03-26 08:31:04.9290312740 +0100 meson-1.3.2-1.gpkg.tar size=2734080
2024-03-26 08:32:08.1786841950 +0100 cython-3.0.9-1.gpkg.tar size=4648960
2024-03-26 08:32:28.7952377290 +0100 python-dateutil-2.9.0_p0-1.gpkg.tar size=409600
2024-03-26 08:32:33.5718781840 +0100 tqdm-4.66.2-1.gpkg.tar size=225280
2024-03-26 08:32:33.9852092490 +0100 pooch-1.8.1-1.gpkg.tar size=194560
2024-03-26 08:32:34.4885398210 +0100 asteval-0.9.32-1.gpkg.tar size=102400
2024-03-26 08:32:44.8318163960 +0100 psutil-5.9.8-1.gpkg.tar size=737280
2024-03-26 08:32:49.0884597040 +0100 pycairo-1.26.0-1.gpkg.tar size=174080
2024-03-26 08:33:13.3083267990 +0100 threadpoolctl-3.3.0-1.gpkg.tar size=92160
2024-03-26 08:33:46.1448132770 +0100 fonttools-4.49.0-1.gpkg.tar size=4075520
2024-03-26 08:34:09.8480165410 +0100 lxml-5.1.0-1.gpkg.tar size=1730560
2024-03-26 08:34:20.0446272540 +0100 numpy-1.26.4-1.gpkg.tar size=10045440
2024-03-26 08:35:28.0075876440 +0100 po4a-0.69-1.gpkg.tar size=1024000
2024-03-26 08:36:17.6106487830 +0100 pam-1.5.3-r1-1.gpkg.tar size=542720
2024-03-26 08:36:35.2972183960 +0100 libcap-2.69-r1-1.gpkg.tar size=133120
2024-03-26 08:37:02.3904030570 +0100 xz-utils-5.6.1-1.gpkg.tar size=563200
2024-03-26 08:37:23.2136221240 +0100 zstd-1.5.5-r1-1.gpkg.tar size=604160
2024-03-26 08:37:37.8835416230 +0100 elt-patches-20240213-1.gpkg.tar size=71680
2024-03-26 08:37:56.3601069010 +0100 attr-2.5.2-r1-1.gpkg.tar size=102400
2024-03-26 08:38:19.0533157060 +0100 libpcre2-10.42-r2-1.gpkg.tar size=1218560
2024-03-26 08:39:02.2297454450 +0100 util-linux-2.39.3-r2-1.gpkg.tar size=5099520
2024-03-26 08:39:21.8096380010 +0100 acl-2.3.2-r1-1.gpkg.tar size=184320
2024-03-26 08:39:52.4961362770 +0100 e2fsprogs-1.47.0-r3-1.gpkg.tar size=1525760
2024-03-26 08:40:28.8426034950 +0100 libarchive-3.7.2-r2-1.gpkg.tar size=624640
2024-03-26 08:40:56.0657874420 +0100 dbus-1.15.8-1.gpkg.tar size=665600
2024-03-26 09:06:33.6340167860 +0100 gcc-13.2.1_p20240210-1.gpkg.tar size=88780800
2024-03-26 09:07:55.8235657750 +0100 coreutils-9.4-r1-1.gpkg.tar size=4167680
2024-03-26 09:08:12.0301435090 +0100 pambase-20240128-1.gpkg.tar size=40960
2024-03-26 09:08:33.7066912270 +0100 make-4.4.1-r1-9.gpkg.tar size=634880
2024-03-26 09:08:53.3465834540 +0100 iputils-20240117-1.gpkg.tar size=143360
2024-03-26 09:09:13.1998078430 +0100 libtool-2.4.7-r3-1.gpkg.tar size=757760
2024-03-26 09:09:38.0963378920 +0100 libpng-1.6.43-1.gpkg.tar size=409600
2024-03-26 09:09:59.5595534470 +0100 xkeyboard-config-2.41-1.gpkg.tar size=1269760
2024-03-26 09:09:59.8062187600 +0100 libinput-1.25.0-1.gpkg.tar size=389120
2024-03-26 09:10:07.6495090540 +0100 libtirpc-1.3.4-r2-1.gpkg.tar size=204800
2024-03-26 09:10:36.6626831790 +0100 libksba-1.6.6-1.gpkg.tar size=194560
2024-03-26 09:10:56.2159092150 +0100 ethtool-6.7-1.gpkg.tar size=266240
2024-03-26 09:11:04.0358663030 +0100 libnvme-1.8-1.gpkg.tar size=266240
2024-03-26 09:11:09.8025013260 +0100 libXext-1.3.6-1.gpkg.tar size=102400
2024-03-26 09:11:26.7090752180 +0100 btrfs-progs-6.7.1-1.gpkg.tar size=1228800
2024-03-26 09:12:52.8452692170 +0100 grub-2.12-r2-1.gpkg.tar size=18012160
2024-03-26 09:13:30.3083969740 +0100 file-5.45-r4-1.gpkg.tar size=1044480
2024-03-26 09:14:01.5148923970 +0100 iptables-1.8.10-1.gpkg.tar size=358400
2024-03-26 09:14:34.5847109280 +0100 curl-8.6.0-r1-1.gpkg.tar size=1392640
2024-03-26 09:15:17.9711395140 +0100 sqlite-3.45.1-r1-1.gpkg.tar size=1781760
2024-03-26 09:15:39.0510238390 +0100 libpciaccess-0.18-1.gpkg.tar size=71680
2024-03-26 09:15:43.6143321310 +0100 xprop-1.2.7-1.gpkg.tar size=71680
2024-03-26 09:15:51.8742868050 +0100 libxkbfile-1.1.3-1.gpkg.tar size=122880
2024-03-26 09:16:04.5508839100 +0100 bluez-5.72-1.gpkg.tar size=1648640
2024-03-26 09:16:09.5308565830 +0100 libpcre-8.45-r3-1.gpkg.tar size=901120
2024-03-26 09:16:30.8040731800 +0100 iwd-2.14-1.gpkg.tar size=552960
2024-03-26 09:16:48.7039749550 +0100 eix-0.36.7-r1-1.gpkg.tar size=911360
2024-03-26 09:16:49.6439697970 +0100 nvme-cli-2.8-1.gpkg.tar size=860160
2024-03-26 09:16:57.9372576220 +0100 strace-6.7-1.gpkg.tar size=1269760
2024-03-26 09:17:03.3572278800 +0100 dracut-060_pre20240104-r4-1.gpkg.tar size=460800
2024-03-26 09:17:11.7738483610 +0100 genkernel-4.3.10-4.gpkg.tar size=188375040
2024-03-26 09:18:42.6466830340 +0100 openssh-9.6_p1-r3-1.gpkg.tar size=1536000
2024-03-26 09:19:22.7864627690 +0100 cmake-3.28.3-4.gpkg.tar size=18227200
2024-03-26 09:20:22.7428004290 +0100 gnupg-2.4.4-r1-1.gpkg.tar size=3952640
2024-03-26 09:20:48.3126601160 +0100 xkbcomp-1.4.7-1.gpkg.tar size=153600
2024-03-26 09:21:06.5558933400 +0100 xdg-utils-1.1.3_p20210805-r2-1.gpkg.tar size=102400
2024-03-26 09:21:20.3658175590 +0100 pillow-10.2.0-r1-1.gpkg.tar size=1044480
2024-03-26 09:22:02.9155840690 +0100 mupdf-1.23.7-r1-1.gpkg.tar size=33853440
2024-03-26 09:22:52.8253101920 +0100 slang-2.3.3-r1-1.gpkg.tar size=983040
2024-03-26 09:23:27.0117892630 +0100 lapack-3.12.0-r1-1.gpkg.tar size=2437120
2024-03-26 09:23:51.4949882460 +0100 emacs-29.3-1.gpkg.tar size=48066560
2024-03-26 09:24:21.8148218670 +0100 mesa-23.3.6-1.gpkg.tar size=13527040
2024-03-26 09:31:21.5158521130 +0100 qtbase-6.6.2-r1-1.gpkg.tar size=18298880
2024-03-26 09:31:49.2257000570 +0100 gklib-5.1.1_p20230327-r1-1.gpkg.tar size=153600
2024-03-26 09:32:36.0254432460 +0100 smartmontools-7.4-r1-1.gpkg.tar size=778240
2024-03-26 09:33:29.7284818870 +0100 libpulse-17.0-1.gpkg.tar size=1024000
2024-03-26 09:33:41.0017533590 +0100 git-2.43.2-1.gpkg.tar size=15185920
2024-03-26 09:33:43.3417405180 +0100 libjxl-0.10.2-1.gpkg.tar size=2816000
2024-03-26 09:34:24.5548476970 +0100 qttools-6.6.2-2.gpkg.tar size=2979840
2024-03-26 09:35:17.1512257440 +0100 metis-5.2.1-r2-1.gpkg.tar size=522240
2024-03-26 09:35:49.3110492680 +0100 matplotlib-3.8.3-1.gpkg.tar size=33935360
2024-03-26 09:36:08.9109417150 +0100 pulseaudio-daemon-17.0-r1-1.gpkg.tar size=1187840
2024-03-26 09:37:28.5971711080 +0100 ffmpeg-6.0.1-r4-1.gpkg.tar size=10127360
2024-03-26 09:37:36.2837955940 +0100 imagemagick-7.1.1.25-4.gpkg.tar size=9379840
2024-03-26 09:39:03.1133191220 +0100 firefox-bin-124.0.1-1.gpkg.tar size=93440000
2024-03-26 09:39:09.0232866910 +0100 obs-studio-30.1.0-1.gpkg.tar size=6758400
2024-03-26 09:39:40.3197816200 +0100 slime-2.29.1-1.gpkg.tar size=921600
2024-03-26 12:00:02.9968959620 +0100 libpcap-1.10.4-2.gpkg.tar size=358400

```



- create new image

```
export TODAY=20240326
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
lib/modules/6.6.12-gentoo-x86_64 \
usr/lib/modules/6.3.12-gentoo-x86_64 \
usr/lib/modules/6.6.12-gentoo-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
opt/rust-bin* \
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
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia} \
persistent \
initramfs-with-squashfs.img

```

- a bit bigger (5MB compressed):

```
# old:
Filesystem size 1849509.25 Kbytes (1806.16 Mbytes)
        30.03% of uncompressed filesystem size (6158277.84 Kbytes)
real    1m7.399s	
# new:
Filesystem size 1855391.59 Kbytes (1811.91 Mbytes)
        30.00% of uncompressed filesystem size (6185094.38 Kbytes)
real    1m10.935s

```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.17-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.17-gentoo-x86_64.img"

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240326 6.6.17 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.17-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/kernel-6.6.17-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off  amd_pstate=active
	initrd	/initramfs20240326_squash_crypt-6.6.17-gentoo-x86_64.img
}

```

# proposal

- x11 usb mouse input not working
- sudo usermod -aG input martin

```
sci-libs/armadillo arpack blas -doc -examples lapack -mkl superlu -test
```
# update 2024-04-21
```
eix-sync
(ql:update-dist "quicklisp") # no change
(ql:update-client) # no change
sbcl --eval "(map nil 'ql-dist:clean (ql-dist:all-dists))"
```
- note: i actually had the backdoored xz-utils version installed
- add armadillo

```
dispatch-conf

emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
sudo emerge -av armadillo
emerge --depclean
revdep-rebuild
eclean-dist #425Mb 58 files
eclean-pkg #1G 63 files
```

```
find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 


2024-04-21 00:43:06.9034268590 +0200 linux-headers-6.6-r1-1.gpkg.tar size=1699840
2024-04-21 00:46:30.9723070420 +0200 glibc-2.38-r12-1.gpkg.tar size=15861760
2024-04-21 00:46:49.8022037140 +0200 ensurepip-setuptools-69.2.0-1.gpkg.tar size=747520
2024-04-21 00:47:11.4820847470 +0200 clang-common-17.0.6-r5-1.gpkg.tar size=40960
2024-04-21 00:49:17.1613950880 +0200 openssl-3.0.13-r2-1.gpkg.tar size=6758400
2024-04-21 00:50:30.1643278220 +0200 linux-firmware-20240410-1.gpkg.tar size=473917440
2024-04-21 00:51:20.9073827050 +0200 libxcrypt-4.4.36-r3-1.gpkg.tar size=163840
2024-04-21 00:51:49.3638932180 +0200 xz-utils-5.4.2-1.gpkg.tar size=542720
2024-04-21 00:52:05.5571376920 +0200 elt-patches-20240315-1.gpkg.tar size=71680
2024-04-21 00:52:52.2402148550 +0200 util-linux-2.39.3-r7-1.gpkg.tar size=5099520
2024-04-21 00:53:13.7567634500 +0200 libseccomp-2.5.5-r1-1.gpkg.tar size=153600
2024-04-21 00:53:55.8398658550 +0200 libtool-2.4.7-r4-1.gpkg.tar size=757760
2024-04-21 00:54:17.8064119810 +0200 libXdmcp-1.1.5-1.gpkg.tar size=61440
2024-04-21 00:56:04.9258241690 +0200 python-3.12.3-1.gpkg.tar size=29317120
2024-04-21 00:56:44.4589405670 +0200 libxcb-1.16.1-1.gpkg.tar size=604160
2024-04-21 00:57:11.2821267090 +0200 at-spi2-core-2.50.2-1.gpkg.tar size=788480
2024-04-21 00:58:09.3284748500 +0200 libunistring-1.2-1.gpkg.tar size=737280
2024-04-21 00:58:29.8016958370 +0200 pax-utils-1.3.7-4.gpkg.tar size=133120
2024-04-21 00:58:33.4883422740 +0200 less-643-r2-1.gpkg.tar size=184320
2024-04-21 00:59:04.9715028450 +0200 libfontenc-1.1.8-1.gpkg.tar size=61440
2024-04-21 00:59:31.7913556720 +0200 glibmm-2.66.7-1.gpkg.tar size=1761280
2024-04-21 00:59:42.8812948170 +0200 pixman-0.43.4-1.gpkg.tar size=593920
2024-04-21 00:59:48.0412665020 +0200 vala-common-0.56.16-1.gpkg.tar size=20480
2024-04-21 00:59:53.5079031710 +0200 opus-1.5.1-r1-1.gpkg.tar size=4526080
2024-04-21 00:59:59.6812026280 +0200 iceauth-1.0.10-1.gpkg.tar size=61440
2024-04-21 01:00:22.8744086900 +0200 nss-3.99-1.gpkg.tar size=3399680
2024-04-21 01:00:27.0177192870 +0200 mkfontscale-1.2.3-1.gpkg.tar size=71680
2024-04-21 01:00:31.9410256040 +0200 iso-codes-4.16.0-1.gpkg.tar size=5488640
2024-04-21 01:00:32.1176913010 +0200 imlib2-1.11.0-1.gpkg.tar size=624640
2024-04-21 01:01:04.7741787670 +0200 xauth-1.1.3-1.gpkg.tar size=81920
2024-04-21 01:01:08.7408236670 +0200 libXcursor-1.2.2-1.gpkg.tar size=81920
2024-04-21 01:01:08.7774901330 +0200 encodings-1.1.0-1.gpkg.tar size=655360
2024-04-21 01:01:12.4141368440 +0200 libXaw-1.0.16-1.gpkg.tar size=307200
2024-04-21 01:01:30.9107020110 +0200 rust-1.75.0-r1-1.gpkg.tar size=30720
2024-04-21 01:01:38.8806582770 +0200 packaging-24.0-1.gpkg.tar size=174080
2024-04-21 01:02:01.7771993000 +0200 xmessage-1.0.7-1.gpkg.tar size=51200
2024-04-21 01:02:22.8770835150 +0200 pango-1.52.1-1.gpkg.tar size=1300480
2024-04-21 01:02:29.9570446640 +0200 libarchive-3.7.2-r3-1.gpkg.tar size=624640
2024-04-21 01:02:46.2836217400 +0200 typing-extensions-4.11.0-1.gpkg.tar size=122880
2024-04-21 01:03:02.0635351480 +0200 jaraco-context-5.1.0-1.gpkg.tar size=51200
2024-04-21 01:03:18.2801128270 +0200 idna-3.7-1.gpkg.tar size=194560
2024-04-21 01:03:35.1066871590 +0200 trove-classifiers-2024.3.25-1.gpkg.tar size=61440
2024-04-21 01:03:50.8666006770 +0200 wheel-0.43.0-1.gpkg.tar size=112640
2024-04-21 01:04:08.9598347250 +0200 gtk-update-icon-cache-3.24.40-1.gpkg.tar size=122880
2024-04-21 01:04:13.9031409320 +0200 hatchling-1.22.5-r1-1.gpkg.tar size=256000
2024-04-21 01:04:21.6997648150 +0200 libplacebo-6.338.2-4.gpkg.tar size=624640
2024-04-21 01:04:42.2229855290 +0200 setuptools-69.2.0-r1-1.gpkg.tar size=1269760
2024-04-21 01:05:45.2893061220 +0200 cython-3.0.10-1.gpkg.tar size=4638720
2024-04-21 01:06:05.9225262320 +0200 dill-0.3.8-1.gpkg.tar size=327680
2024-04-21 01:06:14.7724776680 +0200 pillow-10.3.0-1.gpkg.tar size=1064960
2024-04-21 01:06:36.7723569450 +0200 lxml-5.2.1-1.gpkg.tar size=1628160
2024-04-21 01:06:55.2589221680 +0200 threadpoolctl-3.4.0-1.gpkg.tar size=92160
2024-04-21 01:07:03.4322106510 +0200 lmfit-1.3.1-1.gpkg.tar size=307200
2024-04-21 01:07:39.1420146950 +0200 fonttools-4.51.0-1.gpkg.tar size=4085760
2024-04-21 01:08:20.6151204470 +0200 libXaw3d-1.6.6-1.gpkg.tar size=256000
2024-04-21 01:08:57.8282495750 +0200 vala-0.56.16-1.gpkg.tar size=3399680
2024-04-21 01:10:13.2278358240 +0200 ffmpeg-6.1.1-r5-1.gpkg.tar size=10352640
2024-04-21 01:10:44.8276624220 +0200 mesa-24.0.4-1.gpkg.tar size=13813760
2024-04-21 01:11:39.0240316890 +0200 grub-2.12-r4-1.gpkg.tar size=18012160
2024-04-21 01:13:08.6968729470 +0200 nghttp2-1.61.0-1.gpkg.tar size=225280
2024-04-21 01:13:23.3467925570 +0200 xorg-server-21.1.13-1.gpkg.tar size=3553280
2024-04-21 01:15:21.8328090370 +0200 matplotlib-3.8.4-1.gpkg.tar size=33873920
2024-04-21 01:15:22.8461368100 +0200 gtk+-3.24.41-1.gpkg.tar size=12636160
2024-04-21 01:16:38.9757190530 +0200 scikit-learn-1.4.2-1.gpkg.tar size=13701120
2024-04-21 01:16:54.9022983240 +0200 qtcore-5.15.13-1.gpkg.tar size=8089600
2024-04-21 01:17:38.9720564930 +0200 xf86-input-libinput-1.4.0-8.gpkg.tar size=81920
2024-04-21 01:18:47.4716806050 +0200 firefox-bin-125.0.1-1.gpkg.tar size=93706240
2024-04-21 01:20:17.3511873960 +0200 qtdbus-5.15.13-1.gpkg.tar size=450560
2024-04-21 01:20:18.5078477160 +0200 gtkmm-3.24.9-1.gpkg.tar size=2457600
2024-04-21 01:20:32.1977725930 +0200 emacs-29.3-r1-1.gpkg.tar size=48076800
2024-04-21 01:20:45.4010334740 +0200 gtk-4.12.5-1.gpkg.tar size=14233600
2024-04-21 01:20:49.6810099880 +0200 curl-8.7.1-r1-1.gpkg.tar size=1402880
2024-04-21 01:21:34.1907657430 +0200 xf86-video-amdgpu-23.0.0-10.gpkg.tar size=174080
2024-04-21 01:21:46.2940326600 +0200 gentoolkit-0.6.5-1.gpkg.tar size=3532800
2024-04-21 01:22:01.7606144550 +0200 xf86-video-ati-22.0.0-10.gpkg.tar size=471040
2024-04-21 01:25:05.9929368230 +0200 qtgui-5.15.13-1.gpkg.tar size=5068800
2024-04-21 01:29:33.3281365020 +0200 qtbase-6.7.0-r1-1.gpkg.tar size=19005440
2024-04-21 01:31:41.5907660010 +0200 qttools-6.7.0-1.gpkg.tar size=3102720
2024-04-21 01:31:41.7407651780 +0200 qtwidgets-5.15.13-1.gpkg.tar size=3471360
2024-04-21 01:32:06.0706316690 +0200 qttranslations-6.7.0-1.gpkg.tar size=2529280
2024-04-21 01:32:47.6937365980 +0200 qt5compat-6.7.0-1.gpkg.tar size=645120
2024-04-21 01:32:55.0170297450 +0200 qtsvg-6.7.0-1.gpkg.tar size=348160
2024-04-21 01:32:55.2370285380 +0200 qtshadertools-6.7.0-1.gpkg.tar size=2242560
2024-04-21 01:34:53.2063811880 +0200 qtmultimedia-6.7.0-1.gpkg.tar size=3082240
2024-04-21 01:35:05.9763111130 +0200 obs-studio-30.1.1-1.gpkg.tar size=6768640
2024-04-21 09:19:31.0800695080 +0200 superlu-5.3.0-1.gpkg.tar size=225280
2024-04-21 09:19:48.8233054760 +0200 armadillo-12.4.0-r1-1.gpkg.tar size=645120

```
- cleanup /mnt4/persistent
  - package.use had libpcap with bluetooth support. i don't think i need that

- create new image

```
export TODAY=20240421
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
lib/modules/6.6.12-gentoo-x86_64 \
usr/lib/modules/6.3.12-gentoo-x86_64 \
usr/lib/modules/6.6.12-gentoo-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
opt/rust-bin* \
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
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia} \
persistent \
initramfs-with-squashfs.img

```

- a bit bigger (12MB compressed):

```
# old:
Filesystem size 1855391.59 Kbytes (1811.91 Mbytes)
        30.00% of uncompressed filesystem size (6185094.38 Kbytes)
real    1m10.935s
# new:
Filesystem size 1868857.18 Kbytes (1825.06 Mbytes)
        30.02% of uncompressed filesystem size (6224371.91 Kbytes)
real    1m12.648s

```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.17-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.17-gentoo-x86_64.img"

```
- in first attempt TODAY, was empty. so i have an initramfs file without timestamp
- but i copied it to the correct name


- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240421 6.6.17 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.17-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/kernel-6.6.17-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off  amd_pstate=active
	initrd	/initramfs20240421_squash_crypt-6.6.17-gentoo-x86_64.img
}

```


# update 2024-06-15



```
eix-sync

dispatch-conf

emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
```
```
!!! Your current profile is deprecated and not supported anymore.
!!! Use eselect profile to update your profile.
!!! Please upgrade to the following profile if possible:

        default/linux/amd64/23.0/no-multilib/systemd

To upgrade do the following steps:

A profile upgrade to version 23.0 is available for your architecture.
The new 23.0 profiles enable some toolchain hardening features and 
performance enhancements by default, and standardize settings.
You can find the list of changes on the wiki tracking page [1].


Upgrade instructions

Note 1: If you have manually changed your CHOST to a value different from 
what the stages and profiles set, you may have to do that in the future too.
In that case you should know what you are doing, hopefully; please read the 
instructions with a critical eye then.

Note 2: In case you are already familiar with binary packages, you should be
able to add "--getbinpkg" to the emerge calls to speed things up.
The use of binary packages is completely optional though, and also not
as much tested as the source-based upgrade path yet.

1. Ensure your system backups are up to date. Please also update
   your system fully and depclean before proceeding.
   glibc older than 2.36 and musl older than 1.2.4 is not supported anymore.

2. If you are still using one of the long-deprecated amd64 17.0 profiles 
   (other than x32 or musl), then first complete the migration to the 
   corresponding 17.1 profile. Instructions can be found at [3].

[ my current profile:   [15]  default/linux/amd64/17.1/no-multilib/systemd/merged-usr (exp) * ]

3. If you are currently using systemd in a split-usr configuration, then first 
   complete the migration to the corresponding merged-usr profile of the 
   same profile version. Details on how to do this can be found in the news 
   item [4].
   If you are currently using openrc, migrate to 23.0 first, keeping your disk
   layout. If you want to move from split-usr to merged-usr, do that afterwards.

4. Run "emerge --info" and note down the value of the CHOST variable.

[ CHOST="x86_64-pc-linux-gnu" ]

5. Edit /etc/portage/make.conf; if there is a line defining the CHOST variable,
   remove it. Also delete all lines defining CHOST_... variables.

6. Select the 23.0 profile corresponding to your current profile, either using
   "eselect profile" or by manually setting the profile symlink.
   Note that old profiles are by default split-usr and the 23.0 profiles by
   default merged-usr. Do NOT change directory scheme now, since this will
   mess up your system! 
   Instead, make sure that the new profile has the same property: for example, 
   OLD default/linux/amd64/17.1  
        ==>  NEW default/linux/amd64/23.0/split-usr
             (added "split-usr")
   OLD default/linux/amd64/17.1/systemd/merged-usr  
        ==>  NEW default/linux/amd64/23.0/systemd
             (removed "merged-usr")
   A detailed table of the upgrade paths can be found at [5]. Please consult it.
   In some cases (hppa, x86) the table will tell you to pick between two choices. 
   What you need should be obvious from your *old* CHOST value (from step 4).

[ eselect profile set default/linux/amd64/23.0/no-multilib/systemd ]

7. Delete the contents of your binary package cache at ${PKGDIR}
     rm -r /var/cache/binpkgs/*

[ 7.9G    /var/cache/binpkgs/
  emerge --depclean
  rm -rf /var/cache/binpkgs
]

8. In the file or directory /etc/portage/binrepos.conf (if existing), update
   the URI in all configuration such that they point to 23.0 profile binhost 
   directories. The exact paths can be found in the table at [5], too.

[ file doesn't exist ]

9. Rebuild or reinstall from binary (if available) the following packages in
   this order, with the same version as already active:
     emerge --ask --oneshot sys-devel/binutils
   (you may have to run binutils-config and re-select your binutils now)
   emerge --ask --oneshot sys-devel/gcc
   (IMPORTANT: If this command wants to rebuild glibc first, do *not* let it do 
    that; instead, abort and try again with --nodeps added to the command line.)
   (you may have to run gcc-config and re-select your gcc now)
   and the C library, i.e. for glibc-based systems
     emerge --ask --oneshot sys-libs/glibc
   or for musl-based systems
     emerge --ask --oneshot sys-libs/musl
     

10. Re-run "emerge --info" and check if CHOST has changed compared to step 4.

[ CHOST="x86_64-pc-linux-gnu ]

If the CHOST has NOT changed, skip to step 13 (env-update). Otherwise, 

11. Recheck with binutils-config and gcc-config that valid installed versions
   of binutils and gcc are selected.

12. Check /etc/env.d, /etc/env.d/binutils, and /etc/env.d/gcc for files that
   refer to the *OLD* CHOST value, and remove them. 
   Examples how to do this can be found in the similar procedure at [6].

13. Run env-update && source /etc/profile

14. Re-emerge libtool:
   emerge --ask --oneshot libtool
reboot

15. Just for safety, delete the contents of your binary package cache at 
    ${PKGDIR} again:
     rm -r /var/cache/binpkgs/*

16. Rebuild world:
   emerge --ask --emptytree @world
   emerge --jobs=6 --load-average=10 --ask --emptytree @world 


real    284m48.142s
user    1561m53.091s
sys     202m0.573s


[1] https://wiki.gentoo.org/wiki/Project:Toolchain/23.0_profile_transition
[2] https://wiki.gentoo.org/wiki/Project:Toolchain/23.0_profile_timeline
[3] https://www.gentoo.org/support/news-items/2019-06-05-amd64-17-1-profiles-are-now-stable.html
[4] https://www.gentoo.org/support/news-items/2022-12-01-systemd-usrmerge.html
[5] https://wiki.gentoo.org/wiki/Project:Toolchain/23.0_update_table
[6] https://wiki.gentoo.org/wiki/Changing_the_CHOST_variable#Verifying_things_work



```


```
emerge --depclean
revdep-rebuild
eclean-dist # 1.3G
eclean-pkg # nothing

find /var/cache/binpkgs/ -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %Tz %f size=%s\n"|sort -n 

/etc/portage/savedconfig/x11-dwm/dwm-6.5 needs editing (previous version was 6.4)
```

```
2024-06-15 09:26:26.9311689130 +0200 gnuconfig-20230731-1.gpkg.tar size=51200
2024-06-15 09:26:27.0311683640 +0200 libintl-0-r2-1.gpkg.tar size=30720
...
2024-06-15 14:05:55.0124882960 +0200 sudo-1.9.15_p5-1.gpkg.tar size=2170880
2024-06-15 14:06:10.0000000000 +0200 Packages size=793141
2024-06-15 14:06:10.4357369950 +0200 openssh-9.6_p1-r3-1.gpkg.tar size=1546240
```


- create new image

```
export TODAY=20240615
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
lib/modules/6.6.12-gentoo-x86_64 \
usr/lib/modules/6.3.12-gentoo-x86_64 \
usr/lib/modules/6.6.12-gentoo-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
opt/rust-bin* \
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
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia} \
persistent \
initramfs-with-squashfs.img

```

- a bit smaller (11MB compressed):

```
# old:
Filesystem size 1849509.25 Kbytes (1806.16 Mbytes)
        30.03% of uncompressed filesystem size (6158277.84 Kbytes)
real    1m7.399s	
# new:
Filesystem size 1838651.36 Kbytes (1795.56 Mbytes)
        29.89% of uncompressed filesystem size (6152097.00 Kbytes)
real    1m5.901s
```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.17-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.17-gentoo-x86_64.img"

```



- check grub config, add the new entry

```
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240615 6.6.17 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.17-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/kernel-6.6.17-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240615_squash_crypt-6.6.17-gentoo-x86_64.img
}

```

# update 2024-06-16

- update kernel from 6.6.17 to 6.6.30

```
# modify portage gentoo-sources and linux-headers (not in ~amd64 accept keywords anymore)
emerge sys-kernel/gentoo-sources
eselect kernel set 2
cd /usr/src/linux
make oldconfig
make -j 12
make modules_install install
emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240616 6.6.30 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.30-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.6.30-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240616_squash_crypt-6.6.30-gentoo-x86_64.img
}

menuentry 'Gentoo GNU/Linux 6.6.30 from disk' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.30-gentoo ...'
	linux	/vmlinuz-6.6.30-gentoo-x86_64 root=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 ro  
}


```


```
export TODAY=20240616
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
lib/modules/6.6.12-gentoo-x86_64 \
lib/modules/6.6.17-gentoo-x86_64 \
usr/lib/modules/6.3.12-gentoo-x86_64 \
usr/lib/modules/6.6.12-gentoo-x86_64 \
usr/lib/modules/6.6.17-gentoo-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
opt/rust-bin* \
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
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia} \
persistent \
initramfs-with-squashfs.img

```

```
Filesystem size 1835228.57 Kbytes (1792.22 Mbytes)
        29.81% of uncompressed filesystem size (6156208.55 Kbytes)

```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh

dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.30-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.30-gentoo-x86_64.img"

```
# update 2024-07-25
```
eix-sync


dispatch-conf

emerge --deselect grpc
emerge --deselect lmfit

emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world
emerge --depclean
# clang is now 18
# kernel 6.6.38 instead of 6.6.30
# python 3.12.3-r1 instead of python 3.11.8_p1
# protobuf, re2, asteval, lmfit, grpc gone
revdev-rebuild # nothing

eselect kernel list
eselect kernel set 2 # set 6.6.38
cd /usr/src/linux
cp ../linux-6.6.30/.config .
make menuconfig
make -j12
make modules_install
make install
cp /boot/vmlinuz /boot/vmlinuz-6.6.38-gentoo-x86_64

update emacs packages # (2)

# i tried updating slime, didn't help

emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240727 6.6.38 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.38-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.6.38-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240727_squash_crypt-6.6.38-gentoo-x86_64.img
}

menuentry 'Gentoo GNU/Linux 6.6.38 from disk' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.38-gentoo ...'
	linux	/vmlinuz-6.6.38-gentoo-x86_64 root=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 ro  
}

export TODAY=20240727
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
lib/modules/6.6.12-gentoo-x86_64 \
lib/modules/6.6.17-gentoo-x86_64 \
lib/modules/6.6.30-gentoo-x86_64 \
usr/lib/modules/6.3.12-gentoo-x86_64 \
usr/lib/modules/6.6.12-gentoo-x86_64 \
usr/lib/modules/6.6.17-gentoo-x86_64 \
usr/lib/modules/6.6.30-gentoo-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
opt/rust-bin* \
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
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia} \
persistent \
initramfs-with-squashfs.img


```

- 50Mb smaller than before (compressed)
```
Filesystem size 1797389.54 Kbytes (1755.26 Mbytes)
        30.32% of uncompressed filesystem size (5927673.71 Kbytes)

```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh



dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.38-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.38-gentoo-x86_64.img"

```
- reboot into 6.6.38 from disk and build the ryzen monitor module


# Proposal 2024-09-05

 Important: In case the wireless configuration API (CONFIG_CFG80211) is
built into the kernel (<*>) instead as a module (<M>), the driver
won't be able to load regulatory.db from /lib/firmware resulting in
broken regulatory domain support. Please set CONFIG_CFG80211=m or add
regulatory.db and regulatory.db.p7s (from net-wireless/wireless-regdb)
to CONFIG_EXTRA_FIRMWARE.



# Update 2024-09-06


```
eix-sync


dispatch-conf


emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y --fetchonly @world

emerge --jobs=6 --load-average=10  --ask --verbose --update --newuse --deep --with-bdeps=y @world


 * Checking for suitable kernel configuration options ...
 *   CONFIG_IP_NF_NAT:   is not set when it should be.
 *   CONFIG_IP_NF_TARGET_MASQUERADE:     is not set when it should be.
 CONFIG_UHID:        is not set when it should be.
 make 8852BE a module

 * NOTE: To capture traffic with wireshark as normal user you have to
 * add yourself to the pcap group.
# 6.6.47 new kernel

emerge --depclean -va
perl-cleaner --all



revdev-rebuild # nothing

eselect kernel list
eselect kernel set 1

cd /usr/src/linux
cp ../linux-6.6.38-gentoo/.config .
make menuconfig
make -j12 # 13m14s
make modules_install
make install
cp /boot/vmlinuz /boot/vmlinuz-6.6.47-gentoo-x86_64

update emacs packages 

# i tried updating quicklisp, nothing new. ql has slime 28 and emacs has 30

emacs /boot/grub/grub.cfg

menuentry 'Gentoo GNU/Linux 20240727 6.6.47 ram squash persist crypt ssd ' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.47-gentoo-x86_64 ...'
# the kernel and initramfs is loaded from nvme0n1p3 (unencrypted)
# the initramfs asks for password and gets the squashfs from nvme0n1p4 (encrypted)
	linux	/vmlinuz-6.6.47-gentoo-x86_64 root=/dev/nvme0n1p3 init=/init mitigations=off
	initrd	/initramfs20240727_squash_crypt-6.6.47-gentoo-x86_64.img
}

menuentry 'Gentoo GNU/Linux 6.6.47 from disk' --class gentoo --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-80b66b33-ce31-4a54-9adc-b6c72fe3a826' {
	load_video
	if [ "x$grub_platform" = xefi ]; then
		set gfxpayload=keep
	fi
	insmod gzio
	insmod part_gpt
	insmod fat
	search --no-floppy --fs-uuid --set=root F63D-5318
	echo	'Loading Linux 6.6.47-gentoo ...'
	linux	/vmlinuz-6.6.47-gentoo-x86_64 root=UUID=80b66b33-ce31-4a54-9adc-b6c72fe3a826 ro  
}

cryptsetup luksOpen /dev/nvme0n1p4 p4
mount /dev/mapper/p4 /mnt4

export TODAY=20240906
export INDIR=/
export OUTFILE=/mnt4/gentoo_$TODAY.squashfs
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
lib/modules/6.3.12-gentoo-x86_64 \
lib/modules/6.6.12-gentoo-x86_64 \
lib/modules/6.6.17-gentoo-x86_64 \
lib/modules/6.6.30-gentoo-x86_64 \
lib/modules/6.6.38-gentoo-x86_64 \
usr/lib/modules/6.3.12-gentoo-x86_64 \
usr/lib/modules/6.6.12-gentoo-x86_64 \
usr/lib/modules/6.6.17-gentoo-x86_64 \
usr/lib/modules/6.6.30-gentoo-x86_64 \
usr/lib/modules/6.6.38-gentoo-x86_64 \
usr/src/linux* \
var/cache/binpkgs/* \
var/cache/distfiles/* \
gentoo*squashfs \
usr/share/genkernel/distfiles/* \
opt/rust-bin* \
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
home/martin/stage \
var/log/journal/* \
var/cache/genkernel/* \
var/tmp/portage/* \
tmp/* \
mnt/ \
mnt4/ \
mnt5/ \
usr/lib/firmware/{qcom,netronome,mellanox,mrvl,mediatek,ath11k,ath10k,ath12k,qed,dpaa2,brcm,ti-connectivity,cypress,liquidio,cxgb4,bnx2x,nvidia,i915,qca,cirrus} \
usr/lib/firmwar/{iwlwifi,phanfw}* \
persistent \
initramfs-with-squashfs.img


```

- 20Mb bigger than before (compressed)
```
Filesystem size 1816616.11 Kbytes (1774.04 Mbytes)
        30.44% of uncompressed filesystem size (5967557.51 Kbytes)
1m17s

```

```
emacs init_dracut_crypt.sh
cp init_dracut_crypt.sh  /usr/lib/dracut/modules.d/99base/init.sh
chmod a+x /usr/lib/dracut/modules.d/99base/init.sh



dracut \
  -m " kernel-modules base rootfs-block crypt dm " \
  --filesystems " squashfs vfat overlay " \
  --kver=6.6.47-gentoo-x86_64 \
  --force \
  "/boot/initramfs"$TODAY"_squash_crypt-6.6.47-gentoo-x86_64.img"

```
- reboot into 6.6.47 from disk and build the ryzen monitor module

- look with qdirstat what can be removed in /usr/lib/firmware and in general

- a lot of stuff in /var/db and /var/tmp/portage, maybe i need to 

```
eclean packages
eclean distfiles 


very big

/usr/lib64/libwireshark 91M 
libmupdf 47M

perl5 (52MB!)

ctest cpack binaries wireshark obs
```

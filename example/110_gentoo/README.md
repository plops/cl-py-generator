# script to install gentoo into a chroot

- i am running this on an arch linux 

```
wget https://bouncer.gentoo.org/fetch/root/all/releases/amd64/autobuilds/20230611T170207Z/stage3-amd64-nomultilib-systemd-mergedusr-20230611T170207Z.tar.xz
mkdir gentoo
cd gentoo
sudo tar xpvf ../stage3-*.tar.xz --xattrs-include='*.*' --numeric-owner
# 1.2GB extracted

cd /mnt/gentoo
cat <<EOF > etc/portage/make.conf
COMMON_FLAGS="-march=native -fomit-frame-pointer -O2 -pipe"
CFLAGS="${COMMON_FLAGS}"
CXXFLAGS="${COMMON_FLAGS}"
FCFLAGS="${COMMON_FLAGS}"
FFLAGS="${COMMON_FLAGS}"
LC_MESSAGES=C.utf8
MAKEOPTS="-j7"
USE="X"
VIDEO_CARDS="radeon"
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
rm /bin/cpio
emerge sys-kernel/genkernel

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

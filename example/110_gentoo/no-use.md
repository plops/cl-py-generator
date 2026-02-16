https://forums.gentoo.org/viewtopic-p-8790781.html?sid=d43656936be7d6cffb8864cce3ae37dd

Abstract: Implementing a Global USE="-*" Strategy in Gentoo Linux
This documentation thread explores the advanced configuration of the Gentoo make.conf file using the USE="-*" flag to override default profile settings. The primary objective of this approach is to minimize system "bloat" by disabling all non-essential USE flags, thereby reducing the total package count—in one reported case, by 6.5% to 13%.

The process requires the user to explicitly define every required global and package-specific USE flag, as well as manually manage USE_EXPAND variables such as PYTHON_TARGETS, CPU_FLAGS_X86, and VIDEO_CARDS. While the method offers superior control and system transparency, it introduces significant maintenance overhead, including frequent dependency blockers during world updates and potential breakages in unstated dependencies. Furthermore, the discussion highlights a rift within the Gentoo community: while power users value the minimalism, administrators warn that this "self-inflicted" complexity complicates community support efforts when users fail to disclose their non-standard configurations.



## Helpful tools

genlop -ltn|grep -B1 minute



```
Wed Feb  4 16:31:20 2026 >>> sys-devel/gcc-16.0.9999
       merge time: 27 minutes and 56 seconds.
--
     Wed Feb  4 18:16:42 2026 >>> acct-group/systemd-timesync-0-r3
       merge time: 1 minute.
--
     Wed Feb  4 18:20:14 2026 >>> dev-db/sqlite-3.50.4
       merge time: 1 minute.
```

emerge --info -v
```
Portage 3.0.72 (python 3.13.11-final-0, default/linux/amd64/23.0/no-multilib/systemd, gcc-16, glibc-2.41-r6, 6.12.58-gentoo x86_64)
=================================================================
System uname: Linux-6.12.58-gentoo-x86_64-AMD_Ryzen_Threadripper_PRO_7955WX_16-Cores-with-glibc2.41
KiB Mem:    65206196 total,   9694716 free
KiB Swap:          0 total,         0 free
Timestamp of repository gentoo: Tue, 03 Feb 2026 00:45:00 +0000
Head commit of repository gentoo: 9fdf21a29669c0cae3e45b22a87b74a5ea977285
sh bash 5.3_p9
ld GNU ld (Gentoo 2.45.1 p1) 2.45.1
app-misc/pax-utils:        1.3.8::gentoo
app-shells/bash:           5.3_p9::gentoo
dev-build/autoconf:        2.72-r6::gentoo
dev-build/automake:        1.18.1::gentoo
dev-build/cmake:           4.1.4::gentoo
dev-build/libtool:         2.5.4::gentoo
dev-build/make:            4.4.1-r102::gentoo
dev-build/meson:           1.9.1::gentoo
dev-lang/perl:             5.42.0-r1::gentoo
dev-lang/python:           3.13.11::gentoo, 3.14.0_p1::gentoo
sys-apps/baselayout:       2.18::gentoo
sys-apps/openrc:           0.62.10::gentoo
sys-apps/sandbox:          2.46::gentoo
sys-apps/systemd:          258.3::gentoo
sys-devel/binutils:        2.45.1::gentoo
sys-devel/binutils-config: 5.6::gentoo
sys-devel/gcc:             16.0.9999::gentoo
sys-devel/gcc-config:      2.12.2::gentoo
sys-kernel/linux-headers:  6.12::gentoo (virtual/os-headers)
sys-libs/glibc:            2.41-r6::gentoo
Repositories:

gentoo
    location: /var/db/repos/gentoo
    sync-type: rsync
    sync-uri: rsync://rsync.gentoo.org/gentoo-portage
    priority: -1000
    volatile: False
    sync-rsync-verify-max-age: 3
    sync-rsync-extra-opts: 
    sync-rsync-verify-metamanifest: no
    sync-rsync-verify-jobs: 1
Binary Repositories:

gentoo
    location: /var/cache/binhost/gentoo
    priority: 1
    sync-uri: https://distfiles.gentoo.org/releases/amd64/binpackages/23.0/x86-64

ACCEPT_KEYWORDS="amd64"
ACCEPT_LICENSE="@FREE"
CBUILD="x86_64-pc-linux-gnu"
CFLAGS=" -march=znver3 -fomit-frame-pointer -O2 -pipe "
CHOST="x86_64-pc-linux-gnu"
CONFIG_PROTECT="/etc"
CONFIG_PROTECT_MASK="/etc/ca-certificates.conf /etc/env.d /etc/gconf /etc/gentoo-release /etc/sandbox.d"
CXXFLAGS=" -march=znver3 -fomit-frame-pointer -O2 -pipe "
DISTDIR="/var/cache/distfiles"
EMERGE_DEFAULT_OPTS="--jobs 24 --load-average 32"
ENV_UNSET="CARGO_HOME DBUS_SESSION_BUS_ADDRESS DISPLAY GDK_PIXBUF_MODULE_FILE GOBIN GOPATH PERL5LIB PERL5OPT PERLPREFIX PERL_CORE PERL_MB_OPT PERL_MM_OPT XAUTHORITY XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_RUNTIME_DIR XDG_STATE_HOME"
FCFLAGS=" -march=znver3 -fomit-frame-pointer -O2 -pipe "
FEATURES="assume-digests binpkg-docompress binpkg-dostrip binpkg-logs binpkg-multi-instance buildpkg-live config-protect-if-modified distlocks ebuild-locks fixlafiles ipc-sandbox merge-sync merge-wait multilib-strict network-sandbox news nodoc noinfo noman parallel-fetch pid-sandbox pkgdir-index-trusted preserve-libs protect-owned qa-unresolved-soname-deps sandbox strict unknown-features-warn unmerge-logs unmerge-orphans userfetch userpriv usersandbox usersync"
FFLAGS=" -march=znver3 -fomit-frame-pointer -O2 -pipe "
GENTOO_MIRRORS="https://mirror.init7.net/gentoo/     rsync://mirror.init7.net/gentoo/     http://mirrors.ircam.fr/pub/gentoo-distfiles/     http://ftp.fau.de/gentoo     rsync://ftp.fau.de/gentoo     http://ftp-stud.hs-esslingen.de/pub/Mirrors/gentoo/     rsync://ftp-stud.hs-esslingen.de/gentoo/     http://ftp.uni-stuttgart.de/gentoo-distfiles/"
LANG="C.UTF-8"
LDFLAGS="-Wl,-O1 -Wl,--as-needed -Wl,-z,pack-relative-relocs"
LEX="flex"
MAKEOPTS="-j33"
PKGDIR="/var/cache/binpkgs"
PORTAGE_COMPRESS="bzip2"
PORTAGE_CONFIGROOT="/"
PORTAGE_RSYNC_OPTS="--recursive --links --safe-links --perms --times --omit-dir-times --compress --force --whole-file --delete --stats --human-readable --timeout=180 --exclude=/distfiles --exclude=/local --exclude=/packages --exclude=/.git"
PORTAGE_TMPDIR="/var/tmp"
USE="amd64 gnu minimal reference test-rust" ABI_X86="64" CPU_FLAGS_X86="aes avx avx2 bmi1 bmi2 f16c fma3 mmx mmxext pclmul popcnt rdrand sha sse sse2 sse3 sse4_1 sse4_2 sse4a ssse3 vpclmulqdq" ELIBC="glibc" KERNEL="linux" L10N="en-GB" LLVM_TARGETS="X86" PYTHON_SINGLE_TARGET="python3_13" PYTHON_TARGETS="python3_13" RUBY_TARGETS="ruby33"
Unset:  ADDR2LINE, AR, ARFLAGS, AS, ASFLAGS, CC, CCLD, CONFIG_SHELL, CPP, CPPFLAGS, CTARGET, CXX, CXXFILT, ELFEDIT, EXTRA_ECONF, F77FLAGS, FC, GCOV, GPROF, INSTALL_MASK, LC_ALL, LD, LFLAGS, LIBTOOL, LINGUAS, MAKE, MAKEFLAGS, NM, OBJCOPY, OBJDUMP, PORTAGE_BINHOST, PORTAGE_BUNZIP2_COMMAND, PORTAGE_COMPRESS_FLAGS, PORTAGE_RSYNC_EXTRA_OPTS, PYTHONPATH, RANLIB, READELF, RUSTFLAGS, SHELL, SIZE, STRINGS, STRIP, YACC, YFLAGS


```


emerge -epvt --complete-graph @system
```
These are the packages that would be merged, in reverse order:

Calculating dependencies... done!
Dependency resolution took 1.23 s (backtrack: 0/20).

[nomerge       ] virtual/package-manager-2::gentoo 
[nomerge       ]  sys-apps/portage-3.0.72-r1::gentoo  USE="(ipc) -apidoc -build -doc -gentoo-dev -native-extensions -rsync-verify (-selinux) -test -xattr" PYTHON_TARGETS="python3_13 -python3_12 -python3_14" 
[nomerge       ]   dev-build/meson-1.9.1::gentoo  USE="-test -test-full -verify-sig" PYTHON_TARGETS="python3_13 (-pypy3_11) -python3_11 -python3_12 -python3_14" 
[nomerge       ]    dev-python/setuptools-80.9.0-r1::gentoo  USE="-test" PYTHON_TARGETS="python3_13 (-pypy3_11) -python3_11 -python3_12 (-python3_13t) -python3_14 (-python3_14t)" 
[ebuild   R    ]     dev-python/trove-classifiers-2025.12.1.14::gentoo  USE="-test -verify-provenance" PYTHON_TARGETS="python3_13 (-pypy3_11) -python3_11 -python3_12 (-python3_13t) -python3_14 (-python3_14t)" 0 KiB
[ebuild   R    ]     dev-python/setuptools-scm-9.2.2::gentoo  USE="-test -verify-provenance" PYTHON_TARGETS="python3_13 (-pypy3_11) -python3_11 -python3_12 (-python3_13t) -python3_14 (-python3_14t)" 0 KiB
[ebuild   R    ] virtual/ssh-0-r2::gentoo  USE="minimal" 0 KiB
[ebuild   R    ] virtual/service-manager-2::gentoo  USE="-systemd" 0 KiB
[ebuild   R    ] virtual/pager-0-r1::gentoo  0 KiB
...
```
# alsa-utils Fails With "No linkable libasound" Walkthrough

This walkthrough records how `media-sound/alsa-utils` was installed on the e14
(AMD Ryzen 7 7735HS) running the image
`/1p3/yt/gentoo-z6-min-openrc_20261115` (kernel `6.18.63-gentoo-dist`), why the
build failed, and how the Dockerfile was changed so later images do not have
the problem. Date: November 19, 2026.

## Symptom

```bash
sudo emerge --ask=n media-sound/alsa-utils
```

fails in the configure phase:

```text
checking for libasound headers version >= 1.2.13 (1.2.13)... found.
checking for snd_ctl_open in -lasound... no
configure: error: No linkable libasound was found.
```

The error message is misleading. `media-libs/alsa-lib-1.2.16.1` is installed
correctly, and its library and headers are present:

```bash
ls -la /usr/lib64/libasound*
grep VERSION_STR /usr/include/alsa/version.h   # "1.2.16.1"
qfile /usr/lib64/libasound.so.2.0.0            # media-libs/alsa-lib
```

## 1. Find the Real Error in config.log

```bash
grep -n -A5 "snd_ctl_open in -lasound" \
  /var/tmp/portage/media-sound/alsa-utils-1.2.16/work/alsa-utils-1.2.16/config.log
```

```text
x86_64-pc-linux-gnu-gcc -o conftest ... conftest.c -lasound -lasound -lm -lpthread
/usr/x86_64-pc-linux-gnu/binutils-bin/2.47/ld: cannot find -lpthread: No such file or directory
```

So libasound links fine. The link test fails because of `-lpthread`, which
alsa-lib's pkg-config file adds.

## 2. Check glibc

Since glibc 2.34, the code from `libpthread`, `libdl`, `librt`, `libutil` and
`libanl` is part of `libc.so.6`. glibc still installs small empty archives
(`libpthread.a` is 8 bytes) so that `-lpthread` and the other flags keep
linking. There is no `libpthread.so` development symlink, so the empty `.a`
file is the only thing that satisfies `-lpthread`.

```bash
ls -la /usr/lib64/libpthread*      # only libpthread.so.0, no libpthread.a
qcheck sys-libs/glibc | grep AFK   # all *.a files and docs are missing
```

Missing (`AFK`) files included:

```text
/usr/lib64/libpthread.a  /usr/lib64/libdl.a  /usr/lib64/librt.a
/usr/lib64/libutil.a     /usr/lib64/libanl.a /usr/lib64/libc.a
/usr/lib64/libm.a        /usr/lib64/glibc-2.43/libm-2.43.a  ...
```

`INSTALL_MASK` is empty (`portageq envvar INSTALL_MASK`), so Portage did not
skip them. They were deleted after installation while the image was built.

## 3. Root Cause in the Dockerfile

Both image variants (hpz6 and e14) in `../../Dockerfile` remove static archives
to make the squashfs smaller:

```dockerfile
 && find /usr/lib /usr/lib64 -type f -name '*.a' \
      ! -name '*_nonshared.a' \
      ! -path '/usr/lib/gcc/*' \
      ! -path '/usr/lib64/gcc/*' \
      -delete \
```

This keeps `libc_nonshared.a`, but it also deletes glibc's stub archives. Any
build that passes `-lpthread`, `-ldl`, `-lrt`, `-lutil` or `-lanl` then fails
to link. Autoconf checks often report this as a missing library unrelated to
pthread.

The C++ smoke test after the cleanup did not catch it, because `g++` does not
pass `-lpthread` by default.

## 4. Fix the Running System

Reinstall glibc to restore the deleted files. `getbinpkg` is enabled, so this
uses the upstream binary package `glibc-2.43-r4`, the same one listed in the
image's `upstream_binpkgs.txt`:

```bash
sudo emerge --ask=n --oneshot sys-libs/glibc
ls -la /usr/lib64/libpthread.a     # -rw-r--r-- 1 root root 8 ...
```

`--oneshot` keeps glibc out of the world file.

Then install alsa-utils. It is built from source because there is no binary
package for it in `/1p3/yt/gentoo-z6-min-openrc_20261115/binpkgs`:

```bash
sudo emerge --ask=n media-sound/alsa-utils
aplay --version     # aplay: version 1.2.16
aplay -l            # card 0: Generic [HD-Audio Generic], HDMI 0/1/2 ...
```

On a live image with a persistent overlay, the restored glibc files and
alsa-utils go into the overlay. They are lost when you switch to a new
squashfs that does not contain them.

## 5. Fix the Image Build

In `../../Dockerfile`, at both cleanup sites (hpz6 and e14):

1. Exclude the glibc stub archives from the deletion:

   ```dockerfile
    && find /usr/lib /usr/lib64 -type f -name '*.a' \
         ! -name '*_nonshared.a' \
         ! -name 'libpthread.a' ! -name 'libdl.a' ! -name 'librt.a' \
         ! -name 'libutil.a' ! -name 'libanl.a' \
         ! -path '/usr/lib/gcc/*' \
         ! -path '/usr/lib64/gcc/*' \
         -delete \
   ```

2. Make the smoke test link against the stubs, so an image without them fails
   during the Docker build and never reaches a machine:

   ```dockerfile
    && printf '#include <iostream>\nint main(){std::cout << "cpp-ok\\n"; return 0;}\n' \
       | g++ -x c++ - -o /tmp/cpp_smoke -lpthread -ldl -lrt -lutil -lanl \
   ```

The stubs are a few bytes each, so the image size does not change noticeably.
The other static archives (`libc.a`, `libm.a`, ...) are still deleted. That is
fine unless you need fully static linking.

These changes have not been verified with a new image build yet.

## Quick Check on Any Image

```bash
echo 'int main(){return 0;}' | gcc -x c - -o /tmp/t -lpthread -ldl -lrt -lutil -lanl && echo ok
```

If this prints `ld: cannot find -l...`, the image has the problem described
above. Fix it with step 4.

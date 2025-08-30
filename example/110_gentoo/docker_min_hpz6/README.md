
# Current Build information

#64 195.2 Filesystem size 7664398.69 Kbytes (7484.76 Mbytes)
#64 195.2       36.94% of uncompressed filesystem size (20746320.61 Kbytes)

real    142m11.172s


# How to configure a kernel given the output of lsmod of another kernel

make LSMOD=~/lsmod.txt localmodconfig


# How to configure colemak keyboard layout in the linux command line

Configuring the Linux command line to use the Colemak keyboard layout
without a graphical interface involves a couple of steps: a temporary
change to test the layout and a permanent one to make it persist
across reboots. The commands and configuration files can vary slightly
depending on your Linux distribution.

## 1. Temporary Change (for the Current Session)

Before making any permanent changes, you can test the Colemak layout
in your current console session. This change will revert to the
default layout upon logging out or rebooting.

Open your terminal and type the following command:
```bash
loadkeys colemak
```

Your keyboard layout should now be set to Colemak for the current TTY
session. You can test this by typing in the console. To switch back to
the standard US QWERTY layout temporarily, you can use `loadkeys us`.

## 2. Persistent Change (to Survive Reboots)

To ensure the Colemak layout is active every time you boot your system
into the command line, you need to modify system configuration
files. The method for this varies by Linux distribution.


### For Gentoo with Systemd, Arch, Fedora, CentOS, and RHEL

Modern versions of Fedora, CentOS, and Red Hat Enterprise Linux also
use `systemd` and the `localectl` command for persistent keyboard
layout configuration in the console.

Use the `localectl` command to set the Colemak layout:
    ```bash
    sudo localectl set-keymap colemak
    ```

This will update the `/etc/vconsole.conf` file, making the Colemak
layout the default for all console sessions. You can check the current
settings with `localectl status`.

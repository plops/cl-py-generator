
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

### For Debian and Ubuntu-based Systems

The recommended way to configure the keyboard layout on Debian and its derivatives is by using the `dpkg-reconfigure` tool. This will guide you through a text-based interface.

1.  Run the following command with `sudo`:
    ```bash
    sudo dpkg-reconfigure keyboard-configuration
    ```
2.  A series of dialogs will appear. You'll be asked to select your keyboard model, country of origin, and then the keyboard layout.
3.  In the layout selection screen, choose "USA" and then "Colemak" from the list of variants.
4.  Follow the remaining prompts to complete the configuration.

This tool will automatically update the necessary configuration files, typically `/etc/default/keyboard`.

Alternatively, you can manually edit the `/etc/default/keyboard` file. Open it with a text editor like `nano`:
```bash
sudo nano /etc/default/keyboard
```
And modify the `XKBLAYOUT` line to:
```
XKBLAYOUT="colemak"
```
Save the file and exit. For the changes to take effect, you may need to run `sudo setupcon` or reboot.

##### For Arch Linux and Derivatives

On Arch Linux and systems that use `systemd-vconsole-setup`, the keyboard layout is configured in the `/etc/vconsole.conf` file. The easiest and recommended way to manage this is with the `localectl` utility.

1.  To set the Colemak keymap, execute the following command:
    ```bash
    sudo localectl set-keymap colemak
    ```

This command will automatically create or edit the `/etc/vconsole.conf` file and set the `KEYMAP` variable to `colemak`. You can verify the setting with `localectl status`.

##### For Fedora, CentOS, and RHEL

Modern versions of Fedora, CentOS, and Red Hat Enterprise Linux also use `systemd` and the `localectl` command for persistent keyboard layout configuration in the console.

1.  Use the `localectl` command to set the Colemak layout:
    ```bash
    sudo localectl set-keymap colemak
    ```

This will update the `/etc/vconsole.conf` file, making the Colemak layout the default for all console sessions. You can check the current settings with `localectl status`.

By following these instructions, you can successfully switch your Linux command-line environment to the Colemak keyboard layout.

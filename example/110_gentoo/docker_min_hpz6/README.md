
# Current Build information

#64 195.2 Filesystem size 7664398.69 Kbytes (7484.76 Mbytes)
#64 195.2       36.94% of uncompressed filesystem size (20746320.61 Kbytes)

real    142m11.172s


# How to configure a kernel given the output of lsmod of another kernel

make LSMOD=~/lsmod.txt localmodconfig


# How to configure colemak keyboard layout in the linux command line

Open your terminal and type the following command:
```bash
loadkeys /usr/share/keymaps/i386/colemak/en-latin9.map.gz 
```

Your keyboard layout should now be set to Colemak for the current TTY
session. You can test this by typing in the console. To switch back to
the standard US QWERTY layout temporarily, you can use `loadkeys us`.

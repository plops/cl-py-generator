# -vnc unix:/home/user/.qemu-vnc-socket
qemu-system-x86_64 \
-cpu host \
-enable-kvm \
-smp 15 \
-drive file=qemu/sda1.img,format=raw \
-m 8G \
-nographic
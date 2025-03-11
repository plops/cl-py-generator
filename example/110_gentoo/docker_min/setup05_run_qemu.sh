# -vnc unix:/home/user/.qemu-vnc-socket
# -cpu host 
qemu-system-x86_64 \
-cpu host \
-enable-kvm \
-drive file=qemu/sda1.img,format=raw \
-m 8G \
-nographic
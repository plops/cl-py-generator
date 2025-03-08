# -vnc unix:/home/user/.qemu-vnc-socket
# -cpu host 
qemu-system-x86_64 \
 -drive file=qemu/nvme0n1.img,format=raw \
 -m 8G
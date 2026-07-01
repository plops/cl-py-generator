docker save antigravity-sandbox:26.04 -o /dev/shm/antigravity-sandbox-env.tar
zstd /dev/shm/antigravity-sandbox-env.tar
# 555MB -> 159MB
# docker load -i antigravity-sandbox-env.tarx

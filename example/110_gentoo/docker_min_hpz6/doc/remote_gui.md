install sunshine on server
and moonlight-embedded on client (doesn't require qt)
sudo usermod -a -G input kiel # allow mouse access

connect client and server via tailscale (if you need vpn)

you can quit with C-M-S q (remember this!)
you can release mouse with C-M-S z

moonlight stream -platform sdl -windowed -1080 -app "Desktop" 100.110.241.38


configure the server to not send all 4 screens
emacs ~/.config/sunshine/sunshine.conf

dd_configuration_option = verify_only
controller = disabled
fec_percentage = 3
max_bitrate = 3000
min_threads = 3
minimum_fps_target = 12
nvenc_preset = 5
stream_audio = disabled


i can't move mouse yet
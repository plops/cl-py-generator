rfkill unblock wlan
iwctl -P 'wifipassword' station wlan0 connect mi
sudo iw dev wlan0 set power_save on

sudo iw dev wlan0 set txpower limit 300
sudo iw dev
sudo iw dev wlan0 link

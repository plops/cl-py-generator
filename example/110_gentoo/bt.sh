rfkill unblock bluetooth
#systemctl enable bluetooth
#systemctl start bluetooth

bluetoothctl power on
# get the id of your device with a scan
#bluetoothctl scan on

bluetoothctl pair 2C:FD:B3:A0:B8:44 
#Attempting to pair with 2C:FD:B3:A0:B8:44
#Failed to pair: org.bluez.Error.AlreadyExists
bluetoothctl trust 2C:FD:B3:A0:B8:44 
#Changing 2C:FD:B3:A0:B8:44 trust succeeded
bluetoothctl connect 2C:FD:B3:A0:B8:44 
#Attempting to connect to 2C:FD:B3:A0:B8:44

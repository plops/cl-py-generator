#!/bin/bash

# Path to the battery capacity file
BATTERY_PATH="/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:27/PNP0C09:00/PNP0C0A:00/power_supply/BAT0/capacity"

# Fetch battery percentage from the file
BATTERY_PERCENT=$(cat $BATTERY_PATH)

# Check if battery percentage is less than or equal to 20
if [ "$BATTERY_PERCENT" -le 20 ]; then
    # Beep for alert
    beep
    # Change background to solid red
    xsetroot -solid red
else
    # Set background to black when charge is >= 20%
    xsetroot -solid black
fi

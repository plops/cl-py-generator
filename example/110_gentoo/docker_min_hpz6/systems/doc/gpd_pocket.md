https://www.reddit.com/r/GPDPocket/comments/cfyekh/limit_battery_charge/

I have figured out a way to do it.

It is possible to configure the charger to stop charging at a specific voltage:

Write your desired charge stop voltage (in microvolts) to /sys/class/power_supply/bq24190-charger/constant_charge_voltage.

You can also set the charging current, so effectively creating a slow charge, if you want, or change charging to trickle charge at any time.

(Be careful to not write a 0 to online while playing around -- this will cut off all power from the machine, also resetting the CMOS clock and maybe some BIOS settings (I have not checked the latter), and it needs a few minutes until you can power it on again).

I made a script which shows the charging and battery related settings and makes it more easy to change them:

ix.io/1P4r (web.archive.org)

Since settings get re-set by kernel or firmware sometimes, an udev rule which (re)applies settings on power supply change may make user defined settings more persistent.

Example of the output of that script:

-- Battery: --
Percentage:           91%                  
Health:               Good.                
Cycles:               565                  
Voltage:              4.14/4.15            Avg/Now (V)
V. limits:            4.38/3.28 4.25/3.50  Max/Min Open_Circuit/Min_Design (V)
Current:              -.68/-.78            Avg/Now (A)
Power:                -2.81/-3.23          Avg/Now (W)
Charge:               6.98/6.67            Curr./Counter (Ah)
Ch. limits:           7.66/6.87            Full/Design (Ah)

-- Charger (Values with '(*)' can be changed by user, see option '-h'): --
Status:               Connected.           
Action:               Not charging.        
Health:               Over voltage.        
Charging type:        Fast.                (*)
Chg. end det. I:      .256 A               
Pre-/trickle chg. I:  .256 A               
Voltage limits:       4.000/4.400          Curr.(*)/Max (V)
Current limits:       4.544/4.544          Curr.(*)/Max (A)
Input current limit:  3.00 A               (*)

-- USB-C: --
Status:               Connected.           
Type:                 C [PD] PD_PPS        
Voltage:              12 12/12             Now Max/Min (V)
Negotiated Current:   2.0 2.0              Now Max (A)
Usage:
  To print all information, invoke without arguments:
    /usr/local/sbin/batt

  To get or set a single properties:
    /usr/local/sbin/batt <property> [<value>]

  (invokation without '<value>' prints the current setting)
  where '<property>' is one of:
  * input_cur -- sets the charger's input current limit in A
  * chg_cur   -- sets the charge current limit in A
  * chg_v     -- sets the charge voltage limit in V
  * chg_type  -- Set charge type. Supported: 'Trickle', 'Fast'.

  To get this help text:
    /usr/local/sbin/batt -h|-help|--help



#!/bin/bash

bat_dir="/sys/class/power_supply/max170xx_battery"
chg_dir="/sys/class/power_supply/bq24190-charger"
typec_dir="/sys/class/power_supply/tcpm-source-psy-i2c-fusb302"

toint() {
  printf "%.0f" "$1"
}

tomicro() {
  printf "%.0f" "$(bc <<<"${1}*1000000")"
}

frommicro() {
  if [ $# -ge 2 ]; then
    _digits="$2"
  else
    _digits='2'
  fi
  bc <<<"scale=${_digits}; ${1}/1000000"
}

msg() {
  printf '%s\n' "$@"
}

errmsg() {
  msg "$@" > /dev/stderr
}

msg_format_data() {
  printf '%-21s %-19s  %s\n' "$@"
}


printusage() {
  msg "Usage:"
  msg "  To print all information, invoke without arguments:"
  msg "    $0"
  msg ""
  msg "  To get or set a single properties:"
  msg "    $0 <property> [<value>]"
  msg ""
  msg "  (invokation without '<value>' prints the current setting)"
  msg "  where '<property>' is one of:"
  msg "  * input_cur -- sets the charger's input current limit in A"
  msg "  * chg_cur   -- sets the charge current limit in A"
  msg "  * chg_v     -- sets the charge voltage limit in V"
  msg "  * chg_type  -- Set charge type. Supported: 'Trickle', 'Fast'."
  msg ""
  msg "  To get this help text:"
  msg "    $0 -h|-help|--help"
}


# If arguments are supplied, modify settings and exit. Otherwise go through and print the information.
if [ $# -eq 1 ] || [ $# -eq 2 ]; then
  property="$1"
  if [ $# -eq 2 ]; then
    value="$2"
  fi
  case "${property}" in
    input_cur)
      if [ $# -eq 2 ]; then
        printf '%s' "$(tomicro "${value}")" > "${chg_dir}/input_current_limit"
      else
        msg "$(frommicro "$(<"${chg_dir}/input_current_limit")")"
      fi
      _exitcode=$?
    ;;
    chg_cur)
      if [ $# -eq 2 ]; then
        printf '%s' "$(tomicro "${value}")" > "${chg_dir}/constant_charge_current"
      else
        msg "$(frommicro "$(<"${chg_dir}/constant_charge_current")")"
      fi
      _exitcode=$?
    ;;
    chg_v)
      if [ $# -eq 2 ]; then
        printf '%s' "$(tomicro "${value}")" > "${chg_dir}/constant_charge_voltage"
      else
        msg "$(frommicro "$(<"${chg_dir}/constant_charge_voltage")")"
      fi
      _exitcode=$?
    ;;
    chg_type)
      if [ $# -eq 2 ]; then
        printf '%s' "${value}" > "${chg_dir}/charge_type"
      else
        msg "$(<"${chg_dir}/charge_type")"
      fi
      _exitcode=$?
    ;;
    '-h'|'--help'|'-help')
      printusage
    ;;
    *)
      errmsg "$0: Error: Argument '${property}' not supported."
      errmsg "Invoke with '-h' to get help."
      _exitcode=2
    ;;
  esac
  exit ${_exitcode}
elif [ $# -eq 0 ]; then
  true
else
  errmsg "$0: Error: Wrong amount of arguments ($#) specified."
  errmsg ""
  errmsg "$(printusage)"
  exit 1
fi

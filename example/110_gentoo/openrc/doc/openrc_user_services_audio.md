# OpenRC User Services for D-Bus / PipeWire / WirePlumber (E14)

This walkthrough records how missing sound in Firefox on the E14
(`kiel@e14`, Gentoo, OpenRC, X11 via `startx` + dwm) was diagnosed and fixed
on 2026-11-19, and how the per-user audio stack is meant to run as **OpenRC
user services**.

Context: the old `/etc` changes in the overlayfs upper layer had just been
moved away (`/run/overlayfs/etc_20261119`, visible as `/etc_20261119`), so
`/etc` was back to what the image ships.

## Target design

One owner per daemon. Nothing in `.xinitrc` or an autostart file may start the
audio stack a second time.

```text
system runlevel default
 ├─ user-runtime      creates /run/user/1000, (re)creates the user runlevel links
 └─ user.kiel         -> /etc/init.d/user (supervise-daemon + openrc-user kiel)
                         opens a PAM/elogind session (Service=openrc-user)
                         runs the *user* runlevel "default":
     ~/.config/rc/runlevels/default/
      ├─ dbus            -> /etc/user/init.d/dbus            (session bus at /run/user/1000/bus)
      ├─ pipewire        -> /etc/user/init.d/pipewire        (need dbus)
      ├─ pipewire-pulse  -> /etc/user/init.d/pipewire-pulse  (need pipewire dbus)
      └─ wireplumber     -> /etc/user/init.d/wireplumber     (need pipewire dbus)
```

Sources in this repo (installed by `openrc/Dockerfile`):

| Repo file                          | Installed as                               |
|------------------------------------|--------------------------------------------|
| `config/user-runtime.initd`        | `/etc/init.d/user-runtime`                 |
| `config/user-dbus.initd`           | `/etc/user/init.d/dbus`                    |
| `config/user-pipewire.initd`       | `/etc/user/init.d/pipewire`                |
| `config/user-pipewire-pulse.initd` | `/etc/user/init.d/pipewire-pulse`          |
| `config/user-wireplumber.initd`    | `/etc/user/init.d/wireplumber`             |
| `config/zz-openrc-user-session.sh` | `/etc/profile.d/zz-openrc-user-session.sh` |
| `config/xinitrc`                   | `/home/kiel/.xinitrc`                      |

`zz-openrc-user-session.sh` exports `XDG_RUNTIME_DIR=/run/user/<uid>`,
`DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus` and
`PULSE_SERVER=unix:$XDG_RUNTIME_DIR/pulse/native` for login shells, so X11
apps started from `startx` (Firefox among them) talk to the service-managed
bus and PulseAudio socket.

## Symptom

Firefox played audio (stream shown as active) but nothing was audible.

## Diagnosis

All commands run as root on the host (`ssh host`). User commands use
`su - kiel -c "export XDG_RUNTIME_DIR=/run/user/1000; ..."`.

### 1. Is the stack running, and who started it?

```bash
ps aux | grep -E 'pipewire|wireplumber|dbus-daemon' | grep -v grep
cat /proc/$(pgrep -u kiel -x wireplumber)/environ | tr '\0' '\n' | grep '^_='
```

`pipewire`, `pipewire-pulse` and `wireplumber` were running since boot, started
by `start-stop-daemon`, so by the OpenRC user services. There was also a
second session `dbus-daemon` from `dbus-launch --exit-with-session dwm`.

### 2. Permissions (ruled out)

```bash
id kiel                       # audio group present
ls -l /dev/snd; getfacl /dev/snd/controlC1   # user:kiel:rw- ACL from elogind
grep Groups /proc/$(pgrep -u kiel -x wireplumber)/status
```

Everything was fine: `kiel` was in the `audio` group and had the ACL.

### 3. What does PipeWire see?

```bash
su - kiel -c "export XDG_RUNTIME_DIR=/run/user/1000; wpctl status; pactl list short sinks"
```

```text
Audio
 ├─ Devices:
 │
 ├─ Sinks:
 │  *   55. Dummy Output
 └─ Streams:
        49. Firefox
             48. output_FL > Dummy Output:playback_FL [active]
```

**No ALSA devices at all.** Firefox was playing into `auto_null`
(Dummy Output). `aplay -l` showed the cards (`HD-Audio Generic` HDMI and
`CX11880 Analog`), so the kernel side was fine. WirePlumber had not created any
devices.

### 4. User service state

```bash
su - kiel -c "export XDG_RUNTIME_DIR=/run/user/1000; rc-status --user"
```

```text
 dbus             [  crashed  ]
 pipewire         [  started  ]
 pipewire-pulse   [  started  ]
 wireplumber      [  started  ]
```

`/run/user/1000/bus` did not exist. The user D-Bus had started at boot
(`/run/user/1000/openrc/rc.log` shows `Starting dbus ... [ ok ]` at 23:09:56)
and died at some point afterwards.

### 5. Run WirePlumber in the foreground with debug output

```bash
su - kiel -c "export XDG_RUNTIME_DIR=/run/user/1000
  pkill -u kiel -x wireplumber; sleep 1
  timeout 6 env WIREPLUMBER_DEBUG=3 wireplumber 2>&1 | grep -iE 'alsa|monitor|dbus|^W|^E'"
```

When started fresh (without a session bus), it logs
`optional component 'support.dbus' failed to load`, skips
`monitor.alsa.reserve-device`, and then finds and opens both cards. So the ALSA
monitor itself works.

Probable mechanism at boot (not proven): WirePlumber started while the user
D-Bus was still up, so it enabled ALSA **device reservation**
(`org.freedesktop.ReserveDevice1` on the session bus). Then the bus went away
and the reservation never finished, so the devices were never created.

## Fix (runtime)

```bash
su - kiel -c "export XDG_RUNTIME_DIR=/run/user/1000; rc-service --user dbus restart"
# dependents (pipewire, pipewire-pulse, wireplumber) are restarted automatically
```

Or restart the whole user session from root:

```bash
rc-service user.kiel restart
```

After that, `wpctl status` lists `Radeon High Definition Audio Controller` and
`Ryzen HD Audio Controller`, with the default sink
`Ryzen HD Audio Controller Speaker` (or `Headphones` when plugged in).
Running streams lose their connection when PipeWire restarts, so reload the
Firefox tab or restart Firefox.

## Fix (persistent)

### 1. `.xinitrc`: do not start a second stack or a second bus

The host's `~/.xinitrc` (persistent home, dated 2026-10-25) was still the old
one:

```sh
/usr/bin/gentoo-pipewire-launcher &
exec dbus-launch --exit-with-session dwm
```

That conflicts with the user services. `gentoo-pipewire-launcher` either
exits ("already running") or, if the services are slow, starts duplicate
daemons. `dbus-launch` creates a second, private session bus, so dwm children
such as Firefox end up on a different bus than PipeWire and WirePlumber.

The image's `config/xinitrc` already does the right thing:

```sh
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"
export PULSE_SERVER="${PULSE_SERVER:-unix:${XDG_RUNTIME_DIR}/pulse/native}"
...
if [ -S "${XDG_RUNTIME_DIR}/bus" ]; then
  exec dwm
else
  exec dbus-run-session dwm
fi
```

Installed on the host:

```bash
cp -a /home/kiel/.xinitrc /home/kiel/.xinitrc.20261119.bak
cat config/xinitrc | ssh host 'cat > /home/kiel/.xinitrc && chown kiel:kiel /home/kiel/.xinitrc'
```

Note: `/etc/xdg/autostart/pipewire.desktop` runs
`gentoo-pipewire-launcher restart`, which **kills** the service-managed
daemons. dwm ignores XDG autostart, so it is harmless here. With a desktop
environment that honours autostart, add a `Hidden=true` override in
`~/.config/autostart/pipewire.desktop`.

### 2. Log files for the user services

`dbus-daemon` ran with `--nosyslog` and the other services had no log, so a
crash left no trace. All four user init scripts now set:

```sh
output_log="${XDG_RUNTIME_DIR}/<service>.log"
error_log="${XDG_RUNTIME_DIR}/<service>.log"
```

That gives `/run/user/1000/{dbus,pipewire,pipewire-pulse,wireplumber}.log`
(tmpfs, so the logs last until reboot). They were changed in the repo
(`config/user-*.initd`) and copied to the host's `/etc/user/init.d/`.

## Checklist after the next reboot

```bash
ssh host
su - kiel -c "export XDG_RUNTIME_DIR=/run/user/1000; rc-status --user"   # all 'started', none 'crashed'
ls -l /run/user/1000/bus                                                   # socket exists
su - kiel -c "export XDG_RUNTIME_DIR=/run/user/1000; wpctl status" | sed -n '/Audio/,/Streams/p'
pgrep -a -u kiel dbus-daemon        # exactly one session bus (the --address=unix:path=/run/user/1000/bus one)
cat /run/user/1000/dbus.log /run/user/1000/wireplumber.log   # if dbus crashed again: the reason
```

If `dbus` shows `crashed` again, check `dbus.log` and the timing against
the tty1 login (`loginctl show-session <id> -p Timestamp`). On 2026-11-19 the
`openrc-user` session started at 23:09:56 and the tty1 `login` session at
23:10:03. Restarting `user.kiel` later on did **not** reproduce the crash, so
the cause is specific to boot.

## Useful commands

```bash
# user service control (as kiel, XDG_RUNTIME_DIR must be set)
rc-status --user
rc-service --user wireplumber restart
rc-update --user add|del <svc> default      # or edit ~/.config/rc/runlevels/default/

# whole user session (as root)
rc-service user.kiel restart

# openrc-user's own log for the user runlevel
cat /run/user/1000/openrc/rc.log
```

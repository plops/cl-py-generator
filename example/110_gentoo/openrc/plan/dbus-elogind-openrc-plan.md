# D-Bus Session Plan on OpenRC

## Goal

Make the user D-Bus session bus work reliably when using this login flow:

```sh
sudo ~/activate
~/start2
startx
```

The target outcome is that GUI applications started under X can talk to the user session bus through:

```sh
XDG_RUNTIME_DIR=/run/user/1000
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
```

and that `/run/user/1000/bus` actually exists.

## Current Problem

The shell environment points to a standard user runtime path:

```sh
XDG_RUNTIME_DIR=/run/user/1000
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
```

but `/run/user/1000` was not being created in the active startup path, so the bus socket did not exist and commands like:

```sh
dbus-monitor --session
```

failed.

## Important Architectural Decision

We are not currently trying to fully switch the machine to an `elogind`-managed login session model.

Instead, we are making the existing custom OpenRC-based startup path consistent:

1. Root/system setup should make sure `/run/user/1000` exists.
2. The user X session should export `XDG_RUNTIME_DIR=/run/user/1000`.
3. `dbus-run-session` should create the actual per-session user bus for X.

This keeps the fix aligned with the current workflow instead of requiring a broader login/session redesign.

## What Has Been Changed

### 1. `/home/kiel/activate`

This now starts `user-runtime` if that init script exists.

Purpose:

- Create `/run/user/1000` early in the root-side bring-up path.

### 2. `/home/kiel/start2`

This now explicitly creates:

```sh
/run/user/$(id -u)
```

with the correct ownership and permissions before `startx`.

Purpose:

- Guarantee that the runtime directory exists even if the OpenRC user-service path did not run.

### 3. `/home/kiel/.xinitrc`

This now exports:

```sh
XDG_RUNTIME_DIR=/run/user/$(id -u)
```

before starting the X session, and uses:

```sh
exec dbus-run-session -- dwm
```

Purpose:

- Ensure the X session sees a valid runtime directory.
- Start a clean session bus tied to the lifetime of the X session.

### 4. `/home/kiel/verify.sh`

This script verifies:

- `XDG_RUNTIME_DIR`
- `DBUS_SESSION_BUS_ADDRESS`
- ownership and existence of `/run/user/$UID`
- existence of `/run/user/$UID/bus`
- whether `dbus-monitor --session` can connect

## Expected Runtime Sequence

1. Log into the Linux console as `kiel`.
2. Run:

```sh
sudo ~/activate
```

3. Run:

```sh
~/start2
```

4. Run:

```sh
startx
```

5. Inside X, `dbus-run-session` should create the session bus at:

```sh
/run/user/1000/bus
```

## Verification

After X is up, run:

```sh
~/verify.sh
```

Success means:

- `/run/user/1000` exists
- `/run/user/1000/bus` exists
- `dbus-monitor --session` reports the session bus is reachable

## If It Still Fails

The next likely causes are:

1. A shell init file exports a stale `DBUS_SESSION_BUS_ADDRESS` before `startx`.
2. `/run/user/1000` exists but has wrong ownership or permissions.
3. Some process started before `dbus-run-session` is expecting a different session bus model.
4. The custom OpenRC user-service stack and the ad hoc X-session stack are still partially overlapping.

## Next Debugging Step If Needed

If verification still fails, inspect:

- `~/.profile`
- `~/.bash_profile`
- `~/.bashrc`
- any script that exports `DBUS_SESSION_BUS_ADDRESS`
- any script that starts PipeWire, WirePlumber, or user OpenRC services before `startx`

## Longer-Term Cleanup Option

If this setup needs to become simpler and more robust later, pick one model and use it consistently:

1. `elogind` + PAM + `dbus-run-session`
2. custom OpenRC user runtime + custom user services + no mixed session assumptions

Right now the short-term objective is narrower:

Make the existing console -> `activate` -> `start2` -> `startx` path produce a working user D-Bus session.

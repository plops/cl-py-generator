## Summary of the Dracut init script

In essence, this script is a highly configurable, modular, and event-driven system for setting up the very early user-space environment, finding and mounting the root filesystem, and then handing control over to the real init system of the operating system. The use of hooks allows for great extensibility and customization.


*   **Setup and Environment:**
    *   Saves the current environment variables.
    *   Creates a `/sysroot` directory (this will become the new root).
    *   Resets the `PATH` to a minimal, safe set of directories.

*   **Mounts Essential Filesystems:**
    *   Mounts `/proc` (process information) - *critical for system operation*.  Fails if `/proc` can't be mounted.
    *   Mounts `/sys` (kernel and device information) - *critical for system operation*. Fails if `/sys` can't be mounted.
    *   Mounts `devtmpfs` on `/dev` (dynamic device nodes) - *critical for device access*.  Fails if `devtmpfs` can't be mounted.
    *   Creates symbolic links in `/dev` for `fd`, `stdin`, `stdout`, and `stderr`.
    *   Mounts `devpts` on `/dev/pts` (pseudo-terminal support).
    *   Mounts `tmpfs` on `/dev/shm` (shared memory).
    *   Mounts `tmpfs` on `/run` (runtime data), handling cases where the initramfs itself is located in `/run` to avoid `noexec` issues.  Copies existing `/run` contents.
    *  Creates device nodes using information from `kmod static-nodes`, if available.

*   **Error Handling and Debugging:**
    *   Sets up a trap to call `emergency_shell` if a signal is caught.
    *   Optionally configures detailed logging using `loginit` (if `RD_DEBUG` is set).
    *   Redirects standard input, output, and error to `/dev/console`.

*   **Configuration and Command Line Parsing:**
    *   Reads initramfs release information.
    *   Sources configuration files from `/etc/conf.d`.
    *   Allows the user to enter additional kernel command-line parameters interactively (if `rd.cmdline=ask` is specified).
    *   Handles the `rd.hostonly` parameter, potentially removing files not specific to the host.
    *   Parses kernel command line arguments using `getarg` and `getargbool`. This is *crucially important* for how Dracut behaves, as it's how options are passed from the bootloader to the initramfs.
    * Calls hook `cmdline`.

*   **Root Filesystem Determination:**
    *   Checks for the `root=` argument on the kernel command line.  Exits if it's missing or empty.  This is the *core* of identifying the target root filesystem.
    *   Sets shell variables (`root`, `rflags`, `fstype`, `netroot`, `NEWROOT`) based on the `root=` argument and other parsed parameters.

*   **Pre-udev Setup:**
    * Calls hook `pre-udev`
    *   Runs scripts in the `pre-udev` hook directory.  These scripts run *before* udev starts.

*   **udev Initialization:**
    *   Starts the `systemd-udevd` daemon (udev is the device manager).
    *   Sets udev logging level based on kernel parameters (`rd.udev.info`, `rd.udev.debug`).
    * Calls hook `pre-trigger`.
    *   Runs scripts in the `pre-trigger` hook directory.

*   **udev Triggering and Event Handling:**
    *   Reloads udev rules.
    *   Triggers udev events to populate `/dev` with device nodes.
    * Calls hook `initqueue`.
    *   Enters a loop that handles udev events and runs scripts from the `initqueue` hook directories:
        *   `initqueue/*.sh`:  Scripts run on each iteration of the loop.
        *   `initqueue/settled/*.sh`: Scripts run after udev has settled (no more events).
        *   `initqueue/timeout/*.sh`: Scripts run after a timeout period, used for handling devices that take a long time to appear.
    *   The loop continues until the root filesystem is successfully mounted (checked by `check_finished`) or a timeout is reached.  If the timeout is reached, an emergency shell is invoked.

*   **Pre-Mount and Mount:**
    * Calls hook `pre-mount`
    *   Runs scripts in the `pre-mount` hook directory *before* attempting to mount the root filesystem.
    * Calls hook `mount`.
    *   Enters a loop that tries to mount the root filesystem:
        *   Executes scripts in the `mount` hook directory.  These scripts are responsible for *actually mounting* the root filesystem.  Different scripts handle different filesystem types, network mounts, encrypted volumes, etc.
        *   If a mount script succeeds (and `usable_root` returns true), the loop breaks.
        *   If mounting fails repeatedly, an emergency shell is invoked.
        *   Prints a message indicating the mounted root filesystem.

*   **Pre-Pivot and Cleanup:**
    * Calls hook `pre-pivot`.
    *   Runs scripts in the `pre-pivot` hook directory *just before* switching to the new root.
    * Calls hook `cleanup`.
    *   Runs scripts in the `cleanup` hook directory, also just before switching root.

* **Martin's init script:**
    * Custom mounting of partitions and an overlay filesystem.

*   **Finding Init and Final Steps:**
    *   Searches for the init program (usually `/sbin/init`, but can be overridden by kernel parameters).
    *   If init is not found, displays an error message and enters an emergency shell.
    *   Exits udev.
    *   Cleans up the environment by unsetting most variables.
    *   Restores the original environment variables saved at the beginning.
    *   Parses the kernel command line again to extract arguments to be passed to the init program.
    *   Mounts `/run/initramfs` to `/dev/.initramfs` if `/run` does not exist.
    *   Waits for loginit to exit.
    *   Removes the symlink `/dev/root`.

*   **Switching Root:**
    *   Calls `emergency_shell` if the `rd.break` parameter is set.
    *   Uses either `capsh` (if available and configured via `/etc/capsdrop`) or `switch_root` to:
        *   Change the root filesystem to `/sysroot` (the `NEWROOT`).
        *   Execute the chosen init program (`$INIT`) with the extracted arguments.
    *   If `switch_root` fails, enters an emergency shell.


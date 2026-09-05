# Scanner USB NIC Setup

This note records the commands used on 2026-04-28 to move the scanner subnet to the USB Ethernet adapter `enp133s0u2u2c2`, verify reachability to the scanner at `192.168.254.131`, and confirm that `scanner_cli` can connect.

## Goal

Use the USB adapter `enp133s0u2u2c2` for the scanner network with the static host address `192.168.254.123/24`.

## One-time live reconfiguration

These commands remove the scanner IP from the old interface, assign it to the USB adapter, and bring the USB adapter up.

```bash
sudo ip addr del 192.168.254.123/24 dev enp65s0
sudo ip addr replace 192.168.254.123/24 dev enp133s0u2u2c2
sudo ip link set dev enp133s0u2u2c2 up
```

## Verify routing and connectivity

The route lookup should show the scanner IP going directly over `enp133s0u2u2c2` with source address `192.168.254.123`.

```bash
ip route get 192.168.254.131
ping -c 3 -W 1 192.168.254.131
arping -c 3 -I enp133s0u2u2c2 192.168.254.131
```

Expected route shape:

```text
192.168.254.131 dev enp133s0u2u2c2 src 192.168.254.123
```

Expected connectivity result:

- `ping` returns replies from `192.168.254.131`
- `arping` returns replies from the scanner MAC

Observed ARP reply during verification:

```text
A4:DB:01:02:03:04
```

## Run scanner_cli

Start the CLI from the build directory:

```bash
cd /home/kiel/stage/ls-laserscanner-sw/build/linux-cuda-debug/apps/scanner_cli
./scanner_cli --raw --8bit
```

## Successful connection indicators

During the successful test run, `scanner_cli` reported these key states:

```text
Connecting to 192.168.254.131:1028/1024
Scanner connection state changed to: streamConnected
Scanner state changed to: ready
Press any key to stop the tool
```

## Troubleshooting

If the route is wrong, check whether the USB adapter is still up and still owns `192.168.254.123/24`:

```bash
ip -br a
ip route get 192.168.254.131
```

If `scanner_cli` fails before connecting, verify network reachability first:

```bash
ping -c 3 192.168.254.131
arping -c 3 -I enp133s0u2u2c2 192.168.254.131
```

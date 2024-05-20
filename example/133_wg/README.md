
create a python script that takes a list of N hosts ip's or ip-patterns
(first one will be server and must be full ip) and generates a shell
script that performs the wireguard configuration for each of the
computers.

execution levels: level 0 happens on my laptop, level 1 happens on the
server or client.

in level 0 the private and public keys for all computers are
generated. the N scripts can be copied to the computers where they will write the config file 
and configure network devices and systemctl to start wireguard when linux boots

all computers are running linux.

here are the instructions from wg documentation. use this to define the scripts:

Command-line Interface
A new interface can be added via ip-link(8), which should automatically handle module loading:

# ip link add dev wg0 type wireguard
(Non-Linux users will instead write wireguard-go wg0.)

An IP address and peer can be assigned with ifconfig(8) or ip-address(8)

# ip address add dev wg0 192.168.2.1/24
Or, if there are only two peers total, something like this might be more desirable:

# ip address add dev wg0 192.168.2.1 peer 192.168.2.2
The interface can be configured with keys and peer endpoints with the included wg(8) utility:

# wg setconf wg0 myconfig.conf
or

# wg set wg0 listen-port 51820 private-key /path/to/private-key peer ABCDEF... allowed-ips 192.168.88.0/24 endpoint 209.202.254.14:8172
Finally, the interface can then be activated with ifconfig(8) or ip-link(8):

# ip link set up dev wg0
There are also the wg show and wg showconf commands, for viewing the current configuration. Calling wg with no arguments defaults to calling wg show on all WireGuard interfaces.

￼

Consult the man page of wg(8) for more information.

Much of the routine bring-up and tear-down dance of wg(8) and ip(8) can be automated by the included wg-quick(8) tool:

￼

Key Generation
WireGuard requires base64-encoded public and private keys. These can be generated using the wg(8) utility:

$ umask 077
$ wg genkey > privatekey
This will create privatekey on stdout containing a new private key.

You can then derive your public key from your private key:

$ wg pubkey < privatekey > publickey
This will read privatekey from stdin and write the corresponding public key to publickey on stdout.

Of course, you can do this all at once:

$ wg genkey | tee privatekey | wg pubkey > publickey




```python
import os
import argparse

def generate_keys(hostname):
    """Generates private and public keys for a host."""
    os.system(f"umask 077; wg genkey | tee {hostname}_privatekey | wg pubkey > {hostname}_publickey")

def generate_server_config(server_ip, server_private_key, clients):
    """Generates WireGuard configuration for the server."""
    config = f"""[Interface]
PrivateKey = {server_private_key}
Address = {server_ip}/24
ListenPort = 51820

"""
    for client_ip, client_public_key in clients.items():
        config += f"""[Peer]
PublicKey = {client_public_key}
AllowedIPs = {client_ip}/32

"""
    return config

def generate_client_config(client_ip, client_private_key, server_ip, server_public_key):
    """Generates WireGuard configuration for a client."""
    config = f"""[Interface]
PrivateKey = {client_private_key}
Address = {client_ip}/24

[Peer]
PublicKey = {server_public_key}
Endpoint = {server_ip}:51820
AllowedIPs = 0.0.0.0/0
"""
    return config

def generate_setup_script(hostname, config, is_server):
    """Generates a shell script for setting up WireGuard on a host."""
    script = f"""#!/bin/bash

# Install WireGuard if not already installed
if ! command -v wg &> /dev/null; then
    echo "WireGuard is not installed. Installing..."
    # Install WireGuard using your distribution's package manager
    # For example, on Debian-based systems:
    sudo apt update
    sudo apt install wireguard -y
fi

# Create the WireGuard interface
sudo ip link add dev wg0 type wireguard

# Configure the WireGuard interface
echo '{config}' | sudo tee /etc/wireguard/wg0.conf

# Set the interface IP address
sudo ip address add {hostname}/24 dev wg0

# Bring up the WireGuard interface
sudo ip link set up dev wg0

"""
    if is_server:
        script += "# Enable IP forwarding\n"
        script += "sudo sysctl -w net.ipv4.ip_forward=1\n"
    
    script += """
# Enable WireGuard service on boot
sudo systemctl enable wg-quick@wg0
"""
    return script

def main():
    """Main function for generating WireGuard configuration."""
    parser = argparse.ArgumentParser(description="Generate WireGuard configuration for multiple hosts.")
    parser.add_argument("hosts", metavar="host_ip", type=str, nargs="+",
                        help="IP addresses or IP patterns of the hosts (first one is the server)")
    args = parser.parse_args()

    hosts = args.hosts
    server_ip = hosts[0]
    clients = {}

    # Generate keys for all hosts
    for i, host in enumerate(hosts):
        hostname = f"host{i+1}"
        generate_keys(hostname)
        if i > 0:
            clients[host] = open(f"{hostname}_publickey").read().strip()

    # Generate server configuration
    server_private_key = open(f"host1_privatekey").read().strip()
    server_config = generate_server_config(server_ip, server_private_key, clients)
    with open("server_wg0.conf", "w") as f:
        f.write(server_config)
    server_setup_script = generate_setup_script("host1", server_config, True)
    with open("server_setup.sh", "w") as f:
        f.write(server_setup_script)

    # Generate client configurations and scripts
    for i, (client_ip, client_public_key) in enumerate(clients.items()):
        hostname = f"host{i+2}"
        client_private_key = open(f"{hostname}_privatekey").read().strip()
        client_config = generate_client_config(client_ip, client_private_key, server_ip, client_public_key)
        with open(f"{hostname}_wg0.conf", "w") as f:
            f.write(client_config)
        client_setup_script = generate_setup_script(hostname, client_config, False)
        with open(f"{hostname}_setup.sh", "w") as f:
            f.write(client_setup_script)

if __name__ == "__main__":
    main()
```

**How to use this script:**

1. **Save the script** as a Python file (e.g., `wg_config_generator.py`).
2. **Run the script with host IPs as arguments:**
   ```bash
   python wg_config_generator.py 192.168.1.1 192.168.1.2 192.168.1.3
   ```
   Replace the IP addresses with your actual host IPs. The first IP is the server's IP.

The script will generate the same output files as before, using the IP addresses provided as arguments. 

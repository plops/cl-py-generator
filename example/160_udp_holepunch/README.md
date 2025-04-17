|       |                             |
| gen01 | too complex                 |
| gen02 | just execute itself via ssh |
| gen03 | send packets (2x)           |
| gen04 | send many packets           |
| gen05 | determine ip addresses      |
|       |                             |

https://gist.github.com/gordol/d3f66e5fa0a53a9ccef30b77885042be


symmetric nat on both sides of the connection

basically random ports go through internet
maybe i could try many ports with several servers in order to
try and figure out the underlying assignment pattern. but it
sounds daunting

# tcpdump for udp

```
sudo tcpdump -nn -i any udp
sudo tcpdump -nn -i eth0 udp

-nn: don't resolve IP addresses to host names and port numbers to service names
```

# summary of RPC ont NATs
https://www.rfc-editor.org/rfc/rfc4787.txt

REQ-1: NATs MUST use "Endpoint-Independent Mapping", meaning the same external IP:port is reused for subsequent packets from the same internal IP:port, regardless of the destination.

REQ-3: NATs MUST NOT use "Port overloading" (forcing port preservation even on collision). It is RECOMMENDED to preserve the port range (0-1023 vs 1024-65535) if possible.

REQ-4: It is RECOMMENDED that NATs preserve port parity (mapping even internal ports to even external ports, and odd to odd) to aid RTP/RTCP compatibility with older implementations.

REQ-5: UDP mapping timers MUST NOT expire in less than 2 minutes (default RECOMMENDED >= 5 minutes). Shorter timers MAY be used for specific well-known ports. Timer MAY be configurable.

REQ-6: Mappings MUST be refreshed by outbound traffic. Mappings MAY be refreshed by inbound traffic (with security caveats).

REQ-8: Recommends "Endpoint-Independent Filtering" (allows inbound from any source once mapping exists) for transparency, or "Address-Dependent Filtering" (allows inbound only from IPs previously sent to) for stricter security. Filtering MAY be configurable.


REQ-9: NATs MUST support "Hairpinning", allowing two internal hosts to communicate via their external mapped addresses. The source address presented to the receiving internal host MUST be the external mapped address of the sending internal host.

REQ-10: ALGs for UDP protocols SHOULD be turned off by default to avoid interference with modern NAT traversal techniques. It is RECOMMENDED that ALGs be individually configurable by the administrator.

REQ-11: NATs MUST behave deterministically; mapping and filtering rules MUST NOT change based on load or conflict situations without explicit configuration changes.

REQ-12: Receiving ICMP messages MUST NOT terminate a UDP mapping. NATs SHOULD NOT filter incoming ICMP based on source IP by default. It is RECOMMENDED that NATs process and forward ICMP Destination Unreachable messages.



# stun tool
https://sourceforge.net/projects/stun
```
apt install stun

```

## hetzner
```
sudo stun stun.1und1.de:3478
STUN client version 0.97
Primary: Open
Return value is 0x000001
```

The machine running the test has a public IP address directly assigned
or is behind a very simple firewall that doesn't perform NAT
address/port translation. Port preservation is indicated (which is
expected when there's no port translation). This environment is ideal
for P2P communication as the host is directly reachable.


## azure
```
sudo stun stun.1und1.de:3478
STUN client version 0.97
Primary: Independent Mapping, Port Dependent Filter, preserves ports, will hairpin
Return value is 0x000007
```

Azure is using a Port Restricted Cone NAT.
        *   *Independent Mapping:* The same internal IP:Port maps to the same external IP:Port regardless of destination.
        *   *Port Dependent Filter:* Incoming packets are only allowed if they come from the *exact* IP address *and port* that the client previously sent a packet to.
        *   *Preserves ports:* The NAT tries to use the same external port as the internal source port. This is helpful but not guaranteed.
        *   *Will hairpin:* Allows communication between two clients behind the same Azure NAT by using their public IP addresses.
        *   This is a relatively common and moderately P2P-friendly NAT type. UDP hole punching can usually work.


## enterprise
```
sudo stun stun.1und1.de:3478
STUN client version 0.97
Primary: Independent Mapping, Port Dependent Filter, preserves ports, no hairpin
Return value is 0x000017
```

The enterprise network uses the same type of NAT as Azure (Port
Restricted Cone) regarding mapping and filtering, and it also tries to
preserve ports. However, it explicitly *disables hairpinning*. This
means two clients inside the enterprise network cannot directly
communicate using their STUN-discovered public IP addresses; the
enterprise firewall/NAT prevents this loopback traffic. P2P with
external peers is possible (same as Azure), but internal P2P via the
public mapping is blocked.


## mobile hotspot (iphone)
```
sudo ./client stun.1und1.de:3478
STUN client version 0.97
Primary: Dependent Mapping, random port, no hairpin
Return value is 0x000018
```

This indicates a Symmetric NAT, which is common for mobile carriers (often part of Carrier-Grade NAT - CGNAT).
        *   *Dependent Mapping:* The external IP:Port mapping depends on the destination IP:Port. Sending from the same internal socket to two different servers will result in two different external source IP:Ports being used by the NAT.
        *   *Random port:* The NAT assigns unpredictable external ports.
        *   *No hairpin:* Loopback is blocked.
        *   This is the most restrictive type of NAT for P2P communication. UDP hole punching is often very difficult or impossible because predicting the port mapping used for the peer is extremely hard. Applications often need to rely on a relay server (like TURN) in this scenario.



# Decoding the Output Components

Based on the `client.cxx` source code, we can decode the output:

*   **NAT Type (Base Value):** The core NAT/Firewall type determines a base hexadecimal value (`retval[nic]` in the code):
    *   `0x00`: Open Internet (No NAT or basic firewall)
    *   `0x02`: Independent Mapping, Independent Filter (Full Cone NAT)
    *   `0x04`: Independent Mapping, Address Dependent Filter (Restricted Cone NAT)
    *   `0x06`: Independent Mapping, Port Dependent Filter (Port Restricted Cone NAT)
    *   `0x08`: Dependent Mapping (Symmetric NAT)
    *   `0x0A`: Firewall (Blocks incoming unsolicited, but mapping is like Open)
    *   `0x0C`: Blocked (Cannot reach STUN server or test fails)
    *   `0x0E`: Unknown NAT type (Test results ambiguous)
    *   `-1` (0xFFFFFFFF): Failure (Error during test execution)

*   **Flags (Bitwise ORed with Base Value):**
    *   `preserves ports`: If the NAT attempts to keep the same source port number for the external mapping, bit 0 (`0x01`) is *set*. If it assigns a `random port`, bit 0 is *cleared*.
    *   `no hairpin`: If the NAT *does not* allow a client to send packets to its own external mapped address and receive them back, bit 4 (`0x10`) is *set*. If it `will hairpin`, bit 4 is *cleared*.

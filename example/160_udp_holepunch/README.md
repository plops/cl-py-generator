|       |                             |
| gen01 | too complex                 |
| gen02 | just execute itself via ssh |
| gen03 | send packets (2x)           |
| gen04 | send many packets           |
|       |                             |

https://gist.github.com/gordol/d3f66e5fa0a53a9ccef30b77885042be


symmetric nat on both sides of the connection

basically random ports go through internet
maybe i could try many ports with several servers in order to
try and figure out the underlying assignment pattern. but it
sounds daunting
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


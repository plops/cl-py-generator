# this is only needed if the generateai google server must run in US and the website host is outside of US
# tinyproxy allows to route the transaction via SSH socks5 proxy to an exit host in the US
tinyproxy -d -c tinyproxy.conf

# tinyproxy is only needed if the generateai google server must run in
# US and the website host is outside of US tinyproxy allows to route
# the transaction via SSH socks5 proxy to an exit host in the US

# tinyproxy is not needed if the hosting also happens on a US server
# (or when google rolls out AI in the rest of the world)
tinyproxy -d -c tinyproxy.conf



idf_component_register(
    SRCS
        "main.cpp"
	"Wifi.cpp"
        "TcpConnection.cpp"
	"Ransac.cpp"
	"Display.cpp"
	"Uart.cpp"
	"BmeSensor.cpp"
	"Graph.cpp"
	"data.pb.c"

   INCLUDE_DIRS
        "." "include"
)

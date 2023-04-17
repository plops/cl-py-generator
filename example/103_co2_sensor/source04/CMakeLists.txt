
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
        "."
	"include"
	"/home/martin/src/nanopb/"
)

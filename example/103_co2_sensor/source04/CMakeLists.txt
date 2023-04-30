
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-g3 -Wall -Wextra -Wconversion -Wdouble-promotion -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -fsanitize=undefined -fsanitize-trap=undefined -fsanitize=address)
endif()
if(CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(-Wall -Wextra -Wconversion -Wdouble-promotion -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -fsanitize=undefined )
endif()
# -fsanitize-trap=undefined


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
		/home/martin/src/nanopb/pb_decode.c	
			/home/martin/src/nanopb/pb_encode.c	
				/home/martin/src/nanopb/pb_common.c

   INCLUDE_DIRS
        "."
	"include"
	"/home/martin/src/nanopb/"
)

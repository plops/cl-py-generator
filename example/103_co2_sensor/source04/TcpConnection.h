#ifndef TCPCONNECTION_H
#define TCPCONNECTION_H

#include <pb.h>

class TcpConnection  {
        public:
        static bool read_callback (pb_istream_t* stream, uint8_t* buf, size_t count)       ;  
        static bool write_callback (pb_ostream_t* stream, const pb_byte_t* buf, size_t count)       ;  
        void set_socket_timeout (int fd, float timeout_seconds)       ;  
        pb_istream_t pb_istream_from_socket (int fd)       ;  
        pb_ostream_t pb_ostream_from_socket (int fd)       ;  
        void send_data (float pressure, float humidity, float temperature, float co2_concentration)       ;  
        explicit  TcpConnection ()       ;  
};

#endif /* !TCPCONNECTION_H */
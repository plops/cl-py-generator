#ifndef OFDMTRANSMITTER_H
#define OFDMTRANSMITTER_H

#include "OfdmConstants.h"
#include <vector>

class OfdmTransmitter  {
        public:
        explicit  OfdmTransmitter ()       ;  
        std::vector<Cplx> transmit (const std::vector<Cplx>& data)       ;  
        private:
        static std::vector<Cplx> generatePreamble ()       ;  
};

#endif /* !OFDMTRANSMITTER_H */
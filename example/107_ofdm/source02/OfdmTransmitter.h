#ifndef OFDMTRANSMITTER_H
#define OFDMTRANSMITTER_H

#include "OfdmConstants.h"

class OfdmTransmitter  {
        public:
        explicit  OfdmTransmitter ()       ;  
        std::vector<Cplx> transmit (const std::vector<Cplx>& data)       ;  
        private:
        std::vector<Cplx> generatePreamble ()       ;  
};

#endif /* !OFDMTRANSMITTER_H */
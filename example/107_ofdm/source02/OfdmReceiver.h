#ifndef OFDMRECEIVER_H
#define OFDMRECEIVER_H

#include "OfdmConstants.h"
#include <vector>

class OfdmReceiver  {
        public:
        explicit  OfdmReceiver ()       ;  
        std::vector<Cplx> receive (const std::vector<Cplx>& receivedData)       ;  
        private:
        static size_t schmidlCoxSynchronization (const std::vector<Cplx>& receivedData)       ;  
};

#endif /* !OFDMRECEIVER_H */
#ifndef EPICORRSENDEPIMSGGADGET_H
#define EPICORRSENDEPIMSGGADGET_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "hoArmadillo.h"
#include "gadgetron_epi_export.h"
#include "SMS_utils.h"
#include <ismrmrd.h>
#include "xml.h"
#include <complex>

#define _USE_MATH_DEFINES

#include <math.h>

namespace Gadgetron {

    class EXPORTGADGETS_EPI EPICorrSendEPIMsgGadget :
            public Gadget2<ISMRMRD::AcquisitionHeader, hoNDArray<std::complex<float> > > {
    public:
        EPICorrSendEPIMsgGadget();

        virtual ~EPICorrSendEPIMsgGadget();

    protected:
        virtual int process_config(ACE_Message_Block *mb);

        virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
                            GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2);
    };
}
#endif //EPICORRSENDEPIMSGGADGET_H

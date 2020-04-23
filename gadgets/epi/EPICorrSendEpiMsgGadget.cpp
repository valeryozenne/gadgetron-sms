//    TO DO: - For 3D sequences (E2>1), the 3 navigators should be equivalent for all e2 partition
//             encoding steps (other than the mean phase).  So we should average across them.  I guess
//             one way to do so would be to include all partition encoding steps (that have been acquired
//             up to that one), also from the previous repetitions, in the robust fit, being careful with
//             the column representing the slope.  The problem with that is that the matrix to invert is
//             longer, so it could take longer to compute.
//           - Test the case that more repetitions are sent than the number specified in the xml header.

#include "EPICorrSendEpiMsgGadget.h"
#include "ismrmrd/xml.h"
#include "hoNDArray_fileio.h"

namespace Gadgetron {

#define OE_PHASE_CORR_POLY_ORDER 4

    EPICorrSendEPIMsgGadget::EPICorrSendEPIMsgGadget() {}

    EPICorrSendEPIMsgGadget::~EPICorrSendEPIMsgGadget() {}

    int EPICorrSendEPIMsgGadget::process_config(ACE_Message_Block *mb) {
        ISMRMRD::IsmrmrdHeader h;
        ISMRMRD::deserialize(mb->rd_ptr(), h);

        if (h.encoding.size() == 0) {
            GDEBUG("Number of encoding spaces: %d\n", h.encoding.size());
            GDEBUG("This Gadget needs an encoding description\n");
            return GADGET_FAIL;
        }

        GDEBUG_STREAM("EPICorrSendEPIMsgGadget configured");
        return 0;
    }

    int EPICorrSendEPIMsgGadget::process(
            GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
            GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2) {

        GadgetContainerMessage<t_EPICorrection> *messageEPI = new GadgetContainerMessage<t_EPICorrection>();

        this->next()->putq(m1);
        this->next()->putq(messageEPI);
        return 0;
    }

    GADGET_FACTORY_DECLARE(EPICorrSendEPIMsgGadget)
}

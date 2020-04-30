//    TO DO: - For 3D sequences (E2>1), the 3 navigators should be equivalent for all e2 partition
//             encoding steps (other than the mean phase).  So we should average across them.  I guess
//             one way to do so would be to include all partition encoding steps (that have been acquired
//             up to that one), also from the previous repetitions, in the robust fit, being careful with
//             the column representing the slope.  The problem with that is that the matrix to invert is
//             longer, so it could take longer to compute.
//           - Test the case that more repetitions are sent than the number specified in the xml header.

#include "GenericReconCartesianReceiveEpiMsgGadget.h"
#include "ismrmrd/xml.h"
#include "hoNDArray_fileio.h"



namespace Gadgetron {

#define OE_PHASE_CORR_POLY_ORDER 4

    GenericReconCartesianReceiveEpiMsgGadget::GenericReconCartesianReceiveEpiMsgGadget() {}

    GenericReconCartesianReceiveEpiMsgGadget::~GenericReconCartesianReceiveEpiMsgGadget() {}

    int GenericReconCartesianReceiveEpiMsgGadget::process_config(ACE_Message_Block *mb) {
        ISMRMRD::IsmrmrdHeader h;
        ISMRMRD::deserialize(mb->rd_ptr(), h);

        if (h.encoding.size() == 0) {
            GDEBUG("Number of encoding spaces: %d\n", h.encoding.size());
            GDEBUG("This Gadget needs an encoding description\n");
            return GADGET_FAIL;
        }

        GDEBUG_STREAM("GenericReconCartesianReceiveEpiMsgGadget configured");
        return 0;
    }

    int GenericReconCartesianReceiveEpiMsgGadget::process(
            Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1) {


        GDEBUG("##ReconData obtained##\n");

        //this->next()->putq(m1);
        this->next()->putq(m1);
		return (0);
    }

    int GenericReconCartesianReceiveEpiMsgGadget::process(
            GadgetContainerMessage<t_EPICorrection> *m1) {


        GDEBUG_STREAM("Message obtained" << std::endl);

        //this->next()->putq(m1);
        //this->next()->putq(m2);
		return (0);
    }

    GADGET_FACTORY_DECLARE(GenericReconCartesianReceiveEpiMsgGadget)
}

#ifndef GENERICRECONARTESIANRECEIVEEPIMSGGADGET_H
#define GENERICRECONARTESIANRECEIVEEPIMSGGADGET_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "hoArmadillo.h"
#include "gadgetron_smscore_export.h"
#include "SMS_utils.h"
#include <ismrmrd/ismrmrd.h>
#include "ismrmrd/xml.h"
#include <complex>
#include "mri_core_data.h"

#define _USE_MATH_DEFINES

#include <math.h>

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconCartesianReceiveEpiMsgGadget :
            public Gadget1Of2<IsmrmrdReconData, t_EPICorrection> {
    public:
        GenericReconCartesianReceiveEpiMsgGadget();

        virtual ~GenericReconCartesianReceiveEpiMsgGadget();

    protected:
        

        virtual int process_config(ACE_Message_Block *mb);

        virtual int process(
            GadgetContainerMessage<IsmrmrdReconData> *m2);

        virtual int process(
            GadgetContainerMessage<t_EPICorrection > *m2);
    };
}
#endif //GENERICRECONARTESIANRECEIVEEPIMSGGADGET_H

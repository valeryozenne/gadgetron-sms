/** \file   GenericReconSMSBase.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to prepare the reference data, working on the IsmrmrdReconData.
    \author Hui Xue
*/

#pragma once

#include "GenericReconBase.h"
#include "gadgetron_smscore_export.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconSMSBase : public GenericReconDataBase
    {
    public:
        GADGET_DECLARE(GenericReconSMSBase);

        typedef GenericReconDataBase BaseClass;

        GenericReconSMSBase();
        ~GenericReconSMSBase();

    protected:




        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
    };
}

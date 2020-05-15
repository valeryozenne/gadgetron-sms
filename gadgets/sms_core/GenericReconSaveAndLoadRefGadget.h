/** \file   GenericReconSaveAndLoadRefGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to convert the data into eigen channel, working on the IsmrmrdReconData.
            If incoming data has the ref, ref data will be used to compute KLT coefficients
    \author Hui Xue
*/

#pragma once

#include "GenericReconBase.h"

#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDKLT.h"

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconSaveAndLoadRefGadget : public GenericReconDataBase
    {
    public:
        GADGET_DECLARE(GenericReconSaveAndLoadRefGadget);

        typedef GenericReconDataBase BaseClass;

        GenericReconSaveAndLoadRefGadget();
        ~GenericReconSaveAndLoadRefGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------
        GADGET_PROPERTY(save, bool, "Whether to send out SNR map", false);
        GADGET_PROPERTY(save_number, int, "Whether to send out SNR map", 0);
        GADGET_PROPERTY(load, bool, "Whether to send out SNR map", false);
        GADGET_PROPERTY(load_number, int, "Whether to send out SNR map", 0);
        GADGET_PROPERTY(use_FLASH, bool, "Whether to send out SNR map", false);

    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
    };
}

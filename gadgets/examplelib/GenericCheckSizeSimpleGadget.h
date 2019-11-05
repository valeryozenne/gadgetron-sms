/** \file   GenericCheckSizeSimpleGadget.h
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

    class EXPORTGADGETSEXAMPLELIB GenericCheckSizeSimpleGadget : public GenericReconDataBase
    {
    public:
        GADGET_DECLARE(GenericCheckSizeSimpleGadget);

        typedef GenericReconDataBase BaseClass;

        int compteur;

        GenericCheckSizeSimpleGadget();
        ~GenericCheckSizeSimpleGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------

    protected:     

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
    };
}

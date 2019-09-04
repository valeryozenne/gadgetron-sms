/** \file   GenericReconSMSPostGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to convert the data into eigen channel, working on the IsmrmrdReconData.
            If incoming data has the ref, ref data will be used to compute KLT coefficients
    \author Hui Xue
*/

#pragma once

#include "GenericReconSMSBase.h"

#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDKLT.h"

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconSMSPostGadget : public GenericReconSMSBase
    {
    public:
        GADGET_DECLARE(GenericReconSMSPostGadget);

        typedef GenericReconSMSBase BaseClass;

        GenericReconSMSPostGadget();
        ~GenericReconSMSPostGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------


    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        // for every encoding space
        // calibration mode
        //std::vector<Gadgetron::ismrmrdCALIBMODE> calib_mode_;

        // --------------------------------------------------
        // variable for recon
        // --------------------------------------------------

        // store the KLT coefficients for N, S, SLC at every encoding space
        //std::vector< std::vector< std::vector< std::vector< KLTType > > > > KLT_;

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);

        virtual void undo_stacks_ordering_to_match_gt_organisation(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> > &output);
    };
}

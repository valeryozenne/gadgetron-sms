/**
\file   ZeroFillingGadget.h
\brief  This is the class gadget for cardiac T2 mapping, working on the IsmrmrdImageArray.
\author Hui Xue
*/

#pragma once

#include "ThermoMappingGadget.h"
#include "GenericReconBase.h"
#include "hoNDFFT.h"
#include "hoNDArray_math.h"

namespace Gadgetron {

    class EXPORTGADGETSTHERMOCORE ZeroFillingGadget : public GenericReconImageBase
    {
    public:
        GADGET_DECLARE(ZeroFillingGadget);

        typedef GenericReconImageBase BaseClass;

        ZeroFillingGadget();
        ~ZeroFillingGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the mapping
        /// ------------------------------------------------------------------------------------



    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        // --------------------------------------------------
        // functional functions
        // --------------------------------------------------

        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1);

        // function to perform the mapping
        // data: input image array [RO E1 E2 CHA N S SLC]
        // map and map_sd: mapping result and its sd
        // para and para_sd: other parameters of mapping and its sd
        //virtual int perform_zerofilling(IsmrmrdImageArray& data, IsmrmrdImageArray& map);
        virtual void perform_zerofilling(IsmrmrdImageArray& in, hoNDArray< std::complex<float> > & data_out, int scaling);
        //virtual void perform_zerofilling(IsmrmrdImageArray& in, IsmrmrdImageArray& out);


  };
}

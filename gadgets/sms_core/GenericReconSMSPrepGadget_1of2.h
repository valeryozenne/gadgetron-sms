/** \file   GenericReconSMSPrepGadget_1of2.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to convert the data into eigen channel, working on the IsmrmrdReconData.
            If incoming data has the ref, ref data will be used to compute KLT coefficients
    \author Hui Xue
*/

#pragma once

#include "GenericReconSMSBase_gadget1of2.h"

#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDKLT.h"

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconSMSPrepGadget_1of2 : public GenericReconSMSBase_gadget1of2
    {
    public:
        GADGET_DECLARE(GenericReconSMSPrepGadget_1of2);

        typedef GenericReconSMSBase_gadget1of2 BaseClass;

        GADGET_PROPERTY(useDiskData, bool, "Whether to use data saved on disk or not", false);
        GADGET_PROPERTY(useEPICorrData, bool, "Whether to use data from struct EPICorrection or not", false);

        GenericReconSMSPrepGadget_1of2();
        ~GenericReconSMSPrepGadget_1of2();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------


    protected:

        // --------------------------------------------------
        // variables for protocol
        // --------------------------------------------------

        // --------------------------------------------------
        // variable for recon
        // --------------------------------------------------       

        hoNDArray< std::complex<float> > ref_8D;
        hoNDArray< std::complex<float> > sb_8D;
        hoNDArray< std::complex<float> > mb_8D;

        // --------------------------------------------------       
        // variable to verify if the number of messages sent is the same as the number of messages received
        // --------------------------------------------------       

        size_t process_called_times_epicorr;

        int nb_occurences;

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);

        virtual int process(Gadgetron::GadgetContainerMessage< s_EPICorrection >* m1);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m2);

        //main functions
        virtual void pre_process_ref_data(hoNDArray< std::complex<float> >& ref, hoNDArray< std::complex<float> >& ref_8D, size_t e);
        virtual void pre_process_sb_data(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, size_t encoding);
        virtual void pre_process_mb_data(hoNDArray< std::complex<float> >& mb, hoNDArray< std::complex<float> >& mb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb, size_t encoding);

        //sb functions
        virtual void reorganize_sb_data_to_8D(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_8D, size_t encoding);
        virtual void apply_blip_caipi_shift_sb(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, size_t e);
        virtual void apply_averaged_epi_ghost_correction_sb(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > & headers_sb, size_t e);

         //mb functions
        virtual void reorganize_mb_data_to_8D(hoNDArray< std::complex<float> >& mb, hoNDArray< std::complex<float> >& mb_8D, size_t encoding);
        virtual void apply_averaged_epi_ghost_correction_mb(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > & headers_sb, size_t e);

          };
}

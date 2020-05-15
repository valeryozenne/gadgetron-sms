/** \file   GenericReconSMSPostGadget_1of2.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to convert the data into eigen channel, working on the IsmrmrdReconData.
            If incoming data has the ref, ref data will be used to compute KLT coefficients
    \author Hui Xue
*/

#pragma once

#include "GenericReconSMSBase_gadget1of2.h"

#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDKLT.h"
#include "SMS_utils.h"

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconSMSPostGadget_1of2 : public GenericReconSMSBase_gadget1of2
    {
    public:
        GADGET_DECLARE(GenericReconSMSPostGadget_1of2);

        typedef GenericReconSMSBase_gadget1of2 BaseClass;

        GADGET_PROPERTY(useDiskData, bool, "Whether to use data saved on disk or not", false);
        GADGET_PROPERTY(useEPICorrData, bool, "Whether to use data from struct EPICorrection or not", false);

        GenericReconSMSPostGadget_1of2();
        ~GenericReconSMSPostGadget_1of2();

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


        hoNDArray< ISMRMRD::AcquisitionHeader > headers_buffered;
        size_t process_called_times_epicorr;
        bool first_occurence;

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
        virtual int process(Gadgetron::GadgetContainerMessage<s_EPICorrection> *m2);
        //virtual void undo_stacks_ordering_to_match_gt_organisation(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> > &output);

        //virtual void undo_stacks_ordering_to_match_gt_organisation_open(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> > &output);

        virtual void undo_blip_caipi_shift(hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers_sb, size_t e, bool undo_absolute);

        virtual void post_process_sb_data(hoNDArray< std::complex<float> >& data_8D, hoNDArray< std::complex<float> >& data_7D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers, size_t e);

        virtual void post_process_mb_data(hoNDArray< std::complex<float> >& data_8D, hoNDArray< std::complex<float> >& data_7D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers, size_t e);

        virtual void post_process_ref_data(hoNDArray< std::complex<float> >& data_8D, hoNDArray< std::complex<float> >& data_7D, size_t e);

        virtual void set_idx(hoNDArray< ISMRMRD::AcquisitionHeader > headers_, unsigned int rep, unsigned int set);

    };
}

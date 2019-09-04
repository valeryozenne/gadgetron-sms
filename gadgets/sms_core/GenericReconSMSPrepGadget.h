/** \file   GenericReconSMSPrepGadget.h
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

    class EXPORTGADGETSSMSCORE GenericReconSMSPrepGadget : public GenericReconSMSBase
    {
    public:
        GADGET_DECLARE(GenericReconSMSPrepGadget);

        typedef GenericReconSMSBase BaseClass;

        GenericReconSMSPrepGadget();
        ~GenericReconSMSPrepGadget();

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

        unsigned int center_k_space_xml;

        arma::vec z_offset_geo;
        arma::vec z_gap;

        // store the KLT coefficients for N, S, SLC at every encoding space
        //std::vector< std::vector< std::vector< std::vector< KLTType > > > > KLT_;

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
        virtual void remove_extra_dimension_and_permute_stack_dimension(hoNDArray< std::complex<float> >& data);
        virtual void reorganize_data(hoNDArray< std::complex<float> >& data, arma::uvec indice);

        virtual void create_stacks_of_slices(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& fid_stack_SB);
        virtual void apply_relative_phase_shift(hoNDArray< std::complex<float> >& data);
        virtual void apply_absolute_phase_shift(hoNDArray< std::complex<float> >& data);
        virtual void get_header_and_position_and_gap(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > headers_, size_t E1);



    };
}

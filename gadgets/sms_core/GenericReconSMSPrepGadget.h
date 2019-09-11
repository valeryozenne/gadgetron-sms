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



        // store the KLT coefficients for N, S, SLC at every encoding space
        //std::vector< std::vector< std::vector< std::vector< KLTType > > > > KLT_;

        hoNDArray< std::complex<float> > ref_8D;

        hoNDArray< std::complex<float> > sb_8D;

        hoNDArray< std::complex<float> > mb_8D;


        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);

        //virtual void extract_sb_and_mb_from_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb);
        virtual void pre_process_sb_data(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, size_t encoding);
        virtual void pre_process_mb_data(hoNDArray< std::complex<float> >& mb, hoNDArray< std::complex<float> >& mb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb, size_t encoding);
        virtual void pre_process_ref_data(hoNDArray< std::complex<float> >& ref, hoNDArray< std::complex<float> >& ref_8D, size_t e);

        //sb functions
        virtual void reorganize_sb_data_to_8D(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_8D, size_t encoding);
        virtual void permute_slices_index(hoNDArray< std::complex<float> >& data, arma::uvec indice);
        virtual void create_stacks_of_slices(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& fid_stack_SB);

        //mb functions
        virtual void remove_extra_dimension_and_permute_stack_dimension(hoNDArray< std::complex<float> >& data);
        virtual void reorganize_mb_data_to_8D(hoNDArray< std::complex<float> >& mb,hoNDArray< std::complex<float> >& mb_8D );

        //common functions
        virtual void extract_sb_and_mb_from_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb);
        virtual void fusion_sb_and_mb_in_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb);

        virtual void apply_blip_caipi_shift(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, size_t e);
        virtual void apply_averaged_epi_ghost_correction(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > & headers_sb, size_t e);



          };
}

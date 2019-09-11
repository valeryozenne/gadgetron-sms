/** \file   GenericReconSMSv0Base.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to convert the data into eigen channel, working on the IsmrmrdReconData.
            If incoming data has the ref, ref data will be used to compute KLT coefficients
    \author Hui Xue
*/

#pragma once

#include "GenericReconBase.h"
#include "mri_core_slice_grappa.h"
#include "mri_core_utility_interventional.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDKLT.h"
#include "hoArmadillo.h"

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconSMSv0Base : public GenericReconDataBase
    {
    public:
        GADGET_DECLARE(GenericReconSMSv0Base);

        typedef GenericReconDataBase BaseClass;
        //typedef hoNDKLT< std::complex<float> > KLTType;

        GenericReconSMSv0Base();
        ~GenericReconSMSv0Base();

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

        std::vector<size_t> dimensions_;


        bool is_wip_sequence ;
        bool is_cmrr_sequence ;

        unsigned int MB_factor;
        unsigned int Blipped_CAIPI;

        unsigned int lNumberOfStacks_;
        unsigned int lNumberOfSlices_;
        unsigned int lNumberOfChannels_;

        // acceleration factor for E1 and E2
        std::vector<double> acceFactorSMSE1_;
        std::vector<double> acceFactorSMSE2_;

        // ordre des coupes
        arma::imat MapSliceSMS;
        arma::ivec order_of_acquisition_sb;
        arma::ivec order_of_acquisition_mb;

        arma::uvec indice_mb;
        arma::uvec indice_sb;
        arma::uvec indice_slice_mb;

        unsigned int center_k_space_xml;
        unsigned int center_k_space_E1;

        //
        size_t reduced_E1_;
        size_t start_E1_;
        size_t end_E1_;


        float slice_thickness;

        hoNDArray<std::complex<float>> epi_nav_neg_;
        hoNDArray<std::complex<float>> epi_nav_pos_;

        hoNDArray<std::complex<float>> epi_nav_neg_no_exp_;
        hoNDArray<std::complex<float>> epi_nav_pos_no_exp_;

        hoNDArray< std::complex<float> > epi_nav_neg_STK_;
        hoNDArray< std::complex<float> > epi_nav_pos_STK_;

        hoNDArray<std::complex<float>> epi_nav_neg_no_exp_STK_;
        hoNDArray<std::complex<float>> epi_nav_pos_no_exp_STK_;

        hoNDArray< std::complex<float> > epi_nav_neg_STK_mean_;
        hoNDArray< std::complex<float> > epi_nav_pos_STK_mean_;

        hoNDArray<std::complex<float>> epi_nav_neg_no_exp_STK_mean_;
        hoNDArray<std::complex<float>> epi_nav_pos_no_exp_STK_mean_;


        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);

        virtual arma::ivec map_interleaved_acquisitions(int number_of_slices, bool no_reordering );
        virtual arma::imat get_map_slice_single_band(int MB_factor, int lNumberOfStacks, arma::ivec order_of_acquisition_mb, bool no_reordering);

        virtual void save_4D_data(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_with_SLC_7(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_with_STK_5(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_with_STK_6(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_with_STK_7(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_with_STK_8(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_8D_kspace(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void show_size(hoNDArray< std::complex<float> >& input, std::string name);

        virtual void load_epi_data();

        virtual void compute_mean_epi_nav(hoNDArray< std::complex<float> >& nav, hoNDArray< std::complex<float> >& nav_mean);

        virtual void create_stacks_of_nav(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack);

        virtual void reorganize_nav(hoNDArray< std::complex<float> >& data, arma::uvec indice);

        virtual void prepare_epi_data();

        virtual void apply_ghost_correction_with_STK6(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc , bool optimal);

        virtual void apply_ghost_correction_with_STK7(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc , bool optimal);

        virtual void define_usefull_parameters(IsmrmrdReconBit &recon_bit, size_t e);

        virtual int get_reduced_E1_size(size_t start_E1 , size_t end_E1, size_t acc );

    };
}

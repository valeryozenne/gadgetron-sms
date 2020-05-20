/** \file   GenericReconSMSBase_gadget1of2.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian reconstruction to convert the data into eigen channel, working on the IsmrmrdReconData.
            If incoming data has the ref, ref data will be used to compute KLT coefficients
    \author Hui Xue
*/

#pragma once

//#include "GenericReconBase.h"
#include "mri_core_slice_grappa.h"
#include "mri_core_utility_interventional.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDKLT.h"
#include "hoArmadillo.h"
#include "test_slice_grappa.h"
#include "gadgetron_smscore_export.h"
#include "GenericReconBase_gadget1of2.h"

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconSMSBase_gadget1of2 : public GenericReconDataBase_gadget1of2
    {
    public:
        GADGET_DECLARE(GenericReconSMSBase_gadget1of2);

        typedef GenericReconDataBase_gadget1of2 BaseClass;
        //typedef hoNDKLT< std::complex<float> > KLTType;

        GADGET_PROPERTY(use_omp, bool, "Whether to use omp acceleration", false);
        GADGET_PROPERTY(use_gpu, bool, "Whether to use gpu acceleration", false);

        GenericReconSMSBase_gadget1of2();
        ~GenericReconSMSBase_gadget1of2();

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
        int Blipped_CAIPI;

        unsigned int lNumberOfStacks_;
        unsigned int lNumberOfSlices_;
        unsigned int lNumberOfChannels_;

        // acceleration factor for E1 and E2
        std::vector<double> acceFactorSMSE1_;
        std::vector<double> acceFactorSMSE2_;

        // ordre des coupes
        std::vector< std::vector<unsigned int> > MapSliceSMS;
        std::vector<unsigned int> order_of_acquisition_sb;
        std::vector<unsigned int> order_of_acquisition_mb;

        std::vector<unsigned int> indice_mb;
        std::vector<unsigned int> indice_sb;
        std::vector<unsigned int> indice_slice_mb;

        unsigned int center_k_space_xml;
        unsigned int center_k_space_E1;

        //
        size_t reduced_E1_;
        size_t start_E1_;
        size_t end_E1_;


        float slice_thickness;

        arma::cx_fmat corrneg_all_;
        arma::cx_fmat corrpos_all_;

        arma::cx_fmat corrneg_all_no_exp_;
        arma::cx_fmat corrpos_all_no_exp_;

        arma::cx_fmat corrneg_all_STK_mean_;
        arma::cx_fmat corrpos_all_STK_mean_;

        arma::cx_fcube corrneg_all_STK_;
        arma::cx_fcube corrpos_all_STK_;

        arma::cx_fcube corrneg_all_no_exp_STK_;
        arma::cx_fcube corrpos_all_no_exp_STK_;

        arma::cx_fmat corrneg_all_no_exp_STK_mean_;
        arma::cx_fmat corrpos_all_no_exp_STK_mean_;

        hoNDArray<std::complex<float>> epi_nav_neg_;
        hoNDArray<std::complex<float>> epi_nav_pos_;

        hoNDArray<std::complex<float>> epi_nav_neg_no_exp_;
        hoNDArray<std::complex<float>> epi_nav_pos_no_exp_;

        //tmp for debug

        hoNDArray<std::complex<float>> epi_nav_neg_debug_;
        hoNDArray<std::complex<float>> epi_nav_pos_debug_;

        hoNDArray<std::complex<float>> epi_nav_neg_no_exp_debug_;
        hoNDArray<std::complex<float>> epi_nav_pos_no_exp_debug_;

        //end of tmp

        hoNDArray< std::complex<float> > epi_nav_neg_STK_;
        hoNDArray< std::complex<float> > epi_nav_pos_STK_;

        hoNDArray<std::complex<float>> epi_nav_neg_no_exp_STK_;
        hoNDArray<std::complex<float>> epi_nav_pos_no_exp_STK_;

        hoNDArray< std::complex<float> > epi_nav_neg_STK_mean_;
        hoNDArray< std::complex<float> > epi_nav_pos_STK_mean_;

        hoNDArray<std::complex<float>> epi_nav_neg_no_exp_STK_mean_;
        hoNDArray<std::complex<float>> epi_nav_pos_no_exp_STK_mean_;

        //prepare epi and correction epi
        hoNDArray< std::complex<float> > correction_pos_hoND;
        hoNDArray< std::complex<float> > correction_neg_hoND;
        hoNDArray< std::complex<float> > phase_shift;
        hoNDArray< std::complex<float> > tempo_hoND;
        hoNDArray< std::complex<float> > tempo_1D_hoND;



        cuNDArray<float_complext> device_epi_nav_pos_STK_test ;
        cuNDArray<float_complext> device_epi_nav_neg_STK_test ;
        cuNDArray<float_complext> device_epi_nav_pos_STK_mean_test ;
        cuNDArray<float_complext> device_epi_nav_neg_STK_mean_test ;

        cuNDArray<float_complext> device_d_epi_sb;
        cuNDArray<float_complext> device_d_epi_mb;


        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);

        virtual std::vector<unsigned int> map_interleaved_acquisitions(int number_of_slices, bool no_reordering );
        virtual std::vector< std::vector<unsigned int> > get_map_slice_single_band(int MB_factor, int lNumberOfStacks, std::vector<unsigned int> order_of_acquisition_mb, bool no_reordering);
        virtual std::vector<unsigned int> sort_index(std::vector<unsigned int> array);
        arma::vec z_offset_geo;
        arma::vec z_gap;

        virtual void reorganize_arma_nav(arma::cx_fmat &data, std::vector<unsigned int> indice);

        virtual void compute_mean_epi_arma_nav(arma::cx_fcube &input,  arma::cx_fmat& output_no_exp,  arma::cx_fmat& output);

        virtual void create_stacks_of_arma_nav(arma::cx_fmat &data, arma::cx_fcube &new_stack);

        virtual void save_4D_data(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_with_STK_5(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_with_STK_6(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_with_STK_7(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_8D_kspace(hoNDArray< std::complex<float> >& input, std::string name, std::string encoding_number);

        virtual void save_4D_data(hoNDArray<float >& input, std::string name, std::string encoding_number);

        virtual void show_size(hoNDArray< std::complex<float> >& input, std::string name);

        virtual void load_epi_data();

        virtual void compute_mean_epi_nav(hoNDArray< std::complex<float> >& input,  hoNDArray< std::complex<float> >& output_no_exp ,  hoNDArray< std::complex<float> >& output );

        virtual void create_stacks_of_nav(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack);

        virtual void reorganize_nav(hoNDArray< std::complex<float> >& data, std::vector<unsigned int> indice);

        virtual void prepare_epi_data(size_t e, size_t E1, size_t E2, size_t CHA );

        virtual void apply_ghost_correction_with_STK6_old(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool undo , bool optimal);

        virtual void apply_ghost_correction_with_STK6(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool undo, bool optimal, bool ifft , std::string msg);

        virtual void apply_ghost_correction_with_STK6_gpu(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool undo, bool optimal , bool ifft , std::string msg);

        virtual void apply_ghost_correction_with_STK6_open(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool undo, bool optimal, bool ifft , std::string msg);

        virtual void apply_ghost_correction_with_arma_STK6(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc, bool undo, bool optimal , std::string msg);

        virtual void apply_ghost_correction_with_STK7(hoNDArray< std::complex<float> >& data,  hoNDArray< ISMRMRD::AcquisitionHeader > headers_ , size_t acc , bool optimal);

        virtual void define_usefull_parameters(IsmrmrdReconBit &recon_bit, size_t e);
        virtual void define_usefull_parameters_simple_version(IsmrmrdReconBit &recon_bit, size_t e);

        virtual bool detect_first_repetition(IsmrmrdReconBit &recon_bit);
        virtual bool detect_single_band_data(IsmrmrdReconBit &recon_bit);

        virtual int get_reduced_E1_size(size_t start_E1 , size_t end_E1, size_t acc );

        virtual void apply_relative_phase_shift(hoNDArray< std::complex<float> >& data, bool is_positive );
        virtual void apply_relative_phase_shift_test(hoNDArray< std::complex<float> >& data, bool is_positive );
        virtual void apply_absolute_phase_shift(hoNDArray< std::complex<float> >& data, bool is_positive );

        virtual void get_header_and_position_and_gap(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > headers_);

        virtual int CheckComplexNumberEqualInVector(hoNDArray< std::complex<float> >& input , arma::cx_fvec  input_arma);
        virtual int CheckComplexNumberEqualInMatrix(hoNDArray< std::complex<float> >& input , arma::cx_fmat  input_arma);
        virtual int CheckComplexNumberEqualInCube(hoNDArray< std::complex<float> >& input , arma::cx_fcube  input_arma);

        virtual int CheckComplexNumberEqualInVector(hoNDArray< std::complex<float> >& input ,hoNDArray< std::complex<float> >& input2 );
        virtual int CheckComplexNumberEqualInMatrix(hoNDArray< std::complex<float> >& input , hoNDArray< std::complex<float> >& input2);


        virtual void do_fft_for_ref_scan(hoNDArray< std::complex<float> >& data);
    };
}

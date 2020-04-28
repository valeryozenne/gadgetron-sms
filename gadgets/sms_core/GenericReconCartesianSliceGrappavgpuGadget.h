/** \file   GenericReconCartesianSliceGrappavgpuGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian grappa and grappaone reconstruction, working on the IsmrmrdReconData.
    \author Hui Xue
*/

#pragma once

#include "GenericReconGadget.h"
#include "hoArmadillo.h"
#include "GenericReconSMSBase.h"
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

namespace Gadgetron {

    /// define the recon status
    template <typename T>
    class EXPORTGADGETSSMSCORE GenericReconCartesianSliceGrappaGpuObj
    {
    public:

        GenericReconCartesianSliceGrappaGpuObj() {}
        virtual ~GenericReconCartesianSliceGrappaGpuObj()
        {
            if (this->recon_res_.data_.delete_data_on_destruct()) this->recon_res_.data_.clear();
            if (this->recon_res_.headers_.delete_data_on_destruct()) this->recon_res_.headers_.clear();
            this->recon_res_.meta_.clear();

            if (this->gfactor_.delete_data_on_destruct()) this->gfactor_.clear();
            if (this->sb_e1_reduce_.delete_data_on_destruct()) this->sb_e1_reduce_.clear();
            if (this->mb_e1_reduce_.delete_data_on_destruct()) this->mb_e1_reduce_.clear();
            if (this->block_SB_.delete_data_on_destruct()) this->block_SB_.clear();
            if (this->block_MB_.delete_data_on_destruct()) this->block_MB_.clear();
            if (this->missing_data_.delete_data_on_destruct()) this->missing_data_.clear();
            if (this->kernel_.delete_data_on_destruct()) this->kernel_.clear();
            if (this->kernelonov_.delete_data_on_destruct()) this->kernelonov_.clear();


            if (this->unfolded_image_.delete_data_on_destruct()) this->unfolded_image_.clear();
            if (this->unfolded_image_permute_.delete_data_on_destruct()) this->unfolded_image_permute_.clear();

            if (this->ref_compression_.delete_data_on_destruct()) this->ref_compression_.clear();
            if (this->sb_compression_.delete_data_on_destruct()) this->sb_compression_.clear();
            if (this->mb_compression_.delete_data_on_destruct()) this->mb_compression_.clear();



        }

        // ------------------------------------
        /// recon outputs
        // ------------------------------------
        /// reconstructed images, headers and meta attributes
        IsmrmrdImageArray recon_res_;

        /// gfactor, [RO E1 E2 uncombinedCHA+1 N S SLC]
        hoNDArray<typename realType<T>::Type> gfactor_;

        // ------------------------------------
        /// buffers used in the recon
        // ------------------------------------

         /// [RO, reduced_E1_, CHA, MB, STK, N, S]
        hoNDArray<T> sb_e1_reduce_;

        /// [ RO, reduced_E1_, CHA, 1, STK, N, S]
        hoNDArray<T> mb_e1_reduce_;

        ///  [voxels_number_per_image_, kernel_size_, CHA, MB, STK, N, S]
        hoNDArray<T > block_SB_;

        /// [voxels_number_per_image_, kernel_size_, CHA, 1, STK, N, S]
        hoNDArray<T> block_MB_;

        /// [voxels_number_per_image_, CHA, MB, STK, N, S]
        hoNDArray<T> missing_data_;

        /// [ CHA*kernel_size_, CHA, MB, STK, N, S]
        hoNDArray<T> kernel_;

        /// [ CHA*kernel_size_, CHA, MB, STK, N, S]
        hoNDArray<T> kernelonov_;

        /// voxels_number_per_image_, CHA, MB_factor, STK,  N, S]
        hoNDArray<T > unfolded_image_;

        /// blocks_RO_,blocks_E1_,CHA, MB_factor, STK,  N, S]
        hoNDArray<T > unfolded_image_permute_;

        hoNDArray<T > ref_compression_;
        hoNDArray<T > sb_compression_;
        hoNDArray<T > mb_compression_;

    };
}

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconCartesianSliceGrappavgpuGadget : public GenericReconSMSBase
    {
    public:
        GADGET_DECLARE(GenericReconCartesianSliceGrappavgpuGadget);

        typedef GenericReconSMSBase BaseClass;
        typedef Gadgetron::GenericReconCartesianSliceGrappaGpuObj< std::complex<float> > ReconObjType;

        GenericReconCartesianSliceGrappavgpuGadget();
        ~GenericReconCartesianSliceGrappavgpuGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------

        /// ------------------------------------------------------------------------------------
        /// image sending
        GADGET_PROPERTY(blocking, bool, "Whether to do blocking reco", false);
        GADGET_PROPERTY(dual, bool, "Whether to dual dual", false);
        GADGET_PROPERTY(use_slice_grappa_gpu, bool, "Whether to slice_grappa on the gpu", false);

        /// ------------------------------------------------------------------------------------
        /// Grappa parameters
        GADGET_PROPERTY(grappa_kSize_RO, int, "Grappa kernel size RO", 5);
        GADGET_PROPERTY(grappa_kSize_E1, int, "Grappa kernel size E1", 5);
        GADGET_PROPERTY(calib_fast, bool, "calibration with Tikhonov", false);

        GADGET_PROPERTY(grappa_reg_lamda, double, "Grappa regularization threshold", 0.0005);
        //GADGET_PROPERTY(grappa_calib_over_determine_ratio, double, "Grappa calibration overdermination ratio", 45);


        GADGET_PROPERTY(send_out_lfactor, bool, "Whether to send out lfactor map", false);

        /// ------------------------------------------------------------------------------------
        /// down stream coil compression
        /// if downstream_coil_compression==true, down stream coil compression is used
        /// if downstream_coil_compression_num_modesKept > 0, this number of channels will be used as the dst channels
        /// if downstream_coil_compression_num_modesKept==0 and downstream_coil_compression_thres>0, the number of dst channels will be determined  by this threshold
        GADGET_PROPERTY(downstream_coil_compression, bool, "Whether to perform downstream coil compression", true);
        GADGET_PROPERTY(downstream_coil_compression_thres, double, "Threadhold for downstream coil compression", 0.002);
        GADGET_PROPERTY(downstream_coil_compression_num_modesKept, size_t, "Number of modes to keep for downstream coil compression", 0);

    protected:

        // --------------------------------------------------
        // variable for recon
        // --------------------------------------------------
        // record the recon kernel, coil maps etc. for every encoding space
        std::vector< ReconObjType > recon_obj_;

        size_t kernel_size_;
        size_t blocks_RO_;
        size_t blocks_E1_;
        size_t voxels_number_per_image_;


        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);

        // --------------------------------------------------
        // recon step functions
        // --------------------------------------------------


        unsigned int compteur;

        //TODO necessay for the pinv operation
        arma::cx_fmat CMK_matrix;
        arma::cx_fmat measured_data_matrix;

        void define_kernel_parameters(IsmrmrdReconBit &recon_bit, size_t e);

        void perform_slice_grappa_unwrapping(IsmrmrdReconBit& recon_bit, ReconObjType& recon_obj, size_t encoding);

        void remove_unnecessary_kspace_gpu(hoNDArray<std::complex<float> >& input, hoNDArray<std::complex<float> >& output, const size_t acc, const size_t startE1, const size_t endE1, bool is_mb );

        void perform_slice_grappa_unwrapping_gpu(IsmrmrdReconBit &recon_bit, ReconObjType &recon_obj, size_t e);

        void perform_slice_grappa_calib(IsmrmrdReconBit &recon_bit,  ReconObjType &recon_obj, size_t e);

        void recopy_kspace( ReconObjType &recon_obj, hoNDArray< std::complex<float> >& output, size_t acc );

        void prepare_down_stream_coil_compression_ref_data(hoNDArray<std::complex<float> > &ref_src, hoNDArray<std::complex<float> > &ref_dst, size_t e);

        void im2col_gpu(hoNDArray<std::complex<float> >& input, hoNDArray<std::complex<float> >& output, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1 );

        void do_gpu_test(hoNDArray<std::complex<float> > & input);

        void do_gpu_unmix(hoNDArray<std::complex<float> > & input, hoNDArray<std::complex<float> > & output, hoNDArray<std::complex<float> > & kernel );
        ///////////

        GADGET_PROPERTY(deviceno,int,"GPU device number", 0);
        int device_number_;

    };
}

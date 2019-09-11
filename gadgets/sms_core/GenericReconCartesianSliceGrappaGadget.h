/** \file   GenericReconCartesianSliceGrappaGadget.h
    \brief  This is the class gadget for both 2DT and 3DT cartesian grappa and grappaone reconstruction, working on the IsmrmrdReconData.
    \author Hui Xue
*/

#pragma once

#include "GenericReconGadget.h"
#include "hoArmadillo.h"
#include "GenericReconSMSBase.h"

namespace Gadgetron {

    /// define the recon status
    template <typename T>
    class EXPORTGADGETSSMSCORE GenericReconCartesianGrappaObj
    {
    public:

        GenericReconCartesianGrappaObj() {}
        virtual ~GenericReconCartesianGrappaObj()
        {
            if (this->recon_res_.data_.delete_data_on_destruct()) this->recon_res_.data_.clear();
            if (this->recon_res_.headers_.delete_data_on_destruct()) this->recon_res_.headers_.clear();
            this->recon_res_.meta_.clear();

            if (this->gfactor_.delete_data_on_destruct()) this->gfactor_.clear();
            if (this->ref_calib_.delete_data_on_destruct()) this->ref_calib_.clear();
            if (this->ref_calib_dst_.delete_data_on_destruct()) this->ref_calib_dst_.clear();
            if (this->ref_coil_map_.delete_data_on_destruct()) this->ref_coil_map_.clear();
            if (this->kernel_.delete_data_on_destruct()) this->kernel_.clear();
            if (this->kernelIm_.delete_data_on_destruct()) this->kernelIm_.clear();
            if (this->unmixing_coeff_.delete_data_on_destruct()) this->unmixing_coeff_.clear();
            if (this->coil_map_.delete_data_on_destruct()) this->coil_map_.clear();            
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

        /// [RO E1 E2 srcCHA Nor1 Sor1 SLC]
        hoNDArray<T> ref_calib_;
        /// [RO E1 E2 dstCHA Nor1 Sor1 SLC]
        hoNDArray<T> ref_calib_dst_;

        /// reference data ready for coil map estimation
        /// [RO E1 E2 dstCHA Nor1 Sor1 SLC]
        hoNDArray<T> ref_coil_map_;

        /// for combined imgae channel
        /// convolution kernel, [RO E1 E2 srcCHA - uncombinedCHA dstCHA - uncombinedCHA Nor1 Sor1 SLC]
        hoNDArray<T> kernel_;
        /// image domain kernel, [RO E1 E2 srcCHA - uncombinedCHA dstCHA - uncombinedCHA Nor1 Sor1 SLC]
        hoNDArray<T> kernelIm_;
        /// image domain unmixing coefficients, [RO E1 E2 srcCHA - uncombinedCHA Nor1 Sor1 SLC]
        hoNDArray<T> unmixing_coeff_;

        /// coil sensitivity map, [RO E1 E2 dstCHA - uncombinedCHA Nor1 Sor1 SLC]
        hoNDArray<T> coil_map_;
    };
}

namespace Gadgetron {

    class EXPORTGADGETSSMSCORE GenericReconCartesianSliceGrappaGadget : public GenericReconSMSBase
    {
    public:
        GADGET_DECLARE(GenericReconCartesianSliceGrappaGadget);

        typedef GenericReconSMSBase BaseClass;
        typedef Gadgetron::GenericReconCartesianGrappaObj< std::complex<float> > ReconObjType;

        GenericReconCartesianSliceGrappaGadget();
        ~GenericReconCartesianSliceGrappaGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the reconstruction
        /// ------------------------------------------------------------------------------------

        /// ------------------------------------------------------------------------------------
        /// image sending
        GADGET_PROPERTY(blocking, bool, "Whether to do blocking reco", false);
        GADGET_PROPERTY(dual, bool, "Whether to dual dual", false);

        /// ------------------------------------------------------------------------------------
        /// Grappa parameters
        GADGET_PROPERTY(grappa_kSize_RO, int, "Grappa kernel size RO", 5);
        GADGET_PROPERTY(grappa_kSize_E1, int, "Grappa kernel size E1", 5);
        GADGET_PROPERTY(calib_fast, int, "calibration with Tikhonov", 0);

        GADGET_PROPERTY(grappa_reg_lamda, double, "Grappa regularization threshold", 0.0005);
        //GADGET_PROPERTY(grappa_calib_over_determine_ratio, double, "Grappa calibration overdermination ratio", 45);

        /// ------------------------------------------------------------------------------------
        /// down stream coil compression
        /// if downstream_coil_compression==true, down stream coil compression is used
        /// if downstream_coil_compression_num_modesKept > 0, this number of channels will be used as the dst channels
        /// if downstream_coil_compression_num_modesKept==0 and downstream_coil_compression_thres>0, the number of dst channels will be determined  by this threshold
        //GADGET_PROPERTY(downstream_coil_compression, bool, "Whether to perform downstream coil compression", true);
        //GADGET_PROPERTY(downstream_coil_compression_thres, double, "Threadhold for downstream coil compression", 0.002);
        //GADGET_PROPERTY(downstream_coil_compression_num_modesKept, size_t, "Number of modes to keep for downstream coil compression", 0);

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

          hoNDArray< std::complex<float> > kernel;
          hoNDArray< std::complex<float> > kernelonov;

          hoNDArray< std::complex<float> > unfolded_image;
          hoNDArray< std::complex<float> > unfolded_image_permute;

        arma::cx_fmat CMK_matrix;
        arma::cx_fmat measured_data_matrix;

        virtual void define_kernel_parameters(IsmrmrdReconBit &recon_bit, size_t e);

        virtual void remove_unnecessary_kspace_sb(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_reduce, size_t acc);

        virtual void remove_unnecessary_kspace_mb(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_reduce, size_t acc);

        virtual void perform_slice_grappa_unwrapping(IsmrmrdReconBit& recon_bit, ReconObjType& recon_obj, size_t encoding);

        virtual void  perform_slice_grappa_calib(IsmrmrdReconBit &recon_bit, ReconObjType &recon_obj, size_t e);

        virtual void im2col(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& block_SB);

        virtual void extract_milieu_kernel(hoNDArray< std::complex<float> >& block_SB, hoNDArray< std::complex<float> >& missing_data);

        virtual void extract_sb_and_mb_from_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb);

        virtual void remove_unnecessary_kspace_mb2(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& output, size_t acc );

        virtual void recopy_kspace( hoNDArray< std::complex<float> >& output, size_t acc );       

    };
}

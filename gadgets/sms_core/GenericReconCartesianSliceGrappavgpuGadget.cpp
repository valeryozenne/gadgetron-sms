
#include "GenericReconCartesianSliceGrappavgpuGadget.h"
#include "mri_core_grappa.h"
#include "hoNDArray_reductions.h"
#include "hoNDArray_linalg.h"
#include "mri_core_slice_grappa.h"
#include "GPUTimer.h"
#include "cuNDArray_operators.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_blas.h"
#include "cuNDArray_utils.h"
#include "cuNDArray_reductions.h"
#include "cuNDFFT.h"
#include "b1_map.h"
#include "setup_grid.h"
#include "test_slice_grappa.h"

/*
    The input is IsmrmrdReconData and output is single 2D or 3D ISMRMRD images

    If required, the gfactor map can be sent out

    If the  number of required destination channel is 1, the GrappaONE recon will be performed

    The image number computation logic is implemented in compute_image_number function, which can be overloaded
*/

namespace Gadgetron {

GenericReconCartesianSliceGrappavgpuGadget::GenericReconCartesianSliceGrappavgpuGadget() : BaseClass() {
}

GenericReconCartesianSliceGrappavgpuGadget::~GenericReconCartesianSliceGrappavgpuGadget() {
}

int GenericReconCartesianSliceGrappavgpuGadget::process_config(ACE_Message_Block *mb) {
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    // -------------------------------------------------

    ISMRMRD::IsmrmrdHeader h;
    try {
        deserialize(mb->rd_ptr(), h);
    }
    catch (...) {
        GDEBUG("Error parsing ISMRMRD Header");
    }

    size_t NE = h.encoding.size();
    num_encoding_spaces_ = NE;
    GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);

    recon_obj_.resize(NE);


    compteur=0;

    GDEBUG_STREAM("Allocation OK : " << NE);

    return GADGET_OK;
}

int GenericReconCartesianSliceGrappavgpuGadget::process(Gadgetron::GadgetContainerMessage<IsmrmrdReconData> *m1) {

    if (perform_timing.value()) { gt_timer_.start("GenericReconCartesianSliceGrappavgpuGadget::process"); }

    process_called_times_++;

    IsmrmrdReconData *recon_bit_ = m1->getObjectPtr();

    if (recon_bit_->rbit_.size() > num_encoding_spaces_) {
        GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size()
                     << " instead of "
                     << num_encoding_spaces_);
    }


    for (size_t e = 0; e < recon_bit_->rbit_.size(); e++) {
        std::stringstream os;
        os << "_encoding_" << e;

        GDEBUG_CONDITION_STREAM(verbose.value(),
                                "Calling " << process_called_times_ << " , encoding space : " << e);
        GDEBUG_CONDITION_STREAM(verbose.value(),
                                "======================================================================");


        if (recon_bit_->rbit_[e].ref_)
        {
            // std::cout << " je suis la structure qui contient les données acs" << std::endl;
            this->prepare_down_stream_coil_compression_ref_data(recon_bit_->rbit_[e].ref_->data_, recon_obj_[e].ref_compression_, e);
            recon_bit_->rbit_[e].ref_->data_=recon_obj_[e].ref_compression_;
        }


        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {

            bool is_single_band=false;
            bool is_first_repetition=detect_first_repetition(recon_bit_->rbit_[e]);
            if (is_first_repetition==true) {  is_single_band=detect_single_band_data(recon_bit_->rbit_[e]);    }

            if (compteur==0)
            {
                define_usefull_parameters_simple_version(recon_bit_->rbit_[e], e);
                this->define_kernel_parameters(recon_bit_->rbit_[e], e);
                compteur++;
                //std::cout <<" compteur  "<<  compteur << "  is_single_band " <<  is_single_band << std::endl;
            }



            if (is_single_band==true)  //presence de single band
            {
                save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(recon_bit_->rbit_[e].data_.data_, "FID_SB_Avant_Calib_sms", os.str());

                //TODO this must be done before
                this->prepare_down_stream_coil_compression_ref_data(recon_bit_->rbit_[e].data_.data_, recon_obj_[e].sb_compression_, e);
                recon_bit_->rbit_[e].data_.data_=recon_obj_[e].sb_compression_;

                if (perform_timing.value()) { gt_timer_.start("GenericReconCartesianSliceGrappavgpuGadget::perform_calib"); }
                this->perform_slice_grappa_calib(recon_bit_->rbit_[e], recon_obj_[e], e);
                if (perform_timing.value()) { gt_timer_.stop(); }

            }
            else
            {
                save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(recon_bit_->rbit_[e].data_.data_, "FID_MB_Avant_Unwrap_sms", os.str());

                this->prepare_down_stream_coil_compression_ref_data(recon_bit_->rbit_[e].data_.data_, recon_obj_[e].mb_compression_, e);
                recon_bit_->rbit_[e].data_.data_=recon_obj_[e].mb_compression_;


                if (use_slice_grappa_gpu.value()==true)
                {
                    if (perform_timing.value()) { gt_timer_.start("GenericReconCartesianSliceGrappavgpuGadget::perform_slice_grappa_unwrapping gpu time: "); }
                    this->perform_slice_grappa_unwrapping_gpu(recon_bit_->rbit_[e], recon_obj_[e], e);
                    if (perform_timing.value()) { gt_timer_.stop();}
                }
                else {

                    if (perform_timing.value()) { gt_timer_.start("GenericReconCartesianSliceGrappavgpuGadget::perform_slice_grappa_unwrapping cpu time: "); }
                    this->perform_slice_grappa_unwrapping(recon_bit_->rbit_[e], recon_obj_[e], e);
                    if (perform_timing.value()) { gt_timer_.stop();}
                }


                ///////////////

                recopy_kspace( recon_obj_[e], recon_bit_->rbit_[e].data_.data_, acceFactorSMSE1_[e] );

                save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(m1->getObjectPtr()->rbit_[e].data_.data_, "FID_MB_Slice_Fin", os.str());

            }
        }
    }

    if (perform_timing.value()) { gt_timer_.stop();  }

    if (this->next()->putq(m1) < 0)
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}


void GenericReconCartesianSliceGrappavgpuGadget::define_kernel_parameters(IsmrmrdReconBit &recon_bit, size_t e)
{

    hoNDArray< std::complex<float> >& data = recon_bit.data_.data_;

    size_t RO = data.get_size(0);
    size_t E1 = data.get_size(1);

    size_t kRO = grappa_kSize_RO.value();
    size_t kNE1 = grappa_kSize_E1.value();

    kernel_size_=kRO*kNE1;
    blocks_RO_=RO-kRO+1;
    blocks_E1_=reduced_E1_-kNE1+1;
    voxels_number_per_image_=blocks_RO_*blocks_E1_;

    GDEBUG("Kernel_size %d, blocks_RO:  %d,  blocks_E1: %d, voxels_number_per_image_: %d\n", kernel_size_, blocks_RO_,  blocks_E1_ ,voxels_number_per_image_  );
}


//TODO the mecanism for coil compression should be simplified and optmized
//TODO
void GenericReconCartesianSliceGrappavgpuGadget::prepare_down_stream_coil_compression_ref_data(
        hoNDArray<std::complex<float> > &ref_src, hoNDArray<std::complex<float> > &ref_dst, size_t e) {

    if (!downstream_coil_compression.value()) {
        GDEBUG_CONDITION_STREAM(verbose.value(), "Downstream coil compression is not prescribed ... ");
        ref_dst = ref_src;
        return;
    }

    if (downstream_coil_compression_thres.value() < 0 && downstream_coil_compression_num_modesKept.value() == 0) {
        GDEBUG_CONDITION_STREAM(verbose.value(),
                                "Downstream coil compression is prescribed to use all input channels ... ");
        ref_dst = ref_src;
        return;
    }


    // determine how many channels to use
    size_t RO = ref_src.get_size(0);
    size_t E1 = ref_src.get_size(1);
    size_t E2 = ref_src.get_size(2);
    size_t CHA = ref_src.get_size(3);
    size_t N = ref_src.get_size(4);
    size_t S = ref_src.get_size(5);
    size_t SLC = ref_src.get_size(6);

    //size_t recon_RO = ref_coil_map.get_size(0);
    //size_t recon_E1 = ref_coil_map.get_size(1);
    //size_t recon_E2 = ref_coil_map.get_size(2);

    std::complex<float> *pRef = const_cast< std::complex<float> * >(ref_src.begin());

    size_t dstCHA = CHA;
    if (downstream_coil_compression_num_modesKept.value() > 0 &&
            downstream_coil_compression_num_modesKept.value() <= CHA) {
        dstCHA = downstream_coil_compression_num_modesKept.value();
    } else {
        std::vector<float> E(CHA, 0);
        long long cha;

#pragma omp parallel default(none) private(cha) shared(RO, E1, E2, CHA, pRef, E)
        {
            hoNDArray<std::complex<float> > dataCha;
#pragma omp for
            for (cha = 0; cha < (long long) CHA; cha++) {
                dataCha.create(RO, E1, E2, pRef + cha * RO * E1 * E2);
                float v = Gadgetron::nrm2(dataCha);
                E[cha] = v * v;
            }
        }

        for (cha = 1; cha < (long long) CHA; cha++) {
            if (std::abs(E[cha]) < downstream_coil_compression_thres.value() * std::abs(E[0])) {
                break;
            }
        }

        dstCHA = cha;
    }

    GDEBUG_CONDITION_STREAM(verbose.value(),
                            "Downstream coil compression is prescribed to use " << dstCHA << " out of " << CHA
                            << " channels ...");

    if (dstCHA < CHA) {
        ref_dst.create(RO, E1, E2, dstCHA, N, S, SLC);
        //hoNDArray<std::complex<float> > ref_coil_map_dst;
        //ref_coil_map_dst.create(recon_RO, recon_E1, recon_E2, dstCHA, N, S, SLC);

        size_t slc, s, n;
        for (slc = 0; slc < SLC; slc++) {
            for (s = 0; s < S; s++) {
                for (n = 0; n < N; n++) {
                    std::complex<float> *pDst = &(ref_dst(0, 0, 0, 0, n, s, slc));
                    const std::complex<float> *pSrc = &(ref_src(0, 0, 0, 0, n, s, slc));
                    memcpy(pDst, pSrc, sizeof(std::complex<float>) * RO * E1 * E2 * dstCHA);

                    //pDst = &(ref_coil_map_dst(0, 0, 0, 0, n, s, slc));
                    //pSrc = &(ref_coil_map(0, 0, 0, 0, n, s, slc));
                    //memcpy(pDst, pSrc, sizeof(std::complex<float>) * recon_RO * recon_E1 * recon_E2 * dstCHA);
                }
            }
        }

        //ref_coil_map = ref_coil_map_dst;
    } else {
        ref_dst = ref_src;
    }

}


void GenericReconCartesianSliceGrappavgpuGadget::perform_slice_grappa_unwrapping_gpu(IsmrmrdReconBit &recon_bit, ReconObjType &recon_obj, size_t e)
{
    // unwrapping function : liste des opérations:
    // 1) remove_unnecessary_kspace (in presence of grappa)
    // 2) im2col
    // 3) reshape
    // 4) apply kernel : SB= MB*kernel -> gemm (SB, MB , Kernel)
    // 5) reshape
    // 6) permute

    //hoNDArray< std::complex<float> >& data = recon_bit.data_.data_;
    hoNDArray< std::complex<float> >& data = recon_obj.mb_compression_;

    size_t RO = data.get_size(0);
    size_t E1 = data.get_size(1);
    size_t E2 = data.get_size(2);
    size_t CHA = data.get_size(3);
    size_t MB = data.get_size(4);
    size_t STK = data.get_size(5);
    size_t N = data.get_size(6);
    size_t S = data.get_size(7);

    GDEBUG_STREAM("GenericReconCartesianSliceGrappavgpuGadget - incoming data array data : [RO E1 E2 CHA MB STK N S] - [" << RO << " " << E1 << " " << E2 << " " << CHA <<  " " << MB <<  " " << STK << " " << N << " " << S<<  "]");

    recon_obj.mb_e1_reduce_.create(RO, reduced_E1_, CHA, 1, STK, N, S);

    if (use_omp.value()==true)
    {
        remove_unnecessary_kspace_open( recon_obj.mb_compression_ ,  recon_obj.mb_e1_reduce_,  acceFactorSMSE1_[e], start_E1_, end_E1_, true);
    }
    else
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: remove_unnecessary_kspace global");}
        remove_unnecessary_kspace( recon_obj.mb_compression_ ,  recon_obj.mb_e1_reduce_,  acceFactorSMSE1_[e], start_E1_, end_E1_, true);
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }

    ///////


    //TODO ceci devrait etre alloué une seule fois et non a chaque passage
    recon_obj.block_MB_.create(voxels_number_per_image_, kernel_size_, CHA, 1, STK, N, S);
    recon_obj.unfolded_image_.create(voxels_number_per_image_, CHA, MB_factor, STK,  N, S);
    recon_obj.unfolded_image_permute_.create(blocks_RO_,blocks_E1_,CHA, MB_factor, STK,  N, S);

    gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::process unmix allocation  ");

    typedef std::complex<float> T;

    // utilisé dans la fonction im2col gpu
    cuNDArray<float_complext> device_mb_e1_reduce(RO, reduced_E1_, CHA, 1, STK);
    cuNDArray<float_complext> device_block_MB(voxels_number_per_image_, kernel_size_, CHA, 1, STK);

    // utilisé dans la fonction gemm
    cuNDArray<float_complext> device_mb(voxels_number_per_image_, kernel_size_*CHA);
    cuNDArray<float_complext> device_kernel(kernel_size_*CHA, CHA);
    cuNDArray<float_complext> device_unfolded(voxels_number_per_image_, CHA);

    gt_timer_local_.stop();

    unsigned int number_element_kernel = kernel_size_*CHA* CHA;
    unsigned int number_element_mb = voxels_number_per_image_*kernel_size_*CHA;
    unsigned int number_element_unfolded = voxels_number_per_image_*CHA;

    gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::process unmix v4  ");

    for (long long s = 0; s < S; s++)    {

        for (long long n = 0; n < N; n++)  {

            //[ RO, E1, CHA 1, STK, N, S]
            T *pIn = &( recon_obj.mb_e1_reduce_(0, 0, 0, 0, 0, n, s));

            if(cudaMemcpy(device_mb_e1_reduce.get_data_ptr(),
                          pIn,
                          RO*reduced_E1_*CHA*STK*sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice)!= cudaSuccess )   {
                GERROR_STREAM("Upload to device for device_mb_e1_reduce failed\n");}


            prepare_im2col_5D(device_mb_e1_reduce , device_block_MB ,  blocks_RO_, blocks_E1_,  grappa_kSize_RO.value(), grappa_kSize_E1.value());

            for (size_t a = 0; a < STK; a++) {

                //[voxels_number_per_image_, kernel_size_, CHA, 1, STK]
                if(cudaMemcpy(device_mb.get_data_ptr(),
                              device_block_MB.get_data_ptr() + a* number_element_mb,
                              number_element_mb*sizeof(std::complex<float>),
                              cudaMemcpyDeviceToDevice)!= cudaSuccess )   {
                    GERROR_STREAM("Upload to device for device_mb failed\n");}

                for (size_t m = 0; m < MB_factor; m++) {

                    //[ CHA*kernel_size_, CHA, MB, STK, N, S]
                    T *pkernel = &(recon_obj.kernelonov_(0, 0, m, a, n, s));

                    if(cudaMemcpy(device_kernel.get_data_ptr(),
                                  pkernel,
                                  number_element_kernel*sizeof(std::complex<float>),
                                  cudaMemcpyHostToDevice)!= cudaSuccess )   {
                        GERROR_STREAM("Upload to device for device_kernel failed\n");}

                    //ici on envoie des matrices 2D pour effectuer C = A*B
                    prepare_gpu_unmix(device_mb, device_kernel, device_unfolded);

                    //[voxels_number_per_image_, CHA, MB_factor, STK,  N, S]
                    T *pOut = &(recon_obj.unfolded_image_(0, 0, m, a, n, s));

                    if(cudaMemcpy(pOut,
                                  device_unfolded.get_data_ptr(),
                                  number_element_unfolded*sizeof(std::complex<float>),
                                  cudaMemcpyDeviceToHost)!= cudaSuccess )   {
                        GERROR_STREAM("Upload to host for device_unfolded failed\n");}

                }
            }
        }
    }

    gt_timer_local_.stop();


    // lui il doit être créé à chaque fois car on le reshape

    /*
    if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::gpuExample:: im2col global");}
    im2col_gpu(recon_obj.mb_e1_reduce_,recon_obj.block_MB_, blocks_RO_,  blocks_E1_,  grappa_kSize_RO.value(), grappa_kSize_E1.value() );
    if (perform_timing.value()) { gt_timer_local_.stop();}


    if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::gpuExample:: reshape global");}
    std::vector<size_t> newdims;
    newdims.push_back(voxels_number_per_image_);
    newdims.push_back(kernel_size_*CHA);
    newdims.push_back(1);
    newdims.push_back(1);
    newdims.push_back(STK); //STK
    newdims.push_back(N); //N
    newdims.push_back(S); //S
    recon_obj.block_MB_.reshape(&newdims);

    if (perform_timing.value()) { gt_timer_local_.stop();}

    gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::process  unmix version 3 gpu  ");

    do_gpu_unmix(recon_obj.block_MB_,recon_obj.unfolded_image_, recon_obj.kernelonov_);

    gt_timer_local_.stop();


*/



    if (perform_timing.value()) {gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: reshape 2 ");}

    std::vector<size_t> newdims2;
    newdims2.push_back(blocks_E1_);
    newdims2.push_back(blocks_RO_);
    newdims2.push_back(CHA);
    newdims2.push_back(MB_factor);
    newdims2.push_back(STK); //N
    newdims2.push_back(N); //S
    newdims2.push_back(S); //STK
    recon_obj.unfolded_image_.reshape(&newdims2);

    if (perform_timing.value()) {gt_timer_local_.stop();}



    if (perform_timing.value()) {gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: permute");}

    std::vector<size_t> newdims3;
    newdims3.push_back(1);
    newdims3.push_back(0);
    newdims3.push_back(2);
    newdims3.push_back(3);
    newdims3.push_back(4); //N
    newdims3.push_back(5); //S
    newdims3.push_back(6); //STK

    recon_obj.unfolded_image_permute_=permute(recon_obj.unfolded_image_,newdims3);

    if (perform_timing.value()) {gt_timer_local_.stop();}



}



void GenericReconCartesianSliceGrappavgpuGadget::im2col_gpu(hoNDArray<std::complex<float> >& input, hoNDArray<std::complex<float> >& output, const size_t blocks_RO, const size_t blocks_E1, const size_t grappa_kSize_RO, const size_t grappa_kSize_E1 )
{

    bool do_2D=false;


    if (do_2D==true)
    {
        //////////////////////////////
        try
        {
            std::cout << "coucou im2col gpu 2D"<< std::endl;
            size_t RO = input.get_size(0);
            size_t E1 = input.get_size(1);
            size_t CHA = input.get_size(2);
            size_t MB = input.get_size(3);
            size_t STK = input.get_size(4);
            size_t N = input.get_size(5);
            size_t S = input.get_size(6);

            // input [RO, reduced_E1_, CHA, MB, STK, N, S]
            // output [voxels_number_per_image_, kernel_size_, CHA, MB, STK, N, S]

            hoNDArray<std::complex<float> > data;
            data.create(RO,E1);

            hoNDArray<std::complex<float> > data_out;
            data_out.create(blocks_RO*blocks_E1,grappa_kSize_RO*grappa_kSize_E1);

            hoNDArray<float_complext>* host_data_out =
                    reinterpret_cast< hoNDArray<float_complext>* >(&data_out);

            cuNDArray<float_complext> device_data_out(host_data_out);

            for (size_t s = 0; s < S; s++)    {

                for (size_t n = 0; n < N; n++)  {

                    for (size_t stk = 0; stk < STK; stk++)  {

                        for (size_t mb = 0; mb < MB; mb++)  {

                            for (size_t cha = 0; cha < CHA; cha++)  {

                                std::complex<float> * in = &(input(0, 0, cha, mb, stk, n,s));
                                std::complex<float> * out = &(data(0, 0 ));
                                memcpy(out , in, sizeof(std::complex<float>)*RO*E1);

                                hoNDArray<float_complext>* host_data =
                                        reinterpret_cast< hoNDArray<float_complext>* >(&data);

                                cuNDArray<float_complext> device_data(host_data);

                                prepare_im2col_2D(device_data , device_data_out ,  blocks_RO, blocks_E1,  grappa_kSize_RO, grappa_kSize_E1);

                                device_data_out.to_host(host_data_out);

                                //boost::shared_ptr< hoNDArray<float_complext  > >  host_result = device_data_out.to_host();

                                //hoNDArray<std::complex<float>> *test=reinterpret_cast<hoNDArray<std::complex<float> >*>(host_result.get());

                                //std::cout << " voxels_number_per_image_ ùkernel" << test->get_number_of_elements()<< std::endl;
                                //std::cout << " voxels_number_per_image_ ùkernel" << voxels_number_per_image_ *kernel_size_<< std::endl;

                                std::complex<float> * out2 = &(output(0, 0, cha, mb, stk, n, s ));
                                memcpy(out2 , data_out.get_data_ptr(), sizeof(std::complex<float>)*voxels_number_per_image_*kernel_size_ );

                            }
                        }
                    }
                }

            }
        }
        catch(...)
        {
            GERROR_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!! unable to im2col gpu");
        }

    }
    else
    {

        try
        {
            std::cout << "coucou im2col gpu 5D"<< std::endl;
            size_t RO = input.get_size(0);
            size_t E1 = input.get_size(1);
            size_t CHA = input.get_size(2);
            size_t MB = input.get_size(3);
            size_t STK = input.get_size(4);
            size_t N = input.get_size(5);
            size_t S = input.get_size(6);

            // input [RO, reduced_E1_, CHA, MB, STK, N, S]
            // output [voxels_number_per_image_, kernel_size_, CHA, MB, STK, N, S]

            hoNDArray<std::complex<float> > data;
            data.create(RO, E1, CHA, MB, STK );

            hoNDArray<std::complex<float> > data_out;
            data_out.create(blocks_RO*blocks_E1,grappa_kSize_RO*grappa_kSize_E1, CHA, MB, STK);

            hoNDArray<float_complext>* host_data_out =
                    reinterpret_cast< hoNDArray<float_complext>* >(&data_out);

            cuNDArray<float_complext> device_data_out(host_data_out);

            for (size_t s = 0; s < S; s++)    {

                for (size_t n = 0; n < N; n++)  {

                    std::complex<float> * in = &(input(0, 0, 0, 0, 0, n,s));
                    std::complex<float> * out = &(data(0, 0, 0, 0, 0 ));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*MB*CHA*STK);

                    hoNDArray<float_complext>* host_data =
                            reinterpret_cast< hoNDArray<float_complext>* >(&data);

                    cuNDArray<float_complext> device_data(host_data);

                    prepare_im2col_5D(device_data , device_data_out ,  blocks_RO, blocks_E1,  grappa_kSize_RO, grappa_kSize_E1);

                    device_data_out.to_host(host_data_out);

                    //boost::shared_ptr< hoNDArray<float_complext  > >  host_result = device_data_out.to_host();

                    //hoNDArray<std::complex<float>> *test=reinterpret_cast<hoNDArray<std::complex<float> >*>(host_result.get());

                    std::complex<float> * out2 = &(output(0, 0, 0, 0, 0, n, s ));
                    memcpy(out2 , data_out.get_data_ptr(), sizeof(std::complex<float>)*voxels_number_per_image_*kernel_size_*CHA*MB*STK );

                }
            }
        }
        catch(...)
        {
            GERROR_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!! unable to perform_slice_grappa_unwrapping_gpu");
        }

    }

}

void GenericReconCartesianSliceGrappavgpuGadget::remove_unnecessary_kspace_gpu(hoNDArray<std::complex<float> >& input, hoNDArray<std::complex<float> >& output, const size_t acc, const size_t startE1, const size_t endE1, bool is_mb )
{

    //////////////////////////////
    try
    {

        size_t RO = input.get_size(0);
        size_t E1 = input.get_size(1);
        size_t E2 = input.get_size(2);
        size_t CHA = input.get_size(3);
        size_t MB = input.get_size(4);
        size_t STK = input.get_size(5);
        size_t N = input.get_size(6);
        size_t S = input.get_size(7);

        if (is_mb==true)
        {
            MB=1; //TODO il y a une erreur par ici

            hoNDArray<std::complex<float> > data;
            data.create(RO,E1,E2,CHA,STK);

            for (size_t s = 0; s < S; s++)    {

                for (size_t n = 0; n < N; n++)  {

                    std::complex<float> * in = &(input(0, 0, 0, 0, 0, 0, n,s));
                    std::complex<float> * out = &(data(0, 0, 0, 0, 0 ));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA*STK);

                    hoNDArray<float_complext>* host_data =
                            reinterpret_cast< hoNDArray<float_complext>* >(&data);

                    cuNDArray<float_complext> device_data(host_data);
                    device_data.squeeze();

                    size_t ROg = device_data.get_size(0);
                    size_t E1g = device_data.get_size(1);
                    size_t CHAg = device_data.get_size(2);
                    size_t STKg = device_data.get_size(3);



                    GDEBUG_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: [RO E1 CHA STK] - [" << ROg << " " << E1g << " " << CHAg << " "  << " " << STKg  << "]");

                    cuNDArray<float_complext> device_data_out;
                    device_data_out.create(ROg,reduced_E1_,CHAg, STKg);

                    boost::shared_ptr<GPUTimer> process_timer;
                    process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::estimate_feasibility_no_STK (MB=1)") );
                    //TODO
                    estimate_feasibility_no_STK<float>( device_data, device_data_out,  acceFactorSMSE1_[0], start_E1_, end_E1_, reduced_E1_);
                    process_timer.reset();

                    boost::shared_ptr< hoNDArray<float_complext  > >  host_result = device_data_out.to_host();

                    hoNDArray<std::complex<float>> *test=reinterpret_cast<hoNDArray<std::complex<float> >*>(host_result.get());

                    /*std::cout << test->get_size(0) << std::endl;
                    std::cout << test->get_size(1) << std::endl;
                    std::cout << test->get_size(2) << std::endl;
                    std::cout << test->get_size(3) << std::endl;
                    std::cout << test->get_size(4) << std::endl;

                    hoNDArray<std::complex<float>> test2;
                    test2.create(RO,reduced_E1_,CHA,STK);

                    memcpy(&test2(0,0,0,0) , test->get_data_ptr(), sizeof(std::complex<float>)*RO*reduced_E1_*CHA*STK);

                    std::cout << output.get_size(0) << std::endl;
                    std::cout << output.get_size(1) << std::endl;
                    std::cout << output.get_size(2) << std::endl;
                    std::cout << output.get_size(3) << std::endl;
                    std::cout << output.get_size(4) << std::endl;*/

                    for (size_t stk = 0; stk < STK; stk++)
                    {
                        //std::complex<float> * in2 = &test2(0, 0, 0, stk);
                        std::complex<float> * out2 = &(output(0, 0, 0, 0, stk, n, s ));
                        //memcpy(out2 , in2, sizeof(std::complex<float>)*RO*reduced_E1_*CHA);
                        memcpy(out2 , test->get_data_ptr() + stk*RO*reduced_E1_*CHA , sizeof(std::complex<float>)*RO*reduced_E1_*CHA);
                    }
                    //recon_obj.mb_e1_reduce_.create(RO, reduced_E1_, CHA, 1, STK, N, S);


                    // sinon faire la copie avec le pointeur direct et rajouter l'offset
                }
            }

        }
        else
        {
            hoNDArray<std::complex<float> > data;
            data.create(RO,E1,E2,CHA,MB,STK);

            for (size_t s = 0; s < S; s++)    {

                for (size_t n = 0; n < N; n++)  {

                    std::complex<float> * in = &(input(0, 0, 0, 0, 0, 0, n,s));
                    std::complex<float> * out = &(data(0, 0, 0, 0, 0, 0 ));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA*MB*STK);

                    hoNDArray<float_complext>* host_data =
                            reinterpret_cast< hoNDArray<float_complext>* >(&data);

                    cuNDArray<float_complext> device_data(host_data);
                    device_data.squeeze();

                    size_t ROg = device_data.get_size(0);
                    size_t E1g = device_data.get_size(1);
                    size_t CHAg = device_data.get_size(2);
                    size_t MBg = device_data.get_size(3);
                    size_t STKg = device_data.get_size(4);

                    GDEBUG_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: [RO E1 CHA MB STK] - [" << ROg << " " << E1g << " " << CHAg << " " << MBg  << " " << STKg  << "]");

                    cuNDArray<float_complext> device_data_out;
                    device_data_out.create(ROg,reduced_E1_,CHAg, MBg, STKg);

                    boost::shared_ptr<GPUTimer> process_timer;
                    process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::estimate_feasibility_with_STK (MB>1)") );

                    //TODO
                    estimate_feasibility_with_STK<float>( device_data, device_data_out,  acceFactorSMSE1_[0], start_E1_, end_E1_, reduced_E1_);
                    process_timer.reset();

                    //boost::shared_ptr< hoNDArray<complext<float> > >  host_result_tempo = device_data_out.to_host();
                    boost::shared_ptr< hoNDArray<float_complext  > >  host_result = device_data_out.to_host();

                    //boost::shared_ptr< hoNDArray<complext<float> > >  host_result = reinterpret_cast<hoNDArray<std::complex<float> > >(device_data_out.to_host());

                    hoNDArray<std::complex<float>> *test=reinterpret_cast<hoNDArray<std::complex<float> >*>(host_result.get());

                    //hoNDArray<std::complex<float>> test2;
                    //test2.create(RO,reduced_E1_,CHA, MB, STK);
                    //hoNDArray<std::complex<float>> test2(RO,reduced_E1_,CHA, MB, STK,test->get_data_ptr() );
                    //memcpy(&test2(0,0,0,0,0) , test->get_data_ptr(), sizeof(std::complex<float>)*RO*reduced_E1_*CHA*MB*STK);
                    //std::complex<float> * in2 = &test2(0, 0, 0, 0, 0);

                    std::complex<float> * out2 = &(output(0, 0, 0, 0, 0, n, s ));
                    memcpy(out2 , test->get_data_ptr(), sizeof(std::complex<float>)*RO*reduced_E1_*CHA*MB*STK);

                }
            }

        }

    }
    catch(...)
    {
        GERROR_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!! unable to perform_slice_grappa_unwrapping_gpu");
    }

}


void GenericReconCartesianSliceGrappavgpuGadget::do_gpu_unmix(hoNDArray<std::complex<float> > & input, hoNDArray<std::complex<float> > & output, hoNDArray<std::complex<float> > & kernel )
{


    size_t vsize = input.get_size(0);
    size_t ksize = input.get_size(1);
    size_t CHA_i = input.get_size(2);
    size_t MB_i = input.get_size(3);
    size_t STK_i = input.get_size(4);
    size_t N_i = input.get_size(5);
    size_t S_i = input.get_size(6);

    GDEBUG_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: input [voxels ksize CHA MB STK N S] - [" << vsize << " " << ksize << " " << CHA_i << " " << MB_i << " " << STK_i << " " << N_i<< " " << S_i << "]");


    size_t RO_o = output.get_size(0);
    size_t CHA = output.get_size(1);
    size_t MB = output.get_size(2);
    size_t STK = output.get_size(3);
    size_t N = output.get_size(4);
    size_t S = output.get_size(5);

    GDEBUG_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: ouput [RO E1 CHA MB STK N S] - [" << RO_o <<  " " << CHA << " " << MB << " " << STK << " " << N<< " " << S << "]");


    size_t RO_k = kernel.get_size(0);
    size_t CHA_k = kernel.get_size(2);
    size_t MB_k = kernel.get_size(3);
    size_t STK_k = kernel.get_size(4);
    size_t N_k = kernel.get_size(5);
    size_t S_k = kernel.get_size(6);

    GDEBUG_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: kernel [RO CHA MB STK N S] - [" << RO_k << " " << CHA_k << " " << MB_k << " " << STK_k << " " << N_k<< " " << S_k << "]");

    //cuNDArray<float_complext> device_MB(kernel_size_, voxels_number_per_image_ );
    // cuNDArray<float_complext> device_kernel(kernel_size_*CHA, CHA);
    // cuNDArray<float_complext> device_unfolded(voxels_number_per_image_, CHA);

    //boost::shared_ptr<GPUTimer> process_timer;
    //process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::allocation gpu unmix") );


    // https://us02web.zoom.us/j/81383000327
    cuNDArray<float_complext> device_MB(voxels_number_per_image_, kernel_size_*CHA);
    cuNDArray<float_complext> device_kernel(kernel_size_*CHA, CHA);
    cuNDArray<float_complext> device_unfolded(voxels_number_per_image_, CHA);

    //process_timer.reset();

    size_t ref_N = input.get_size(5);
    size_t ref_S = input.get_size(6);

    typedef std::complex<float> T;

    for (long long a = 0; a < STK; a++) {

        for (long long s = 0; s < S; s++)    {

            for (long long n = 0; n < N; n++)  {

                size_t usedN = n;
                if (n >= ref_N) usedN = ref_N - 1;

                size_t usedS = s;
                if (s >= ref_S) usedS = ref_S - 1;

                T *pIn = &(input(0, 0, 0, 0, a, n, s));

                for (size_t m = 0; m < MB_factor; m++) {

                    T *pkernel = &(kernel(0, 0, m, a, n, s));


                    T *pOut = &(output(0, 0, m, a, n, s));

                    //hoNDArray<std::complex<float> > tempo_MB(voxels_number_per_image_, kernel_size_*CHA, pIn);
                    //hoNDArray<std::complex<float> > tempo_kernel(kernel_size_*CHA, CHA, pkernel);
                    //hoNDArray<std::complex<float> > tempo_kspace_unfolded(voxels_number_per_image_, CHA, pOut);

                    //hoNDArray<float_complext>* host_MB =reinterpret_cast< hoNDArray<float_complext>* >(&tempo_MB);
                    //hoNDArray<float_complext>* host_kernel =reinterpret_cast< hoNDArray<float_complext>* >(&tempo_kernel);
                    // hoNDArray<float_complext>* host_unfolded =reinterpret_cast< hoNDArray<float_complext>* >(&tempo_kspace_unfolded);

                    //  cuNDArray<float_complext> device_MB(host_MB);
                    //  cuNDArray<float_complext> device_kernel(host_kernel);
                    // cuNDArray<float_complext> device_unfolded(host_unfolded);

                    if(cudaMemcpy(device_MB.get_data_ptr(),
                                  pIn,
                                  voxels_number_per_image_*kernel_size_*CHA*sizeof(std::complex<float>),
                                  cudaMemcpyHostToDevice)!= cudaSuccess )   {
                        GERROR_STREAM("Upload to device for device_MB failed\n");}

                    if(cudaMemcpy(device_kernel.get_data_ptr(),
                                  pkernel,
                                  kernel_size_*CHA* CHA*sizeof(std::complex<float>),
                                  cudaMemcpyHostToDevice)!= cudaSuccess )   {
                        GERROR_STREAM("Upload to device for device_kernel failed\n");}

                    /*  if(cudaMemcpy(device_kernel.get_data_ptr(),
                               host_kernel->get_data_ptr(),
                               kernel_size_*CHA* CHA*sizeof(std::complex<float>),
                               cudaMemcpyHostToDevice)!= cudaSuccess )   {
                            GERROR_STREAM("Upload to device for device_kernel failed\n");}*/

                    if(cudaMemcpy(device_unfolded.get_data_ptr(),
                                  pOut,
                                  voxels_number_per_image_*CHA*sizeof(std::complex<float>),
                                  cudaMemcpyHostToDevice)!= cudaSuccess )   {
                        GERROR_STREAM("Upload to host for device_unfolded failed\n");}



                    prepare_gpu_unmix(device_MB, device_kernel, device_unfolded);

                    if(cudaMemcpy(pOut,
                                  device_unfolded.get_data_ptr(),
                                  voxels_number_per_image_*CHA*sizeof(std::complex<float>),
                                  cudaMemcpyDeviceToHost)!= cudaSuccess )   {
                        GERROR_STREAM("Upload to host for device_kernel failed\n");}


                }
            }
        }
    }

}

void GenericReconCartesianSliceGrappavgpuGadget::do_gpu_test(hoNDArray<std::complex<float> > & input)
{
    //////////////////////////////

    //https://forums.developer.nvidia.com/t/how-do-you-allocate-4d-or-5d-arrays-on-gpu/27690/2
    //https://stackoverflow.com/questions/11798067/segmentation-of-4d-data-on-gpu
    //https://stackoverflow.com/questions/34529387/kernel-for-processing-a-4d-tensor-in-cuda
    //https://rvr.typepad.com/wind/2017/04/using-pycuda-to-boost-fits-image-processing-time.html
    //http://www-personal.umich.edu/~smeyer/cuda/grid.pdf
    //https://www.olcf.ornl.gov/wp-content/uploads/2018/03/Intro_to_CUDA.pdf
    try
    {

        //////////////////////////////////////////


        size_t RO = input.get_size(0);
        size_t E1 = input.get_size(1);
        size_t E2 = input.get_size(2);
        size_t CHA = input.get_size(3);
        size_t MB = input.get_size(4);
        size_t STK = input.get_size(5);
        size_t N = input.get_size(6);
        size_t S = input.get_size(7);



        //noter que l'on peut boucler sur les stacks pour que ce soit plus facile:


        hoNDArray< std::complex<float> > reduce_host_data;
        reduce_host_data.create(RO,E1,E2,CHA,MB);

        for (int a = 0; a < lNumberOfStacks_; a++)
        {

            std::complex<float> * in = &(input(0, 0, 0, 0, 0, a, 0 ,0 ));
            std::complex<float> * out = &(reduce_host_data(0, 0, 0, 0,  0));

            memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA *MB);

        }



        GDEBUG_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: [RO E1 E2 CHA MB STK N S] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N<< " " << S << "]");

        hoNDArray<float_complext>* host_data =
                reinterpret_cast< hoNDArray<float_complext>* >(&reduce_host_data);


        cuNDArray<float_complext> fourD_device_data(host_data);
        fourD_device_data.squeeze();

        cuNDArray<float_complext> fourD_device_data_two(host_data);
        fourD_device_data_two.squeeze();

        //std::vector<size_t> ftdims(2,0); ftdims[1] = 1;

        //Go to image space
        //cuNDFFT<float>::instance()->ifft( &device_data, &ftdims);

        size_t ROg = fourD_device_data.get_size(0);
        size_t E1g = fourD_device_data.get_size(1);
        //size_t E2g = device_data.get_size(2);
        size_t CHAg = fourD_device_data.get_size(2);
        size_t MBg = fourD_device_data.get_size(3);
        size_t STKg = fourD_device_data.get_size(4);
        size_t Ng = fourD_device_data.get_size(5);
        size_t Sg = fourD_device_data.get_size(6);

        GDEBUG_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: [RO E1 E2 CHA MB STK N S] - [" << ROg << " " << E1g << " " << CHAg << " " << MBg << " " << STKg << " " << Ng<< " " << Sg << "]");

        //std::vector<size_t> ftdims(2,0); ftdims[1] = 1;



        unsigned int dim_to_transform=0;
        if (perform_timing.value()) { gt_timer_local_.start("gpuExample:: fft and ifft cpu time");}
        boost::shared_ptr<GPUTimer> process_timer;
        process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::fft and ifft gpu time") );


        hoNDArray<float_complext>* host =
                reinterpret_cast< hoNDArray<float_complext>* >(&input);

        cuNDArray<float_complext> device_d(host);
        cuNDFFT<float>::instance()->fft(&device_d, dim_to_transform);
        cuNDFFT<float>::instance()->ifft(&device_d, dim_to_transform);


        boost::shared_ptr< hoNDArray<float_complext  > >  host_result = device_d.to_host();

        hoNDArray<std::complex<float>> *host_result_cast=reinterpret_cast<hoNDArray<std::complex<float> >*>(host_result.get());

        process_timer.reset();


        gt_timer_local_.stop();





        if (perform_timing.value()) { gt_timer_local_.start("cpuExample:: fft and ifft");}

        hoNDFFT<float>::instance()->fft(&input,0);
        hoNDFFT<float>::instance()->ifft(&input,0);

        gt_timer_local_.stop();

        // du coup il faut ecrire les fonctions
        //remove() t //im2col pour commencer

        //int target_coils_=10;

        //auto device_data_test = estimate_feasibility<float,2>( device_data,target_coils_);

        process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::estimate_feasibility_no_STK") );
        //TODO
        estimate_feasibility_no_STK<float>( fourD_device_data, fourD_device_data_two,  acceFactorSMSE1_[0], start_E1_, end_E1_, reduced_E1_);
        process_timer.reset();



        //cuNDArray<float_complext> device_data(reduce_host_data);
        //device_data.squeeze();

        //cuNDArray<float_complext> device_data_two(reduce_host_data);
        //device_data.squeeze();

        //process_timer = boost::shared_ptr<GPUTimer>( new GPUTimer("gpuExample::estimate_feasibility_with_STK") );
        //TODO
        //estimate_feasibility_with_STK<float>( device_data, device_data_two,  acceFactorSMSE1_[0], start_E1_, end_E1_, reduced_E1_);
        //process_timer.reset();

        // unsigned int number_of_batches = device_data.get_size(CHAg);
        //unsigned int number_of_elements = device_data.get_number_of_elements()/number_of_batches;

        //std::cout<< " number_of_batches :  "  << number_of_batches << " number_of_elements :" << number_of_elements << std::endl;

        // Setup block/grid dimensions
        //dim3 blockDim;
        //dim3 gridDim;
        //setup_grid( number_of_elements, &blockDim, &gridDim );

        // Find element stride
        //unsigned int stride; std::vector<size_t> dims;
        //find_stride< std::complex<float> >( device_data, dim, &stride, &dims );

        //std::cout<< " stride :  "  << stride  << std::endl;

    }
    catch(...)
    {
        GERROR_STREAM("!!!!!!!!!!!!!!!!!!!!!!!!! unable to do the gpu stuff");
    }

    ////////////////////////////////
    //boost::shared_array< hoNDArray<float_complext> >  csm_host_;
    //int slices_=10;
    //int sets_=5;
    //csm_host_ = boost::shared_array< hoNDArray<float_complext> >(new hoNDArray<float_complext>[slices_*sets_]);
    //boost::shared_ptr< cuNDArray<float_complext> > csm(new cuNDArray<float_complext> (csm_host_.get()));
    //boost::shared_ptr< cuNDArray<float_complext> > device_samples(new cuNDArray<float_complext> (j->dat_host_.get()));

    //////////////////////////////////////////

}


void GenericReconCartesianSliceGrappavgpuGadget::perform_slice_grappa_unwrapping(IsmrmrdReconBit &recon_bit, ReconObjType &recon_obj, size_t e)
{
    // unwrapping function : liste des opérations:
    // 1) remove_unnecessary_kspace (in presence of grappa)
    // 2) im2col
    // 3) reshape
    // 4) apply kernel : SB= MB*kernel -> gemm (SB, MB , Kernel)
    // 5) reshape
    // 6) permute

    //hoNDArray< std::complex<float> >& data = recon_bit.data_.data_;
    hoNDArray< std::complex<float> >& data = recon_obj.mb_compression_;

    //do_gpu_test(data);

    size_t RO = data.get_size(0);
    size_t E1 = data.get_size(1);
    size_t E2 = data.get_size(2);
    size_t CHA = data.get_size(3);
    size_t MB = data.get_size(4);
    size_t STK = data.get_size(5);
    size_t N = data.get_size(6);
    size_t S = data.get_size(7);

    GDEBUG_STREAM("GenericReconCartesianSliceGrappavgpuGadget - incoming data array data : [RO E1 E2 CHA MB STK N S] - [" << RO << " " << E1 << " " << E2 << " " << CHA <<  " " << MB <<  " " << STK << " " << N << " " << S<<  "]");

    recon_obj.mb_e1_reduce_.create(RO, reduced_E1_, CHA, 1, STK, N, S);




    /*if (use_gpu.value()==true)
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::gpuExample:: remove_unnecessary_kspace global");}
        remove_unnecessary_kspace_gpu( recon_obj.mb_compression_  ,  recon_obj.mb_e1_reduce_,  acceFactorSMSE1_[e], start_E1_, end_E1_, true);
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }
    else
    {*/

    if (use_omp.value()==true)
    {
        remove_unnecessary_kspace_open( recon_obj.mb_compression_ ,  recon_obj.mb_e1_reduce_,  acceFactorSMSE1_[e], start_E1_, end_E1_, true);
    }
    else
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: remove_unnecessary_kspace global");}
        remove_unnecessary_kspace( recon_obj.mb_compression_ ,  recon_obj.mb_e1_reduce_,  acceFactorSMSE1_[e], start_E1_, end_E1_, true);
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }
    //}

    if (!debug_folder_full_path_.empty())
    {
        //show_size(mb_reduce,"mb_reduce");
        save_4D_with_STK_5(recon_obj.mb_e1_reduce_, "mb_e1_reduce_", "0");
    }

    // lui il doit être créé à chaque fois car on le reshape
    recon_obj.block_MB_.create(voxels_number_per_image_, kernel_size_, CHA, 1, STK, N, S);

    if (use_gpu.value()==true)
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::gpuExample:: im2col global");}
        im2col_gpu(recon_obj.mb_e1_reduce_,recon_obj.block_MB_, blocks_RO_,  blocks_E1_,  grappa_kSize_RO.value(), grappa_kSize_E1.value() );
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }
    else
    {
        if (use_omp.value()==true)
        {
            im2col_open(recon_obj.mb_e1_reduce_,recon_obj.block_MB_, blocks_RO_,  blocks_E1_,  grappa_kSize_RO.value(), grappa_kSize_E1.value() );
        }
        else
        {
            if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: im2col global");}
            im2col(recon_obj.mb_e1_reduce_,recon_obj.block_MB_, blocks_RO_,  blocks_E1_,  grappa_kSize_RO.value(), grappa_kSize_E1.value() );
            if (perform_timing.value()) { gt_timer_local_.stop();}
        }
    }


    if (!debug_folder_full_path_.empty())
    {
        //show_size(block_MB,"block_MB");
        save_4D_with_STK_5(recon_obj.block_MB_,"block_MB_", "0");
    }

    if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: reshape global");}
    std::vector<size_t> newdims;
    newdims.push_back(blocks_RO_*blocks_E1_);
    newdims.push_back(kernel_size_*CHA);
    newdims.push_back(1);
    newdims.push_back(1);
    newdims.push_back(STK); //STK
    newdims.push_back(N); //N
    newdims.push_back(S); //S
    recon_obj.block_MB_.reshape(&newdims);

    if (perform_timing.value()) { gt_timer_local_.stop();}

    if (!debug_folder_full_path_.empty())
    {
        //show_size(recon_obj.block_MB_,"block_MB reshape");
        save_4D_with_STK_5(recon_obj.block_MB_,"", "0");
    }

    //TODO ceci devrait etre alloué une seule fois et non a chaque passage
    recon_obj.unfolded_image_.create(voxels_number_per_image_, CHA, MB_factor, STK,  N, S);
    recon_obj.unfolded_image_permute_.create(blocks_RO_,blocks_E1_,CHA, MB_factor, STK,  N, S);


    /*
    //old version part 1/3
    //TODO à simplifier, ne pas créer tout ce monde à chaque fois
    hoNDArray<std::complex<float> > tempo_MB(voxels_number_per_image_, kernel_size_*CHA);
    hoNDArray<std::complex<float> > tempo_kernel(kernel_size_*CHA, CHA );
    hoNDArray<std::complex<float> > tempo_kspace_unfolded(voxels_number_per_image_, CHA);

    if (perform_timing.value()) {gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: unmix version 1 ");}

    for (long long a = 0; a < STK; a++) {

        for (long long s = 0; s < S; s++)    {

            for (long long n = 0; n < N; n++)  {

                //old version part 2/3
                Gadgetron::clear(tempo_MB);
                std::complex<float> * in = &(recon_obj.block_MB_(0, 0, 0, 0, a, n, s));  //[voxels_number_per_image_, kernel_size_, CHA, 1, STK, N, S ]
                std::complex<float> * out = &(tempo_MB(0, 0));
                memcpy(out , in, sizeof(std::complex<float>)*voxels_number_per_image_*kernel_size_*CHA );

                //old version part 3/3
                for (size_t m = 0; m < MB_factor; m++) {
                    Gadgetron::clear(tempo_kernel);
                    std::complex<float> * in2;

                    if (calib_fast.value()==true)
                    {
                        in2 = &(recon_obj.kernelonov_(0, 0, m, a, n, s));
                    }
                    else
                    {
                        in2 = &(recon_obj.kernel_(0, 0, m, a, n, s));
                    }
                    std::complex<float> * out2 = &(tempo_kernel(0, 0));
                    memcpy(out2 , in2, sizeof(std::complex<float>)*kernel_size_*CHA*CHA );
                    Gadgetron::clear(tempo_kspace_unfolded);
                    Gadgetron::gemm(tempo_kspace_unfolded, tempo_MB, false, tempo_kernel, false);
                    std::complex<float> * image_in = &(tempo_kspace_unfolded(0, 0));
                    std::complex<float> * image_out = &(recon_obj.unfolded_image_(0, 0, m, a, n, s));

                    memcpy(image_out , image_in, sizeof(std::complex<float>)*voxels_number_per_image_*CHA);
                }
            }
        }
    }


    /// on va essayer de faire un code gpu de gemm

    if (perform_timing.value()) {gt_timer_local_.stop();}*/

    /*gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::process  unmix version 3 gpu  ");

    do_gpu_unmix(recon_obj.block_MB_,recon_obj.unfolded_image_, recon_obj.kernelonov_);

    gt_timer_local_.stop();
    */

     gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::process  unmix version 2 ");

    size_t ref_N = recon_obj.block_MB_.get_size(5);
    size_t ref_S = recon_obj.block_MB_.get_size(6);

    typedef std::complex<float> T;

    for (long long a = 0; a < STK; a++) {

        for (long long s = 0; s < S; s++)    {

            for (long long n = 0; n < N; n++)  {

                size_t usedN = n;
                if (n >= ref_N) usedN = ref_N - 1;

                size_t usedS = s;
                if (s >= ref_S) usedS = ref_S - 1;

                T *pIn = &(recon_obj.block_MB_(0, 0, 0, 0, a, n, s));

                for (size_t m = 0; m < MB_factor; m++) {

                    T *pkernel;

                    if (calib_fast.value()==true)
                    {
                        pkernel = &(recon_obj.kernelonov_(0, 0, m, a, n, s));
                    }
                    else
                    {
                        pkernel = &(recon_obj.kernel_(0, 0, m, a, n, s));
                    }

                    //T *pkernel = &(recon_obj.kernelonov_(0, 0, m, a, n, s));

                    T *pOut = &(recon_obj.unfolded_image_(0, 0, m, a, n, s));

                    hoNDArray<std::complex<float> > tempo_MB(voxels_number_per_image_, kernel_size_*CHA, pIn);
                    hoNDArray<std::complex<float> > tempo_kernel(kernel_size_*CHA, CHA, pkernel);
                    hoNDArray<std::complex<float> > tempo_kspace_unfolded(voxels_number_per_image_, CHA, pOut);

                    Gadgetron::apply_unmix_coeff_kspace_SMS(tempo_MB, tempo_kernel, tempo_kspace_unfolded );

                }
            }
        }
    }

    gt_timer_local_.stop();




    ///////////////

    if (!debug_folder_full_path_.empty())
    {
        //show_size(recon_obj.unfolded_image_,"unfolded_image");
        save_4D_with_STK_5(recon_obj.unfolded_image_, "unfolded_image", "0");
        save_4D_data(recon_obj.unfolded_image_, "unfolded_image", "0");
    }

    if (perform_timing.value()) {gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: reshape 2 ");}

    std::vector<size_t> newdims2;
    newdims2.push_back(blocks_E1_);
    newdims2.push_back(blocks_RO_);
    newdims2.push_back(CHA);
    newdims2.push_back(MB_factor);
    newdims2.push_back(STK); //N
    newdims2.push_back(N); //S
    newdims2.push_back(S); //STK
    recon_obj.unfolded_image_.reshape(&newdims2);

    if (perform_timing.value()) {gt_timer_local_.stop();}

    if (!debug_folder_full_path_.empty())
    {
        //show_size(recon_obj.unfolded_image_,"unfolded_image_reshape");
        save_4D_with_STK_5(recon_obj.unfolded_image_, "unfolded_image_reshape", "0");
    }

    if (perform_timing.value()) {gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: permute");}
    std::vector<size_t> newdims3;
    newdims3.push_back(1);
    newdims3.push_back(0);
    newdims3.push_back(2);
    newdims3.push_back(3);
    newdims3.push_back(4); //N
    newdims3.push_back(5); //S
    newdims3.push_back(6); //STK

    recon_obj.unfolded_image_permute_=permute(recon_obj.unfolded_image_,newdims3);

    if (perform_timing.value()) {gt_timer_local_.stop();}

    if (!debug_folder_full_path_.empty())
    {
        //show_size(recon_obj.unfolded_image_permute_,"unfolded_image_permute");
    }

}



void  GenericReconCartesianSliceGrappavgpuGadget::perform_slice_grappa_calib(IsmrmrdReconBit &recon_bit,  ReconObjType &recon_obj, size_t e)
{

    hoNDArray< std::complex<float> >& sb = recon_obj.sb_compression_;

    size_t RO = sb.get_size(0);
    size_t E1 = sb.get_size(1);
    size_t E2 = sb.get_size(2);
    size_t CHA = sb.get_size(3);
    size_t MB = sb.get_size(4);
    size_t STK = sb.get_size(5);
    size_t N = sb.get_size(6);
    size_t S = sb.get_size(7);

    GDEBUG_STREAM("GenericReconCartesianSliceGrappavgpuGadget - incoming data array sb : [RO E1 E2 CHA MB STK N S] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N<< " " << S << "]");

    recon_obj.sb_e1_reduce_.create(RO, reduced_E1_, CHA, MB, STK, N, S);

    //---------------------------------
    // TODO attention dimension MB =1 car c'est mb par définition il n'y a pas plusieurs coupes dans cette direction
    // recon_obj.mb_e1_reduce_.create(RO, reduced_E1_, CHA, 1, STK, N, S);
    // TODO ceci devrait etre alloué une seule fois et non a chaque passage
    // recon_obj.unfolded_image_.create(voxels_number_per_image_, CHA, MB_factor, STK,  N, S);
    // recon_obj.unfolded_image_permute_.create(blocks_RO_,blocks_E1_,CHA, MB_factor, STK,  N, S);
    // allocation unmixing comme cela c est fait une seule fois
    //---------------------------------

    if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::cpuExample:: remove_unnecessary_kspace calib");}

    if (use_omp.value()==true)
    {
        remove_unnecessary_kspace_open( recon_obj.sb_compression_ ,  recon_obj.sb_e1_reduce_,  acceFactorSMSE1_[e], start_E1_, end_E1_ , false);
    }
    else
    {
        remove_unnecessary_kspace( recon_obj.sb_compression_ ,  recon_obj.sb_e1_reduce_,  acceFactorSMSE1_[e], start_E1_, end_E1_ , false);
    }

    //if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::gpuExample:: remove_unnecessary_kspace calib");}
    //remove_unnecessary_kspace_gpu( recon_obj.sb_compression_ ,  recon_obj.sb_e1_reduce_,  acceFactorSMSE1_[e], start_E1_, end_E1_, false);
    //if (perform_timing.value()) { gt_timer_local_.stop();}

    if (!debug_folder_full_path_.empty())
    {
        //show_size(sb_reduce,"sb_reduce");
        save_4D_with_STK_5(recon_obj.sb_e1_reduce_, "sb_e1_reduce_", "0");
    }

    ///////////////

    recon_obj.block_SB_.create(voxels_number_per_image_, kernel_size_, CHA, MB, STK, N, S);
    recon_obj.missing_data_.create( voxels_number_per_image_, CHA, MB, STK, N, S);

    recon_obj.kernel_.create( CHA*kernel_size_, CHA, MB, STK, N, S);
    recon_obj.kernelonov_.create( CHA*kernel_size_, CHA, MB, STK, N, S);

    /////////
    if (use_omp.value()==true)
    {
        im2col_open(recon_obj.sb_e1_reduce_, recon_obj.block_SB_, blocks_RO_,  blocks_E1_,  grappa_kSize_RO.value(), grappa_kSize_E1.value() );
    }
    else
    {
        im2col(recon_obj.sb_e1_reduce_, recon_obj.block_SB_, blocks_RO_,  blocks_E1_,  grappa_kSize_RO.value(), grappa_kSize_E1.value() );
    }


    if (!debug_folder_full_path_.empty())
    {
        save_4D_8D_kspace(recon_obj.block_SB_, "block_SB_", "0");
    }

    if (blocking==true)
    {
        GDEBUG("blocking == true\n");
    }
    else
    {
        GDEBUG("blocking == false\n");
        extract_milieu_kernel(recon_obj.block_SB_,  recon_obj.missing_data_, kernel_size_,  voxels_number_per_image_);
    }

    if (!debug_folder_full_path_.empty())
    {
        save_4D_8D_kspace(recon_obj.missing_data_, "missing_data", "0");
    }

    CMK_matrix.set_size(CHA*kernel_size_, voxels_number_per_image_);
    measured_data_matrix.set_size(voxels_number_per_image_, CHA*kernel_size_);

    long long s, n, a, m, cha, e2, e1;

    hoNDArray<std::complex<float> > tempo_matrix(voxels_number_per_image_, kernel_size_,  CHA, MB);

    for (s = 0; s < S; s++)    {

        for (n = 0; n < N; n++)  {

            for (a = 0; a < STK; a++) {

                Gadgetron::clear(tempo_matrix);
                hoNDArray<std::complex<float> > measured_data;

                //show_size(tempo," tempo ");
                std::complex<float> * in = &(recon_obj.block_SB_(0, 0, 0 ,0 , a, n, s)); // [7316 25 12 2 4 1 1 1]
                std::complex<float> * out = &(tempo_matrix(0, 0, 0 ));

                memcpy(out, in, sizeof(std::complex<float>)*voxels_number_per_image_*kernel_size_*CHA*MB );

                // sum over the MB dimension to average the kspace data (simulation of sms aliasing)
                Gadgetron::sum_over_dimension(tempo_matrix, measured_data, 3);

                if (!debug_folder_full_path_.empty())
                {
                    std::stringstream stk;
                    stk << "_stack_" << a;
                    gt_exporter_.export_array_complex(measured_data, debug_folder_full_path_ + "measured_data" + stk.str());
                }

                std::vector<size_t> newdims;
                newdims.push_back(measured_data.get_size(0));
                newdims.push_back(measured_data.get_size(1)* measured_data.get_size(2));
                measured_data.reshape(&newdims);

                if (!debug_folder_full_path_.empty())
                {
                    std::stringstream stk;
                    stk << "_stack_" << a;
                    gt_exporter_.export_array_complex(measured_data, debug_folder_full_path_ + "measured_data_reorganise" + stk.str());
                }

                //TODO rajouter les différentes implémentations
                if (calib_fast.value()==true)
                {
                    size_t rowA=voxels_number_per_image_;
                    size_t colA=kernel_size_*CHA;

                    size_t colB=CHA;

                    double thres=grappa_reg_lamda.value();

                    gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::process  Tikhonov ");

                    for (long long m = 0; m < MB; m++)
                    {
                        hoMatrix<std::complex<float>> A, B, x(colA, colB);

                        hoNDArray<std::complex<float>> A_mem(rowA, colA);
                        A.createMatrix(rowA, colA, A_mem.begin());
                        std::complex<float>* pA = A.begin();

                        hoNDArray<std::complex<float>> B_mem(rowA, colB);
                        B.createMatrix(rowA, colB, B_mem.begin());
                        std::complex<float>* pB = B.begin();

                        size_t i,j;

                        for (j = 0; j < colA; j++) {
                            for (i = 0; i < rowA; i++) {

                                pA[i + j*rowA]=measured_data(i,j);
                            }
                        }

                        for (j = 0; j < colB; j++) {
                            for (i = 0; i < rowA; i++) {
                                pB[i + j*rowA  ]   = recon_obj.missing_data_(i, j, m,  a, n, s);
                            }
                        }

                        SolveLinearSystem_Tikhonov(A, B, x, thres);

                        std::complex<float> * x_in = &(x(0, 0));
                        std::complex<float> * x_out = &(recon_obj.kernelonov_(0, 0, m, a, n, s));

                        memcpy(x_out , x_in, sizeof(std::complex<float>)*colA*colB);

                    }

                    gt_timer_local_.stop();

                }
                else
                {
                    /// solver
                    ///
                    size_t p, q;

                    gt_timer_local_.start("GenericReconCartesianSliceGrappavgpuGadget::process  pinv ");

                    for (p = 0; p < size(measured_data_matrix,0); p++)
                    {
                        for (q = 0; q < size(measured_data_matrix,1); q++)
                        {
                            measured_data_matrix(p,q)=measured_data(p,q);
                        }
                    }

                    CMK_matrix = pinv(measured_data_matrix.t()*measured_data_matrix)*measured_data_matrix.t();

                    hoNDArray<std::complex<float> > CMK(measured_data.get_size(1),measured_data.get_size(0));

                    for (p = 0; p < measured_data.get_size(1); p++)
                    {
                        for (q = 0; q < measured_data.get_size(0); q++)
                        {
                            CMK(p,q)=CMK_matrix(p,q);
                        }
                    }

                    for (long long m = 0; m < MB; m++)
                    {
                        hoNDArray< std::complex<float> > miss(voxels_number_per_image_, CHA);

                        std::complex<float> * in = &(recon_obj.missing_data_(0, 0, m,  a, n, s));
                        std::complex<float> * out = &(miss(0, 0));

                        memcpy(out , in, sizeof(std::complex<float>)*voxels_number_per_image_*CHA);

                        //show_size(miss," miss ");

                        hoNDArray< std::complex<float> > lala(CHA*kernel_size_, CHA);

                        Gadgetron::gemm(lala, CMK, false, miss, false);
                        //gemm( lala, CMK,miss);

                        std::complex<float> * kernel_in = &(lala(0, 0));
                        std::complex<float> * kernel_out = &(recon_obj.kernel_(0, 0, m, a, n, s));

                        memcpy(kernel_out , kernel_in, sizeof(std::complex<float>)*CHA*kernel_size_*CHA);

                    }
                    gt_timer_local_.stop();
                }
            }
        }

        if (!debug_folder_full_path_.empty())
        {
            save_4D_data(recon_obj.kernel_, "kernel", "0");
            save_4D_data(recon_obj.kernelonov_, "kernelonov", "0");
        }

    }

    // -----------------------------------
}





void GenericReconCartesianSliceGrappavgpuGadget::recopy_kspace(  ReconObjType &recon_obj, hoNDArray< std::complex<float> >& output, size_t acc )
{

    size_t RO = recon_obj.unfolded_image_.get_size(1);
    size_t E1 = recon_obj.unfolded_image_.get_size(0);
    size_t CHA = recon_obj.unfolded_image_.get_size(2);
    size_t MB = recon_obj.unfolded_image_.get_size(3);
    size_t STK = recon_obj.unfolded_image_.get_size(4);
    size_t N = recon_obj.unfolded_image_.get_size(5);
    size_t S = recon_obj.unfolded_image_.get_size(6);

    //show_size(recon_obj.unfolded_image_," input ");
    //show_size(output," output ");

    size_t index_x, index_y ;

    index_x=2;

    size_t s,n,a,m,cha,e1,ro;

    //std::cout << "start_E1_ " << start_E1_+2*acc<<  " end_E1_ " <<  end_E1_-2*acc<< " acc "   << acc << " E1 "   << E1<< std::endl;

    for (s = 0; s < S; s++) {

        for (n = 0; n < N; n++) {

            for (a = 0; a < STK; a++) {

                for (m = 0; m < MB; m++) {

                    for (cha = 0; cha < CHA; cha++) {

                        index_y=start_E1_+2*acc;  // on se déplace de 2 lignes * accélération selon y
                        //TODO  il faut ajuster le nombre 2 en fonction de la taille du kernel

                        for (e1 = 0; e1 < E1; e1++) {

                            std::complex<float> * in2 = & (recon_obj.unfolded_image_permute_(0, e1, cha, m, a, n, s));
                            std::complex<float> * out2 = &(output(index_x, index_y, 0, cha, m, a, n, s));

                            memcpy(out2 , in2, sizeof(std::complex<float>)*RO);

                            //if (m==0 && cha==0 && a==0) { std::cout << " e1 " << e1 <<  " index_y "<< index_y << " RO "<< RO<< std::endl;}

                            index_y+=acc;
                        }
                    }
                }
            }
        }
    }

    //std::cout << "reduced_E1_-4  "<< reduced_E1_-2*acc<< "  index_x "<< index_x  << "  index_y "<< index_y  << std::endl;

    //recopy_kspace( m1->getObjectPtr()->rbit_[e].data_.data_ ,  acceFactorSMSE1_[e] );}

    //GADGET_CHECK_THROW(reduced_E1_ == index+1);

}



GADGET_FACTORY_DECLARE(GenericReconCartesianSliceGrappavgpuGadget)
}


#include "GenericReconCartesianSliceGrappav0Gadget.h"
#include "mri_core_grappa.h"
#include "hoNDArray_reductions.h"
#include "hoNDArray_linalg.h"


/*
    The input is IsmrmrdReconData and output is single 2D or 3D ISMRMRD images

    If required, the gfactor map can be sent out

    If the  number of required destination channel is 1, the GrappaONE recon will be performed

    The image number computation logic is implemented in compute_image_number function, which can be overloaded
*/

namespace Gadgetron {

GenericReconCartesianSliceGrappav0Gadget::GenericReconCartesianSliceGrappav0Gadget() : BaseClass() {
}

GenericReconCartesianSliceGrappav0Gadget::~GenericReconCartesianSliceGrappav0Gadget() {
}

int GenericReconCartesianSliceGrappav0Gadget::process_config(ACE_Message_Block *mb) {
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

    return GADGET_OK;
}

int GenericReconCartesianSliceGrappav0Gadget::process(Gadgetron::GadgetContainerMessage<IsmrmrdReconData> *m1) {
    if (perform_timing.value()) { gt_timer_local_.start("GenericReconCartesianSliceGrappav0Gadget::process"); }

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

        os << "_encoding_" << e;

        GDEBUG_CONDITION_STREAM(verbose.value(),
                                "Calling " << process_called_times_ << " , encoding space : " << e);
        GDEBUG_CONDITION_STREAM(verbose.value(),
                                "======================================================================");

        if (recon_bit_->rbit_[e].sb_  &&  recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
            define_usefull_parameters(recon_bit_->rbit_[e], e);
            this->define_kernel_parameters(recon_bit_->rbit_[e], e);
        }

        if (recon_bit_->rbit_[e].sb_)
        {
            save_4D_with_STK_8(recon_bit_->rbit_[e].sb_->data_, "FID_SB_Slice", os.str());

            // std::cout << " je suis la structure qui contient les données single band" << std::endl;
            if (perform_timing.value()) { gt_timer_.start("GenericReconCartesianSliceGrappav0Gadget::perform_calib"); }
            this->perform_slice_grappa_calib(recon_bit_->rbit_[e], recon_obj_[e], e);
            if (perform_timing.value()) { gt_timer_.stop(); }
        }

        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
            save_4D_with_SLC_7(recon_bit_->rbit_[e].data_.data_, "FID_MB_Slice", os.str());

            if (perform_timing.value()) { gt_timer_.start("GenericReconCartesianSliceGrappav0Gadget::perform_slice_grappa_unwrapping"); }
            this->perform_slice_grappa_unwrapping(recon_bit_->rbit_[e], recon_obj_[e], e);
            if (perform_timing.value()) { gt_timer_.stop(); }
        }

    }

    if (perform_timing.value()) { gt_timer_local_.stop(); }


    if (this->next()->putq(m1) < 0)
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }


    return GADGET_OK;
}


void GenericReconCartesianSliceGrappav0Gadget::define_kernel_parameters(IsmrmrdReconBit &recon_bit, size_t e)
{

hoNDArray< std::complex<float> >& data = recon_bit.sb_->data_;

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

void GenericReconCartesianSliceGrappav0Gadget::perform_slice_grappa_unwrapping(IsmrmrdReconBit &recon_bit, ReconObjType &recon_obj, size_t e)
{
    hoNDArray< std::complex<float> >& data = recon_bit.data_.data_;

    size_t RO = data.get_size(0);
    size_t E1 = data.get_size(1);
    size_t E2 = data.get_size(2);
    size_t CHA = data.get_size(3);
    size_t N = data.get_size(4);
    size_t S = data.get_size(5);
    size_t STK = data.get_size(6);

    GDEBUG_STREAM("GenericReconCartesianSliceGrappav0Gadget - incoming data array data : [RO E1 E2 CHA N S STK] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S<< " " << STK << "]");

    hoNDArray< std::complex<float> > mb_reduce(RO, reduced_E1_, CHA, 1, STK, N, S);
    remove_unnecessary_kspace_mb( recon_bit.data_.data_ ,  mb_reduce,  acceFactorSMSE1_[e] );

    if (!debug_folder_full_path_.empty())
    {
        show_size(mb_reduce,"mb_reduce");
        save_4D_with_STK_5(mb_reduce, "mb_reduce", "0");
    }

    hoNDArray< std::complex<float> > block_MB;
    block_MB.create(voxels_number_per_image_, kernel_size_, CHA, 1, STK, N, S);

    im2col(mb_reduce,block_MB);
    show_size(block_MB,"block_MB");
    save_4D_with_STK_5(block_MB,"block_MB", "0");

    std::vector<size_t> newdims;
    newdims.push_back(blocks_RO_*blocks_E1_);
    newdims.push_back(kernel_size_*CHA);
    newdims.push_back(1);
    newdims.push_back(1);
    newdims.push_back(STK); //STK
    newdims.push_back(N); //N
    newdims.push_back(S); //S
    block_MB.reshape(&newdims);

    show_size(block_MB,"block_MB reshape");
    save_4D_with_STK_5(block_MB,"block_MB_reshape", "0");

    unfolded_image.create(voxels_number_per_image_,  1, CHA, MB_factor, STK,  N, S);

    show_size(kernel,"kernel");

    // attention les dimensions ne sont pas les mêmes

    size_t s, n, a, m;

    for (a = 0; a < STK; a++) {

        for (s = 0; s < S; s++)    {

            for (n = 0; n < N; n++)  {

                hoNDArray<std::complex<float> > tempo_MB(voxels_number_per_image_, kernel_size_*CHA);

                std::complex<float> * in = &(block_MB(0, 0, 0, 0, a, n, s));
                std::complex<float> * out = &(tempo_MB(0, 0));

                memcpy(out , in, sizeof(std::complex<float>)*voxels_number_per_image_*kernel_size_*CHA );

                for (m = 0; m < MB_factor; m++)    {

                    hoNDArray<std::complex<float> > tempo_kernel(kernel_size_*CHA, CHA );

                     std::complex<float> * in2;

                    if (calib_fast.value()==1)
                    {
                        in2 = &(kernelonov(0, 0, m, a, n, s));
                    }
                    else
                    {
                        in2 = &(kernel(0, 0, m, a, n, s));
                    }
                    std::complex<float> * out2 = &(tempo_kernel(0, 0));

                    memcpy(out2 , in2, sizeof(std::complex<float>)*kernel_size_*CHA*CHA );

                    hoNDArray< std::complex<float> > lili(voxels_number_per_image_, CHA);

                    Gadgetron::gemm(lili, tempo_MB, false, tempo_kernel, false);

                    std::complex<float> * image_in = &(lili(0, 0));
                    std::complex<float> * image_out = &(unfolded_image(0, 0, 0, m,  a, n, s));

                    memcpy(image_out , image_in, sizeof(std::complex<float>)*voxels_number_per_image_*CHA);

                }

            }
        }
    }

    show_size(unfolded_image,"unfolded_image");

    save_4D_with_STK_5(unfolded_image, "unfolded_image", "0");

    std::vector<size_t> newdims2;
    newdims2.push_back(blocks_E1_);
    newdims2.push_back(blocks_RO_);
    newdims2.push_back(CHA);
    newdims2.push_back(MB_factor);
    newdims2.push_back(STK); //N
    newdims2.push_back(N); //S
    newdims2.push_back(S); //STK
    unfolded_image.reshape(&newdims2);

    show_size(unfolded_image,"unfolded_image_reshape");

    save_4D_with_STK_5(unfolded_image, "unfolded_image_reshape", "0");

}




void  GenericReconCartesianSliceGrappav0Gadget::perform_slice_grappa_calib(IsmrmrdReconBit &recon_bit, ReconObjType &recon_obj, size_t e) {


    hoNDArray< std::complex<float> >& sb = recon_bit.sb_->data_;

    size_t RO = sb.get_size(0);
    size_t E1 = sb.get_size(1);
    size_t E2 = sb.get_size(2);
    size_t CHA = sb.get_size(3);
    size_t MB = sb.get_size(4);
    size_t STK = sb.get_size(5);
    size_t N = sb.get_size(6);
    size_t S = sb.get_size(7);

    GDEBUG_STREAM("GenericReconCartesianSliceGrappav0Gadget - incoming data array sb : [RO E1 E2 CHA MB STK N S] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << MB << " " << STK << " " << N<< " " << S << "]");

    hoNDArray< std::complex<float> > sb_reduce(RO, reduced_E1_, CHA, MB, STK, N, S);

    remove_unnecessary_kspace_sb( recon_bit.sb_->data_ ,  sb_reduce,  acceFactorSMSE1_[e] );

    if (!debug_folder_full_path_.empty())
    {
        show_size(sb_reduce,"sb_reduce");
        save_4D_with_STK_5(sb_reduce, "sb_reduce", "0");
    }


    ///////////////

    hoNDArray< std::complex<float> > block_SB;
    block_SB.create(voxels_number_per_image_, kernel_size_, CHA, MB, STK, N, S);

    hoNDArray< std::complex<float> > missing_data;
    missing_data.create( voxels_number_per_image_, CHA, MB, STK, N, S);

    kernel.create( CHA*kernel_size_, CHA, MB, STK, N, S);
    kernelonov.create( CHA*kernel_size_, CHA, MB, STK, N, S);

    /////////

    im2col(sb_reduce,block_SB);
    //show_size(block_SB, " block_SB ");

    save_4D_8D_kspace(block_SB, "block_SB", "0");

    if (blocking==true)
    {

    }
    else
    {
        //GDEBUG("blocking == false");
        extract_milieu_kernel(block_SB,  missing_data);
    }

    CMK_matrix.set_size(CHA*kernel_size_, voxels_number_per_image_);
    measured_data_matrix.set_size(voxels_number_per_image_, CHA*kernel_size_);

    size_t s, n, a, m, cha, e2, e1;

    for (s = 0; s < S; s++)    {

        for (n = 0; n < N; n++)  {

            for (a = 0; a < STK; a++) {

                hoNDArray<std::complex<float> > tempo(voxels_number_per_image_, kernel_size_,  CHA, MB);
                hoNDArray<std::complex<float> > measured_data;

                //show_size(tempo," tempo ");
                std::complex<float> * in = &(block_SB(0, 0, 0,  a, n, s));
                std::complex<float> * out = &(tempo(0, 0, 0 ));

                memcpy(out, in, sizeof(std::complex<float>)*voxels_number_per_image_*kernel_size_*CHA*MB );

                // sum over the MB dimension to average the kspace data (simulation of sms)
                Gadgetron::sum_over_dimension(tempo, measured_data, 3);
                //show_size(measured_data," measured_data ");


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

                //show_size(measured_data," measured_data ");

                if (!debug_folder_full_path_.empty())
                {
                    std::stringstream stk;
                    stk << "_stack_" << a;
                    gt_exporter_.export_array_complex(measured_data, debug_folder_full_path_ + "measured_data_reorganise" + stk.str());
                }

                //TODO rajouter les différentes implémentations
                if (calib_fast.value()==1)
                {
                    size_t rowA=voxels_number_per_image_;
                    size_t colA=kernel_size_*CHA;

                    size_t colB=CHA;

                    double thres=grappa_reg_lamda.value();

                    gt_timer_local_.start("GenericReconCartesianSliceGrappav0Gadget::process  Tikhonov ");

                    for (unsigned long m = 0; m < MB; m++)
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
                                pB[i + j*rowA  ]   = missing_data(i, j, m,  a, n, s);
                            }
                        }

                        SolveLinearSystem_Tikhonov(A, B, x, thres);

                        std::complex<float> * x_in = &(x(0, 0));
                        std::complex<float> * x_out = &(kernelonov(0, 0, m,  a, n, s));

                        memcpy(x_out , x_in, sizeof(std::complex<float>)*colA*colB);

                    }

                    gt_timer_local_.stop();
                    save_4D_data(kernelonov, "kernelonov", "0");
                }
                else
                {
                    /// solver
                    ///
                    size_t p, q;

                    gt_timer_local_.start("GenericReconCartesianSliceGrappav0Gadget::process  pinv ");

                    for (p = 0; p < size(measured_data_matrix,0); p++)
                    {
                        for (q = 0; q < size(measured_data_matrix,1); q++)
                        {
                            measured_data_matrix(p,q)=measured_data(p,q);
                        }
                    }

                    CMK_matrix = pinv(measured_data_matrix.t()*measured_data_matrix)*measured_data_matrix.t();

                    //std::cout << size(CMK_matrix,0 )<< "  "<< size(CMK_matrix,1) <<std::endl;

                    hoNDArray<std::complex<float> > CMK(measured_data.get_size(1),measured_data.get_size(0));

                    for (p = 0; p < measured_data.get_size(1); p++)
                    {
                        for (q = 0; q < measured_data.get_size(0); q++)
                        {
                            CMK(p,q)=CMK_matrix(p,q);
                        }
                    }




                    for (unsigned long m = 0; m < MB; m++)
                    {
                        hoNDArray< std::complex<float> > miss(voxels_number_per_image_, CHA);

                        std::complex<float> * in = &(missing_data(0, 0, m,  a, n, s));
                        std::complex<float> * out = &(miss(0, 0));

                        memcpy(out , in, sizeof(std::complex<float>)*voxels_number_per_image_*CHA);

                        //show_size(miss," miss ");

                        hoNDArray< std::complex<float> > lala(CHA*kernel_size_, CHA);

                        Gadgetron::gemm(lala, CMK, false, miss, false);
                        //gemm( lala, CMK,miss);

                        std::complex<float> * kernel_in = &(lala(0, 0));
                        std::complex<float> * kernel_out = &(kernel(0, 0, m,  a, n, s));

                        memcpy(kernel_out , kernel_in, sizeof(std::complex<float>)*CHA*kernel_size_*CHA);

                    }


                    gt_timer_local_.stop();
                }



                //show_size(missing_data," missing_data ");
                //show_size(kernel," kernel ");
            }
        }
    }

    show_size(kernel, " kernel out ");
    save_4D_data(kernel, "kernel", "0");
    // -----------------------------------
}


void GenericReconCartesianSliceGrappav0Gadget::remove_unnecessary_kspace_sb(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& output, size_t acc )
{

    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t MB = input.get_size(4);
    size_t STK = input.get_size(5);
    size_t N = input.get_size(6);
    size_t S = input.get_size(7);

    size_t index;

    size_t s,n,a,m,cha,e2,e1;

    for (s = 0; s < S; s++)
    {
        for (n = 0; n < N; n++)
        {
            for (a = 0; a < STK; a++) {

                for (m = 0; m < MB; m++) {

                    for (cha = 0; cha < CHA; cha++) {

                        index=0;

                        for (e1 = start_E1_; e1 <= end_E1_; e1+=acc) {

                            std::complex<float> * in = &(input(0, e1, 0, cha, m, a, n, s));
                            std::complex<float> * out = &(output(0, index, cha, m, a, n, s));

                            memcpy(out , in, sizeof(std::complex<float>)*RO);

                            index++;
                        }
                    }
                }
            }
        }
    }

    GADGET_CHECK_THROW(reduced_E1_ == index+1);

}



void GenericReconCartesianSliceGrappav0Gadget::remove_unnecessary_kspace_mb(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& output, size_t acc )
{

    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t E2 = input.get_size(2);
    size_t CHA = input.get_size(3);
    size_t N = input.get_size(4);
    size_t S = input.get_size(5);
    size_t STK = input.get_size(6);

    size_t index;

    size_t s,n,a,cha,e1;

    size_t m=0;

    for (s = 0; s < S; s++)
    {
        for (n = 0; n < N; n++)
        {
            for (a = 0; a < STK; a++) {

                for (cha = 0; cha < CHA; cha++) {

                    index=0;

                    for (e1 = start_E1_; e1 <= end_E1_; e1+=acc) {

                        std::complex<float> * in = &(input(0, e1, 0, cha,  n, s, a));
                        std::complex<float> * out = &(output(0, index, cha, m, a, n, s));

                        memcpy(out , in, sizeof(std::complex<float>)*RO);

                        index++;
                    }
                }
            }
        }
    }

    GADGET_CHECK_THROW(reduced_E1_ == index+1);

}

void GenericReconCartesianSliceGrappav0Gadget::extract_milieu_kernel(hoNDArray< std::complex<float> >& block_SB, hoNDArray< std::complex<float> >& missing_data)
{

    ///------------------------------------------------------------------------
    /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
    /// UK
    ///

    size_t CHA = block_SB.get_size(2);
    size_t MB = block_SB.get_size(3);
    size_t STK = block_SB.get_size(4);
    size_t N = block_SB.get_size(5);
    size_t S = block_SB.get_size(6);

    size_t c,m,a,n,s;
    size_t p;

    GADGET_CHECK_THROW(CHA == lNumberOfChannels_);
    GADGET_CHECK_THROW(STK == lNumberOfStacks_);

    size_t milieu=round(float(kernel_size_)/2)-1;

    GDEBUG("Le milieu du kernel est %d  \n",milieu );

    // TODO #pragma omp parallel for default() private() shared()
    for (s = 0; s < S; s++)    {

        for (n = 0; n < N; n++)  {

            for (a = 0; a < STK; a++) {

                for (m = 0; m < MB; m++) {

                    for ( c = 0; c < CHA; c++)
                    {
                        for ( p = 0; p < voxels_number_per_image_; p++)
                        {
                            missing_data(p,c,m,a,n,s)=block_SB(p, m, c, m, a, n,s);

                        }
                    }
                }
            }
        }
    }
}



void GenericReconCartesianSliceGrappav0Gadget::im2col(hoNDArray< std::complex<float> >& input, hoNDArray< std::complex<float> >& block_SB)
{


    ///------------------------------------------------------------------------
    /// FR on recoit l'image folded pour une coupe uniquement [E1 Readout Coil ]
    /// UK
    ///

    size_t RO = input.get_size(0);
    size_t E1 = input.get_size(1);
    size_t CHA = input.get_size(2);
    size_t MB = input.get_size(3);
    size_t STK = input.get_size(4);
    size_t N = input.get_size(6);
    size_t S = input.get_size(7);

    GADGET_CHECK_THROW(CHA==lNumberOfChannels_)
            GADGET_CHECK_THROW(STK==lNumberOfStacks_);

    GDEBUG_STREAM("GenericReconCartesianSliceGrappav0Gadget - im2col : [RO E1 CHA MB STK N S] - [" << RO << " " << E1 << " " << CHA << " " << MB << " " << STK << " " << N<< " " << S << "]");

    size_t rowIdx, colIdx;
    size_t c,m,a,n,s;
    size_t i,j,xx,yy;
    // input ici vaut : e1, readout, cha

    //long long num = MB * STK * N * S;
    //long long ii;
    // TODO #pragma omp parallel for default() private() shared()
    for (s = 0; s < S; s++)    {

        for (n = 0; n < N; n++)  {

            for (a = 0; a < STK; a++) {

                for (m = 0; m < MB; m++) {

                    for ( c = 0; c < CHA; c++)
                    {
                        for ( j = 0; j < blocks_RO_; j++)
                        {
                            for (i = 0; i < blocks_E1_; i++)
                            {
                                rowIdx = i + j*blocks_E1_;

                                for ( yy = 0; yy < grappa_kSize_RO.value(); yy++)
                                {
                                    for (xx = 0; xx < grappa_kSize_E1.value(); xx++)
                                    {
                                        colIdx = xx + yy*grappa_kSize_E1.value();

                                        block_SB(rowIdx, colIdx, c, m, a, n,s)=input(j+yy , i+xx, c, m, a, n, s);

                                        //if (c==0 && a==0 && m==0 && xx==3 && yy==3)
                                        //{
                                        //  std::cout<<   i <<  " " <<  j << " "<<  rowIdx  << " " <<   xx <<  " " <<  yy<< " "<<  colIdx <<   " "<< abs(block_SB(rowIdx, colIdx, c, m, a, n,s)) <<std::endl;
                                        //}

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


GADGET_FACTORY_DECLARE(GenericReconCartesianSliceGrappav0Gadget)
}

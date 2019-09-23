
#include "GenericReconSMSPrepGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericReconSMSPrepGadget::GenericReconSMSPrepGadget() : BaseClass()
{
}

GenericReconSMSPrepGadget::~GenericReconSMSPrepGadget()
{
}

int GenericReconSMSPrepGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    return GADGET_OK;
}

int GenericReconSMSPrepGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    if (perform_timing.value()) { gt_timer_.start("GenericReconSMSPrepGadget::process"); }

    process_called_times_++;

    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
    if (recon_bit_->rbit_.size() > num_encoding_spaces_)
    {
        GWARN_STREAM("Incoming recon_bit has more encoding spaces than the protocol : " << recon_bit_->rbit_.size() << " instead of " << num_encoding_spaces_);
    }

    // for every encoding space, prepare the recon_bit_->rbit_[e].ref_
    size_t e, n, s, slc;
    for (e = 0; e < recon_bit_->rbit_.size(); e++)
    {
        auto & rbit = recon_bit_->rbit_[e];
        std::stringstream os;
        os << "_encoding_" << e;

        if (recon_bit_->rbit_[e].ref_)
        {
            // std::cout << " je suis la structure qui contient les données acs" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].ref_->data_;

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);

            GDEBUG_STREAM("GenericSMSPrepGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");

            ref_8D.create(RO, E1, E2, CHA, MB_factor, lNumberOfStacks_ , N, S );

            pre_process_ref_data(data, ref_8D,  e);

            recon_bit_->rbit_[e].ref_->data_ = ref_8D;

        }

        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
            // std::cout << " je suis la structure qui contient les données single band et/ou multiband" << std::endl;

            GDEBUG("GenericSMSPrepGadget - |--------------------------------------------------------------------------|\n");

            bool is_single_band=false;

            bool is_first_repetition=detect_first_repetition(recon_bit_->rbit_[e]);

            if (is_first_repetition==true) {

                is_single_band=detect_single_band_data(recon_bit_->rbit_[e]);

                hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;

                size_t RO = data.get_size(0);
                size_t E1 = data.get_size(1);
                size_t E2 = data.get_size(2);
                size_t CHA = data.get_size(3);
                size_t N = data.get_size(4);
                size_t S = data.get_size(5);
                size_t SLC = data.get_size(6);

                //TODO should be done somewhere else
                sb_8D.create(RO, E1, E2, CHA, MB_factor, lNumberOfStacks_ , N, S );
                mb_8D.create(RO, E1, E2, CHA, MB_factor, lNumberOfStacks_ , N, S );
            }

            //TODO mettre recon_bit_->rbit_[e] a la place de data + header
            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;
            hoNDArray< ISMRMRD::AcquisitionHeader > headers_ =recon_bit_->rbit_[e].data_.headers_;  //5D, fixed order [E1, E2, N, S, LOC]

            // create to new hoNDArray [8D] for the sb and mb data

            if (is_single_band==true)
            {                
                define_usefull_parameters_simple_version(recon_bit_->rbit_[e], e);
                pre_process_sb_data(data, sb_8D, headers_, e);
                recon_bit_->rbit_[e].data_.data_ = sb_8D;

            }
            else
            {
                // only mb data
                //then apply standard proccesing on mb                
                define_usefull_parameters_simple_version(recon_bit_->rbit_[e], e);
                pre_process_mb_data(data, mb_8D, headers_ , e);
                recon_bit_->rbit_[e].data_.data_ = mb_8D;                

            }
        }
    }

    if (perform_timing.value()) { gt_timer_.stop(); }

    if (this->next()->putq(m1) < 0)
    {
        GERROR_STREAM("Put IsmrmrdReconData to Q failed ... ");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}



//sur les données single band
void GenericReconSMSPrepGadget::pre_process_ref_data(hoNDArray< std::complex<float> >& ref, hoNDArray< std::complex<float> >& ref_8D, size_t e)
{
    // three steps are necessary
    // 1) to reorganize the slices in the stack of slices according the to the slice acceleration
    // 2) to apply (or not depending of the sequence implementation) a blip caipi shift along the y
    // 3) to apply the averaged epi ghost correction

    reorganize_sb_data_to_8D(ref, ref_8D, e);

    //apply_relative_phase_shift(ref_8D, false);


}


//sur les données single band
void GenericReconSMSPrepGadget::pre_process_sb_data(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, size_t e)
{
    // three steps are necessary
    // 1) to reorganize the slices in the stack of slices according the to the slice acceleration
    // 2) to apply (or not depending of the sequence implementation) a blip caipi shift along the y
    // 3) to apply the averaged epi ghost correction

    reorganize_sb_data_to_8D(sb, sb_8D, e);

    apply_averaged_epi_ghost_correction(sb_8D, h_sb, e);

    apply_blip_caipi_shift(sb_8D, h_sb,  e);



}


void GenericReconSMSPrepGadget::reorganize_sb_data_to_8D(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_8D, size_t e)
{
    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    if (!debug_folder_full_path_.empty())
    {
        save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(sb, "FID_SB4D", os.str());
    }

    //permute_slices_index(sb, indice_sb);

    //if (!debug_folder_full_path_.empty())
    //{
    //    save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(sb, "FID_SB4D_permute_slices", os.str());
    //}
    //create_stacks_of_slices(sb, sb_8D);

    create_stacks_of_slices_directly(sb, sb_8D, indice_sb);

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB4D_create_stacks", os.str());
    }
}


void GenericReconSMSPrepGadget::apply_blip_caipi_shift(hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers_sb, size_t e)
{
    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    if (is_wip_sequence==1)
    {

        if (!debug_folder_full_path_.empty())
        {
            save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB4D_prep_avant_relative_shift", os.str());
        }

        // si WIP on applique le blip caipi
        apply_relative_phase_shift(sb_8D, false);

        if (!debug_folder_full_path_.empty())
        {
            save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB4D_prep_apres_relative_shift", os.str());
        }

        // et on applique aussi l'offset de phase
        // recupération de l'offset de position dans la direction de coupe
        get_header_and_position_and_gap(sb_8D, headers_sb);

        apply_absolute_phase_shift(sb_8D,false);

        if (!debug_folder_full_path_.empty())
        {
            save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB4D_prep_apres_absolute_shift", os.str());
        }

    }
    else if (is_cmrr_sequence==1 && is_wip_sequence==0)
    {
        // si CMMR on ne fait rien
    }
    else
    {
        GERROR("is_wip_sequence && is_cmrr_sequence");
    }
}


void GenericReconSMSPrepGadget::apply_averaged_epi_ghost_correction(hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers_sb, size_t e)
{
    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    //apply the average slice navigator

    load_epi_data();

    //prepare epi data

    prepare_epi_data(e);

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB_avant_epi_nav", os.str());
    }

    //hoNDArray< std::complex<float> > sb_8D_optimal=sb_8D;
    //unapply optimal correction
    //apply_ghost_correction_with_STK6(sb_8D, headers_sb ,  acceFactorSMSE1_[e], true , true);

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB_avant2_epi_nav", os.str());
    }

    apply_ghost_correction_with_arma_STK6(sb_8D, headers_sb ,  acceFactorSMSE1_[e], false , false, "PREP SB");

    //apply_ghost_correction_with_STK6(sb_8D_optimal, headers_sb ,  acceFactorSMSE1_[e] , true);

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB_apres_epi_nav", os.str());
    }

    //if (!debug_folder_full_path_.empty())
    //{
    //save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D_optimal, "FID_SB_optimal_apres_epi_nav", os.str());
    //}

}

void GenericReconSMSPrepGadget::pre_process_mb_data(hoNDArray< std::complex<float> >& mb, hoNDArray< std::complex<float> >& mb_8D,hoNDArray< ISMRMRD::AcquisitionHeader > & headers_mb, size_t e)
{

    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    if (!debug_folder_full_path_.empty())
    {
        save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(mb, "FID_MB4D", os.str());
    }

    reorganize_mb_data_to_8D(mb, mb_8D);   

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(mb_8D, "FID_MB4D_remove", os.str());
    }

    //apply the average slice navigator
    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(mb_8D, "FID_MB_avant_epi_nav", os.str());
    }

    apply_ghost_correction_with_arma_STK6(mb_8D, headers_mb ,  acceFactorSMSE1_[e], false , false , " Prep MB ");

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(mb_8D, "FID_MB_apres_epi_nav", os.str());
    }

}

/*
void GenericReconSMSPrepGadget::fusion_sb_and_mb_in_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb)
{
    size_t RO=sb.get_size(0);
    size_t E1=sb.get_size(1);
    size_t E2=sb.get_size(2);
    size_t CHA=sb.get_size(3);
    size_t MB=sb.get_size(4);
    size_t STK=sb.get_size(5);
    size_t N=sb.get_size(6);
    size_t S=sb.get_size(7);

    hoNDArray< std::complex<float> > data;
    data.create(RO,E1,E2,CHA,MB,STK,N*2,S);

    size_t s, n;

    for (s = 0; s < S; s++)
    {
        for (n = 0; n < N; n++)
        {
            std::complex<float> * in_sb  = &(sb(0, 0, 0, 0, 0, 0, s));
            std::complex<float> * in_mb  = &(mb(0, 0, 0, 0, 0, 0, s));

            if (n==1)
            {
                std::complex<float> * out = &(sb(0, 0, 0, 0, 0, 1, s));
                memcpy(out , in_sb, sizeof(std::complex<float>)*RO*E1*E2*CHA*MB*STK);
            }
            else
            {
                std::complex<float> * out = &(sb(0, 0, 0, 0, 0, 0, s));
                memcpy(out , in_sb, sizeof(std::complex<float>)*RO*E1*E2*CHA*MB*STK);
            }
        }
    }

    recon_bit.data_.data_ = data;
}*/

/*
//sur les données single band
void GenericReconSMSPrepGadget::extract_sb_and_mb_from_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb)
{
    //TODO instead of creating a new sb and sb_header i t would be easier to create a new reconbit

    hoNDArray< std::complex<float> >& data = recon_bit.data_.data_;
    hoNDArray< ISMRMRD::AcquisitionHeader > headers_ =recon_bit.data_.headers_;  //5D, fixed order [E1, E2, N, S, LOC]

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    size_t hE1=headers_.get_size(0);
    size_t hE2=headers_.get_size(1);
    size_t hN=headers_.get_size(2);
    size_t hS=headers_.get_size(3);
    size_t hSLC=headers_.get_size(4);

    GDEBUG_STREAM("GenericSMSPrepGadget - incoming headers_ array : [E1, E2, N, S, LOC] - [" << hE1 << " " << hE2 << " " << hN << " " << hS << " " << hSLC << "]");

    if (N!=2)
    {
        GERROR_STREAM("size(N) should be equal to 2 ");
    }

    size_t n, s, slc;

    for (slc = 0; slc < SLC; slc++)
    {
        for (s = 0; s < S; s++)
        {
            for (n = 0; n < N; n++)
            {
                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, slc));
                std::complex<float> * out_sb = &(sb(0, 0, 0, 0, 0, s, slc));
                std::complex<float> * out_mb = &(mb(0, 0, 0, 0, 0, s, slc));

                ISMRMRD::AcquisitionHeader *in_h=&(headers_(0,0,n,s,slc));
                ISMRMRD::AcquisitionHeader *out_h_sb=&(h_sb(0,0,0,s,slc));
                ISMRMRD::AcquisitionHeader *out_h_mb=&(h_mb(0,0,0,s,slc));

                if (n==1)
                {
                    memcpy(out_sb , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                    memcpy(out_h_sb , in_h, sizeof(ISMRMRD::AcquisitionHeader)*E1*E2);
                }
                else
                {
                    memcpy(out_mb , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                    memcpy(out_h_mb , in_h, sizeof(ISMRMRD::AcquisitionHeader)*E1*E2);
                }
            }
        }
    }
    // how to put for instance
    // hoNDArray< std::complex<float> > sb;
    // and hoNDArray< ISMRMRD::AcquisitionHeader > headers_sb;
    // into a new_recon_bit.data_,  what is the contructor of new_recon_bit.data_, assuming only one recon_bit ? I guess it also require some memory allocation;
    // in order to have new_recon_bit.data_.data=sb;
    // and new_recon_bit.data_.headers_=h_sb;

}*/



void GenericReconSMSPrepGadget::permute_slices_index(hoNDArray< std::complex<float> >& data, arma::uvec indice)
{

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    hoNDArray< std::complex<float> > new_data;
    new_data.create(RO,E1, E2, CHA, N, S, SLC);

    size_t n, s, slc;

    for (slc = 0; slc < SLC; slc++) {

        for (s = 0; s < S; s++)
        {
            size_t usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (n = 0; n < N; n++)
            {
                size_t usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, indice(slc)));
                std::complex<float> * out = &(new_data(0, 0, 0, 0, n, s, slc));

                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

            }
        }
    }

    data = new_data;

}





void GenericReconSMSPrepGadget::remove_extra_dimension_and_permute_stack_dimension(hoNDArray< std::complex<float> >& data)
{
    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);
    size_t SLC=data.get_size(6);

    hoNDArray< std::complex<float> > FID_MB;
    FID_MB.create(RO, E1, E2, CHA,  N, S , lNumberOfStacks_ );

    //size_t nb_elements_multiband = data.get_number_of_elements()/MB_factor;

    size_t index_in;
    size_t index_out;

    size_t n, s;
    for (int a = 0; a < lNumberOfStacks_; a++)
    {
        index_in=indice_slice_mb[a];
        index_out=indice_mb[a];

        for (s = 0; s < S; s++)
        {
            size_t usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (n = 0; n < N; n++)
            {
                size_t usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index_in));
                std::complex<float> * out = &(FID_MB(0, 0, 0, 0,  n, s,index_out));

                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

            }
        }
    }

    data = FID_MB;

}



void GenericReconSMSPrepGadget::reorganize_mb_data_to_8D(hoNDArray< std::complex<float> >& mb,hoNDArray< std::complex<float> >& mb_8D )
{
    size_t RO=mb.get_size(0);
    size_t E1=mb.get_size(1);
    size_t E2=mb.get_size(2);
    size_t CHA=mb.get_size(3);
    size_t N=mb.get_size(4);
    size_t S=mb.get_size(5);
    size_t SLC=mb.get_size(6);

    //mb_8D.create(RO, E1, E2, CHA, MB_factor, lNumberOfStacks_ , new_N, S );

    //hoNDArray< std::complex<float> > FID_MB;
    //FID_MB.create(RO, E1, E2, CHA,  N, S , lNumberOfStacks_ );

    //size_t nb_elements_multiband = data.get_number_of_elements()/MB_factor;

    size_t index_in;
    size_t index_out;

    size_t n, s;
    for (int a = 0; a < lNumberOfStacks_; a++)
    {
        index_in=indice_slice_mb[a];
        index_out=indice_mb[a];

        for (s = 0; s < S; s++)
        {
            size_t usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (n = 0; n < N; n++)
            {
                size_t usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(mb(0, 0, 0, 0, n, s, index_in));
                std::complex<float> * out = &(mb_8D(0, 0, 0, 0,  0, index_out, 0, s));

                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

            }
        }
    }

}



//sur les données single band
void GenericReconSMSPrepGadget::create_stacks_of_slices_directly(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack, arma::uvec indice)
{

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);

    size_t MB=new_stack.get_size(4);
    size_t STK=new_stack.get_size(5);

    size_t n, s, a, m;
    size_t index;

    // copy of the data in the 8D array

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> * in = &(data(0, 0, 0, 0, n, s, indice(index)));
                    //std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index));
                    std::complex<float> * out = &(new_stack(0, 0, 0, 0, m, a, n, s));

                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                }
            }
        }
    }
}


//sur les données single band
void GenericReconSMSPrepGadget::create_stacks_of_slices(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack)
{

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t N=data.get_size(4);
    size_t S=data.get_size(5);

    size_t MB=new_stack.get_size(4);
    size_t STK=new_stack.get_size(5);

    size_t n, s, a, m;
    size_t index;

    // copy of the data in the 8D array

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index));
                    std::complex<float> * out = &(new_stack(0, 0, 0, 0, m, a, n, s));

                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);
                }
            }
        }
    }
}


GADGET_FACTORY_DECLARE(GenericReconSMSPrepGadget)
}


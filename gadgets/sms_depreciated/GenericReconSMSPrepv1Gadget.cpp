
#include "GenericReconSMSPrepv1Gadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericReconSMSPrepv1Gadget::GenericReconSMSPrepv1Gadget() : BaseClass()
{
}

GenericReconSMSPrepv1Gadget::~GenericReconSMSPrepv1Gadget()
{
}

int GenericReconSMSPrepv1Gadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    return GADGET_OK;
}

int GenericReconSMSPrepv1Gadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    if (perform_timing.value()) { gt_timer_.start("GenericReconSMSPrepv1Gadget::process"); }

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
        }

        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {
            // std::cout << " je suis la structure qui contient les données single band et/ou multiband" << std::endl;

            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;
            hoNDArray< ISMRMRD::AcquisitionHeader > headers_ =recon_bit_->rbit_[e].data_.headers_;  //5D, fixed order [E1, E2, N, S, LOC]

            size_t RO = data.get_size(0);
            size_t E1 = data.get_size(1);
            size_t E2 = data.get_size(2);
            size_t CHA = data.get_size(3);
            size_t N = data.get_size(4);
            size_t S = data.get_size(5);
            size_t SLC = data.get_size(6);
            size_t new_N=1;

            GDEBUG_STREAM("GenericSMSPrepGadget - incoming data array sb : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");

            // create to new hoNDArray for the sb and mb data
            // ideally the memory allocation should occurs once for the mb hoNDArray

            hoNDArray< std::complex<float> > sb;
            hoNDArray< std::complex<float> > sb_8D;
            hoNDArray< ISMRMRD::AcquisitionHeader > headers_sb;

            hoNDArray< std::complex<float> > mb;
            hoNDArray< std::complex<float> > mb_8D;
            hoNDArray< ISMRMRD::AcquisitionHeader > headers_mb;

            headers_mb.create(E1, E2, new_N, S , SLC );
            mb.create(RO, E1, E2, CHA, new_N, S , SLC );
            mb_8D.create(RO, E1, E2, CHA, MB_factor, lNumberOfStacks_ , new_N, S );

            if (N>1)
            {
                define_usefull_parameters_simple_version(recon_bit_->rbit_[e], e);

                // now if the dimension along mb is >1 , it means that the containers got sb and mb data
                // please remind that the user_0 paramter must be defined in N_dimension in BucketToBuffer
                // so we need to separate them
                // then process the singleband data
                // latter we wiil process the mb data
                // and finally concatenate them into a new containers with a higher dimension (for the stack dim)            }

                headers_sb.create(E1, E2, new_N, S , SLC );
                sb.create(RO, E1, E2, CHA, new_N, S , SLC );
                sb_8D.create(RO, E1, E2, CHA, MB_factor, lNumberOfStacks_ , new_N, S );

                //TODO instead of using sb and headers_sb , it should be better to create something like  "recon_bit_sb->rbit_[e] "
                extract_sb_and_mb_from_data( recon_bit_->rbit_[e], sb,  mb, headers_sb,  headers_mb);

                /*Gadgetron::IsmrmrdReconData array_sb_data;
                array_sb_data.rbit_.resize(1);
                array_sb_data.rbit_[0].data_.data_=sb;
                array_sb_data.rbit_[0].data_.headers_=headers_sb;

                Gadgetron::IsmrmrdReconData array_mb_data;
                array_mb_data.rbit_.resize(1);
                array_mb_data.rbit_[0].data_.data_=mb;
                array_mb_data.rbit_[0].data_.headers_=headers_mb;*/


                //TODO and then send it to the next function
                pre_process_sb_data(sb, sb_8D, headers_sb, e);

            }
            else
            {
                // only mb data
                 mb = recon_bit_->rbit_[e].data_.data_;
                 headers_mb = recon_bit_->rbit_[e].data_.headers_;
            }

            //then apply standard proccesing on mb

            if (!debug_folder_full_path_.empty())
            {
            save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(mb, "FID_MB4D", os.str());
            }

            pre_process_mb_data(mb, mb_8D,headers_mb , e);

            // il nous faut remettre les données avec les dimensions de N = 2 pour les envoyer aux gadgets suivant
            // est-ce vraiment nécessaire ??

            //show_size(sb_8D,"sb_8D");
            //show_size(mb_8D,"mb_8D");

            // m1->getObjectPtr()->rbit_[e].data_.data_ = data;
            // m1->getObjectPtr()->rbit_[e].sb_->data_ = data8D;

            if (N>1)
            {
            fusion_sb_and_mb_in_data(recon_bit_->rbit_[e], sb_8D, mb_8D);
            }
            else
            {
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
void GenericReconSMSPrepv1Gadget::pre_process_sb_data(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, size_t e)
{
    // three steps are necessary
    // 1) to reorganize the slices in the stack of slices according the to the slice acceleration
    // 2) to apply (or not depending of the sequence implementation) a blip caipi shift along the y
    // 3) to apply the averaged epi ghost correction

    reorganize_sb_data_to_8D(sb, sb_8D, e);

    apply_blip_caipi_shift(sb_8D, h_sb,  e);

    apply_averaged_epi_ghost_correction(sb_8D, h_sb, e);

}


void GenericReconSMSPrepv1Gadget::reorganize_sb_data_to_8D(hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& sb_8D, size_t e)
{
    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    if (!debug_folder_full_path_.empty())
    {
    save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(sb, "FID_SB4D", os.str());
    }

    permute_slices_index(sb, indice_sb);

    if (!debug_folder_full_path_.empty())
    {
    save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(sb, "FID_SB4D_permute_slices", os.str());
    }

    create_stacks_of_slices(sb, sb_8D);

    if (!debug_folder_full_path_.empty())
    {
    save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB4D_create_stacks", os.str());
    }
}


void GenericReconSMSPrepv1Gadget::apply_blip_caipi_shift(hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers_sb, size_t e)
{
    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    if (is_wip_sequence==1)
    {
        // si WIP on applique le blip caipi
        apply_relative_phase_shift(sb_8D);

        if (!debug_folder_full_path_.empty())
        {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB4D_relative_shift", os.str());
        }

        // et on applique aussi l'offset de phase
        // recupération de l'offset de position dans la direction de coupe
        get_header_and_position_and_gap(sb_8D, headers_sb);

        apply_absolute_phase_shift(sb_8D);

        if (!debug_folder_full_path_.empty())
        {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB4D_absolute_shift", os.str());
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


void GenericReconSMSPrepv1Gadget::apply_averaged_epi_ghost_correction(hoNDArray< std::complex<float> >& sb_8D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers_sb, size_t e)
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

    apply_ghost_correction_with_STK6(sb_8D, headers_sb ,  acceFactorSMSE1_[e] , false, false);

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

void GenericReconSMSPrepv1Gadget::pre_process_mb_data(hoNDArray< std::complex<float> >& mb, hoNDArray< std::complex<float> >& mb_8D,hoNDArray< ISMRMRD::AcquisitionHeader > & headers_mb, size_t e)
{

    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    // a remplacer par mb_8;

    reorganize_mb_data_to_8D(mb, mb_8D);
    //remove_extra_dimension_and_permute_stack_dimension(mb);

    if (!debug_folder_full_path_.empty())
    {
    save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(mb_8D, "FID_MB4D_remove", os.str());
    }

    size_t STK = mb_8D.get_size(5);

    // code usefull only for matlab comparison
    // reorganize_data(data, indice_mb);
    // save_4D_data(data, "FID_MB4D_reorganize", os.str());
    // reorganize_data(data, arma::conv_to<arma::uvec>::from(order_of_acquisition_mb));

    show_size(mb_8D,"FID_MB4D_remove" );

    //apply the average slice navigator
    if (!debug_folder_full_path_.empty())
    {
    save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(mb_8D, "FID_MB_avant_epi_nav", os.str());
    }

    apply_ghost_correction_with_STK6(mb_8D, headers_mb ,  acceFactorSMSE1_[e] , false, false);
    //apply_ghost_correction_with_STK7(data, recon_bit_->rbit_[e].data_.headers_ ,  acceFactorSMSE1_[e] , false);

    if (!debug_folder_full_path_.empty())
    {
    save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(mb_8D, "FID_MB_apres_epi_nav", os.str());
    }

}


void GenericReconSMSPrepv1Gadget::fusion_sb_and_mb_in_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb)
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

}


//sur les données single band
void GenericReconSMSPrepv1Gadget::extract_sb_and_mb_from_data(IsmrmrdReconBit &recon_bit, hoNDArray< std::complex<float> >& sb, hoNDArray< std::complex<float> >& mb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_sb, hoNDArray< ISMRMRD::AcquisitionHeader > & h_mb)
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

}


void GenericReconSMSPrepv1Gadget::get_header_and_position_and_gap(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > headers_)
{
    size_t E1 = data.get_size(1);
    size_t SLC=lNumberOfSlices_;
    size_t STK=lNumberOfStacks_;
    size_t MB=MB_factor;

    size_t e1,s,a,m;

    size_t start_E1(0), end_E1(0);
    auto t = Gadgetron::detect_sampled_region_E1(data);
    start_E1 = std::get<0>(t);
    end_E1 = std::get<1>(t);

    //std::cout <<  " start_E1 " <<  start_E1 << "  end_E1  "    << end_E1 << std::endl;

    hoNDArray<float > shift_from_isocenter;
    shift_from_isocenter.create(3);
    hoNDArray<float > read_dir;
    read_dir.create(3);
    hoNDArray<float > phase_dir;
    phase_dir.create(3);
    hoNDArray<float > slice_dir;
    slice_dir.create(3);

    arma::fvec z_offset(SLC);

    for (s = 0; s < SLC; s++)
    {
        ISMRMRD::AcquisitionHeader& curr_header = headers_(start_E1, 0, 0, 0, s);
        for (int j = 0; j < 3; j++) {

            shift_from_isocenter(j)=curr_header.position[j];
            read_dir(j)=curr_header.read_dir[j];
            phase_dir(j)=curr_header.phase_dir[j];
            slice_dir(j)=curr_header.slice_dir[j];
            //std::cout <<  curr_header.position[j]<< " "  <<  curr_header.read_dir[j] << " "  <<  curr_header.phase_dir[j]  << " "  <<  curr_header.slice_dir[j]  << std::endl;
        }

        z_offset(s) = dot(shift_from_isocenter,slice_dir);
    }

    //std::cout << z_offset <<std::endl;
    //std::cout << "   " <<z_offset.max()<< " " <<z_offset.min() << std::endl;

    //reorientation dans l'espace
    z_offset_geo.set_size(SLC);

    for (s = 0; s < SLC; s++)
    {
        z_offset_geo(s)=z_offset(indice_sb(s));
    }

    //std::cout << z_offset_geo << std::endl;

    arma::fvec delta_slice(SLC-1);

    for (s = 1; s < SLC; s++)
    {
        delta_slice(s-1)=z_offset_geo(s)-z_offset_geo(s-1);
    }

    //std::cout << delta_slice << std::endl;

    float  slice_gap_factor=(delta_slice(0)-slice_thickness)/slice_thickness*100;

    GDEBUG_STREAM("slice thickness is "<<  slice_thickness);
    GDEBUG_STREAM("slice distance is "<<  delta_slice(0));
    GDEBUG_STREAM("slice gap factor is "<<  slice_gap_factor<< " %" );

    //selection d'un jeux de données :
    arma::ivec index(MB);

    z_gap.set_size(1);

    for (a = 0; a < 1; a++)
    {        
        index=MapSliceSMS.row(a);

        for (m = 0; m < MB-1; m++)
        {
            if (z_offset_geo(index(m+1))>z_offset_geo(index(m)))
            {
                 GDEBUG_STREAM("distance au centre de la coupe la proche: " <<z_offset_geo(index(m))) ;
                 GDEBUG_STREAM("distance entre les coupes simultanées: " <<  z_offset_geo(index(m+1))-z_offset_geo(index(m))) ;

                z_gap(m)=z_offset_geo(index(m+1))-z_offset_geo(index(m));
            }

        }
    }

   // std::cout << z_gap<< std::endl;
}


void GenericReconSMSPrepv1Gadget::permute_slices_index(hoNDArray< std::complex<float> >& data, arma::uvec indice)
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

    size_t n, s;

    for (int i = 0; i < SLC; i++) {

        for (s = 0; s < S; s++)
        {
            size_t usedS = s;
            if (usedS >= S) usedS = S - 1;

            for (n = 0; n < N; n++)
            {
                size_t usedN = n;
                if (usedN >= N) usedN = N - 1;

                std::complex<float> * in = &(data(0, 0, 0, 0, n, s, indice[i]));
                std::complex<float> * out = &(new_data(0, 0, 0, 0, n, s, i));

                memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

            }
        }
    }

    data = new_data;

}





void GenericReconSMSPrepv1Gadget::remove_extra_dimension_and_permute_stack_dimension(hoNDArray< std::complex<float> >& data)
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



void GenericReconSMSPrepv1Gadget::reorganize_mb_data_to_8D(hoNDArray< std::complex<float> >& mb,hoNDArray< std::complex<float> >& mb_8D )
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
void GenericReconSMSPrepv1Gadget::create_stacks_of_slices(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> >& new_stack)
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






void GenericReconSMSPrepv1Gadget::apply_relative_phase_shift(hoNDArray< std::complex<float> >& data)
{

    // vecteur_in_E1_direction=exp(1i*([1:1:size(reconSB,2)]- 1*center_k_space_sample   )*2*pi/PE_shift*(nband-1)) ;
    //  test=repmat( vecteur_in_E1_direction ,[ size(reconSB,1) 1 size(reconSB,3) size(reconSB,4) size(reconSB,5) size(reconSB,6)]  );
    //  reconSB(:,:,:,:,:,1,nt,nband)=reconSB(:,:,:,:,:,:,nt,nband).*test;

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    hoNDArray< std::complex<float> > tempo;
    tempo.create(RO, E1, E2, CHA);

    hoNDArray< std::complex<float> > phase_shift;
    phase_shift.create(RO, E1, E2, CHA);

    center_k_space_E1=round(E1/2);

    GDEBUG_STREAM("  center_k_space_xml  "<<   center_k_space_xml  << " center_k_space_E1    "<<  center_k_space_E1  );

    // à définir dans SMS Base car c'est aussi utilisé dans SMSPostGadget
    arma::fvec index_imag = arma::linspace<arma::fvec>( 1, E1, E1 )  - center_k_space_E1 ;

    arma::cx_fvec phase;
    arma::cx_fvec shift_to_apply;

    phase.set_size(E1);
    phase.zeros();
    phase.set_imag(index_imag);

    size_t m,a,n,s,cha,e2,e1,ro;
    float caipi_factor;

    for (m = 0; m < MB_factor; m++) {

        caipi_factor=2*arma::datum::pi/(Blipped_CAIPI)*(m);
        GDEBUG_STREAM("  Blipped_CAIPI  "<<   Blipped_CAIPI  << " caipi_factor    "<<  caipi_factor  );
        shift_to_apply=exp(phase*caipi_factor);

        for (e1 = start_E1_; e1 < end_E1_; e1++)
        {
            for (ro = 0; ro < RO; ro++)
            {
                for (e2 = 0; e2 < E2; e2++)
                {
                    for (cha = 0; cha < CHA; cha++)
                    {
                        phase_shift(ro,e1,e2,cha)=shift_to_apply(e1);
                    }
                }
            }
        }

        for (a = 0; a < lNumberOfStacks_; a++) {

            for (s = 0; s < S; s++)
            {
                size_t usedS = s;
                if (usedS >= S) usedS = S - 1;

                for (n = 0; n < N; n++)
                {
                    size_t usedN = n;
                    if (usedN >= N) usedN = N - 1;

                    std::complex<float> * in = &(data(0, 0, 0, 0, m, a, n, s));
                    std::complex<float> * out = &(tempo(0, 0, 0, 0));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                    Gadgetron::multiply(tempo, phase_shift, tempo);

                    memcpy(in , out, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                }
            }
        }
    }
}





void GenericReconSMSPrepv1Gadget::apply_absolute_phase_shift(hoNDArray< std::complex<float> >& data)
{

    // fid_stack_SB_corrected(:, :, :, c, :, :, a, m)=fid_stack.SB_shift(:, :, :, c, :, :, a, m) * exp(1i*pi*z_offset_geo(index)/z_gap(1));

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    size_t m, a, n, s;
    size_t index;

    std::complex<double> ii(0,1);

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            std::complex<double> lala=  exp(arma::datum::pi*ii*z_offset_geo(index)/z_gap(0));
            std::complex<float>  lili=  static_cast< std::complex<float> >(lala) ;

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> *in = &(data(0, 0, 0, 0, m, a, n, s));

                    for (size_t ii = 0; ii < RO * E1 * E2 * CHA; ii++) {

                        data[ii] = data[ii]*lili  ;
                    }
                }
            }
        }
    }

}

GADGET_FACTORY_DECLARE(GenericReconSMSPrepv1Gadget)
}




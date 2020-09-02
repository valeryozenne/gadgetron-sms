
#include "GenericReconSMSPostGadget.h"
#include <iomanip>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"

namespace Gadgetron {

GenericReconSMSPostGadget::GenericReconSMSPostGadget() : BaseClass()
{
}

GenericReconSMSPostGadget::~GenericReconSMSPostGadget()
{
}

int GenericReconSMSPostGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    return GADGET_OK;
}

int GenericReconSMSPostGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    if (perform_timing.value()) { gt_timer_.start("GenericReconSMSPostGadget::process"); }

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

            hoNDArray< std::complex<float> >& ref_8D = recon_bit_->rbit_[e].ref_->data_;

            size_t RO = ref_8D.get_size(0);
            size_t E1 = ref_8D.get_size(1);
            size_t E2 = ref_8D.get_size(2);
            size_t CHA = ref_8D.get_size(3);
            size_t MB = ref_8D.get_size(4);
            size_t STK = ref_8D.get_size(5);
            size_t N = ref_8D.get_size(6);
            size_t S = ref_8D.get_size(7);

            //GDEBUG_STREAM("GenericReconSMSPostGadget - incoming data array ref : [RO E1 E2 CHA N S SLC] - [" << RO << " " << E1 << " " << E2 << " " << CHA << " " << N << " " << S << " " << SLC << "]");

            hoNDArray< std::complex<float> > ref_7D;

            ref_7D.create(RO, E1, E2, CHA, N, S, STK*MB);

            post_process_ref_data(ref_8D, ref_7D  ,e);

            m1->getObjectPtr()->rbit_[e].ref_->data_ = ref_7D;

            if (!debug_folder_full_path_.empty())
            {
                save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(ref_7D, "FID_REF_fin", os.str());
            }

        }

        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {

            bool is_single_band=false;
            bool is_first_repetition=detect_first_repetition(recon_bit_->rbit_[e]);
            if (is_first_repetition==true) {  is_single_band=detect_single_band_data(recon_bit_->rbit_[e]);    }

            show_size(recon_bit_->rbit_[e].data_.data_, "GenericReconSMSPostGadget - incoming data array data");

            hoNDArray< std::complex<float> >& data_8D = recon_bit_->rbit_[e].data_.data_;
            //hoNDArray< ISMRMRD::AcquisitionHeader > headers_ =recon_bit_->rbit_[e].data_.headers_;  //5D, fixed order [E1, E2, N, S, LOC]

            size_t RO = data_8D.get_size(0);
            size_t E1 = data_8D.get_size(1);
            size_t E2 = data_8D.get_size(2);
            size_t CHA = data_8D.get_size(3);
            size_t MB = data_8D.get_size(4);
            size_t STK = data_8D.get_size(5);
            size_t N = data_8D.get_size(6);
            size_t S = data_8D.get_size(7);

            if (is_single_band==true)  //presence de single band
            {
                headers_buffered=recon_bit_->rbit_[e].data_.headers_;

                hoNDArray< std::complex<float> > data_7D;

                data_7D.create(RO, E1, E2, CHA, N, S, STK*MB);

                define_usefull_parameters_simple_version(recon_bit_->rbit_[e], e);

                post_process_sb_data(data_8D, data_7D , recon_bit_->rbit_[e].data_.headers_ ,e);

                m1->getObjectPtr()->rbit_[e].data_.data_ = data_7D;

                if (!debug_folder_full_path_.empty())
                {
                    save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(m1->getObjectPtr()->rbit_[e].data_.data_, "FID_SB_fin", os.str());
                }

                /*for (size_t ii=0; ii<m1->getObjectPtr()->rbit_[e].data_.headers_.get_number_of_elements(); ii++)
                {
                    if (m1->getObjectPtr()->rbit_[e].data_.headers_(ii).sample_time_us>0)
                    {
                        m1->getObjectPtr()->rbit_[e].data_.headers_(ii).idx.set=1;
                    }
                }*/

            }
            else
            {

                hoNDArray< std::complex<float> > data_7D;

                data_7D.create(RO, E1, E2, CHA, N, S, STK*MB);

                define_usefull_parameters_simple_version(recon_bit_->rbit_[e], e);

                post_process_mb_data(data_8D, data_7D , recon_bit_->rbit_[e].data_.headers_, e);

                m1->getObjectPtr()->rbit_[e].data_.data_ = data_7D;

                //int repetition=get_max_repetition_number(recon_bit_->rbit_[e].data_.headers_);
                //GDEBUG_STREAM("--------- repetition  "<< repetition );

                //set_idx(headers_buffered,  repetition , 0);

                m1->getObjectPtr()->rbit_[e].data_.headers_=headers_buffered;

                /*for (size_t ii=0; ii<m1->getObjectPtr()->rbit_[e].data_.headers_.get_number_of_elements(); ii++)
                {
                   if (m1->getObjectPtr()->rbit_[e].data_.headers_(ii).sample_time_us>0)
                   {
                       m1->getObjectPtr()->rbit_[e].data_.headers_(ii).idx.repetition=repetition;
                   }
                }*/

                //GDEBUG_STREAM("--------- repetition  after "<< get_max_repetition_number(m1->getObjectPtr()->rbit_[e].data_.headers_) );

                if (!debug_folder_full_path_.empty())
                {
                    save_7D_containers_as_4D_matrix_with_a_loop_along_the_7th_dim(m1->getObjectPtr()->rbit_[e].data_.data_, "FID_MB_fin", os.str());
                }
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



void GenericReconSMSPostGadget::post_process_ref_data(hoNDArray< std::complex<float> >& data_8D, hoNDArray< std::complex<float> >& data_7D, size_t e)
{

    undo_stacks_ordering_to_match_gt_organisation_open(data_8D, data_7D, MapSliceSMS,  indice_sb);

}


void GenericReconSMSPostGadget::post_process_sb_data(hoNDArray< std::complex<float> >& data_8D, hoNDArray< std::complex<float> >& data_7D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers, size_t e)
{
    // undo the average phase_shift

    // apply slice_optimal phase-shift

    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    undo_blip_caipi_shift(data_8D, headers, e, true);

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data_8D, "FID_SB4D_fin_caipi", os.str());

    }

    load_epi_data();

    prepare_epi_data(e, data_8D.get_size(1),  data_8D.get_size(2) ,  data_8D.get_size(3) );

    if (!debug_folder_full_path_.empty())
    {
        save_4D_data(epi_nav_neg_STK_, "epi_nav_neg_STK", os.str());
        save_4D_data(epi_nav_pos_STK_, "epi_nav_pos_STK", os.str());

        save_4D_data(epi_nav_neg_STK_mean_, "epi_nav_neg_STK_mean", os.str());
        save_4D_data(epi_nav_pos_STK_mean_, "epi_nav_pos_STK_mean", os.str());

        size_t E1 = data_8D.get_size(1);

        hoNDArray<float> reverse_line;
        reverse_line.create(E1);
        reverse_line.fill(0);

        //std::cout<< " start_E1_ "<<  start_E1_ << std::endl;
        //std::cout<< " end_E1_ "<<  end_E1_ << std::endl;

        for (size_t e1 = start_E1_; e1 <= end_E1_; e1+=acceFactorSMSE1_[e])
        {
            ISMRMRD::AcquisitionHeader& curr_header = headers(e1, 0, 0, 0, 0);  //5D, fixed order [E1, E2, N, S, LOC]
            if (curr_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)) {
                reverse_line(e1)=1;
            }
            else
            {
                reverse_line(e1)=0;
            }

            //std::cout << e1 << " "<< reverse_line(e1) << std::endl;
        }
        //std::cout << reverse_line << std::endl;
        save_4D_data(reverse_line, "reverse_line", os.str());

    }


    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data_8D, "FID_SB4D_stk6_avant_cpu", os.str());
        //save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data_compare, "FID_SB4D_stk6_avant_gpu", os.str());
    }

    if (use_gpu.value()==true)
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPrepGadget::apply_ghost_correction_with_STK6 gpu time Post SB ");}
        apply_ghost_correction_with_STK6_gpu(data_8D, headers ,  acceFactorSMSE1_[e], true , false, true, "Post SB" );
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }
    else
    {
        if (use_omp.value()==true)
        {
            if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPrepGadget::apply_ghost_correction_with_STK6 openmp time Post SB ");}
            apply_ghost_correction_with_STK6_open(data_8D, headers ,  acceFactorSMSE1_[e], true , false, true, "Post SB" );
            if (perform_timing.value()) { gt_timer_local_.stop();}
        }
        else
        {

            if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPrepGadget::apply_ghost_correction_with_STK6 cpu time Post SB ");}
            apply_ghost_correction_with_STK6(data_8D, headers ,  acceFactorSMSE1_[e], true , false, true, "Post SB" );
            if (perform_timing.value()) { gt_timer_local_.stop();}

        }
    }
    //

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data_8D, "FID_SB4D_fin_epi", os.str());

        // save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data_8D, "FID_SB4D_stk6_apres_cpu", os.str());
        //save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data_compare, "FID_SB4D_stk6_apres_gpu", os.str());
    }



    if(use_omp.value()==true)
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPrepGadget::undo_stacks_ordering_to_match_gt_organisation_open"); }
        undo_stacks_ordering_to_match_gt_organisation_open(data_8D, data_7D, MapSliceSMS,  indice_sb);
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }
    else
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPostGadget::undo_stacks_ordering_to_match_gt_organisation"); }
        undo_stacks_ordering_to_match_gt_organisation(data_8D, data_7D, MapSliceSMS,  indice_sb);
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }

}


void GenericReconSMSPostGadget::post_process_mb_data(hoNDArray< std::complex<float> >& data_8D, hoNDArray< std::complex<float> >& data_7D, hoNDArray< ISMRMRD::AcquisitionHeader > & headers, size_t e)
{

    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    undo_blip_caipi_shift(data_8D, headers, e, false);

    if (!debug_folder_full_path_.empty())
    {
        save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data_8D, "FID_MB4D_fin_caipi", os.str());
    }


    if (use_gpu.value()==true)
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPrepGadget::apply_ghost_correction_with_STK6 gpu time Post MB ");}
        apply_ghost_correction_with_STK6_gpu(data_8D, headers ,  acceFactorSMSE1_[e], true , false, true, "POST MB" );
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }
    else
    {
        if (use_omp.value()==true)
        {
            if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPrepGadget::apply_ghost_correction_with_STK6 openmp time Post MB ");}
            apply_ghost_correction_with_STK6_open(data_8D, headers ,  acceFactorSMSE1_[e], true , false, true,  "POST MB");
            if (perform_timing.value()) { gt_timer_local_.stop();}
        }
        else
        {
            if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPrepGadget::apply_ghost_correction_with_STK6 cpu time Post MB ");}
            apply_ghost_correction_with_STK6(data_8D, headers ,  acceFactorSMSE1_[e], true , false, true,  "POST MB");
            if (perform_timing.value()) { gt_timer_local_.stop();}
        }
    }


    if(use_omp.value()==true)
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPrepGadget::undo_stacks_ordering_to_match_gt_organisation_open"); }
        undo_stacks_ordering_to_match_gt_organisation_open(data_8D, data_7D, MapSliceSMS,  indice_sb);
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }
    else
    {
        if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPostGadget::undo_stacks_ordering_to_match_gt_organisation"); }
        undo_stacks_ordering_to_match_gt_organisation(data_8D, data_7D, MapSliceSMS,  indice_sb);
        if (perform_timing.value()) { gt_timer_local_.stop();}
    }


}



void GenericReconSMSPostGadget::set_idx(hoNDArray< ISMRMRD::AcquisitionHeader > headers_, unsigned int rep, unsigned int set)
{
    try
    {
        size_t E1 = headers_.get_size(0);
        size_t E2 = headers_.get_size(1);
        size_t N = headers_.get_size(2);
        size_t S = headers_.get_size(3);
        size_t SLC = headers_.get_size(4);

        size_t e1,e2, n, s, slc;

        for (slc=0; slc<SLC; slc++)
        {
            for (s=0; s<S; s++)
            {
                for (n=0; n<N; n++)
                {
                    for (e2=0; e2<E2; e2++)
                    {
                        for (e1=0; e1<E1; e1++)
                        {
                            headers_(e1,e2,n, s, slc).idx.repetition = rep;
                        }
                    }
                }
            }
        }
    }
    catch (...)
    {
        GADGET_THROW("Errors happened in GenericReconSMSPostGadget::set_idx(...) ... ");
    }
}





void GenericReconSMSPostGadget::undo_blip_caipi_shift(hoNDArray< std::complex<float> >& data, hoNDArray< ISMRMRD::AcquisitionHeader > & headers, size_t e, bool undo_absolute)
{
    std::stringstream os;
    os << "_encoding_" << e;
    std::string suffix = os.str();

    if (is_wip_sequence==1)
    {
        // et on applique aussi l'offset de phase
        // recupération de l'offset de position dans la direction de coupe
        if (undo_absolute==true)
        {
            // true means single band data
            get_header_and_position_and_gap(data, headers);
            apply_absolute_phase_shift(data, true);

            apply_relative_phase_shift(data, true);
        }
        else
        {
            // false means multiband data
            apply_relative_phase_shift_test(data, true);
        }


        //if (!debug_folder_full_path_.empty())
        //{
        //save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data, "FID_SB4D_relative_shift", os.str());
        //}

        //if (!debug_folder_full_path_.empty())
        //{
        //save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(sb_8D, "FID_SB4D_absolute_shift", os.str());
        //}

    }
    else if (is_cmrr_sequence==1 && is_wip_sequence==0)
    {
        // si CMMR on ne fait rien


        if (perform_timing.value()) { gt_timer_local_.start("GenericReconSMSPostGadget::apply_relative_phase_shift"); }
        apply_relative_phase_shift(data, false);
        if (perform_timing.value()) { gt_timer_local_.stop();}
        //if (!debug_folder_full_path_.empty())
        //{
        //save_8D_containers_as_4D_matrix_with_a_loop_along_the_6th_dim_stk(data, "FID_SB4D_relative_shift", os.str());
        //}

    }
    else
    {
        GERROR("is_wip_sequence && is_cmrr_sequence");
    }
}




GADGET_FACTORY_DECLARE(GenericReconSMSPostGadget)
}

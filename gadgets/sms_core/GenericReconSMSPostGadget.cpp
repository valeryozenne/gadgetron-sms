
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

            }
            else
            {

                hoNDArray< std::complex<float> > data_7D;

                data_7D.create(RO, E1, E2, CHA, N, S, STK*MB);

                define_usefull_parameters_simple_version(recon_bit_->rbit_[e], e);

                post_process_mb_data(data_8D, data_7D , recon_bit_->rbit_[e].data_.headers_, e);

                m1->getObjectPtr()->rbit_[e].data_.data_ = data_7D;

                //set_idx(headers_buffered,  recon_bit_->rbit_[e].data_.headers_(2, 2, 0, 0, 0).idx.repetition , 0);

                recon_bit_->rbit_[e].data_.headers_=headers_buffered;

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

     undo_stacks_ordering_to_match_gt_organisation(data_8D, data_7D);

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

     prepare_epi_data(e);

     apply_ghost_correction_with_arma_STK6(data_8D, headers ,  acceFactorSMSE1_[e], true , false, "POST SB" );

     undo_stacks_ordering_to_match_gt_organisation(data_8D, data_7D);

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

    apply_ghost_correction_with_arma_STK6(data_8D, headers ,  acceFactorSMSE1_[e], true , false, "POST MB");

    undo_stacks_ordering_to_match_gt_organisation(data_8D, data_7D);

}



void GenericReconSMSPostGadget::set_idx(hoNDArray< ISMRMRD::AcquisitionHeader > headers_, unsigned int rep, unsigned int set)
{
    try
    {
        size_t RO = headers_.get_size(0);
        size_t E1 = headers_.get_size(1);
        size_t N = headers_.get_size(2);
        size_t S = headers_.get_size(3);
        size_t SLC = headers_.get_size(4);

        size_t ro,e1, n, s, slc;
        for (slc=0; slc<SLC; slc++)
        {
            for (s=0; s<S; s++)
            {
                for (n=0; n<N; n++)
                {
                    for (e1=0; e1<E1; e1++)
                    {
                        for (ro=0; ro<RO; ro++)
                        {
                        headers_(ro,e1,n, s, slc).idx.repetition = rep;
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




void GenericReconSMSPostGadget::undo_stacks_ordering_to_match_gt_organisation(hoNDArray< std::complex<float> >& data, hoNDArray< std::complex<float> > &output)
{

    //TODO it should be remplaced by one single copy

    size_t RO=data.get_size(0);
    size_t E1=data.get_size(1);
    size_t E2=data.get_size(2);
    size_t CHA=data.get_size(3);
    size_t MB=data.get_size(4);
    size_t STK=data.get_size(5);
    size_t N=data.get_size(6);
    size_t S=data.get_size(7);

    size_t SLC=output.get_size(6);

    hoNDArray< std::complex<float> > tempo;
    tempo.create(RO,E1,E2,CHA,N,S,SLC);

    //GADGET_CHECK_THROW(lNumberOfSlices_ == STK*MB);

    size_t n, s, a, m, slc;
    int index;

    for (a = 0; a < STK; a++) {

        for (m = 0; m < MB; m++) {

            index = MapSliceSMS(a,m);

            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                    std::complex<float> * in = &(data(0, 0, 0, 0, m, a, n, s));
                    std::complex<float> * out = &(tempo(0, 0, 0, 0, n, s, index));
                    memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);                    

                    //create stack
                    //index = MapSliceSMS(a,m);
                    //std::complex<float> * in = &(data(0, 0, 0, 0, n, s, index));
                    //std::complex<float> * out = &(new_stack(0, 0, 0, 0, m, a, n, s));
                }
            }
        }
    }

    for (slc = 0; slc < SLC; slc++)
    {
            for (s = 0; s < S; s++)
            {
                for (n = 0; n < N; n++)
                {
                   std::complex<float> * in = &(tempo(0, 0, 0, 0, n, s, slc));
                   std::complex<float> * out = &(output(0, 0, 0, 0, n, s, indice_sb(slc)));
                   memcpy(out , in, sizeof(std::complex<float>)*RO*E1*E2*CHA);

                   //permute
                   //std::complex<float> * in = &(data(0, 0, 0, 0, n, s, indice(slc)));
                   //std::complex<float> * out = &(new_data(0, 0, 0, 0, n, s, slc));

                }
            }
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

        apply_relative_phase_shift(data, false);

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

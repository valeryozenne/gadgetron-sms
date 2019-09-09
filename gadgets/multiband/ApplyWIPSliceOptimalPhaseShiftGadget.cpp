/*
* ApplyWIPSliceOptimalPhaseShiftGadget.cpp
*
*  Created on: September 26, 2017
*      Author: Valéry Ozenne
*/

#include "ApplyWIPSliceOptimalPhaseShiftGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"
#include "mri_core_multiband.h"

namespace Gadgetron{

ApplyWIPSliceOptimalPhaseShiftGadget::ApplyWIPSliceOptimalPhaseShiftGadget() {
}

ApplyWIPSliceOptimalPhaseShiftGadget::~ApplyWIPSliceOptimalPhaseShiftGadget() {
}

int ApplyWIPSliceOptimalPhaseShiftGadget::process_config(ACE_Message_Block *mb)
{
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);

    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;

    ISMRMRD::StudyInformation study_info;

    lNumberOfSlices_ = e_limits.slice? e_limits.slice->maximum+1 : 1;  /* Number of slices in one measurement */ //MANDATORY
    lNumberOfAverage_= e_limits.average? e_limits.average->maximum+1 : 1;
    lNumberOfChannels_ = h.acquisitionSystemInformation->receiverChannels ? *h.acquisitionSystemInformation->receiverChannels : 1;

    dimensions_.push_back(e_space.matrixSize.x);
    dimensions_.push_back(e_space.matrixSize.y);
    dimensions_.push_back(e_space.matrixSize.z);

    GDEBUG("Matrix size after reconstruction: %d, %d, %d\n", dimensions_[0], dimensions_[1], dimensions_[2]);

    readout=dimensions_[0];
    encoding=dimensions_[1];

    //TODO ne pas passer tout h
    deal_with_inline_or_offline_situation(h);

    str_home=GetHomeDirectory();
    //GDEBUG(" str_home: %s \n", str_home);

    ///------------------------------------------------------------------------
    /// FR
    /// UK

    lNumberOfStacks_= lNumberOfSlices_/MB_factor_;

    ///------------------------------------------------------------------------
    /// FR flag a définir dans le xml au minimum
    /// UK
    bool no_reordering=0;

    order_of_acquisition_mb = Gadgetron::map_interleaved_acquisitions_mb(lNumberOfStacks_,  no_reordering );

    order_of_acquisition_sb = Gadgetron::map_interleaved_acquisitions(lNumberOfSlices_,  no_reordering );

    indice_mb = sort_index( order_of_acquisition_mb );

    indice_sb = sort_index( order_of_acquisition_sb );


    ///------------------------------------------------------------------------
    /// FR
    /// UK Reordering, Assuming interleaved SMS acquisition, define the acquired slices for each multislice band

    MapSliceSMS=Gadgetron::get_map_slice_single_band( MB_factor_,  lNumberOfStacks_,  order_of_acquisition_mb,  no_reordering);

    vec_MapSliceSMS=vectorise(MapSliceSMS);

    corrpos_.set_size( readout );
    corrneg_.set_size( readout );

    corrpos_all_.set_size( readout , lNumberOfSlices_);
    corrneg_all_.set_size( readout , lNumberOfSlices_);

    corrpos_mean_.set_size( readout , lNumberOfStacks_);
    corrneg_mean_.set_size( readout , lNumberOfStacks_);

    return GADGET_OK;
}


int ApplyWIPSliceOptimalPhaseShiftGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    bool is_parallel = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION);
    bool is_acq_single_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1);
    bool is_acq_multi_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2);
    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    unsigned int slice= m1->getObjectPtr()->idx.slice;

    // Get a reference to the acquisition header
    ISMRMRD::AcquisitionHeader &hdr = *m1->getObjectPtr();

    if (is_parallel)
    {
        ///------------------------------------------------------------------------
        /// FR on fait rien
        /// UK
    }
    else if (is_acq_single_band)
    {

        // Make an armadillo matrix of the data
        arma::cx_fmat adata = as_arma_matrix(*m2->getObjectPtr());

        if (is_first_scan_in_slice)
        {
            ///------------------------------------------------------------------------
            /// FR relecture des corrections single band EPI
            /// UK

            std::ostringstream indice_slice;
            indice_slice << slice;
            str_s = indice_slice.str();

            corrneg_=Gadgetron::LoadCplxVectorFromtheDisk(str_home, "Tempo/", "corrneg_",   str_s,  ".bin");
            corrpos_=Gadgetron::LoadCplxVectorFromtheDisk(str_home, "Tempo/", "corrpos_",   str_s,  ".bin");

            corrneg_all_.col(slice)=corrneg_;
            corrpos_all_.col(slice)=corrpos_;
        }


        ///------------------------------------------------------------------------
        /// FR relecture des corrections multiband moyenne par stack EPI
        /// on le fait seulement une fois on pourrait prendre une autre condition
        /// UK

        if(is_first_scan_in_slice && slice==0)
        {
            std::cout << " relecture des correction EPI moyenne par stack" << std::endl;
            std::cout << " action effectuée a la slice " << slice << std::endl;

            for (int a=0; a<lNumberOfStacks_; a++) {

                std::ostringstream indice_stack;
                indice_stack << a;
                str_a = indice_stack.str();

                corrneg_mean_.col(a)=Gadgetron::LoadCplxVectorFromtheDisk(str_home, "Tempo/", "corrneg_mean_",   str_a,  ".bin");
                corrpos_mean_.col(a)=Gadgetron::LoadCplxVectorFromtheDisk(str_home, "Tempo/", "corrpos_mean_",   str_a,  ".bin");
            }
        }

        ///------------------------------------------------------------------------
        /// FR
        /// UK

        if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)) {
            // Negative readout            
            for (int p=0; p<adata.n_cols; p++) {
                // adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrneg_);
                adata.col(p) %=  corrneg_all_.col(slice) / corrneg_mean_.col(get_stack_number_from_gt_numerotation(slice));
            }
            // Now that we have corrected we set the readout direction to positive
            hdr.clearFlag(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE);

            // std::cout << " adata  " << abs(adata).max()  << std::endl;

        }
        else {            
            // Positive readout
            for (int p=0; p<adata.n_cols; p++) {
                // adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrpos_);
               adata.col(p) %=  corrpos_all_.col(slice) / corrpos_mean_.col(get_stack_number_from_gt_numerotation(slice));
            }

           // std::cout << " adata  " << abs(adata).max()  << std::endl;
        }

    }
    else if (is_acq_multi_band)
    {

        arma::cx_fmat adata = as_arma_matrix(*m2->getObjectPtr());

        ///------------------------------------------------------------------------
        /// FR
        /// UK

        if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE)) {
            // Negative readout
            for (int p=0; p<adata.n_cols; p++) {
                // adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrneg_);
                adata.col(p) %= corrneg_all_.col(slice) / corrneg_mean_.col(get_stack_number_from_gt_numerotation(slice));
            }
            // Now that we have corrected we set the readout direction to positive
            hdr.clearFlag(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE);
        }
        else {
            // Positive readout
            for (int p=0; p<adata.n_cols; p++) {
                // adata.col(p) %= (arma::pow(corrB0_,epiEchoNumber_+RefNav_to_Echo0_time_ES_) % corrpos_);
                adata.col(p) %=  corrpos_all_.col(slice) / corrpos_mean_.col(get_stack_number_from_gt_numerotation(slice));
            }
        }
    }


    if( this->next()->putq(m1) < 0 ){
        GDEBUG("Failed to put message on queue\n");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}





void ApplyWIPSliceOptimalPhaseShiftGadget::get_multiband_parameters(ISMRMRD::IsmrmrdHeader h)
{
    arma::fvec store_info_special_card=Gadgetron::get_information_from_wip_multiband_special_card(h);

    MB_factor_=store_info_special_card(1);
   // MB_Slice_Inc_=store_info_special_card(3);
    Blipped_CAIPI_=store_info_special_card(4);
}


int ApplyWIPSliceOptimalPhaseShiftGadget::get_blipped_factor(int numero_de_slice)
{
    int value=(numero_de_slice-(numero_de_slice%lNumberOfStacks_))/lNumberOfStacks_;

    return value;
}

int ApplyWIPSliceOptimalPhaseShiftGadget::get_stack_number_from_gt_numerotation(int slice)
{

    arma::uvec q1 = arma::find(vec_MapSliceSMS == order_of_acquisition_sb(slice));

    int value= arma::conv_to<int>::from(q1);

    int stack_number= (value-(value%MB_factor_))/MB_factor_;

    return stack_number;
}


int ApplyWIPSliceOptimalPhaseShiftGadget::get_stack_number_from_spatial_numerotation(int slice)
{

    arma::uvec q1 = arma::find(vec_MapSliceSMS == slice);

    int value= arma::conv_to<int>::from(q1);

    int stack_number= (value-(value%MB_factor_))/MB_factor_;

    return stack_number;
}



void ApplyWIPSliceOptimalPhaseShiftGadget::deal_with_inline_or_offline_situation(ISMRMRD::IsmrmrdHeader h)
{
    if (doOffline.value()==1)
    {
        MB_factor_=MB_factor.value();
        Blipped_CAIPI_=Blipped_CAIPI.value();
       // MB_Slice_Inc_=MB_Slice_Inc.value();
    }
    else
    {
        get_multiband_parameters(h);
    }
}



GADGET_FACTORY_DECLARE(ApplyWIPSliceOptimalPhaseShiftGadget)
}



//if (e1==last_scan_in_acs && slice==indice_sb(lNumberOfSlices_-1))
//{
// GDEBUG("Sorting acs  \n");

/*for (unsigned int a = 0; a < lNumberOfStacks_; a++)
    {

        for (unsigned int m = 0; m < MB_factor; m++)
        {
            acs_calibration_reduce.slice(m)=acs_calibration.slice(MapSliceSMS(m,a));
        }

        arma::cx_fcube acs_calibration_reduce_blipped=shifting_of_multislice_data(acs_calibration_reduce, Blipped_CAIPI);

        for (unsigned int m = 0; m < MB_factor; m++)
        {
            acs_calibration_blipped.slice(MapSliceSMS(m,a))=acs_calibration_reduce_blipped.slice(m);
        }
    }*/


/* for (unsigned long s = 0; s < lNumberOfSlices_; s++)
    {

        for (unsigned long e1 = 0; e1 < encoding; e1++)
        {
        //ceci est le numero de dynamique ie de 0 à N-1
        std::ostringstream indice_e1;
        indice_e1 << e1;
        str_e = indice_e1.str();

        std::ostringstream indice_slice;
        indice_slice << s;
        str_s = indice_slice.str();

        std::ostringstream indice_slice2;
        indice_slice2 << indice_sb(s);
        str_s2 = indice_slice2.str();

        //Gadgetron::SaveVectorOntheDisk(vectorise(acs_calibration_unblipped.slice(s).col(e1)), str_home, "Tempo/", "acs_calibration_unblipped_",  str_e, str_s2,  ".bin");
        //Gadgetron::SaveVectorOntheDisk(vectorise(acs_calibration.slice(s).col(e1)), str_home, "Tempo/", "acs_calibration_",  str_e, str_s,  ".bin");
        Gadgetron::SaveVectorOntheDisk(vectorise(acs_calibration_initial.slice(s).col(e1)), str_home, "Tempo/", "acs_calibration_initial_",  str_e, str_s,  ".bin");

        }
    }*/

//}

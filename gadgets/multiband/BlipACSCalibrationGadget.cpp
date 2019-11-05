/*
* BlipACSCalibrationGadget.cpp
*
*  Created on: September 26, 2017
*      Author: Valéry Ozenne
*/

#include "BlipACSCalibrationGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"
#include "mri_core_multiband.h"

namespace Gadgetron{

BlipACSCalibrationGadget::BlipACSCalibrationGadget() {
}

BlipACSCalibrationGadget::~BlipACSCalibrationGadget() {
}

int BlipACSCalibrationGadget::process_config(ACE_Message_Block *mb)
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

    GDEBUG(" encoding center: %d \n", h.encoding[0].encodingLimits.kspace_encoding_step_1->center);

    //TODO ne pas passer tout h
    deal_with_inline_or_offline_situation(h);

    str_home=GetHomeDirectory();
    //GDEBUG(" str_home: %s \n", str_home);

     Gadgetron::DeleteTemporaryFiles( str_home, "Tempo/", "acs_cal*" );

    ///------------------------------------------------------------------------
    /// FR
    /// UK    

    lNumberOfStacks_= lNumberOfSlices_/MB_factor_;

    // flag a définir dans le xml au minimum
    bool no_reordering=0;

    order_of_acquisition_mb = Gadgetron::map_interleaved_acquisitions_mb(lNumberOfStacks_,  no_reordering );

    order_of_acquisition_sb = Gadgetron::map_interleaved_acquisitions(lNumberOfSlices_,  no_reordering );

    indice_mb = sort_index( order_of_acquisition_mb );

    indice_sb = sort_index( order_of_acquisition_sb );

    // Reordering
    // Assuming interleaved SMS acquisition, define the acquired slices for each multislice band
    MapSliceSMS=Gadgetron::get_map_slice_single_band( MB_factor_,  lNumberOfStacks_,  order_of_acquisition_mb,  no_reordering);

    //TODO erreur prendre
    int center_k_space_sample=h.encoding[0].encodingLimits.kspace_encoding_step_1->center+1;
    //int center_k_space_sample=round(float(encoding)/2);

    ///------------------------------------------------------------------------
    /// FR
    /// UK
    ///
    ///

    std::cout << encoding << std::endl;
    arma::fvec index_imag = arma::linspace<arma::fvec>( 1, encoding, encoding )  - center_k_space_sample ;


    phase_.set_size(encoding);
    phase_.zeros();
    phase_.set_imag(index_imag);

    //TODO renommer
    shift_to_apply_.set_size(encoding);
    shift_to_apply_.zeros();

    return GADGET_OK;
}


int BlipACSCalibrationGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    bool is_parallel = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION);
    bool is_acq_single_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1);
    bool is_acq_multi_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2);

    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice= m1->getObjectPtr()->idx.slice;    

    if (is_parallel)
    {
        ///------------------------------------------------------------------------
        /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
        /// UK

        std::ostringstream indice_e1;
        indice_e1 << e1;
        str_e = indice_e1.str();

        std::ostringstream indice_slice;
        indice_slice << slice;
        str_s = indice_slice.str();

        shift_ = get_blipped_factor(order_of_acquisition_sb(slice));

        if (shift_!=0)
        {

        arma::cx_fvec tempo= as_arma_col(*m2->getObjectPtr());

        //Gadgetron::SaveVectorOntheDisk(tempo, str_home, "Tempo/", "acs_calibration_initial_",  str_e, str_s,  ".bin");

        caipi_factor_=2*arma::datum::pi/Blipped_CAIPI_*shift_;

        shift_to_apply_=exp(phase_*caipi_factor_);

        tempo=tempo*shift_to_apply_.row(e1);        

        }

        if( this->next()->putq(m1) < 0 ){
            GDEBUG("Failed to put message on queue\n");
            return GADGET_FAIL;
        }

    }
    else if (is_acq_single_band)
    {
        if( this->next()->putq(m1) < 0 ){
            GDEBUG("Failed to put message on queue\n");
            return GADGET_FAIL;
        }

    }
    else if (is_acq_multi_band)
    {
        if( this->next()->putq(m1) < 0 ){
            GDEBUG("Failed to put message on queue\n");
            return GADGET_FAIL;
        }
    }

    return GADGET_OK;
}


int BlipACSCalibrationGadget::get_blipped_factor(int numero_de_slice)
{

    int value=(numero_de_slice-(numero_de_slice%lNumberOfStacks_))/lNumberOfStacks_;

    return value;
}


void BlipACSCalibrationGadget::get_multiband_parameters(ISMRMRD::IsmrmrdHeader h)
{
    arma::fvec store_info_special_card=Gadgetron::get_information_from_multiband_special_card(h);

    MB_factor_=store_info_special_card(1);
    MB_Slice_Inc_=store_info_special_card(3);
    Blipped_CAIPI_=store_info_special_card(4);
}



void BlipACSCalibrationGadget::deal_with_inline_or_offline_situation(ISMRMRD::IsmrmrdHeader h)
{
    if (doOffline.value()==1)
    {
        MB_factor_=MB_factor.value();
        Blipped_CAIPI_=Blipped_CAIPI.value();
        MB_Slice_Inc_=MB_Slice_Inc.value();
    }
    else
    {
        get_multiband_parameters(h);
    }
}

GADGET_FACTORY_DECLARE(BlipACSCalibrationGadget)
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

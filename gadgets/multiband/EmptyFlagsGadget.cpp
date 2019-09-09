/*
* EmptyFlagsGadget.cpp
*
*  Created on: September 26, 2017
*      Author: Valery Ozenne
*/

#include "EmptyFlagsGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"

namespace Gadgetron{

EmptyFlagsGadget::EmptyFlagsGadget() {
}

EmptyFlagsGadget::~EmptyFlagsGadget() {
}

int EmptyFlagsGadget::process_config(ACE_Message_Block *mb)
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

    str_home=GetHomeDirectory();
    //GDEBUG(" str_home: %s \n", str_home);

    deja_vu.set_size(encoding,lNumberOfSlices_);
    deja_vu.zeros();

    deja_vu_epi_calib.set_size(1,lNumberOfSlices_);
    deja_vu_epi_calib.zeros();

    return GADGET_OK;
}


int EmptyFlagsGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    bool is_parallel = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION);
    bool is_acq_single_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1);
    bool is_acq_multi_band = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2);

    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice= m1->getObjectPtr()->idx.slice;
    unsigned int rep= m1->getObjectPtr()->idx.repetition;

    bool is_phase_correction = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA);
    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);
    bool is_last_scan_in_repetition = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_REPETITION);
    bool is_first_scan_in_repetition = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_REPETITION);

    if (is_phase_correction)
    {
         /*  std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<  "problem phase correction" << std::endl;
           std::cout << "	 -------------------------------------  " << std::endl;
           std::cout << "	!!!!  H O U S T O N ,   O N   A   U N  P R O B L E M E  !!!! " << std::endl;
           std::cout << "	 -------------------------------------  " << std::endl;*/
    }

     arma::cx_fvec tempo= as_arma_col(*m2->getObjectPtr());

    if (is_parallel)
    {
        ///------------------------------------------------------------------------
        /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
        /// UK
       //if (e1==64) std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_repetition " << is_last_scan_in_repetition << "  max  " << abs(tempo).max() << "  min  " << abs(tempo).min() << std::endl;

    }
    else if (is_acq_single_band )
    {



       //if (e1==0) std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_acq_single_band yes " << is_acq_single_band << " is_first_scan_in_repetition " << is_first_scan_in_repetition <<" max: "<< abs(tempo).max()<<" min: "<< abs(tempo).min()  << std::endl;
        //if (is_first_scan_in_slice) std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_acq_single_band yes " << is_acq_single_band << " is_last_scan_in_repetition " << is_last_scan_in_repetition <<" is_first_scan_in_slice: "<< is_first_scan_in_slice<<" is_last_scan_in_slice: "<< is_last_scan_in_slice << "  max  " << abs(tempo).max() << "  min  " << abs(tempo).min() << std::endl;

     //  if (is_last_scan_in_slice) std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_acq_single_band yes " << is_acq_single_band << " is_last_scan_in_repetition " << is_last_scan_in_repetition <<" is_first_scan_in_slice: "<< is_first_scan_in_slice<<" is_last_scan_in_slice: "<< is_last_scan_in_slice << "  max  " << abs(tempo).max() << "  min  " << abs(tempo).min()<< std::endl;

      // std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_acq_single_band yes " << is_acq_single_band << " is_last_scan_in_repetition " << is_last_scan_in_repetition <<" is_first_scan_in_slice: "<< is_first_scan_in_slice<<" is_last_scan_in_slice: "<< is_last_scan_in_slice << "  max  " << abs(tempo).max() << "  min  " << abs(tempo).min()<< std::endl;

    }
    else if (is_acq_multi_band )
    {
        if (e1==0) std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_acq_multi_band yes " << is_acq_multi_band << " is_first_scan_in_repetition " << is_first_scan_in_repetition << std::endl;

        //if (e1==64)std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_acq_multi_band yes " << is_acq_multi_band << " is_last_scan_in_repetition " << is_last_scan_in_repetition << std::endl;


       // if (is_last_scan_in_slice) std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_acq_multi_band yes " << is_acq_multi_band << " is_last_scan_in_repetition " << is_last_scan_in_repetition << std::endl;


    }
    else
    {
        //std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<  "problem unkown "  << std::endl;
        std::cout << "	 -------------------------------------  " << std::endl;
        std::cout << "	!!!!  H O U S T O N ,   O N   A   U N  P R O B L E M E  !!!! " << std::endl;
        std::cout << "	 -------------------------------------  " << std::endl;
    }




    if( this->next()->putq(m1) < 0 ){
        GDEBUG("Failed to put message on queue\n");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}







GADGET_FACTORY_DECLARE(EmptyFlagsGadget)
}

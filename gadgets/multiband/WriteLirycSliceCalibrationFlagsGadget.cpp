/*
* WriteLirycSliceCalibrationFlagsGadget.cpp
*
*  Created on: September 26, 2017
*      Author: Valery Ozenne
*/

#include "WriteLirycSliceCalibrationFlagsGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"

namespace Gadgetron{

WriteLirycSliceCalibrationFlagsGadget::WriteLirycSliceCalibrationFlagsGadget() {
}

WriteLirycSliceCalibrationFlagsGadget::~WriteLirycSliceCalibrationFlagsGadget() {
}

int WriteLirycSliceCalibrationFlagsGadget::process_config(ACE_Message_Block *mb)
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

    return GADGET_OK;
}


int WriteLirycSliceCalibrationFlagsGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    bool is_parallel = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION);

    unsigned int e1 = m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice = m1->getObjectPtr()->idx.slice;
    unsigned int rep = m1->getObjectPtr()->idx.repetition;
    bool is_first_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_FIRST_IN_SLICE);
    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);

    if (rep==0)
    {

        if (is_parallel)
        {
            ///------------------------------------------------------------------------
            /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
            /// UK
            //std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

        }
        else
        {
            if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA))
            {
                // si c'est pas déjà vu c'est SB
                m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
                m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
            }
            else
            {
                // si c'est pas déjà vu c'est SB
                m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
                m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
            }
        }

        //  bool is_acq_user1 = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1);
        //  bool is_acq_user2 = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2);

    }
    else
    {
        ///------------------------------------------------------------------------
        /// FR  si on n'est pas à la repetition 1 c'est forcément des MB mais bon on va quand même être prudent
        /// UK

        if (is_parallel)
        {
            ///------------------------------------------------------------------------
            /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
            /// UK
            ///
            if (is_first_scan_in_slice){
                std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes "   << std::endl;
            }

        }
        else
        {
            if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA))
            {
                // si c'est pas déjà vu c'est MB
                m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
                m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
            }
            else
            {
                // si c'est pas déjà vu c'est MB
                m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
                m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
            }
        }
    }

    if( this->next()->putq(m1) < 0 ){
        GDEBUG("Failed to put message on queue\n");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(WriteLirycSliceCalibrationFlagsGadget)
}

/*
* WriteSliceCalibrationFlagsGadget.cpp
*
*  Created on: September 26, 2017
*      Author: Valery Ozenne
*/

#include "WriteSliceCalibrationFlagsGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"

namespace Gadgetron{

WriteSliceCalibrationFlagsGadget::WriteSliceCalibrationFlagsGadget() {
}

WriteSliceCalibrationFlagsGadget::~WriteSliceCalibrationFlagsGadget() {
}

int WriteSliceCalibrationFlagsGadget::process_config(ACE_Message_Block *mb)
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

    encoding=dimensions_[1];

    deja_vu.set_size(encoding,lNumberOfSlices_);
    deja_vu.zeros();

    deja_vu_epi_calib.set_size(1,lNumberOfSlices_);
    deja_vu_epi_calib.zeros();

    return GADGET_OK;
}


int WriteSliceCalibrationFlagsGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    bool is_parallel = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION);

    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    unsigned int slice= m1->getObjectPtr()->idx.slice;
    unsigned int rep= m1->getObjectPtr()->idx.repetition;   

    bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);

    if (rep==0)
    {

        if (is_parallel)
        {
            ///------------------------------------------------------------------------
            /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
            /// UK if acs calibration scan , do nothing
           // std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

        }
        else
        {

            ///------------------------------------------------------------------------
            /// FR
            /// UK if not acs calibration , two cases, single band (SB) scans or multiband scans (MB)
            /// As SB scans are always played before MB scans, an identical encoding value
            /// This rule only works if  without the presence of multiple contrasts , echos or average.
            /// A proper solution could be to know the mdh tag for MB and SB scans

            //std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel no " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;


            if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA))
            {
                deja_vu_epi_calib(0,slice)++;

                if (deja_vu_epi_calib(0,slice)>3)
                {
                    // si c'est déjà vu c'est MB                    
                    m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
                    m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
                }
                else
                {
                    // si c'est pas déjà vu c'est SB
                    m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
                    m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
                }
            }
            else
            {
                deja_vu(e1,slice)++;

                if (deja_vu(e1,slice)>1)
                {
                    // si c'est déjà vu c'est MB                    
                    m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
                    m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
                }
                else
                {
                    // si c'est pas déjà vu c'est SB
                    m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
                    m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
                }
            }
        }

        bool is_acq_user1 = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER1);
        bool is_acq_user2 = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_USER2);

    }
    else
    {
        ///------------------------------------------------------------------------
        /// FR  si on n'est pas à la repetition 1 c'est forcément des MB mais bon on va quand même être prudent
        /// UK  if repetition number if highter than 0 , it must be a MB scan


        if (is_parallel)
        {
            ///------------------------------------------------------------------------
            /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
            /// UK
           // std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

        }
        else
        {
            m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_USER2);
            m1->getObjectPtr()->clearFlag(ISMRMRD::ISMRMRD_ACQ_USER1);
        }

    }

    if( this->next()->putq(m1) < 0 ){
        GDEBUG("Failed to put message on queue\n");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}


GADGET_FACTORY_DECLARE(WriteSliceCalibrationFlagsGadget)
}

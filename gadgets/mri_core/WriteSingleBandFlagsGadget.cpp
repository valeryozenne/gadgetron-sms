/*
* WriteSingleBandFlagsGadget.cpp
*
*  Created on: September 26, 2017
*      Author: Valery Ozenne
*/

#include "WriteSingleBandFlagsGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"


namespace Gadgetron{

WriteSingleBandFlagsGadget::WriteSingleBandFlagsGadget() {
}

WriteSingleBandFlagsGadget::~WriteSingleBandFlagsGadget() {
}

int WriteSingleBandFlagsGadget::process_config(ACE_Message_Block *mb)
{
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);



    size_t NE = h.encoding.size();
    num_encoding_spaces_ = NE;
    //GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);
    GDEBUG_STREAM("Number of encoding spaces: " << NE);

    if (NE>1){
       GERROR("WriteSingleBandFlagsGadget::process config, Number of encoding spaces > 1");
    }

    size_t e=0;

    ISMRMRD::EncodingSpace e_space = h.encoding[e].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[e].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[e].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;

    ISMRMRD::StudyInformation study_info;

    lNumberOfSlices_ = e_limits.slice? e_limits.slice->maximum+1 : 1;
    lNumberOfAverage_= e_limits.average? e_limits.average->maximum+1 : 1;
    lNumberOfChannels_ = h.acquisitionSystemInformation->receiverChannels ? *h.acquisitionSystemInformation->receiverChannels : 1;
    lNumberOfSegments_ = e_limits.segment? e_limits.segment->maximum+1 : 1;
    lNumberOfSets_ = e_limits.set? e_limits.set->maximum+1 : 1;
    lNumberOfPhases_ = e_limits.phase? e_limits.phase->maximum+1 : 1;

    bool is_cartesian_sampling = (h.encoding[e].trajectory == ISMRMRD::TrajectoryType::CARTESIAN);
    bool is_epi_sampling= (h.encoding[e].trajectory == ISMRMRD::TrajectoryType::EPI);

    size_t E1,E2;



    if (is_cartesian_sampling || is_epi_sampling)
    {

        if (h.encoding[e].encodingLimits.kspace_encoding_step_1.is_present())
        {
          //std::cout <<   (int)h.encoding[e].encodedSpace.matrixSize.y  << std::endl;
          //std::cout <<   (int)h.encoding[e].encodingLimits.kspace_encoding_step_1->center  << std::endl;
          //std::cout <<   (int)h.encoding[e].encodingLimits.kspace_encoding_step_1->maximum << std::endl;
          E1=e_space.matrixSize.y;
        }

        if (h.encoding[e].encodingLimits.kspace_encoding_step_2.is_present())
        {
          E2=e_space.matrixSize.z;
          //std::cout <<   (int)h.encoding[e].encodedSpace.matrixSize.z << std::endl;
          //std::cout <<   (int)h.encoding[e].encodingLimits.kspace_encoding_step_2->center  << std::endl;
          //std::cout <<   (int)h.encoding[e].encodingLimits.kspace_encoding_step_2->maximum  << std::endl;
        }

    }
    else
    {
        GERROR("WriteSingleBandFlagsGadget::process config");
    }

    dimensions_.push_back(e_space.matrixSize.x);
    dimensions_.push_back(e_space.matrixSize.y);
    dimensions_.push_back(e_space.matrixSize.z);

    encoding=dimensions_[1];

    deja_vu.set_size(encoding,lNumberOfSlices_);
    deja_vu.zeros();

    deja_vu_epi_calib.set_size(1,lNumberOfSlices_);
    deja_vu_epi_calib.zeros();

    matrix_deja_vu_data_.create(E1,E2,lNumberOfSlices_); //,lNumberOfAverage_,lNumberOfSegments_,lNumberOfSets_, lNumberOfPhases_);
    matrix_deja_vu_epi_nav_.create(E1,E2,lNumberOfSlices_); //,lNumberOfAverage_,lNumberOfSegments_,lNumberOfSets_, lNumberOfPhases_);

    return GADGET_OK;
}



int WriteSingleBandFlagsGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{

    e1= m1->getObjectPtr()->idx.kspace_encode_step_1;
    e2= m1->getObjectPtr()->idx.kspace_encode_step_2;
    slice= m1->getObjectPtr()->idx.slice;
    repetition= m1->getObjectPtr()->idx.repetition;
    set= m1->getObjectPtr()->idx.set;
    segment= m1->getObjectPtr()->idx.segment;
    phase= m1->getObjectPtr()->idx.phase;
    average= m1->getObjectPtr()->idx.average;
    user= m1->getObjectPtr()->idx.user[0];

   // bool is_last_scan_in_slice = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);

   // std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

    if (repetition==0)
    {

        if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION))
        {
            ///------------------------------------------------------------------------
            /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
            /// UK if acs calibration scan , do nothing
            //std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<   " is_parallel yes " << is_parallel << " is_last_scan_in_slice " << is_last_scan_in_slice << std::endl;

        }
        else
        {

            ///------------------------------------------------------------------------
            /// FR
            /// UK if not acs calibration , two cases, single band (SB) scans or multiband scans (MB)
            /// As SB scans are always played before MB scans, an identical encoding value
            /// This rule only works  without the presence of multiple contrasts , echos or average.
            /// A proper solution could be to know the mdh tag for MB and SB scans

            if (m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA))
            {
                deja_vu_epi_calib(0,slice)++;
                //matrix_deja_vu_epi_nav_(0,e2,slice)=matrix_deja_vu_epi_nav_(0,e2,slice)+1;

                if (deja_vu_epi_calib(0,slice)>3)
                {
                    // si c'est déjà vu c'est MB
                    m1->getObjectPtr()->idx.user[0]=0;
                }
                else
                {
                    // si c'est pas déjà vu c'est SB
                    m1->getObjectPtr()->idx.user[0]=1;
                    //m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION); if commented -> array.data else -> array.ref

                }
            }
            else
            {
                //matrix_deja_vu_data_(e1,e2,slice)=matrix_deja_vu_data_(e1,e2,slice)+1;
                 deja_vu(e1,slice)++;

                if (deja_vu(e1,slice)>1)
                {
                    // si c'est déjà vu c'est MB
                    m1->getObjectPtr()->idx.user[0]=0;
                }
                else
                {
                    // si c'est pas déjà vu c'est SB
                    m1->getObjectPtr()->idx.user[0]=1;
                    //m1->getObjectPtr()->setFlag(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION); if commented -> array.data else -> array.ref
                }
            }
        }


    }
    else
    {
        ///------------------------------------------------------------------------
        /// FR  si on n'est pas à la repetition 1 c'est forcément des MB mais bon on va quand même être prudent
        /// UK  if repetition number if highter than 0 , it must be a MB scan

        // m1->getObjectPtr()->idx.user[0]=0; not necessary
    }



    if( this->next()->putq(m1) < 0 ){
        GDEBUG("Failed to put message on queue\n");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}


GADGET_FACTORY_DECLARE(WriteSingleBandFlagsGadget)
}


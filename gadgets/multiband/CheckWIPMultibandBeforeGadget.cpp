/*
* CheckWIPMultibandBeforeGadget.cpp
*
*  Created on: September 26, 2017
*      Author: Valery Ozenne
*/

#include "CheckWIPMultibandBeforeGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"
#include "mri_core_multiband.h"

namespace Gadgetron{

CheckWIPMultibandBeforeGadget::CheckWIPMultibandBeforeGadget() {
}

CheckWIPMultibandBeforeGadget::~CheckWIPMultibandBeforeGadget() {
}

int CheckWIPMultibandBeforeGadget::process_config(ACE_Message_Block *mb)
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

    ISMRMRD::ParallelImaging p_imaging = *h.encoding[0].parallelImaging;

    acceFactorE1_ = (double)(p_imaging.accelerationFactor.kspace_encoding_step_1);
    acceFactorE2_ = (double)(p_imaging.accelerationFactor.kspace_encoding_step_2);

    readout=dimensions_[0];
    encoding=dimensions_[1];

    deal_with_inline_or_offline_situation(h);

    lNumberOfStacks_= lNumberOfSlices_/MB_factor_;

    str_home=GetHomeDirectory();
    //GDEBUG(" str_home: %s \n", str_home);

    deja_vu.set_size(encoding,lNumberOfSlices_);
    deja_vu.zeros();

    deja_vu_epi_calib.set_size(1,lNumberOfSlices_);
    deja_vu_epi_calib.zeros();


    bool no_reordering=0;

    order_of_acquisition_mb = Gadgetron::map_interleaved_acquisitions_mb(lNumberOfStacks_,  no_reordering );

    order_of_acquisition_sb = Gadgetron::map_interleaved_acquisitions(lNumberOfSlices_,  no_reordering );

    indice_mb = sort_index( order_of_acquisition_mb );

    indice_sb = sort_index( order_of_acquisition_sb );


    slice_number_of_acquisition_mb.set_size(lNumberOfStacks_);
    slice_number_of_acquisition_mb.zeros();

    index_of_acquisition_mb.set_size(lNumberOfStacks_);
    index_of_acquisition_mb.zeros();

    for (unsigned long a = 0; a < lNumberOfStacks_; a++)
    {
       slice_number_of_acquisition_mb(a)=indice_sb(a);
    }

    //this is the number of the slices that will be acquired in mb
    std::cout <<  " this is the number of the slices that will be acquired in mb "<< std::endl;
    std::cout <<  slice_number_of_acquisition_mb << std::endl;

    for (unsigned long a = 0; a < lNumberOfStacks_; a++)
    {
        //std::cout << " a "<< a <<   " order_of_acquisition_mb(a) "<< order_of_acquisition_mb(a) <<   "  acquisition order " <<  slice_number_of_acquisition_mb(order_of_acquisition_mb(a)) << std::endl;
        index_of_acquisition_mb(a)=slice_number_of_acquisition_mb(order_of_acquisition_mb(a));

        //std::cout << " a "<< a << "  order_of_acquisition_sb(a)  "<< order_of_acquisition_sb(a) << std::endl;
    }


    compteur_mb=0;
    compteur_sb=0;



    return GADGET_OK;
}


int CheckWIPMultibandBeforeGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
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


    std::ostringstream indice_e1;
    indice_e1 << e1;
    str_e = indice_e1.str();

    std::ostringstream indice_slice;
    indice_slice << slice;
    str_s = indice_slice.str();


    if (is_phase_correction)
    {



    }
    else
    {

    arma::cx_fvec tempo= as_arma_col(*m2->getObjectPtr());

    if (is_parallel)
    {
        ///------------------------------------------------------------------------
        /// FR ce sont les lignes de calibration grappa acs, on les laisse passer, pas touche
        /// UK
        ///
        //if (is_first_scan_in_slice && rep==0)  {std::cout <<" parallel e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep  << std::endl;}


        Gadgetron::SaveVectorOntheDisk(tempo, str_home, "Tempo/", "debug_before_acs_",  str_e, str_s,  ".bin");

    }
    else if (is_acq_single_band )
    {

        if (is_first_scan_in_slice && rep==0)
        {
          //  std::cout <<" BEFORE single_band e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep<<  " compteur_sb : "<< compteur_sb << std::endl;

           // real_order_of_acquisition_sb(compteur_sb)=slice;
           // compteur_sb++;
        }

        Gadgetron::SaveVectorOntheDisk(tempo, str_home, "Tempo/", "debug_before_sb_",  str_e, str_s,  ".bin");

    }
    else if (is_acq_multi_band )
    {


        Gadgetron::SaveVectorOntheDisk(tempo, str_home, "Tempo/", "debug_before_mb_",  str_e, str_s,  ".bin");


       /* if (is_first_scan_in_slice && rep==0)  {


            std::cout <<" multi_band e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<  " compteur_mb : "<< compteur_mb << std::endl;

            real_order_of_acquisition_mb(compteur_mb)=slice;
            compteur_mb++;

            if (compteur_mb==lNumberOfStacks_)
            {
                //std::cout<< "impression"<< std::endl;

                ofstream fichier(mon_fichier.c_str(), ios::out | ios::app);

                fichier << " order_of_acquisition_mb"<< std::endl;
                fichier << "    index   slice_number    order_acq    guest_order   real_order "<< std::endl;

                for (unsigned long a = 0; a < size(order_of_acquisition_mb,0); a++)
                {
                    fichier << "     " << a <<  "          " <<  slice_number_of_acquisition_mb(a)  << "             "<< order_of_acquisition_mb(a)<< "                  " <<index_of_acquisition_mb(a)<<  "               " <<real_order_of_acquisition_mb(a)<< "    "  << Is_Equal(index_of_acquisition_mb (a) , real_order_of_acquisition_mb (a))<< std::endl;
                }

                fichier << "   " << std::endl;
                fichier << " order_of_acquisition_sb"<< std::endl;
                fichier << "    index   order_acq   guest_order   real_order "<< std::endl;

                for (unsigned long a = 0; a < size(order_of_acquisition_sb,0); a++)
                {
                    fichier << "     " << a <<  "          " <<  order_of_acquisition_sb(a)<< "               " <<indice_sb(a)<<  "               " <<real_order_of_acquisition_sb (a)<<  "    "  << Is_Equal(indice_sb (a) , real_order_of_acquisition_sb (a)) <<std::endl;
                }
            }
        }*/


    }
    else
    {
        std::cout <<" e1 "  <<e1  <<  " slice "  <<slice <<  " rep "<< rep <<  "problem unkown "  << std::endl;
        std::cout << "	 -------------------------------------  " << std::endl;
        std::cout << "	!!!!  H O U S T O N ,   O N   A   U N  P R O B L E M E  !!!! " << std::endl;
        std::cout << "	 -------------------------------------  " << std::endl;
    }

    }

    if( this->next()->putq(m1) < 0 ){
        GDEBUG("Failed to put message on queue\n");
        return GADGET_FAIL;
    }

    return GADGET_OK;
}



std::string  CheckWIPMultibandBeforeGadget::Is_Equal(unsigned int a , unsigned int b)
{
    std::string output;

    if (a==b)
    {
        output="TRUE";
    }
        else
    {
         output="FALSE     Slices:" +  std::to_string(lNumberOfSlices_) + "  MB factor:  " +  std::to_string(MB_factor_)+  "  Stacks:  " +  std::to_string(lNumberOfStacks_)+  "  GRAPPA : " +  std::to_string(acceFactorE2_) ;
    }


return output;

}



void CheckWIPMultibandBeforeGadget::get_multiband_parameters(ISMRMRD::IsmrmrdHeader h)
{
    arma::fvec store_info_special_card=Gadgetron::get_information_from_wip_multiband_special_card(h);

    MB_factor_= store_info_special_card(1);
//    Unknow_2 = store_info_special_card(2);
  //  MB_Slice_Inc_=store_info_special_card(3);
    Blipped_CAIPI_=store_info_special_card(4);
   // Unknow_3 = store_info_special_card(5);
   // Unknow_4 = store_info_special_card(6);

}



void CheckWIPMultibandBeforeGadget::deal_with_inline_or_offline_situation(ISMRMRD::IsmrmrdHeader h)
{
    if (doOffline.value()==1)
    {
        MB_factor_=MB_factor.value();
        Blipped_CAIPI_=Blipped_CAIPI.value();
        //MB_Slice_Inc_=MB_Slice_Inc.value();
    }
    else
    {
        get_multiband_parameters(h);
    }
}


GADGET_FACTORY_DECLARE(CheckWIPMultibandBeforeGadget)
}

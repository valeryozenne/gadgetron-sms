
#include "DisplayGadget.h"
#include <iomanip>
#include <sstream>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "mri_core_utility.h"


namespace Gadgetron {

DisplayGadget::DisplayGadget() : BaseClass()
{
}

DisplayGadget::~DisplayGadget()
{
}

int DisplayGadget::process_config(ACE_Message_Block* mb)
{
    GADGET_CHECK_RETURN(BaseClass::process_config(mb) == GADGET_OK, GADGET_FAIL);

    ISMRMRD::IsmrmrdHeader h;
    try
    {
        deserialize(mb->rd_ptr(), h);
    }
    catch (...)
    {
        GDEBUG("Error parsing ISMRMRD Header");
    }

    //ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;

    //dimensions_.push_back(e_space.matrixSize.x);
   // dimensions_.push_back(e_space.matrixSize.y);
    //dimensions_.push_back(e_space.matrixSize.z);


    /////////////////////////////////
    ///get_user_parameter_from_hdr
    /// assuming only one encoding

    std::vector<ISMRMRD::UserParameterDouble> vector_userParameterDouble;

    if (h.userParameters)
    {
        GDEBUG_STREAM("ParameterDouble in userParameters")
                vector_userParameterDouble =  h.userParameters->userParameterDouble;
    }
    else
    {
        GDEBUG_STREAM("ParameterDouble in trajectoryDescription")
                vector_userParameterDouble = h.encoding[0].trajectoryDescription->userParameterDouble;
    }

    is_wip_sequence=0;
    is_cmrr_sequence=0;

    /////////////////////////////////////
    ///find_acquisition_is_wip_or_cmrr
    ///

    //std::cout << vector_userParameterDouble.size() << std::endl;
    std::vector<ISMRMRD::UserParameterDouble>::const_iterator iter = vector_userParameterDouble.begin();

    for (; iter != vector_userParameterDouble.end(); iter++)
    {
        std::string usrParaName = iter->name;
        double usrParaValue = iter->value;

        //std::cout << " iter->name  "<<  iter->name << " iter->value "<< iter->value << std::endl;
        std::stringstream str;
        str << "MB_factor";

        std::stringstream str2;
        str2 << "NMBSliceBands";

        if (usrParaName == str.str() )
        {
            is_wip_sequence = 1;
            GDEBUG_STREAM("Sequence is Siemens WIP, find MB_factor : " << usrParaValue );
        }

        if (usrParaName == str2.str() )
        {
            is_cmrr_sequence = 1;
            GDEBUG_STREAM("Sequence is Siemens CMRR, find NMBSliceBands : " << usrParaValue );
        }
    }


    ////////////////////////////
    /// read MB_factor && Blipped_CAIPI parameters
    ///

    if (is_wip_sequence==1 && is_cmrr_sequence==0  )
    {
        iter = vector_userParameterDouble.begin();
        for (; iter != vector_userParameterDouble.end(); iter++)
        {
            std::string usrParaName = iter->name;
            double usrParaValue = iter->value;
            if (iter->name == "MB_factor") {
                MB_factor = iter->value;
            } else if (iter->name == "Blipped_CAIPI") {
                Blipped_CAIPI =iter->value;
            }
        }
    }
    else if (is_cmrr_sequence==1 && is_wip_sequence==0   )
    {

        iter = vector_userParameterDouble.begin();
        for (; iter != vector_userParameterDouble.end(); iter++)
        {
            std::string usrParaName = iter->name;
            double usrParaValue = iter->value;
            if (iter->name == "NMBSliceBands") {
                MB_factor = iter->value;
            } else if (iter->name == "BlipFactorSL") {
                Blipped_CAIPI =iter->value;
            }
        }
    }
    else
    {
        GERROR("is_wip_sequence && is_cmrr_sequence ");
    }

    GDEBUG_STREAM("Find MB factor : " << MB_factor << " find Blipped_CAIPI factor: "<< Blipped_CAIPI);

    ////////////////////////////
    /// adjust Blipped_CAIPI parameter if in plane acceleration
    ///

    size_t NE = h.encoding.size();
    num_encoding_spaces_ = NE;
    GDEBUG_CONDITION_STREAM(verbose.value(), "Number of encoding spaces: " << NE);

    acceFactorSMSE1_.resize(NE, 1);
    acceFactorSMSE2_.resize(NE, 1);
    //calib_SMSmode_.resize(NE, ISMRMRD_noacceleration);

    size_t e;
    for (e = 0; e < NE; e++)
    {
        if (!h.encoding[e].parallelImaging)
        {
            GDEBUG_STREAM("Parallel Imaging section not found in header for encoding space " << e);
            //calib_SMSmode_[e] = ISMRMRD_noacceleration;
            acceFactorSMSE1_[e] = 1;
            acceFactorSMSE2_[e] = 1;
        }
        else
        {
            ISMRMRD::ParallelImaging p_imaging = *h.encoding[e].parallelImaging;
            acceFactorSMSE1_[e] = p_imaging.accelerationFactor.kspace_encoding_step_1;
            acceFactorSMSE2_[e] = p_imaging.accelerationFactor.kspace_encoding_step_2;
            GDEBUG_CONDITION_STREAM(verbose.value(), "acceFactorE1 is " << acceFactorSMSE1_[e]);
            GDEBUG_CONDITION_STREAM(verbose.value(), "acceFactorE2 is " << acceFactorSMSE2_[e]);
        }
    }

    if (is_wip_sequence==1 && acceFactorSMSE1_[0]>1)
    {
        Blipped_CAIPI=Blipped_CAIPI*acceFactorSMSE1_[0];
        GDEBUG_STREAM("Find MB factor : " << MB_factor << " adjusted Blipped_CAIPI factor: "<< Blipped_CAIPI);
    }

    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    lNumberOfSlices_ = e_limits.slice? e_limits.slice->maximum+1 : 1;
    lNumberOfChannels_ = h.acquisitionSystemInformation->receiverChannels ? *h.acquisitionSystemInformation->receiverChannels : 1;
    lNumberOfStacks_=lNumberOfSlices_/MB_factor;

    GDEBUG_STREAM(" lNumberOfSlices_  : " << lNumberOfSlices_ << " MB factor: "<< MB_factor << " lNumberOfStacks_: "<< lNumberOfStacks_ );


    //////////////
    /// \brief
    ///

    bool no_reordering=0;

    order_of_acquisition_mb=map_interleaved_acquisitions(lNumberOfStacks_, no_reordering);
    order_of_acquisition_sb=map_interleaved_acquisitions(lNumberOfSlices_, no_reordering);

    //std::cout <<  order_of_acquisition_mb << std::endl;
    //std::cout <<  order_of_acquisition_sb << std::endl;

    // indice_mb =  arma::sort_index( order_of_acquisition_mb );
    // indice_sb =  arma::sort_index( order_of_acquisition_sb );
    // indice_slice_mb=indice_sb.rows(0,lNumberOfStacks_-1);
    indice_mb = sort_index(order_of_acquisition_mb);
    indice_sb = sort_index(order_of_acquisition_sb);

    for (unsigned int i = 0; i < lNumberOfStacks_; i++)
    {
        indice_slice_mb.push_back(indice_sb[i]);
    }







     counter_=0;

    // -------------------------------------------------

    return GADGET_OK;
}

int DisplayGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
{
    if (perform_timing.value()) { gt_timer_local_.start("DisplayGadget::process"); }

    GDEBUG_CONDITION_STREAM(verbose.value(), "DisplayGadget::process(...) starts ... ");

    // -------------------------------------------------------------
    process_called_times_++;

    // -------------------------------------------------------------



    if (counter_==0)
    {
        m1->release();
    }
    else
    {



    IsmrmrdImageArray* data = m1->getObjectPtr();

    hoNDArray< ISMRMRD::ImageHeader > headers_=m1->getObjectPtr()->headers_;

    size_t RO = data->data_.get_size(0);
    size_t E1 = data->data_.get_size(1);
    size_t E2 = data->data_.get_size(2);
    size_t CHA = data->data_.get_size(3);
    size_t N = data->data_.get_size(4);
    size_t S = data->data_.get_size(5);
    size_t SLC = data->data_.get_size(6);


    size_t hN = headers_.get_size(0);
    size_t hS = headers_.get_size(1);
    size_t hSLC = headers_.get_size(2);

    //GDEBUG_STREAM(  " N :  "  << hN  <<" S: "<< hS <<" SLC: " << hSLC);
    size_t repetition;
    size_t image_index;
    size_t image_series_index;

    //TODO must be allocated once
    hoNDArray<float > shift_from_isocenter;
    shift_from_isocenter.create(3);
    hoNDArray<float > slice_dir;
    slice_dir.create(3);

    //TODO must be allocated once
    //hoNDArray<float > shift_from_isocenter_all;
    //shift_from_isocenter_all.create(3, SLC);
    //hoNDArray<float > slice_dir_all;
    //slice_dir.create(3, SLC);

    hoNDArray<float > z_offset;
    z_offset.create(SLC);

    //hoNDArray<std::complex<float>> temporaire= data->data_;

    /*for  (size_t slc = 0; slc <  hSLC; slc++) {

            ISMRMRD::ImageHeader& curr_image_header = headers_(0, 0,  slc);
            for (int j = 0; j < 3; j++) {

                shift_from_isocenter_all(j,slc)=curr_image_header.position[j];
                slice_dir_slc(j,slc)=curr_image_header.slice_dir[j];
            }

     }*/

    for  (size_t slc = 0; slc <  hSLC; slc++) {

        ISMRMRD::ImageHeader& curr_image_header = headers_(0, 0,  slc);

        if ( counter_==0)
        {
            curr_image_header.image_series_index=curr_image_header.image_series_index+2000;
        }
        else
        {
            curr_image_header.image_index=curr_image_header.image_index+hSLC;
        }

        repetition =  curr_image_header.repetition;
        image_index =  curr_image_header.image_index;
        image_series_index =  curr_image_header.image_series_index;

        for (int j = 0; j < 3; j++) {

            shift_from_isocenter(j)=curr_image_header.position[j];
            slice_dir(j)=curr_image_header.slice_dir[j];
        }

        z_offset(slc) = dot(shift_from_isocenter,slice_dir);

        GDEBUG_STREAM( "slc "<<slc  << " repetition :  "  << repetition  <<  " image_index :  "  << image_index   <<  " image_series_index :  "  << image_series_index  );
        GDEBUG_STREAM( "slc "<<slc  << " shift_from_isocenter  " <<shift_from_isocenter(0) <<"  "<< shift_from_isocenter(1) <<"  "<<shift_from_isocenter(2) <<"   z_offset(slc) "<<  z_offset(slc)<< " order_of_acquisition_sb  "<<order_of_acquisition_sb[slc] << " indice _sb "<<indice_sb[slc]);
    }

    // print out data info
    if (verbose.value())
    {
        GDEBUG_STREAM("----> DisplayGadget::process(...) has been called " << process_called_times_ << " times ...");
        std::stringstream os;
        data->data_.print(os);
        //GDEBUG_STREAM(os.str());
    }

    // -------------------------------------------------------------

    if (this->next()->putq(m1) == -1)
    {
        GERROR("DisplayGadget::process, passing map on to next gadget");
        return GADGET_FAIL;
    }

     }

    if (perform_timing.value()) { gt_timer_local_.stop(); }

     counter_++;

    return GADGET_OK;
}


std::vector<unsigned int> DisplayGadget::sort_index(std::vector<unsigned int> array)
{
    std::vector<unsigned int>sortedArray(array);

    std::sort(sortedArray.begin(), sortedArray.end());
    //quicksort(sortedArray, 0, arraySize - 1);
    for (int i = 0; i < (array.size()); i++)
    {
        for (int j = 0; j < (array.size()); j++)
        {
            if (sortedArray[i] == array[j])
            {
                sortedArray[i] = j;
                break;
            }
        }
    }
    return (sortedArray);
}




std::vector<unsigned int> DisplayGadget::map_interleaved_acquisitions(int number_of_slices, bool no_reordering )
{

    std::vector<unsigned int> index(number_of_slices, 0);
    //index.zeros();


    if(no_reordering)
    {
        GDEBUG("CAUTION there is no reordering for single band images \n");

        for (unsigned int i = 0; i < number_of_slices; i++)
        {
            index[i]=i;
        }
    }
    else
    {
        GDEBUG("Reordering with interleaved pattern for single band images \n");

        if (number_of_slices%2)
        {
            index[0]=0;
            GDEBUG("Number of single band images is odd \n");
        }
        else
        {
            index[0]=1;
            GDEBUG("Number of single band images is even \n");
        }

        for (unsigned int i = 1; i < number_of_slices; i++)
        {
            index[i]=index[i-1]+2;

            if (index[i]>=number_of_slices)
            {
                if (number_of_slices%2)
                {
                    index[i]=1;
                }
                else
                {
                    index[i]=0;
                }
            }
        }
    }
    return index;
}

GADGET_FACTORY_DECLARE(DisplayGadget)

}

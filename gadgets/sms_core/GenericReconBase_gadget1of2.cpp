
#include "GenericReconBase_gadget1of2.h"
#include <boost/filesystem.hpp>

namespace Gadgetron {

    template <typename T, typename U> 
    GenericReconBase_gadget1of2<T, U>::GenericReconBase_gadget1of2() : num_encoding_spaces_(1), process_called_times_(0)
    {
        gt_timer_.set_timing_in_destruction(false);
        gt_timer_local_.set_timing_in_destruction(false);
    }

    template <typename T, typename U> 
    GenericReconBase_gadget1of2<T, U>::~GenericReconBase_gadget1of2()
    {
    }

    template <typename T, typename U> 
    int GenericReconBase_gadget1of2<T, U>::process_config(ACE_Message_Block* mb)
    {
        if (!debug_folder.value().empty())
        {
            Gadgetron::get_debug_folder_path(debug_folder.value(), debug_folder_full_path_);
            GDEBUG_CONDITION_STREAM(verbose.value(), "Debug folder is " << debug_folder_full_path_);

            // Create debug folder if necessary
            boost::filesystem::path boost_folder_path(debug_folder_full_path_);
            try
            {
                boost::filesystem::create_directories(boost_folder_path);
            }
            catch (...)
            {
                GERROR("Error creating the debug folder.\n");
                return false;
            }
        }
        else
        {
            GDEBUG_CONDITION_STREAM(verbose.value(), "Debug folder is not set ... ");
        }

        return GADGET_OK;
    }

    template <typename T, typename U>
    int GenericReconBase_gadget1of2<T, U>::process(GadgetContainerMessage<T>* m1)
    {
        return GADGET_OK;
    }


    template <typename T, typename U>
    int GenericReconBase_gadget1of2<T, U>::process(GadgetContainerMessage<U>* m2)
    {
        return GADGET_OK;
    }

    //template class EXPORTGADGETSSMSCORE GenericReconBase_gadget1of2<ISMRMRD::AcquisitionHeader, s_EPICorrection>;
    template class EXPORTGADGETSSMSCORE GenericReconBase_gadget1of2<IsmrmrdReconData, s_EPICorrection>;
    template class EXPORTGADGETSSMSCORE GenericReconBase_gadget1of2<IsmrmrdImageArray, s_EPICorrection>;
    template class EXPORTGADGETSSMSCORE GenericReconBase_gadget1of2<ISMRMRD::ImageHeader, s_EPICorrection>;

    GenericReconKSpaceReadoutBase_gadget1of2::GenericReconKSpaceReadoutBase_gadget1of2() : BaseClass()
    {
    }

    GenericReconKSpaceReadoutBase_gadget1of2::~GenericReconKSpaceReadoutBase_gadget1of2()
    {
    }

    GenericReconDataBase_gadget1of2::GenericReconDataBase_gadget1of2() : BaseClass()
    {
    }

    GenericReconDataBase_gadget1of2::~GenericReconDataBase_gadget1of2()
    {
    }

    GenericReconImageBase_gadget1of2::GenericReconImageBase_gadget1of2() : BaseClass()
    {
    }

    GenericReconImageBase_gadget1of2::~GenericReconImageBase_gadget1of2()
    {
    }

    GenericReconImageHeaderBase_gadget1of2::GenericReconImageHeaderBase_gadget1of2() : BaseClass()
    {
    }

    GenericReconImageHeaderBase_gadget1of2::~GenericReconImageHeaderBase_gadget1of2()
    {
    }

    GADGET_FACTORY_DECLARE(GenericReconKSpaceReadoutBase_gadget1of2)
    GADGET_FACTORY_DECLARE(GenericReconDataBase_gadget1of2)
    GADGET_FACTORY_DECLARE(GenericReconImageBase_gadget1of2)
    GADGET_FACTORY_DECLARE(GenericReconImageHeaderBase_gadget1of2)
}

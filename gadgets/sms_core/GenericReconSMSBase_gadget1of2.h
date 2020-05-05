/** \file   GenericReconSMSBase_gadget1of2.h
    \brief  This serves an optional base class gadget for the generic chain.
            Some common functionalities are implemented here and can be reused in specific recon gadgets.
            This gadget is instantiated for IsmrmrdReconData and IsmrmrdImageArray
    \author Hui Xue
*/

#pragma once

#include <complex>
#include "gadgetron_mricore_export.h"
#include "Gadget.h"
#include "GadgetronTimer.h"

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/xml.h"
#include "ismrmrd/meta.h"

#include "mri_core_def.h"
#include "mri_core_data.h"
#include "mri_core_utility.h"

#include "ImageIOAnalyze.h"

#include "gadgetron_sha1.h"

namespace Gadgetron {

    template <typename T> 
    class EXPORTGADGETSMRICORE GenericReconSMSBase_gadget1of2 : public Gadget1<T>
    {
    public:
        GADGET_DECLARE(GenericReconSMSBase_gadget1of2);

        typedef Gadget1<T> BaseClass;

        GenericReconSMSBase_gadget1of2();
        ~GenericReconSMSBase_gadget1of2();

        /// ------------------------------------------------------------------------------------
        /// debug and timing
        GADGET_PROPERTY(verbose, bool, "Whether to print more information", false);
        GADGET_PROPERTY(debug_folder, std::string, "If set, the debug output will be written out", "");
        GADGET_PROPERTY(perform_timing, bool, "Whether to perform timing on some computational steps", false);

        /// ms for every time tick
        GADGET_PROPERTY(time_tick, float, "Time tick in ms", 2.5);

    protected:

        // number of encoding spaces in the protocol
        size_t num_encoding_spaces_;

        // number of times the process function is called
        size_t process_called_times_;

        // --------------------------------------------------
        // variables for debug and timing
        // --------------------------------------------------

        // clock for timing
        Gadgetron::GadgetronTimer gt_timer_local_;
        Gadgetron::GadgetronTimer gt_timer_;

        // debug folder
        std::string debug_folder_full_path_;

        // exporter
        Gadgetron::ImageIOAnalyze gt_exporter_;

        // --------------------------------------------------
        // gadget functions
        // --------------------------------------------------
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(GadgetContainerMessage<T>* m1);
    };

    class EXPORTGADGETSMRICORE GenericReconKSpaceReadoutBase_gadget1of2 :public GenericReconSMSBase_gadget1of2 < ISMRMRD::AcquisitionHeader >
    {
    public:
        GADGET_DECLARE(GenericReconKSpaceReadoutBase_gadget1of2);

        typedef GenericReconSMSBase_gadget1of2 < ISMRMRD::AcquisitionHeader > BaseClass;

        GenericReconKSpaceReadoutBase_gadget1of2();
        virtual ~GenericReconKSpaceReadoutBase_gadget1of2();
    };

    class EXPORTGADGETSMRICORE GenericReconDataBase_gadget1of2 :public GenericReconSMSBase_gadget1of2 < IsmrmrdReconData >
    {
    public:
        GADGET_DECLARE(GenericReconDataBase_gadget1of2);

        typedef GenericReconSMSBase_gadget1of2 < IsmrmrdReconData > BaseClass;

        GenericReconDataBase_gadget1of2();
        virtual ~GenericReconDataBase_gadget1of2();
    };

    class EXPORTGADGETSMRICORE GenericReconImageBase_gadget1of2 :public GenericReconSMSBase_gadget1of2 < IsmrmrdImageArray >
    {
    public:
        GADGET_DECLARE(GenericReconImageBase_gadget1of2);

        typedef GenericReconSMSBase_gadget1of2 < IsmrmrdImageArray > BaseClass;

        GenericReconImageBase_gadget1of2();
        virtual ~GenericReconImageBase_gadget1of2();
    };

    class EXPORTGADGETSMRICORE GenericReconImageHeaderBase_gadget1of2 :public GenericReconSMSBase_gadget1of2 < ISMRMRD::ImageHeader >
    {
    public:
        GADGET_DECLARE(GenericReconImageHeaderBase_gadget1of2);

        typedef GenericReconSMSBase_gadget1of2 < ISMRMRD::ImageHeader > BaseClass;

        GenericReconImageHeaderBase_gadget1of2();
        virtual ~GenericReconImageHeaderBase_gadget1of2();
    };
}

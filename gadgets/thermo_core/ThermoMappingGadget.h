/** \file   ThermoMappingGadget.h
    \brief  This is the class gadget for cardiac parametric mapping, working on the IsmrmrdImageArray.
    \author Hui Xue
*/

#pragma once

#include "gadgetron_thermocore_export.h"
#include "GenericReconBase.h"

namespace Gadgetron {

    class EXPORTGADGETSTHERMOCORE ThermoMappingGadget : public GenericReconImageBase
    {
    public:
        GADGET_DECLARE(ThermoMappingGadget);

        typedef GenericReconImageBase BaseClass;

        ThermoMappingGadget();
        ~ThermoMappingGadget();

        /// ------------------------------------------------------------------------------------
        /// parameters to control the mapping
        /// ------------------------------------------------------------------------------------

       
        // encoding space size
        ISMRMRD::EncodingCounters meas_max_idx_;
        // ------------------------------------------------------------------------------------

    protected:

        // default interface function
        virtual int process_config(ACE_Message_Block* mb);
        virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1);


        
    };
}

/** \file   ThermoMappingBase.h
    \brief  This is the class gadget for temperature mapping, working on the IsmrmrdImageArray.
    \author Hui Xue
*/

#pragma once

#include "gadgetron_thermocore_export.h"
#include "GenericReconBase.h"

namespace Gadgetron {

    class EXPORTGADGETSTHERMOCORE ThermoMappingBase : public GenericReconImageBase
    {
    public:
        GADGET_DECLARE(ThermoMappingBase);

        typedef GenericReconImageBase BaseClass;

        ThermoMappingBase();
        ~ThermoMappingBase();

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


        unsigned int lNumberOfSlices_;
        unsigned int lNumberOfAverages_;
        unsigned int lNumberOfRepetitions_;
        unsigned int lNumberOfInterventions;

        float Te_;
        float Tr_;
        float B0_;
        float k_value_;
        
    };
}

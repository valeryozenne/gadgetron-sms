
#include "ThermoMappingBase.h"
#include <iomanip>
#include <sstream>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "mri_core_utility.h"


namespace Gadgetron {

    ThermoMappingBase::ThermoMappingBase() : BaseClass()
    {
    }

    ThermoMappingBase::~ThermoMappingBase()
    {
    }

    int ThermoMappingBase::process_config(ACE_Message_Block* mb)
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

        if (!h.acquisitionSystemInformation)
        {
            GDEBUG("acquisitionSystemInformation not found in header. Bailing out");
            return GADGET_FAIL;
        }

        // -------------------------------------------------

        float field_strength_T_ = h.acquisitionSystemInformation.get().systemFieldStrength_T();
        GDEBUG_CONDITION_STREAM(verbose.value(), "field_strength_T_ is read from protocol : " << field_strength_T_);

        if (h.encoding.size() != 1)
        {
            GDEBUG("Number of encoding spaces: %d\n", h.encoding.size());
        }

        size_t e = 0;
        ISMRMRD::EncodingSpace e_space = h.encoding[e].encodedSpace;
        ISMRMRD::EncodingSpace r_space = h.encoding[e].reconSpace;
        ISMRMRD::EncodingLimits e_limits = h.encoding[e].encodingLimits;

        meas_max_idx_.kspace_encode_step_1 = (uint16_t)e_space.matrixSize.y - 1;
        meas_max_idx_.set = (e_limits.set && (e_limits.set->maximum>0)) ? e_limits.set->maximum : 0;
        meas_max_idx_.phase = (e_limits.phase && (e_limits.phase->maximum>0)) ? e_limits.phase->maximum : 0;
        meas_max_idx_.kspace_encode_step_2 = (uint16_t)e_space.matrixSize.z - 1;
        meas_max_idx_.contrast = (e_limits.contrast && (e_limits.contrast->maximum > 0)) ? e_limits.contrast->maximum : 0;
        meas_max_idx_.slice = (e_limits.slice && (e_limits.slice->maximum > 0)) ? e_limits.slice->maximum : 0;
        meas_max_idx_.repetition = e_limits.repetition ? e_limits.repetition->maximum : 0;
        meas_max_idx_.average = e_limits.average ? e_limits.average->maximum : 0;
        meas_max_idx_.segment = 0;

        lNumberOfSlices_ = e_limits.slice? e_limits.slice->maximum+1 : 1;  /* Number of slices in one measurement */ //MANDATORY
        lNumberOfAverages_= e_limits.average? e_limits.average->maximum+1 : 1;
        lNumberOfRepetitions_= e_limits.repetition? e_limits.repetition->maximum+1 : 1;

        field_strength_T_ = h.acquisitionSystemInformation.get().systemFieldStrength_T();
        GDEBUG_CONDITION_STREAM(verbose.value(), "field_strength_T_ is read from protocol : " << field_strength_T_);

        ISMRMRD::SequenceParameters seq_info = *h.sequenceParameters;
        Tr_=seq_info.TR.get().front();
        Te_=seq_info.TE.get().front();

        /// proton shift frenquency parameters
        float Te=Te_*1e-3;
        float B0=field_strength_T_;
        float alpha=0.0094;
        float gamma=42.58;
        k_value_=-1/(gamma*alpha*B0*Te*2*M_PI);

        GDEBUG_CONDITION_STREAM(verbose.value(), "kvalue is calculated from protocol : " << k_value_);





        return GADGET_OK;
    }

    int ThermoMappingBase::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
    {        

        return GADGET_OK;
    }

    // ----------------------------------------------------------------------------------------

     GADGET_FACTORY_DECLARE(ThermoMappingBase)

}

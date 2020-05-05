
#include "ThermoMappingGadget.h"
#include <iomanip>
#include <sstream>

#include "hoNDArray_reductions.h"
#include "mri_core_def.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "mri_core_utility.h"


namespace Gadgetron {

    ThermoMappingGadget::ThermoMappingGadget() : BaseClass()
    {
    }

    ThermoMappingGadget::~ThermoMappingGadget()
    {
    }

    int ThermoMappingGadget::process_config(ACE_Message_Block* mb)
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

        return GADGET_OK;
    }

    int ThermoMappingGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdImageArray >* m1)
    {
        if (perform_timing.value()) { gt_timer_local_.start("ThermoMappingGadget::process"); }

        GDEBUG_CONDITION_STREAM(verbose.value(), "ThermoMappingGadget::process(...) starts ... ");

        // -------------------------------------------------------------

        process_called_times_++;

        // -------------------------------------------------------------

        IsmrmrdImageArray* data = m1->getObjectPtr();

        // print out data info
        if (verbose.value())
        {
            GDEBUG_STREAM("----> ThermoMappingGadget::process(...) has been called " << process_called_times_ << " times ...");
            std::stringstream os;
            data->data_.print(os);
            GDEBUG_STREAM(os.str());
        }

        // -------------------------------------------------------------

	if (this->next()->putq(cm1) == -1)
                {
                    GERROR("ThermoMappingGadget::process, passing map on to next gadget");
                    return GADGET_FAIL;
                }

        if (perform_timing.value()) { gt_timer_local_.stop(); }

        return GADGET_OK;
    }

    // ----------------------------------------------------------------------------------------

    // GADGET_FACTORY_DECLARE(ThermoMappingGadget)

}

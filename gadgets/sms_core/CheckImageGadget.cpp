/*
 *       CheckImageGadget.cpp
 *       Author: Hui Xue
 */

#include "CheckImageGadget.h"
#include "hoNDArray_elemwise.h"

Gadgetron::CheckImageGadget::CheckImageGadget(
    const Gadgetron::Core::Context& context, const Gadgetron::Core::GadgetProperties& props)
    : PureGadget(props) {


};

Gadgetron::Core::Image<ushort> Gadgetron::CheckImageGadget::process_function(
    Gadgetron::Core::Image<ushort> input_image) const {

    auto& header     = std::get<ISMRMRD::ImageHeader>(input_image);
    auto& meta       = std::get<2>(input_image);
    auto& input_data = std::get<hoNDArray<ushort>>(input_image);

   // hoNDArray<ushort> output_data;

    GDEBUG_STREAM("--------------------------------------------------");
    GDEBUG_STREAM("header.repetition                      " <<  header.repetition);
    GDEBUG_STREAM("header.slice                           " <<  header.slice);
    GDEBUG_STREAM("header.data_type                       " <<  header.data_type);
    GDEBUG_STREAM("header.image_index                     " <<  header.image_index);
    GDEBUG_STREAM("header.image_series_index              " <<  header.image_series_index);
    GDEBUG_STREAM("header.image_type                      " <<  header.image_type);
    GDEBUG_STREAM("header.measurement_uid                 " <<  header.measurement_uid);
    GDEBUG_STREAM("header.acquisition_time_stamp          " <<  header.acquisition_time_stamp);

    return { header, input_data, meta };
}

namespace Gadgetron{
    GADGETRON_GADGET_EXPORT(CheckImageGadget)
}

/** \file   CheckImageGadget.h
    \brief  This Gadget converts complex ushort values to ushort format.
    \author Hui Xue
*/

#pragma once
#include "hoNDArray.h"
#include "ismrmrd/meta.h"

#include <ismrmrd/ismrmrd.h>

#include <Types.h>
#include "PureGadget.h"
namespace Gadgetron
{
class CheckImageGadget: public Core::PureGadget<Core::Image<ushort>,Core::Image<ushort>>
    {
    public:
        CheckImageGadget(const Core::Context& context, const Core::GadgetProperties& props);

        Core::Image<ushort> process_function(Core::Image<ushort> args) const override;
    private:

        int toto;
};
}


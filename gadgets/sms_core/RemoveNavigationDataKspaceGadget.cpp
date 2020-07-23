/*
* RemoveNavigationDataKspaceGadget.cpp
*
*  Created on: Dec 5, 2011
*      Author: hansenms
*/

#include "RemoveNavigationDataKspaceGadget.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include "ismrmrd/xml.h"

namespace Gadgetron{

RemoveNavigationDataKspaceGadget::RemoveNavigationDataKspaceGadget() {
}

RemoveNavigationDataKspaceGadget::~RemoveNavigationDataKspaceGadget() {
}

int RemoveNavigationDataKspaceGadget::process_config(ACE_Message_Block *mb)
{
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);

    return GADGET_OK;
}


int RemoveNavigationDataKspaceGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1, GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2)
{
    bool is_navigation_data = m1->getObjectPtr()->isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA );

    if (is_navigation_data)
    {
        m1->release();
    }
    else
    {
        if( this->next()->putq(m1) < 0 ){
            GDEBUG("Failed to put message on queue\n");
            return GADGET_FAIL;
        }
    }

    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(RemoveNavigationDataKspaceGadget)
}



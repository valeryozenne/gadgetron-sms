/*
 * AutoScaleGadget.cpp
 *
 *  Created on: Dec 19, 2011
 *      Author: Michael S. Hansen
 */

#include "MessageContainerNumberGadget.h"

namespace Gadgetron{

MessageContainerNumberGadget::MessageContainerNumberGadget()
{
}

MessageContainerNumberGadget::~MessageContainerNumberGadget() {
	// TODO Auto-generated destructor stub
}

int MessageContainerNumberGadget::process_config(ACE_Message_Block* mb) {
        max_value_ = 0;
		
	return GADGET_OK;
}


int MessageContainerNumberGadget::process(GadgetContainerMessage<ISMRMRD::ImageHeader> *m1, GadgetContainerMessage<hoNDArray< float> > *m2)
{
	max_value_++;
    if (avant.value())
	{
		GDEBUG_STREAM("Number of Message Containers before: " << max_value_ << " ptr : " << m1->getObjectPtr()->image_index << std::endl );
	}
	else
	{
		GDEBUG_STREAM("Number of Message Containers after: " << max_value_ << " ptr: " << m1->getObjectPtr()->image_index << std::endl );
	
	}
	if (this->next()->putq(m1) < 0) {
		GDEBUG("Failed to pass on data to next Gadget\n");
		return GADGET_FAIL;
	}

	return GADGET_OK;
}

GADGET_FACTORY_DECLARE(MessageContainerNumberGadget)

}

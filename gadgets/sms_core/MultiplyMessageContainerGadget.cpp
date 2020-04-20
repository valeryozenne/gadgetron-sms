/*
 * AutoScaleGadget.cpp
 *
 *  Created on: Dec 19, 2011
 *      Author: Michael S. Hansen
 */

#include "MultiplyMessageContainerGadget.h"
#include "mri_core_def.h"

namespace Gadgetron{

MultiplyMessageContainerGadget::MultiplyMessageContainerGadget()
{
}

MultiplyMessageContainerGadget::~MultiplyMessageContainerGadget() {
	// TODO Auto-generated destructor stub
}

int MultiplyMessageContainerGadget::process_config(ACE_Message_Block* mb) {
	return GADGET_OK;
}

int MultiplyMessageContainerGadget::process(GadgetContainerMessage<ISMRMRD::ImageHeader> *m1, GadgetContainerMessage<hoNDArray< float> > *m2)
{
	GadgetContainerMessage<ISMRMRD::ImageHeader> *cm1 = new GadgetContainerMessage<ISMRMRD::ImageHeader> ();
    GadgetContainerMessage<hoNDArray< float>  > *cm2 = new GadgetContainerMessage<hoNDArray< float>  > ();
    Gadgetron::GadgetContainerMessage<ISMRMRD::MetaContainer>* cm3 = new Gadgetron::GadgetContainerMessage<ISMRMRD::MetaContainer>();

	memcpy(cm1->getObjectPtr(), m1->getObjectPtr(), sizeof(ISMRMRD::ImageHeader));
	//memcpy(cm2->getObjectPtr(), &imagearr.headers_(n,s,loc), sizeof(ISMRMRD::ImageHeader));

	boost::shared_ptr< std::vector<size_t> > dims = m2->getObjectPtr()->get_dimensions();	

	try{
		cm2->getObjectPtr()->create(dims);
		}
    catch (std::runtime_error &err)
		{
        	GEXCEPTION(err,"Unable to allocate new image\n");
            cm1->release();
            cm2->release();
            return GADGET_FAIL;
        }

	size_t data_length_2 = 1;

	for (int i = 0; i < dims->size(); i++)
	{
		data_length_2 *= dims->at(i);	
	}

	size_t data_length = m2->getObjectPtr()->get_number_of_elements();

	GDEBUG_STREAM("data_length get_number_of_elements: " << data_length << std::endl);
	GDEBUG_STREAM("data_length for: " << data_length_2 << std::endl);
    memcpy(cm2->getObjectPtr()->get_data_ptr(), m2->getObjectPtr()->get_data_ptr(), data_length*sizeof(float));


    cm1->getObjectPtr()->image_type = ISMRMRD::ISMRMRD_IMTYPE_PHASE;
    cm1->getObjectPtr()->image_series_index += 3000; //Ensure that this will go in a different series

    cm3->getObjectPtr()->set(GADGETRON_IMAGECOMMENT, "Bordeaux Thermometry");
    cm3->getObjectPtr()->set(GADGETRON_SEQUENCEDESCRIPTION, "Phase");
    cm3->getObjectPtr()->set(GADGETRON_IMAGEPROCESSINGHISTORY, "GT");
    //cm3->getObjectPtr()->set(GADGETRON_IMAGE_SCALE_RATIO, (double)(1));
    //cm3->getObjectPtr()->set(GADGETRON_IMAGE_COLORMAP, "Perfusion.pal");
    //cm3->getObjectPtr()->set(GADGETRON_IMAGE_WINDOWCENTER, (long)(38));
    //cm3->getObjectPtr()->set(GADGETRON_IMAGE_WINDOWWIDTH, (long)(46));


    //Chain them
    cm1->cont(cm2);
    cm2->cont(cm3);

	if (this->next()->putq(m1) < 0) {
		GDEBUG("Failed to pass on data to next Gadget\n");
		return GADGET_FAIL;
	}

	if (this->next()->putq(cm1) < 0) {
		GDEBUG("Failed to pass on data to next Gadget\n");
		return GADGET_FAIL;
	}

	return GADGET_OK;
}

GADGET_FACTORY_DECLARE(MultiplyMessageContainerGadget)

}

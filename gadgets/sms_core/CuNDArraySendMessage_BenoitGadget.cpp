#include "CuNDArraySendMessage_BenoitGadget.h"
//#include "test_gpu_benoit.h"

namespace Gadgetron{



int CuNDArraySendMessage_BenoitGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
	                                 GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
{

  hoNDArray< std::complex<float> > *data = m2->getObjectPtr();

  cuNDArray<complext<float>> k_space_data(reinterpret_cast<const hoNDArray<complext<float>> &>(*data));

  GDEBUG_STREAM("Data content begin: " << (*(data))[0] << ", " << (*(data))[1] << ", " << (*(data))[2] << ", " << (*(data))[3]);

  std::vector<size_t> dimensions = *(m2->getObjectPtr()->get_dimensions());

  GadgetContainerMessage< cuNDArray<complext<float> >> *m3 = new GadgetContainerMessage<cuNDArray<complext<float>>>();
  try{m3->getObjectPtr()->create(dimensions);}
  catch (std::runtime_error &err){
  	GEXCEPTION(err,"CombineGadget, failed to allocate new array\n");
    return -1;
  }



  cudaMemcpy(m3->getObjectPtr()->get_data_ptr(), k_space_data.get_data_ptr(), k_space_data.get_number_of_elements() * sizeof(complext<float>), cudaMemcpyDeviceToDevice);

  //adding the new array to the outgoing message
  m1->cont(m3);
  //and releasing the unused one
  m2->release();
  
  if (this->next()->putq(m1) == -1) {  
    m1->release();
    GERROR("AcquisitionPassthroughGadget::process, passing data on to next gadget");
    return -1;
  }

  return 0;
}
GADGET_FACTORY_DECLARE(CuNDArraySendMessage_BenoitGadget)
}



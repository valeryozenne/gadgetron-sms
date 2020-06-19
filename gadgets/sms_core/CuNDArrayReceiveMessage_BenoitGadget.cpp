#include "CuNDArrayReceiveMessage_BenoitGadget.h"
//#include "test_gpu_benoit.h"

namespace Gadgetron{



int CuNDArrayReceiveMessage_BenoitGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
	                                 GadgetContainerMessage< cuNDArray< complext<float> > >* m2)
{
  cuNDArray< complext<float> > *data_d = m2->getObjectPtr();

  //hoNDArray<std::complex<float> > data_h;
  //(reinterpret_cast<hoNDArray< std::complex<float> > *>(data_d));

  boost::shared_ptr< hoNDArray<float_complext  > >  host_result_out = data_d->to_host();
  hoNDArray<std::complex<float>> *data_h=reinterpret_cast<hoNDArray<std::complex<float> >*>(host_result_out.get());

  //data_h = data_d->to_host();

  std::vector<size_t> dimensions = *(m2->getObjectPtr()->get_dimensions());

  GadgetContainerMessage< hoNDArray<std::complex<float> >> *m3 = new GadgetContainerMessage<hoNDArray<std::complex<float>>>();
  try{m3->getObjectPtr()->create(dimensions);}
  catch (std::runtime_error &err){
  	GEXCEPTION(err,"CombineGadget, failed to allocate new array\n");
    return -1;
  }



  cudaMemcpy(m3->getObjectPtr()->get_data_ptr(), m2->getObjectPtr()->get_data_ptr(), m2->getObjectPtr()->get_number_of_elements() * sizeof(complext<float>), cudaMemcpyDeviceToHost);

  GDEBUG_STREAM("Data content end: " << (*(data_h))[0] << ", " << (*(data_h))[1] << ", " << (*(data_h))[2] << ", " << (*(data_h))[3]);

  m1->cont(m3);
  m2->release();
  if (this->next()->putq(m1) == -1) {  
    m1->release();
    GERROR("AcquisitionPassthroughGadget::process, passing data on to next gadget");
    return -1;
  }

  return 0;
}
GADGET_FACTORY_DECLARE(CuNDArrayReceiveMessage_BenoitGadget)
}



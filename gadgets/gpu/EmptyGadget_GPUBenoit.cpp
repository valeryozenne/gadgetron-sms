#include "EmptyGadget_GPUBenoit.h"
//#include "test_gpu_benoit.h"
//#include "cuNDArray_fileio.h"
#include "cuNDArray.h"

namespace Gadgetron{



int EmptyGadget_GPUBenoit::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
	                                 GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
{
  //It is enough to put the first one, since they are linked

   // std::cout << " coucou" << std::endl;

  //1) créer un hondarray de complex<float>
  hoNDArray<std::complex<float>  >data_in(128, 128);
  hoNDArray<std::complex<float>  >data_out(128, 128);

  data_in.fill(0);
  data_out.fill(56);

  //solution 1 pour créer un cuNDArray
  // hoNDArray<std::complex<float> > host_data
  // host_data = reinterpret_cast< hoNDArray<float_complext> * >(&data);

  // cuNDArray<float_complext> device_data(host_data);

  // //solution 2 pour créer un cuNDArray
  // /*
  // **cuNDArray<complext<float>> k_space_data(reinterpret_cast<const hoNDArray<complext<float>> &>(data));
  // */

  // //Retour du device GPu ver l'host CPU
  // boost::shared_ptr< hoNDArray<float_complext  > >  host_result_out = device_data.to_host();
  // hoNDArray<std::complex<float>> *host_result_cast=reinterpret_cast<hoNDArray<std::complex<float> >*>(host_result_out.get());

  //Solution alternative pour conversion CPU-GPU

  
  //cudaSetDevice(0);
  //GDEBUG("Device set");
  cuNDArray< complext<float> > device_mb(128, 128);
  cuNDArray<float_complext> device_mb_out(128, 128);

  

  std::complex<float> *in = &(data_in(0, 0));
  std::complex<float> *out = &(data_out(0, 0));

  if(cudaMemcpy(device_mb.get_data_ptr(),
                          in,//autre solution : data->get_data_ptr()
                          128*128*sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice)!= cudaSuccess )   {
                GERROR_STREAM("Upload to device for device_mb failed\n");}

  CHECK_FOR_CUDA_ERROR();


  // create_and_copy_cuNDArray_benoit(device_mb, device_mb_out);

  // if(cudaMemcpy(out,
  //               device_mb_out.get_data_ptr(),
  //               128*128*sizeof(std::complex<float>),
  //               cudaMemcpyDeviceToHost)!= cudaSuccess )   {
  //   GERROR_STREAM("Upload to host for device_unfolded failed\n");}




  if (this->next()->putq(m1) == -1) {
    m1->release();
    GERROR("AcquisitionPassthroughGadget::process, passing data on to next gadget");
    return -1;
  }

  return 0;
}
GADGET_FACTORY_DECLARE(EmptyGadget_GPUBenoit)
}



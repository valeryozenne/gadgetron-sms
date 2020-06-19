#include "EmptyGadget_GPUBenoit.h"
#include "test_gpu_benoit.h"

namespace Gadgetron{



int EmptyGadget_GPUBenoit::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
	                                 GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
{
  //It is enough to put the first one, since they are linked

   // std::cout << " coucou" << std::endl;

  //1) créer un hondarray de complex<float>
  hoNDArray<std::complex<float>  >data, data2;

  data.create(128, 128);
  data.fill(0);
  data2.create(128, 128);
  data2.fill(56);
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

  

  cuNDArray<float_complext> device_mb(128, 128);
  cuNDArray<float_complext> device_mb_out(128, 128);

  

  std::complex<float> *in = &(data(0, 0));
  std::complex<float> *out = &(data2(0, 0));

  if(cudaMemcpy(device_mb.get_data_ptr(),
                          in,//autre solution : data->get_data_ptr()
                          128*128*sizeof(std::complex<float>),
                          cudaMemcpyHostToDevice)!= cudaSuccess )   {
                GERROR_STREAM("Upload to device for device_mb failed\n");}

  GDEBUG_STREAM("BEFORE CALLING GPU FUNCTION");
  GDEBUG_STREAM("IN: " << in[0] << ", " << in[1] << ", " << in[1] << ", " << in[2] << ", " << in[3] << ", " << in[4] << ", " << in[5] << ", " << in[6] << ", " << in[7] << ", " << in[8] << ", " << in[9]);
  GDEBUG_STREAM("OUT: " << out[0] << ", " << out[1] << ", " << out[1] << ", " << out[2] << ", " << out[3] << ", " << out[4] << ", " << out[5] << ", " << out[6] << ", " << out[7] << ", " << out[8] << ", " << out[9] << std::endl)

  create_and_copy_cuNDArray_benoit(device_mb, device_mb_out);

  

  if(cudaMemcpy(out,
                device_mb_out.get_data_ptr(),
                128*128*sizeof(std::complex<float>),
                cudaMemcpyDeviceToHost)!= cudaSuccess )   {
    GERROR_STREAM("Upload to host for device_unfolded failed\n");}

  GDEBUG_STREAM("AFTER CALLING GPU FUNCTION");
  GDEBUG_STREAM("IN: " << in[0] << ", " << in[1] << ", " << in[1] << ", " << in[2] << ", " << in[3] << ", " << in[4] << ", " << in[5] << ", " << in[6] << ", " << in[7] << ", " << in[8] << ", " << in[9]);
  GDEBUG_STREAM("OUT: " << out[0] << ", " << out[1] << ", " << out[1] << ", " << out[2] << ", " << out[3] << ", " << out[4] << ", " << out[5] << ", " << out[6] << ", " << out[7] << ", " << out[8] << ", " << out[9] << std::endl)



  if (this->next()->putq(m1) == -1) {
    m1->release();
    GERROR("AcquisitionPassthroughGadget::process, passing data on to next gadget");
    return -1;
  }

  return 0;
}
GADGET_FACTORY_DECLARE(EmptyGadget_GPUBenoit)
}



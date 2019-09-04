//CuNDArrayTohoNDArrayGadget.cpp

#include "cuNDArray.h"
#include "hoCuNDArray.h"
#include "hoNDArray_utils.h"
#include "cuNDArray_utils.h"
#include "GenericReconBase.h"
#include "CuNDArrayTohoNDArrayGadget.h"

using namespace Gadgetron;

int CuNDArrayTohoNDArrayGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
{
    IsmrmrdReconData* recon_bit_ = m1->getObjectPtr();
       size_t e;

    for (e = 0; e < recon_bit_->rbit_.size(); e++)
    {
        auto & rbit = recon_bit_->rbit_[e];
        std::stringstream os;
        os << "_encoding_" << e;

        if (recon_bit_->rbit_[e].data_.data_.get_number_of_elements() > 0)
        {

            hoNDArray< std::complex<float> >& gpu_data = recon_bit_->rbit_[e].data_.data_;
            hoNDArray< std::complex<float> > new_data;
            if (gpu_data.get_number_of_dimensions() == 8) {
            new_data.create(gpu_data.get_size(0), gpu_data.get_size(1), gpu_data.get_size(2), gpu_data.get_size(3), gpu_data.get_size(4), gpu_data.get_size(5), gpu_data.get_size(6), gpu_data.get_size(7));
            }

            //7 dimensions
            else {
                new_data.create(gpu_data.get_size(0), gpu_data.get_size(1), gpu_data.get_size(2), gpu_data.get_size(3), gpu_data.get_size(4), gpu_data.get_size(5), gpu_data.get_size(6));
            }
            int elements = gpu_data.get_number_of_elements();
            for (int i = 0; i < elements; i++) {
                new_data[i] = gpu_data[i];
            }

            m1->getObjectPtr()->rbit_[e].data_.data_ = new_data;
        }
    }


  //Now pass on image
  if (this->next()->putq(m1) < 0) {
    return GADGET_FAIL;
  }

  return GADGET_OK;
}

GADGET_FACTORY_DECLARE(CuNDArrayTohoNDArrayGadget)

//hoNDArrayToCuNDArrayGadget.cpp

#include "cuNDArray.h"
#include "hoCuNDArray.h"
#include "hoNDArray_utils.h"
#include "cuNDArray_utils.h"
#include "GenericReconBase.h"
#include "hoNDArrayToCuNDArrayGadget.h"

using namespace Gadgetron;

int hoNDArrayToCuNDArrayGadget::process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1)
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
            hoNDArray< std::complex<float> >& data = recon_bit_->rbit_[e].data_.data_;
            hoCuNDArray< std::complex<float> > gpu_data(data);

            m1->getObjectPtr()->rbit_[e].data_.data_ = gpu_data;
        }
    }


  //Now pass on image
  if (this->next()->putq(m1) < 0) {
    return GADGET_FAIL;
  }

  return GADGET_OK;
}

GADGET_FACTORY_DECLARE(hoNDArrayToCuNDArrayGadget)

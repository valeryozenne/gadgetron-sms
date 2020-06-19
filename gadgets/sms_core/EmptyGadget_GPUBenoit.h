#ifndef EMPTYGADGET_GPUBENOIT_H
#define EMPTYGADGET_GPUBENOIT_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_smscore_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE EmptyGadget_GPUBenoit :
  public Gadget2<ISMRMRD::AcquisitionHeader,hoNDArray< std::complex<float> > >
    {
    public:
      GADGET_DECLARE(EmptyGadget_GPUBenoit);
      
    protected:
      virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
			  GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);
    };
}
#endif //EMPTYGADGET_GPUBENOIT_H

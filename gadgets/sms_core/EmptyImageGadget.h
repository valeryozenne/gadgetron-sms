#ifndef ACQUISITIONPASSTHROUGHGADGET_H
#define ACQUISITIONPASSTHROUGHGADGET_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_smscore_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE EmptyImageGadget :
  public Gadget2<ISMRMRD::AcquisitionImage,hoNDArray< std::complex<float> > >
    {
    public:
      GADGET_DECLARE(EmptyImageGadget);
      
    protected:
      virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionImage>* m1,
			  GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);
    };
}
#endif //ACQUISITIONPASSTHROUGHGADGET_H

#ifndef EMPTYGADGET_GPUBENOIT_H
#define EMPTYGADGET_GPUBENOIT_H

#include "Gadget.h"
#include "hoNDArray.h"


#include <ismrmrd/ismrmrd.h>
#include <complex>

#if defined (WIN32)
#if defined (__BUILD_GADGETRON_GPUGADGET__)
#define EXPORTGPUGADGET __declspec(dllexport)
#else
#define EXPORTGPUGADGET __declspec(dllimport)
#endif
#else
#define EXPORTGPUGADGET
#endif

namespace Gadgetron{

  class EXPORTGPUGADGET EmptyGadget_GPUBenoit :
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

#ifndef CUNDARRAYSENDMESSAGE_BENOITGADGET
#define CUNDARRAYSENDMESSAGE_BENOITGADGET

#include "Gadget.h"
#include "hoNDArray.h"
#include "cuNDArray.h"
#include "gadgetron_smscore_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE CuNDArraySendMessage_BenoitGadget :
  public Gadget2<ISMRMRD::AcquisitionHeader,hoNDArray< std::complex<float> > >
    {
    public:
      GADGET_DECLARE(CuNDArraySendMessage_BenoitGadget);
      
    protected:
      virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
			  GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);
    };
}
#endif //CUNDARRAYSENDMESSAGE_BENOITGADGET

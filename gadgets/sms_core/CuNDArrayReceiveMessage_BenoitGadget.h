#ifndef CUNDARRAYRECEIVEMESSAGE_BENOITGADGET
#define CUNDARRAYRECEIVEMESSAGE_BENOITGADGET

#include "Gadget.h"
#include "hoNDArray.h"
#include "cuNDArray.h"
#include "gadgetron_smscore_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE CuNDArrayReceiveMessage_BenoitGadget :
  public Gadget2<ISMRMRD::AcquisitionHeader,cuNDArray< complext<float> > >
    {
    public:
      GADGET_DECLARE(CuNDArrayReceiveMessage_BenoitGadget);
      
    protected:
      virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
			  GadgetContainerMessage< cuNDArray< complext<float> > >* m2);
    };
}
#endif //CUNDARRAYRECEIVEMESSAGE_BENOITGADGET

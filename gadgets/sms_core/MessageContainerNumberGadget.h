#ifndef AUTOSCALEGADGET_H_
#define AUTOSCALEGADGET_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_smscore_export.h"

#include <ismrmrd/ismrmrd.h>

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE MessageContainerNumberGadget:
    public Gadget2<ISMRMRD::ImageHeader,hoNDArray< float> >
  {
  public:
    GADGET_DECLARE(MessageContainerNumberGadget);

    MessageContainerNumberGadget();
    virtual ~MessageContainerNumberGadget();

  protected:
    GADGET_PROPERTY(max_value, float, "Maximum value (after scaling)", 2048);
    GADGET_PROPERTY(avant, bool, "Whether to average all N for ref generation", true);

    virtual int process(GadgetContainerMessage<ISMRMRD::ImageHeader>* m1,
            GadgetContainerMessage< hoNDArray< float > >* m2);
    virtual int process_config(ACE_Message_Block *mb);

    float max_value_;
  };
}

#endif /* AUTOSCALEGADGET_H_ */

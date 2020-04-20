#ifndef MULTIPLYMESSAGECONTAINERGADGET_H_
#define MULTIPLYMESSAGECONTAINERGADGET_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_smscore_export.h"

#include <ismrmrd/ismrmrd.h>

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE MultiplyMessageContainerGadget:
    public Gadget2<ISMRMRD::ImageHeader,hoNDArray< float > >
  {
  public:
    GADGET_DECLARE(MultiplyMessageContainerGadget);

    MultiplyMessageContainerGadget();
    virtual ~MultiplyMessageContainerGadget();

  protected:
    GADGET_PROPERTY(max_value, float, "Maximum value (after scaling)", 2048);

    virtual int process(GadgetContainerMessage<ISMRMRD::ImageHeader>* m1,
            GadgetContainerMessage< hoNDArray< float> >* m2);
    virtual int process_config(ACE_Message_Block *mb);

    unsigned int histogram_bins_;
    std::vector<size_t> histogram_;
    float current_scale_;
    float max_value_;
  };
}

#endif /* MULTIPLYMESSAGECONTAINERGADGET_H_ */

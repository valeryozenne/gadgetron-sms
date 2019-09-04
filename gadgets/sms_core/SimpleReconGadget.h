#ifndef SIMPLERECONGADGET_H
#define SIMPLERECONGADGET_H

#include "Gadget.h"
#include "gadgetron_smscore_export.h"

#include "mri_core_data.h"

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE SimpleReconGadget :
  public Gadget1<IsmrmrdReconData>
    {
    public:
      GADGET_DECLARE(SimpleReconGadget)
      SimpleReconGadget();
	
    protected:
      virtual int process(GadgetContainerMessage<IsmrmrdReconData>* m1);
      long long image_counter_;
    };
}
#endif //SIMPLERECONGADGET_H

#ifndef IMAGEARRAYSPLIT_H
#define IMAGEARRAYSPLIT_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_smscore_export.h"
#include "mri_core_def.h"
#include "mri_core_data.h"

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE ImageArrayEmptyGadget :
  public Gadget1Of2<IsmrmrdImageArray, ISMRMRD::ImageHeader >
    {
    public:
      GADGET_DECLARE(ImageArrayEmptyGadget)
      ImageArrayEmptyGadget();
	
    protected:
      virtual int process(GadgetContainerMessage<IsmrmrdImageArray>* m1);
      virtual int process(GadgetContainerMessage<ISMRMRD::ImageHeader>* m1);

      bool setISMRMRMetaValues(ISMRMRD::MetaContainer& attrib, const std::string& name, const std::vector<double>& v);
    };
}
#endif //IMAGEARRAYSPLIT_H

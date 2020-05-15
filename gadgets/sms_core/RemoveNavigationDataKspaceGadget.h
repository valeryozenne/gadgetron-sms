#ifndef RemoveNavigationDataKspaceGadget_H_
#define RemoveNavigationDataKspaceGadget_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_smscore_export.h"
#include "hoArmadillo.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

class EXPORTGADGETSSMSCORE RemoveNavigationDataKspaceGadget :
  public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
    {
    public:
      GADGET_DECLARE(RemoveNavigationDataKspaceGadget);
      
      RemoveNavigationDataKspaceGadget();
      virtual ~RemoveNavigationDataKspaceGadget();
      
      virtual int process_config(ACE_Message_Block* mb);
      virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
			  GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);
      
    protected:


    };
}
#endif /* COILREDUCTIONGADGET_H_ */

#ifndef hoNDArrayToCuNDArrayGadget_H
#define hoNDArrayToCuNDArrayGadget_H

#include "gadgetron_examplelib_export.h"
#include "Gadget.h"
#include "GadgetMRIHeaders.h"
#include "hoNDArray.h"
#include <complex>
#include <ismrmrd/ismrmrd.h>

namespace Gadgetron
{

class EXPORTGADGETSEXAMPLELIB hoNDArrayToCuNDArrayGadget : public GenericReconDataBase
{
public:
  GADGET_DECLARE(hoNDArrayToCuNDArrayGadget)

    typedef GenericReconDataBase BaseClass;
    typedef hoNDKLT< std::complex<float> > KLTType;

    hoNDArrayToCuNDArrayGadget();
    ~hoNDArrayToCuNDArrayGadget();

  protected:
    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
};

}
#endif //hoNDArrayToCuNDArrayGadget_H

#ifndef CuNDArrayTohoNDArrayGadget_H
#define CuNDArrayTohoNDArrayGadget_H

#include "gadgetron_examplelib_export.h"
#include "Gadget.h"
#include "GadgetMRIHeaders.h"
#include "hoNDArray.h"
#include <complex>
#include <ismrmrd/ismrmrd.h>

namespace Gadgetron
{

class EXPORTGADGETSEXAMPLELIB CuNDArrayTohoNDArrayGadget : public GenericReconDataBase
{
public:
  GADGET_DECLARE(CuNDArrayTohoNDArrayGadget)

    typedef GenericReconDataBase BaseClass;
    typedef hoNDKLT< std::complex<float> > KLTType;

    CuNDArrayTohoNDArrayGadget();
    ~CuNDArrayTohoNDArrayGadget();

  protected:
    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(Gadgetron::GadgetContainerMessage< IsmrmrdReconData >* m1);
};

}
#endif //CuNDArrayTohoNDArrayGadget_H

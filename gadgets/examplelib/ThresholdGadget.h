#ifndef THRESHOLDGADGET_H
#define THRESHOLDGADGET_H

#include "gadgetron_examplelib_export.h"
#include "Gadget.h"
#include "GadgetMRIHeaders.h"
#include "hoNDArray.h"
#include <complex>
#include <ismrmrd/ismrmrd.h>

namespace Gadgetron
{

class EXPORTGADGETSEXAMPLELIB ThresholdGadget :
public Gadget2<ISMRMRD::ImageHeader, hoNDArray< std::complex<float> > >
{
  public:
    GADGET_DECLARE(ThresholdGadget)

    GADGET_PROPERTY(level, double, "Threshold level", 1.0);

  protected:
    virtual int process( GadgetContainerMessage< ISMRMRD::ImageHeader>* m1,
                         GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);
};

}
#endif //THRESHOLDGADGET_H

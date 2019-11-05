#ifndef IFFTXGadget_H
#define IFFTXGadget_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_epi_liryc_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

  class   EXPORTGADGETS_EPI_LIRYC IFFTXGadget :
  public Gadget2<ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
  {
    public:
      IFFTXGadget();
      virtual ~IFFTXGadget();

    protected:
      virtual int process( GadgetContainerMessage< ISMRMRD::AcquisitionHeader>* m1,
                       GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);

      hoNDArray< std::complex<float> > r_;
      hoNDArray< std::complex<float> > buf_;
  };
}
#endif //IFFTXGadget_H

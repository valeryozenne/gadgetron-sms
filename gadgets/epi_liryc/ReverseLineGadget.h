#ifndef ReverseLineGadget_H
#define ReverseLineGadget_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_epi_liryc_export.h"
#include "hoArmadillo.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>


namespace Gadgetron{

  class EXPORTGADGETS_EPI_LIRYC ReverseLineGadget :
  public Gadget2<ISMRMRD::AcquisitionHeader,hoNDArray< std::complex<float> > >
    {
    public:
      ReverseLineGadget();
      virtual ~ReverseLineGadget();

    protected:
      GADGET_PROPERTY(verboseMode, bool, "Verbose output", false);

      virtual int process_config(ACE_Message_Block* mb);
      virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
              GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);

      // in verbose mode, more info is printed out
      bool verboseMode_;

      unsigned int number_of_channels;

      unsigned int readout;

      unsigned int reconNx_;
    };
}
#endif //ReverseLineGadget_H

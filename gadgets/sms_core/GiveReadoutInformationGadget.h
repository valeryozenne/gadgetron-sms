#ifndef ACQUISITIONPASSTHROUGHGADGET_H
#define ACQUISITIONPASSTHROUGHGADGET_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_smscore_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

  class EXPORTGADGETSSMSCORE GiveReadoutInformationGadget :  public Gadget2<ISMRMRD::AcquisitionHeader,hoNDArray< std::complex<float> > >
    {
    public:
      GADGET_DECLARE(GiveReadoutInformationGadget);

      GiveReadoutInformationGadget();
      ~GiveReadoutInformationGadget();

    protected:
         virtual int process_config(ACE_Message_Block* mb);
      virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
			  GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);

        void display_header_information(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1, bool is_calib, bool is_phase_corr);

    };
}
#endif //ACQUISITIONPASSTHROUGHGADGET_H

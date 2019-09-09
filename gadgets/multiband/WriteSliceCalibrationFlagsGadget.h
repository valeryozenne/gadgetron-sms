#ifndef WriteSliceCalibrationFlagsGadget_H_
#define WriteSliceCalibrationFlagsGadget_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_multiband_export.h"
#include "hoArmadillo.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

class EXPORTGADGETSMULTIBAND WriteSliceCalibrationFlagsGadget :
        public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
{
public:
    GADGET_DECLARE(WriteSliceCalibrationFlagsGadget);

    WriteSliceCalibrationFlagsGadget();
    virtual ~WriteSliceCalibrationFlagsGadget();

    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
                        GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);

protected:

    std::vector<size_t> dimensions_;

    unsigned int lNumberOfSlices_;
    unsigned int lNumberOfAverage_;
    unsigned int lNumberOfStacks_;
    unsigned int lNumberOfChannels_;

    unsigned int encoding;

    arma::fmat deja_vu;
    arma::fmat deja_vu_epi_calib;

};
}
#endif /* WRITESLICECALIBRATIONFLAGS_H_ */

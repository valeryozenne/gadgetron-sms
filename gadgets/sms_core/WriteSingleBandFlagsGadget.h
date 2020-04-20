#ifndef ACQUISITIONPASSTHROUGHGADGET_H
#define ACQUISITIONPASSTHROUGHGADGET_H

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_smscore_export.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

class EXPORTGADGETSSMSCORE WriteSingleBandFlagsGadget : public Gadget2<ISMRMRD::AcquisitionHeader,hoNDArray< std::complex<float> > >
{
public:
    GADGET_DECLARE(WriteSingleBandFlagsGadget);

    WriteSingleBandFlagsGadget();
    virtual ~WriteSingleBandFlagsGadget();


protected:

    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
                        GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);




    unsigned int e1;
    unsigned int e2;
    unsigned int slice;
    unsigned int repetition;
    unsigned int set;
    unsigned int segment;
    unsigned int phase;
    unsigned int average;
    unsigned int user;

    // encoding space size
    ISMRMRD::EncodingCounters meas_size_idx_;

    size_t num_encoding_spaces_;

    hoNDArray<int> matrix_deja_vu_data_;
    hoNDArray<int> matrix_deja_vu_epi_nav_;


};
}
#endif //ACQUISITIONPASSTHROUGHGADGET_H

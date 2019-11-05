#ifndef WriteSingleBandFlagsGadget_H_
#define WriteSingleBandFlagsGadget_H_

#include "Gadget.h"
#include "hoNDArray.h"
#include "gadgetron_mricore_export.h"
#include "hoArmadillo.h"

#include <ismrmrd/ismrmrd.h>
#include <complex>

namespace Gadgetron{

class EXPORTGADGETSMRICORE WriteSingleBandFlagsGadget :
        public Gadget2< ISMRMRD::AcquisitionHeader, hoNDArray< std::complex<float> > >
{
public:
    GADGET_DECLARE(WriteSingleBandFlagsGadget);

    WriteSingleBandFlagsGadget();
    virtual ~WriteSingleBandFlagsGadget();

    virtual int process_config(ACE_Message_Block* mb);
    virtual int process(GadgetContainerMessage< ISMRMRD::AcquisitionHeader >* m1,
                        GadgetContainerMessage< hoNDArray< std::complex<float> > > * m2);

protected:

    std::vector<size_t> dimensions_;

    unsigned int lNumberOfSlices_;
    unsigned int lNumberOfAverage_;
    unsigned int lNumberOfStacks_;
    unsigned int lNumberOfChannels_;
    unsigned int lNumberOfSegments_;
    unsigned int lNumberOfSets_;
    unsigned int lNumberOfPhases_;    

    unsigned int e1;
    unsigned int e2;
    unsigned int slice;
    unsigned int repetition;
    unsigned int set;
    unsigned int segment;
    unsigned int phase;
    unsigned int average;
    unsigned int user;



    size_t num_encoding_spaces_;

    unsigned int encoding;

    arma::fmat deja_vu;
    arma::fmat deja_vu_epi_calib;

    hoNDArray<int> matrix_deja_vu_data_;
    hoNDArray<int> matrix_deja_vu_epi_nav_;

};
}
#endif /* WRITESLICECALIBRATIONFLAGS_H_ */

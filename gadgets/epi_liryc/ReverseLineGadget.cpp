#include "ReverseLineGadget.h"
#include "ismrmrd/xml.h"
#include "mri_core_utility_interventional.h"

#ifdef USE_OMP
#include "omp.h"
#endif // USE_OMP

namespace Gadgetron{

ReverseLineGadget::ReverseLineGadget() {}
ReverseLineGadget::~ReverseLineGadget() {}

int ReverseLineGadget::process_config(ACE_Message_Block* mb)
{
    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);

    if (h.encoding.size() == 0) {
        GDEBUG("Number of encoding spaces: %d\n", h.encoding.size());
        GDEBUG("This Gadget needs an encoding description\n");
        return GADGET_FAIL;
    }

    //GDEBUG("Number of encoding spaces = %d\n", h.encoding.size());

    // Get the encoding space and trajectory description
    ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
    ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
    ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
    ISMRMRD::TrajectoryDescription traj_desc;

    reconNx_= r_space.matrixSize.x ;

    number_of_channels = h.acquisitionSystemInformation->receiverChannels ? *h.acquisitionSystemInformation->receiverChannels : 1;;

    return GADGET_OK;
}

int ReverseLineGadget::process(
        GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
        GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2)
{

    // Get a reference to the acquisition header
    ISMRMRD::AcquisitionHeader &hdr = *m1->getObjectPtr();

    unsigned int e1= m1->getObjectPtr()->idx.kspace_encode_step_1;

    arma::cx_fvec data= as_arma_col(*m2->getObjectPtr());

    if (hdr.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_REVERSE))
    {
        readout=size(data,0)/number_of_channels;
        arma::cx_fmat tempo = reshape(data,readout,number_of_channels);
        arma::cx_fmat inverse_tempo=flipud(tempo);
        data=vectorise(inverse_tempo);
    }

    // It is enough to put the first one, since they are linked
    if (this->next()->putq(m1) == -1) {
        m1->release();
        GERROR("ReverseLineGadget::process, passing data on to next gadget");
        return -1;
    }

    return GADGET_OK;
}

GADGET_FACTORY_DECLARE(ReverseLineGadget)
}


